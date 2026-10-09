
import io
from pathlib import Path
from datetime import date, timedelta

import cv2
import folium
import numpy as np
import rasterio
import requests
import streamlit as st
import torch
from PIL import Image
from rasterio.io import MemoryFile
from streamlit_folium import st_folium

from tiny_unet import TinyUNet


st.set_page_config(
    page_title="VEYORA | Satellite Intelligence",
    page_icon="🛰️",
    layout="wide"
)

st.title("🛰️ VEYORA")
st.subheader("Live Sentinel-1 Oil-Spill Analysis")

TOKEN_URL = (
    "https://identity.dataspace.copernicus.eu/"
    "auth/realms/CDSE/protocol/openid-connect/token"
)
PROCESS_URL = "https://sh.dataspace.copernicus.eu/api/v1/process"

MODEL_PATH = Path(__file__).parent / "best_model.pth"


@st.cache_data(ttl=3000, show_spinner=False)
def get_access_token(client_id, client_secret):
    response = requests.post(
        TOKEN_URL,
        data={
            "grant_type": "client_credentials",
            "client_id": client_id,
            "client_secret": client_secret,
        },
        timeout=30,
    )
    response.raise_for_status()
    return response.json()["access_token"]


def fetch_sentinel1(bbox, start_date, end_date, token):
    # Output channels are VV and VH in linear power.
    evalscript = """
    //VERSION=3
    function setup() {
      return {
        input: [{
          bands: ["VV", "VH"],
          units: "LINEAR_POWER"
        }],
        output: {
          bands: 2,
          sampleType: "FLOAT32"
        }
      };
    }
    function evaluatePixel(s) {
      return [s.VV, s.VH];
    }
    """

    payload = {
        "input": {
            "bounds": {
                "bbox": bbox,
                "properties": {
                    "crs": "http://www.opengis.net/def/crs/EPSG/0/4326"
                }
            },
            "data": [{
                "type": "sentinel-1-grd",
                "dataFilter": {
                    "timeRange": {
                        "from": f"{start_date}T00:00:00Z",
                        "to": f"{end_date}T23:59:59Z",
                    },
                    "acquisitionMode": "IW",
                    "polarization": "DV",
                },
                "processing": {
                    "orthorectify": True,
                    "backCoeff": "SIGMA0_ELLIPSOID",
                }
            }]
        },
        "output": {
            "width": 512,
            "height": 512,
            "responses": [{
                "identifier": "default",
                "format": {"type": "image/tiff"}
            }]
        },
        "evalscript": evalscript
    }

    response = requests.post(
        PROCESS_URL,
        headers={"Authorization": f"Bearer {token}"},
        json=payload,
        timeout=180,
    )

    if not response.ok:
        raise RuntimeError(
            f"Copernicus HTTP {response.status_code}: "
            f"{response.text[:1000]}"
        )

    with MemoryFile(response.content) as memfile:
        with memfile.open() as src:
            bands = src.read().astype(np.float32)
            profile = {
                "crs": str(src.crs) if src.crs else None,
                "transform": tuple(src.transform),
                "width": src.width,
                "height": src.height,
                "bounds": tuple(src.bounds),
            }

    if bands.shape[0] != 2:
        raise ValueError(f"Expected VV/VH bands, got {bands.shape}")

    return bands, profile


@st.cache_resource
def load_model(model_path_string, model_mtime):
    model = UNet(in_channels=2, base=16)
    state = torch.load(
        model_path_string, map_location="cpu", weights_only=True
    )
    model.load_state_dict(state)
    model.eval()
    return model


def to_db(bands):
    # Copernicus returns linear power; training uses dB.
    safe = np.maximum(bands, 1e-10)
    return 10.0 * np.log10(safe)


def prepare_model_input(bands_db):
    # Same normalization as train.py.
    normalized = np.clip((bands_db + 35.0) / 40.0, 0, 1)

    # Training uses 256x256 patches.
    resized = np.stack([
        cv2.resize(
            band, (256, 256), interpolation=cv2.INTER_AREA
        )
        for band in normalized
    ]).astype(np.float32)

    return torch.from_numpy(resized[None]).float()


def predict_with_model(bands_db):
    model = load_model(str(MODEL_PATH), MODEL_PATH.stat().st_mtime)
    tensor = prepare_model_input(bands_db)

    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.sigmoid(logits)[0, 0].numpy()

    # Return mask at original requested raster size.
    h, w = bands_db.shape[1:]
    probabilities = cv2.resize(
        probabilities, (w, h), interpolation=cv2.INTER_LINEAR
    )
    return probabilities


def dark_target_fallback(vv_db, threshold_db, min_pixels):
    valid = np.isfinite(vv_db)
    clean = np.where(valid, vv_db, 0).astype(np.float32)
    smooth = cv2.GaussianBlur(clean, (5, 5), 0)

    binary = ((smooth < threshold_db) & valid).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )

    mask = np.zeros_like(binary)
    for i in range(1, count):
        if stats[i, cv2.CC_STAT_AREA] >= min_pixels:
            mask[labels == i] = 1

    return mask


def display_rgb(vv_db):
    valid = np.isfinite(vv_db)
    if not valid.any():
        return np.zeros((*vv_db.shape, 3), dtype=np.uint8)

    lo, hi = np.percentile(vv_db[valid], [2, 98])
    gray = np.clip(
        (np.nan_to_num(vv_db, nan=lo) - lo)
        / max(hi - lo, 1e-6) * 255,
        0, 255,
    ).astype(np.uint8)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)


with st.sidebar:
    st.header("Satellite search")

    region = st.selectbox(
        "Region", ["Chennai Coast", "Gulf of Mexico", "Custom"]
    )

    defaults = {
        "Chennai Coast": [80.0, 12.5, 81.0, 13.5],
        "Gulf of Mexico": [-92.0, 25.0, -90.0, 27.0],
        "Custom": [80.0, 12.5, 81.0, 13.5],
    }[region]

    west = st.number_input("West longitude", value=float(defaults[0]))
    south = st.number_input("South latitude", value=float(defaults[1]))
    east = st.number_input("East longitude", value=float(defaults[2]))
    north = st.number_input("North latitude", value=float(defaults[3]))

    today = date.today()
    end_date = st.date_input("End date", value=today)
    start_date = st.date_input(
        "Start date", value=today - timedelta(days=14)
    )

    st.header("Screening settings")
    threshold_db = st.slider(
        "Fallback VV threshold (dB)",
        -35.0, -5.0, -18.0, 0.5
    )
    min_pixels = st.slider(
        "Minimum fallback region (pixels)", 5, 500, 25
    )

    run = st.button("Fetch Sentinel-1", type="primary")


if run:
    if not (-180 <= west < east <= 180):
        st.error("Invalid longitude bounds.")
        st.stop()

    if not (-90 <= south < north <= 90):
        st.error("Invalid latitude bounds.")
        st.stop()

    if start_date > end_date:
        st.error("Start date must not be after end date.")
        st.stop()

    if east - west > 3 or north - south > 3:
        st.error("Choose an area no larger than 3 degrees per side.")
        st.stop()

    try:
        client_id = st.secrets["CDSE_CLIENT_ID"]
        client_secret = st.secrets["CDSE_CLIENT_SECRET"]
    except Exception:
        st.error("Set CDSE_CLIENT_ID and CDSE_CLIENT_SECRET in Secrets.")
        st.stop()

    try:
        with st.spinner("Fetching live Sentinel-1 VV/VH data..."):
            token = get_access_token(client_id, client_secret)
            bands, profile = fetch_sentinel1(
                [west, south, east, north],
                start_date.isoformat(),
                end_date.isoformat(),
                token,
            )

        st.session_state["bands"] = bands
        st.session_state["profile"] = profile
        st.session_state["bbox"] = [west, south, east, north]
        st.success("Satellite data received.")

    except Exception as exc:
        st.error(f"Satellite request failed: {exc}")


if "bands" not in st.session_state:
    st.info("Select an area/date and click Fetch Sentinel-1.")
    st.stop()


bands = st.session_state["bands"]
profile = st.session_state["profile"]
bbox = st.session_state["bbox"]

bands_db = to_db(bands)
vv_db = bands_db[0]

st.markdown("### 1. Live satellite imagery")
left, right = st.columns(2)

base_rgb = display_rgb(vv_db)
with left:
    st.image(base_rgb, caption="Sentinel-1 VV intensity (dB)")

# Choose AI only when trained weights are available.
if MODEL_PATH.exists():
    try:
        with st.spinner("Running trained U-Net..."):
            probabilities = predict_with_model(bands_db)
            mask = (probabilities >= 0.5).astype(np.uint8)
        mode = "Trained U-Net segmentation"
        st.success("Trained model loaded and inference completed.")
    except Exception as exc:
        st.warning(f"Model inference failed: {exc}")
        st.info("Using the clearly labelled screening fallback.")
        mask = dark_target_fallback(
            vv_db, threshold_db, min_pixels
        )
        probabilities = None
        mode = "Dark-target screening fallback"
else:
    mask = dark_target_fallback(vv_db, threshold_db, min_pixels)
    probabilities = None
    mode = "Dark-target screening fallback"

with right:
    overlay = base_rgb.copy()
    overlay[mask == 1] = [255, 40, 40]
    overlay = cv2.addWeighted(base_rgb, 0.65, overlay, 0.35, 0)
    st.image(overlay, caption=f"Red candidates — {mode}")

if probabilities is None:
    st.warning(
        "SCREENING MODE: this is a dark-pixel heuristic, not AI. "
        "Dark features can be caused by low wind, natural films, "
        "or other radar effects."
    )
else:
    st.warning(
        "AI candidate mask: not confirmation of oil. "
        "Validate with independent labelled scenes and human review."
    )
    st.image(
        np.clip(probabilities * 255, 0, 255).astype(np.uint8),
        caption="U-Net probability map (0–255)",
    )

pixels = int(mask.sum())
total = mask.size
coverage = 100 * pixels / max(total, 1)

m1, m2, m3 = st.columns(3)
m1.metric("Positive pixels", f"{pixels:,}")
m2.metric("Image coverage", f"{coverage:.2f}%")
m3.metric("Analysis mode", mode)

# This map displays the requested bounding box, not exact mask polygons.
st.markdown("### 2. Region reference")
st.caption(
    "The requested bounding box is shown below. Individual mask pixels "
    "are not yet transformed into geographic polygons."
)

m = folium.Map(
    location=[(bbox[1] + bbox[3]) / 2, (bbox[0] + bbox[2]) / 2],
    zoom_start=7,
)
folium.Rectangle(
    bounds=[[bbox[1], bbox[0]], [bbox[3], bbox[2]]],
    color="cyan",
    fill=False,
).add_to(m)
st_folium(m, height=400, use_container_width=True)

mask_bytes = io.BytesIO()
Image.fromarray(mask * 255).save(mask_bytes, format="PNG")
st.download_button(
    "Download predicted mask",
    mask_bytes.getvalue(),
    file_name="veyora_mask.png",
    mime="image/png",
)

st.caption(
    "VEYORA research prototype • Copernicus Sentinel-1 • "
    "No operational oil-spill alert should be based on this output alone."
)
