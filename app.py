import io
from pathlib import Path
from datetime import date, timedelta

import cv2
import folium
import numpy as np
import requests
import streamlit as st
import torch
from PIL import Image
import rasterio
from rasterio.io import MemoryFile
from streamlit_folium import st_folium

from tiny_unet import TinyUNet


# --------------------------------------------------
# VEYORA CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="VEYORA | Satellite Intelligence",
    page_icon="🛰️",
    layout="wide",
)

st.title("🛰️ VEYORA")
st.subheader("Sentinel-1 Oil-Spill Candidate Screening")

TOKEN_URL = (
    "https://identity.dataspace.copernicus.eu/"
    "auth/realms/CDSE/protocol/openid-connect/token"
)
PROCESS_URL = "https://sh.dataspace.copernicus.eu/api/v1/process"

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "best.pt"


# --------------------------------------------------
# COPERNICUS AUTHENTICATION
# --------------------------------------------------

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


# --------------------------------------------------
# FETCH SENTINEL-1 VV / VH DATA
# --------------------------------------------------

def fetch_sentinel1(bbox, start_date, end_date, token):
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
                    "crs": (
                        "http://www.opengis.net/def/crs/"
                        "EPSG/0/4326"
                    )
                },
            },
            "data": [
                {
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
                    },
                }
            ],
        },
        "output": {
            "width": 512,
            "height": 512,
            "responses": [
                {
                    "identifier": "default",
                    "format": {"type": "image/tiff"},
                }
            ],
        },
        "evalscript": evalscript,
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
        raise ValueError(
            f"Expected two Sentinel-1 bands (VV/VH); "
            f"received shape {bands.shape}"
        )

    return bands, profile


# --------------------------------------------------
# MODEL LOADING
# --------------------------------------------------

@st.cache_resource
def load_model(model_path_string, model_mtime):
    model = TinyUNet(
        in_channels=3,
        num_classes=2,
        base_ch=32,
    )

    checkpoint = torch.load(
        model_path_string,
        map_location="cpu",
        weights_only=True,
    )

    # Training saves a dictionary containing model_state.
    state_dict = checkpoint.get("model_state", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()

    return model


# --------------------------------------------------
# SAR PREPROCESSING
# --------------------------------------------------

def to_db(bands):
    """Convert linear radar power to decibels."""
    safe = np.maximum(bands, 1e-10)
    return 10.0 * np.log10(safe)


def display_rgb(vv_db):
    """Display the VV radar image as grayscale RGB."""
    valid = np.isfinite(vv_db)

    if not valid.any():
        return np.zeros((*vv_db.shape, 3), dtype=np.uint8)

    lo, hi = np.percentile(vv_db[valid], [2, 98])

    gray = np.clip(
        (np.nan_to_num(vv_db, nan=lo) - lo)
        / max(hi - lo, 1e-6)
        * 255,
        0,
        255,
    ).astype(np.uint8)

    return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)


# --------------------------------------------------
# EXPERIMENTAL MODEL INPUT
# --------------------------------------------------

def prepare_model_input(bands_db):
    """
    Convert VV/VH into a pseudo-RGB tensor.

    IMPORTANT:
    The model was trained on RGB SAR-derived images,
    not directly on these live VV/VH arrays.
    This conversion is experimental and needs validation.
    """
    vv = bands_db[0]
    vh = bands_db[1]

    vv_norm = np.clip((vv + 35.0) / 40.0, 0, 1)
    vh_norm = np.clip((vh + 45.0) / 40.0, 0, 1)
    diff_norm = np.clip((vv - vh + 20.0) / 40.0, 0, 1)

    pseudo_rgb = np.stack(
        [vv_norm, vh_norm, diff_norm],
        axis=0,
    ).astype(np.float32)

    resized = np.stack(
        [
            cv2.resize(
                channel,
                (256, 256),
                interpolation=cv2.INTER_AREA,
            )
            for channel in pseudo_rgb
        ],
        axis=0,
    )

    return torch.from_numpy(resized[None]).float()


def predict_with_model(bands_db):
    model = load_model(
        str(MODEL_PATH),
        MODEL_PATH.stat().st_mtime,
    )

    tensor = prepare_model_input(bands_db)

    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)[0, 1]
        probabilities = probabilities.cpu().numpy()

    h, w = bands_db.shape[1:]

    return cv2.resize(
        probabilities,
        (w, h),
        interpolation=cv2.INTER_LINEAR,
    )


# --------------------------------------------------
# DARK-TARGET FALLBACK
# --------------------------------------------------

def dark_target_fallback(vv_db, threshold_db, min_pixels):
    valid = np.isfinite(vv_db)
    clean = np.where(valid, vv_db, 0).astype(np.float32)
    smooth = cv2.GaussianBlur(clean, (5, 5), 0)

    binary = (
        (smooth < threshold_db) & valid
    ).astype(np.uint8)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8,
    )

    mask = np.zeros_like(binary)

    for i in range(1, count):
        if stats[i, cv2.CC_STAT_AREA] >= min_pixels:
            mask[labels == i] = 1

    return mask


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:
    st.header("Satellite search")

    region = st.selectbox(
        "Region",
        ["Chennai Coast", "Gulf of Mexico", "Custom"],
    )

    defaults = {
        "Chennai Coast": [80.0, 12.5, 81.0, 13.5],
        "Gulf of Mexico": [-92.0, 25.0, -90.0, 27.0],
        "Custom": [80.0, 12.5, 81.0, 13.5],
    }[region]

    west = st.number_input(
        "West longitude", value=float(defaults[0])
    )
    south = st.number_input(
        "South latitude", value=float(defaults[1])
    )
    east = st.number_input(
        "East longitude", value=float(defaults[2])
    )
    north = st.number_input(
        "North latitude", value=float(defaults[3])
    )

    today = date.today()

    end_date = st.date_input(
        "End date",
        value=today,
    )
    start_date = st.date_input(
        "Start date",
        value=today - timedelta(days=14),
    )

    st.header("Screening settings")

    threshold_db = st.slider(
        "Fallback VV threshold (dB)",
        -35.0,
        -5.0,
        -18.0,
        0.5,
    )

    min_pixels = st.slider(
        "Minimum fallback region (pixels)",
        5,
        500,
        25,
    )

    confidence_threshold = st.slider(
        "Experimental model threshold",
        0.1,
        0.9,
        0.5,
        0.05,
    )

    run = st.button(
        "Fetch Sentinel-1",
        type="primary",
    )


# --------------------------------------------------
# VALIDATE INPUTS AND FETCH DATA
# --------------------------------------------------

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
        st.error(
            "Choose an area no larger than 3 degrees per side."
        )
        st.stop()

    try:
        client_id = st.secrets["CDSE_CLIENT_ID"]
        client_secret = st.secrets["CDSE_CLIENT_SECRET"]
    except Exception:
        st.error(
            "Set CDSE_CLIENT_ID and CDSE_CLIENT_SECRET "
            "in Streamlit Cloud Secrets."
        )
        st.stop()

    try:
        with st.spinner(
            "Fetching Sentinel-1 VV/VH data from Copernicus..."
        ):
            token = get_access_token(
                client_id,
                client_secret,
            )

            bands, profile = fetch_sentinel1(
                [west, south, east, north],
                start_date.isoformat(),
                end_date.isoformat(),
                token,
            )

        st.session_state["bands"] = bands
        st.session_state["profile"] = profile
        st.session_state["bbox"] = [
            west, south, east, north
        ]

        st.success("Satellite data received successfully.")

    except Exception as exc:
        st.error(f"Satellite request failed: {exc}")


# --------------------------------------------------
# RESULTS
# --------------------------------------------------

if "bands" not in st.session_state:
    st.info(
        "Choose a region and date range, then click "
        "'Fetch Sentinel-1'."
    )
    st.stop()

bands = st.session_state["bands"]
profile = st.session_state["profile"]
bbox = st.session_state["bbox"]

bands_db = to_db(bands)
vv_db = bands_db[0]

st.markdown("### 1. Live Sentinel-1 imagery")

left, right = st.columns(2)

base_rgb = display_rgb(vv_db)

with left:
    st.image(
        base_rgb,
        caption="Sentinel-1 VV intensity (dB)",
        use_container_width=True,
    )


# --------------------------------------------------
# RUN MODEL OR FALLBACK
# --------------------------------------------------

probabilities = None

if MODEL_PATH.exists():
    try:
        with st.spinner(
            "Running experimental TinyUNet segmentation..."
        ):
            probabilities = predict_with_model(bands_db)

        mask = (
            probabilities >= confidence_threshold
        ).astype(np.uint8)

        mode = "Experimental TinyUNet"

    except Exception as exc:
        st.warning(
            f"Model inference failed: {exc}"
        )
        st.info(
            "Using the dark-target screening fallback."
        )

        mask = dark_target_fallback(
            vv_db,
            threshold_db,
            min_pixels,
        )

        mode = "Dark-target screening fallback"

else:
    st.warning(
        "best.pt was not found. Using the screening fallback."
    )

    mask = dark_target_fallback(
        vv_db,
        threshold_db,
        min_pixels,
    )

    mode = "Dark-target screening fallback"


# --------------------------------------------------
# DISPLAY PREDICTIONS
# --------------------------------------------------

with right:
    overlay = base_rgb.copy()
    overlay[mask == 1] = [255, 40, 40]
    overlay = cv2.addWeighted(
        base_rgb, 0.65, overlay, 0.35, 0
    )

    st.image(
        overlay,
        caption=f"Red candidate regions — {mode}",
        use_container_width=True,
    )

if probabilities is not None:
    st.warning(
        "EXPERIMENTAL AI OUTPUT: The model was trained on "
        "RGB SAR-derived images. Live VV/VH inputs use an "
        "approximate conversion, so these predictions are "
        "not validated oil-spill detections. Human review "
        "and testing on independently labelled Sentinel-1 "
        "scenes are required."
    )

    st.image(
        np.clip(
            probabilities * 255, 0, 255
        ).astype(np.uint8),
        caption="Experimental class-1 probability map",
        use_container_width=True,
    )

else:
    st.warning(
        "SCREENING MODE: Dark radar areas are not proof of oil. "
        "Low wind, natural films, and other radar effects can "
        "produce similar signatures."
    )


# --------------------------------------------------
# SUMMARY METRICS
# --------------------------------------------------

positive_pixels = int(mask.sum())
total_pixels = mask.size
coverage = (
    100.0 * positive_pixels / max(total_pixels, 1)
)

m1, m2, m3 = st.columns(3)

m1.metric(
    "Candidate pixels",
    f"{positive_pixels:,}",
)

m2.metric(
    "Image coverage",
    f"{coverage:.2f}%",
)

m3.metric(
    "Analysis mode",
    mode,
)


# --------------------------------------------------
# REGION REFERENCE MAP
# --------------------------------------------------

st.markdown("### 2. Region reference")

st.caption(
    "This map shows the requested bounding box. "
    "Candidate mask pixels have not yet been transformed "
    "into georeferenced oil-spill polygons."
)

map_view = folium.Map(
    location=[
        (bbox[1] + bbox[3]) / 2,
        (bbox[0] + bbox[2]) / 2,
    ],
    zoom_start=7,
)

folium.Rectangle(
    bounds=[
        [bbox[1], bbox[0]],
        [bbox[3], bbox[2]],
    ],
    color="cyan",
    fill=False,
).add_to(map_view)

st_folium(
    map_view,
    height=400,
    use_container_width=True,
)


# --------------------------------------------------
# DOWNLOAD PREDICTED MASK
# --------------------------------------------------

mask_bytes = io.BytesIO()

Image.fromarray(
    (mask * 255).astype(np.uint8)
).save(
    mask_bytes,
    format="PNG",
)

st.download_button(
    "Download candidate mask",
    data=mask_bytes.getvalue(),
    file_name="veyora_mask.png",
    mime="image/png",
)

st.caption(
    "VEYORA research prototype • Copernicus Sentinel-1 • "
    "Outputs are candidate regions, not confirmed oil spills."
)
