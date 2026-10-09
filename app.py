import io
from datetime import date, timedelta

import cv2
import folium
import numpy as np
import rasterio
import requests
import streamlit as st
from rasterio.io import MemoryFile
from streamlit_folium import st_folium


# --------------------------------------------------
# VEYORA: LIVE SENTINEL-1 SAR SCREENING
# --------------------------------------------------

st.set_page_config(
    page_title="VEYORA | Satellite Intelligence",
    page_icon="🛰️",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background: #07111f;
    color: #e6f1ff;
}
h1, h2, h3 {
    color: #65d9ff !important;
}
[data-testid="stMetric"] {
    background: #101f32;
    padding: 16px;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)

TOKEN_URL = (
    "https://identity.dataspace.copernicus.eu/"
    "auth/realms/CDSE/protocol/openid-connect/token"
)
PROCESS_URL = "https://sh.dataspace.copernicus.eu/api/v1/process"


# --------------------------------------------------
# AUTHENTICATION
# --------------------------------------------------

@st.cache_data(ttl=3000, show_spinner=False)
def get_access_token(client_id, client_secret):
    response = requests.post(
        TOKEN_URL,
        data={
            "grant_type": "client_credentials",
            "client_id": client_id,
            "client_secret": client_secret
        },
        timeout=30
    )
    response.raise_for_status()
    return response.json()["access_token"]


# --------------------------------------------------
# DOWNLOAD TWO-BAND SENTINEL-1 DATA
# --------------------------------------------------

def fetch_sentinel1(bbox, start_date, end_date, token):
    """
    bbox order: west, south, east, north.
    Requests VV and VH in linear-power units.
    """
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

    function evaluatePixel(sample) {
        return [sample.VV, sample.VH];
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
                        "to": f"{end_date}T23:59:59Z"
                    },
                    "acquisitionMode": "IW",
                    "polarization": "DV"
                }
            }]
        },
        "output": {
            "width": 512,
            "height": 512,
            "responses": [{
                "identifier": "default",
                "format": {
                    "type": "image/tiff"
                }
            }]
        },
        "evalscript": evalscript
    }

    response = requests.post(
        PROCESS_URL,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        },
        json=payload,
        timeout=180
    )

    if not response.ok:
        detail = response.text[:1500]
        raise RuntimeError(
            f"Sentinel Hub returned HTTP {response.status_code}: {detail}"
        )

    with MemoryFile(response.content) as memfile:
        with memfile.open() as src:
            bands = src.read().astype(np.float32)
            profile = {
                "crs": str(src.crs) if src.crs else None,
                "transform": tuple(src.transform),
                "width": src.width,
                "height": src.height,
                "bounds": tuple(src.bounds)
            }

    if bands.shape[0] != 2:
        raise ValueError(
            f"Expected VV and VH bands, received shape {bands.shape}"
        )

    return bands, profile


# --------------------------------------------------
# FALLBACK: DARK-TARGET SCREENING
# --------------------------------------------------

def screen_dark_targets(vv_linear, threshold_db, min_pixels):
    """
    Heuristic screening only. NOT a trained AI model.
    """
    valid = np.isfinite(vv_linear) & (vv_linear > 0)

    vv_db = np.full(vv_linear.shape, np.nan, dtype=np.float32)
    vv_db[valid] = 10 * np.log10(vv_linear[valid])

    # Fill invalid values before filtering; exclude them afterwards.
    clean = np.where(valid, vv_db, 0).astype(np.float32)
    smooth = cv2.GaussianBlur(clean, (5, 5), 0)

    candidates = (
        (smooth < threshold_db) & valid
    ).astype(np.uint8)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        candidates, connectivity=8
    )

    mask = np.zeros_like(candidates)
    regions = []

    for index in range(1, count):
        pixels = int(stats[index, cv2.CC_STAT_AREA])
        if pixels >= min_pixels:
            mask[labels == index] = 1
            regions.append(pixels)

    return vv_db, mask, regions


# --------------------------------------------------
# DASHBOARD
# --------------------------------------------------

st.title("🛰️ VEYORA")
st.subheader("Live Sentinel-1 Marine Oil-Spill Screening")

st.warning(
    "SCREENING MODE — No trained AI model is installed. "
    "Dark radar targets are not proof of oil; low wind, natural films, "
    "and other effects can produce similar signatures."
)

with st.sidebar:
    st.header("Satellite search")

    region = st.selectbox(
        "Area of interest",
        ["Chennai Coast", "Gulf of Mexico", "Custom region"]
    )

    if region == "Chennai Coast":
        default_bbox = [80.0, 12.5, 81.0, 13.5]
    elif region == "Gulf of Mexico":
        default_bbox = [-92.0, 25.0, -90.0, 27.0]
    else:
        default_bbox = [80.0, 12.5, 81.0, 13.5]

    west = st.number_input(
        "West longitude", value=float(default_bbox[0]),
        min_value=-180.0, max_value=180.0, format="%.4f"
    )
    south = st.number_input(
        "South latitude", value=float(default_bbox[1]),
        min_value=-90.0, max_value=90.0, format="%.4f"
    )
    east = st.number_input(
        "East longitude", value=float(default_bbox[2]),
        min_value=-180.0, max_value=180.0, format="%.4f"
    )
    north = st.number_input(
        "North latitude", value=float(default_bbox[3]),
        min_value=-90.0, max_value=90.0, format="%.4f"
    )

    today = date.today()
    end_date = st.date_input(
        "Acquisition end date", value=today
    )
    start_date = st.date_input(
        "Acquisition start date",
        value=today - timedelta(days=30)
    )

    st.divider()
    st.header("Screening parameters")

    threshold_db = st.slider(
        "VV dark-target threshold (dB)",
        min_value=-35.0, max_value=-5.0,
        value=-18.0, step=0.5
    )
    min_pixels = st.slider(
        "Minimum region size (pixels)",
        min_value=5, max_value=500,
        value=25
    )

    run_search = st.button(
        "🛰️ Fetch satellite data",
        type="primary",
        use_container_width=True
    )


if run_search:
    if not (-180 <= west < east <= 180):
        st.error("Longitude bounds are invalid.")
        st.stop()

    if not (-90 <= south < north <= 90):
        st.error("Latitude bounds are invalid.")
        st.stop()

    if start_date > end_date:
        st.error("Start date must be before end date.")
        st.stop()

    if (east - west) > 3 or (north - south) > 3:
        st.error(
            "Please select a smaller area (maximum 3 degrees "
            "in either dimension) for this prototype."
        )
        st.stop()

    try:
        client_id = st.secrets["CDSE_CLIENT_ID"]
        client_secret = st.secrets["CDSE_CLIENT_SECRET"]
    except Exception:
        st.error(
            "Missing API credentials. Add CDSE_CLIENT_ID and "
            "CDSE_CLIENT_SECRET in Streamlit App Settings → Secrets."
        )
        st.stop()

    bbox = [west, south, east, north]

    try:
        with st.spinner(
            "Authenticating and requesting Sentinel-1 radar data..."
        ):
            token = get_access_token(client_id, client_secret)
            bands, profile = fetch_sentinel1(
                bbox, start_date.isoformat(),
                end_date.isoformat(), token
            )

        st.session_state["sar_bands"] = bands
        st.session_state["sar_profile"] = profile
        st.session_state["sar_bbox"] = bbox
        st.session_state["sar_dates"] = (
            start_date.isoformat(), end_date.isoformat()
        )
        st.success("Radar data received from Sentinel Hub.")

    except Exception as exc:
        st.error(f"Satellite request failed: {exc}")
        st.info(
            "Check OAuth credentials, API access, date range, "
            "data availability, and the service response."
        )


if "sar_bands" not in st.session_state:
    st.info(
        "Choose an area and date range, then click "
        "'Fetch satellite data' to retrieve live SAR data."
    )
    st.stop()


bands = st.session_state["sar_bands"]
profile = st.session_state["sar_profile"]
bbox = st.session_state["sar_bbox"]
dates = st.session_state["sar_dates"]

vv = bands[0]
vh = bands[1]

vv_db, mask, regions = screen_dark_targets(
    vv, threshold_db, min_pixels
)

valid = np.isfinite(vv_db)
if valid.any():
    low, high = np.percentile(vv_db[valid], [2, 98])
    display = np.clip(
        (np.nan_to_num(vv_db, nan=low) - low)
        / max(high - low, 1e-6) * 255,
        0, 255
    ).astype(np.uint8)
else:
    display = np.zeros(vv_db.shape, dtype=np.uint8)

rgb = cv2.cvtColor(display, cv2.COLOR_GRAY2RGB)
overlay = rgb.copy()
overlay[mask == 1] = [255, 35, 35]
overlay = cv2.addWeighted(rgb, 0.65, overlay, 0.35, 0)

st.markdown("### Satellite acquisition request")
st.write(f"**Requested dates:** {dates[0]} to {dates[1]}")
st.write(
    "**Requested area:** "
    f"{bbox[0]:.4f}, {bbox[1]:.4f}, "
    f"{bbox[2]:.4f}, {bbox[3]:.4f}"
)
st.caption(
    "The request window is not necessarily the exact acquisition "
    "time of a single scene. Processing API requests may combine "
    "available observations over the requested interval."
)

left, right = st.columns(2)
with left:
    st.image(display, caption="Sentinel-1 VV intensity (display stretch)")
with right:
    st.image(overlay, caption="Red = dark-target screening candidates")

total = int(valid.sum())
candidate_pixels = int(mask.sum())
coverage = 100 * candidate_pixels / max(total, 1)

m1, m2, m3 = st.columns(3)
m1.metric("Candidate regions", len(regions))
m2.metric("Candidate pixels", f"{candidate_pixels:,}")
m3.metric("Image coverage", f"{coverage:.2f}%")

st.markdown("### Reference map")
st.caption(
    "The map shows the requested region, not automatically "
    "geolocated individual candidate pixels."
)

map_obj = folium.Map(
    location=[(south + north) / 2, (west + east) / 2],
    zoom_start=7
)

folium.Rectangle(
    bounds=[[south, west], [north, east]],
    color="cyan",
    fill=False,
    tooltip="Requested satellite area"
).add_to(map_obj)

st_folium(map_obj, height=420, use_container_width=True)

mask_buffer = io.BytesIO()
from PIL import Image
Image.fromarray(mask * 255).save(mask_buffer, format="PNG")

st.download_button(
    "Download screening mask",
    data=mask_buffer.getvalue(),
    file_name="veyora_screening_mask.png",
    mime="image/png"
)

st.markdown("---")
st.caption(
    "VEYORA • Copernicus Sentinel-1 via Sentinel Hub • "
    "Dark-target screening is not confirmed oil-spill detection."
)
