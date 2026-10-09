from pathlib import Path
from io import BytesIO
from datetime import date, timedelta

import numpy as np
import requests
import streamlit as st
import torch
from PIL import Image

# Keep imports lightweight at startup. The model is imported only when needed.
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "best.pt"
DEVICE = torch.device("cpu")

TOKEN_URL = (
    "https://identity.dataspace.copernicus.eu/"
    "auth/realms/CDSE/protocol/openid-connect/token"
)
PROCESS_URL = "https://sh.dataspace.copernicus.eu/api/v1/process"

st.set_page_config(
    page_title="VEYORA | SAR Oil Spill Screening",
    page_icon="🌊",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp { background-color: #071521; color: #e8f1f8; }
    h1, h2, h3 { color: #62d9e8 !important; }
    div[data-testid="stMetric"] {
        background-color: #102736;
        border: 1px solid #245064;
        padding: 12px;
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def get_secret(*names):
    for name in names:
        try:
            value = st.secrets.get(name)
            if value:
                return str(value).strip()
        except Exception:
            pass
    return None


def get_credentials():
    return (
        get_secret("CDSE_CLIENT_ID", "SH_CLIENT_ID", "SENTINEL_HUB_CLIENT_ID", "CLIENT_ID"),
        get_secret("CDSE_CLIENT_SECRET", "SH_CLIENT_SECRET", "SENTINEL_HUB_CLIENT_SECRET", "CLIENT_SECRET"),
    )


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
    if not response.ok:
        raise RuntimeError(
            f"CDSE authentication failed ({response.status_code}): "
            f"{response.text[:500]}"
        )
    token = response.json().get("access_token")
    if not token:
        raise RuntimeError("Authentication response did not contain an access token.")
    return token


def search_places(query):
    response = requests.get(
        "https://nominatim.openstreetmap.org/search",
        params={"q": query, "format": "jsonv2", "limit": 5},
        headers={"User-Agent": "VEYORA-SAR-Research-App/1.0"},
        timeout=20,
    )
    response.raise_for_status()
    return response.json()


def make_evalscript():
    return """
    //VERSION=3
    function setup() {
      return {
        input: [{bands: ["VV", "VH"], units: "LINEAR_POWER"}],
        output: {bands: 3, sampleType: "UINT8"}
      };
    }
    function clamp01(x) { return Math.max(0.0, Math.min(1.0, x)); }
    function toDb(x) { return 10.0 * Math.log(Math.max(x, 0.00000001)) / Math.LN10; }
    function evaluatePixel(s) {
      let vvDb = toDb(s.VV);
      let vhDb = toDb(s.VH);
      return [
        Math.round(clamp01((vvDb + 30.0) / 30.0) * 255.0),
        Math.round(clamp01((vhDb + 35.0) / 30.0) * 255.0),
        Math.round(clamp01(((vvDb - vhDb) + 5.0) / 25.0) * 255.0)
      ];
    }
    """


def fetch_sentinel1_image(token, bbox, start_date, end_date, size):
    payload = {
        "input": {
            "bounds": {
                "bbox": bbox,
                "properties": {
                    "crs": "http://www.opengis.net/def/crs/EPSG/0/4326"
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
                        "resolution": "HIGH",
                        "acquisitionMode": "IW",
                        "polarization": "DV",
                        "orthorectify": True,
                    },
                }
            ],
        },
        "output": {
            "width": size,
            "height": size,
            "responses": [
                {"identifier": "default", "format": {"type": "image/png"}}
            ],
        },
        "evalscript": make_evalscript(),
    }
    response = requests.post(
        PROCESS_URL,
        headers={"Authorization": f"Bearer {token}"},
        json=payload,
        timeout=120,
    )
    if not response.ok:
        raise RuntimeError(
            f"Sentinel-1 Processing API failed ({response.status_code}): "
            f"{response.text[:1000]}"
        )
    try:
        return Image.open(BytesIO(response.content)).convert("RGB")
    except Exception as exc:
        raise RuntimeError(f"Could not decode the Sentinel Hub image response: {exc}") from exc


def load_model():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Model checkpoint not found at {MODEL_PATH}. "
            "Upload the actual trained best.pt file to models/best.pt."
        )

    # Only use weights_only=False for your own trusted checkpoint.
    checkpoint = torch.load(str(MODEL_PATH), map_location=DEVICE, weights_only=False)

    if not isinstance(checkpoint, dict):
        raise RuntimeError("Checkpoint is not a PyTorch dictionary.")

    state_dict = None
    for key in ("model_state", "model_state_dict", "state_dict", "model"):
        value = checkpoint.get(key)
        if isinstance(value, dict) and value and all(torch.is_tensor(v) for v in value.values()):
            state_dict = value
            break

    if state_dict is None and checkpoint and all(torch.is_tensor(v) for v in checkpoint.values()):
        state_dict = checkpoint

    if state_dict is None:
        raise RuntimeError(
            "Could not find model weights. best.pt may be a Git LFS pointer or invalid download."
        )

    state_dict = {
        (key[7:] if key.startswith("module.") else key): value
        for key, value in state_dict.items()
    }

    # Import lazily so an architecture/import problem does not blank the whole UI.
    try:
        from tiny_unet import TinyUNet
    except Exception as exc:
        raise RuntimeError(f"Could not import TinyUNet from tiny_unet.py: {exc}") from exc

    model = TinyUNet(in_channels=3, num_classes=2, base_ch=32)
    model.load_state_dict(state_dict, strict=True)
    model.to(DEVICE)
    model.eval()
    return model


@st.cache_resource(show_spinner=False)
def cached_model():
    return load_model()


def run_inference(model, image):
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(tensor)
        if logits.ndim != 4 or logits.shape[1] != 2:
            raise RuntimeError(f"Expected model output [1, 2, H, W], got {tuple(logits.shape)}")
        probabilities = torch.softmax(logits, dim=1)
        probability = probabilities[0, 1].cpu().numpy()
        mask = (probabilities.argmax(dim=1)[0].cpu().numpy() == 1).astype(np.uint8)
    return mask, probability


def png_bytes(image):
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def make_overlay(image, mask):
    base = np.asarray(image.convert("RGB"), dtype=np.float32)
    cyan = np.zeros_like(base)
    cyan[:, :, 1] = 255
    cyan[:, :, 2] = 255
    selected = mask.astype(bool)
    base[selected] = 0.55 * base[selected] + 0.45 * cyan[selected]
    return Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))


# ============================================================
# USER INTERFACE — uses native Streamlit components, no Folium
# ============================================================

st.sidebar.title("🌊 VEYORA")
st.sidebar.caption("Satellite SAR Oil Spill Screening")
st.sidebar.subheader("Area of interest")

if "aoi_lat" not in st.session_state:
    st.session_state.aoi_lat = 27.5
    st.session_state.aoi_lon = -90.5
    st.session_state.aoi_name = "Default Gulf of Mexico area"

place_query = st.sidebar.text_input(
    "Search city, coast or region",
    placeholder="Example: Chennai, India",
    key="place_query",
)

if st.sidebar.button("Search location", use_container_width=True):
    if place_query.strip():
        try:
            results = search_places(place_query.strip())
            if results:
                st.session_state["place_results"] = results
            else:
                st.sidebar.warning("No place found. Try another search.")
        except Exception as exc:
            st.sidebar.error(f"Location search failed: {exc}")

place_results = st.session_state.get("place_results", [])
if place_results:
    result_index = st.sidebar.selectbox(
        "Choose matching place",
        range(len(place_results)),
        format_func=lambda index: place_results[index].get(
            "display_name", f"Result {index + 1}"
        ),
    )
    if st.sidebar.button("Use selected place", use_container_width=True):
        chosen = place_results[result_index]
        st.session_state.aoi_lat = float(chosen["lat"])
        st.session_state.aoi_lon = float(chosen["lon"])
        st.session_state.aoi_name = chosen.get("display_name", place_query)
        st.rerun()

st.sidebar.caption(f"Selected place: {st.session_state.aoi_name}")

# Approximate bounding box around the selected city; editable by user.
lat = float(st.session_state.aoi_lat)
lon = float(st.session_state.aoi_lon)
st.sidebar.subheader("Area size")
half_height = st.sidebar.slider("Half-height (latitude degrees)", 0.02, 1.0, 0.10, 0.01)
half_width = st.sidebar.slider("Half-width (longitude degrees)", 0.02, 1.0, 0.10, 0.01)

min_lon = max(-180.0, lon - half_width)
max_lon = min(180.0, lon + half_width)
min_lat = max(-90.0, lat - half_height)
max_lat = min(90.0, lat + half_height)
bbox = [min_lon, min_lat, max_lon, max_lat]

st.title("VEYORA")
st.subheader("Satellite-based SAR Oil Spill Screening")
st.write(
    "Search for a city in the sidebar. The map below shows the selected location "
    "and the approximate area that will be queried."
)

# Native Streamlit map avoids third-party interactive iframe/component issues.
map_data = [{"latitude": lat, "longitude": lon}]
st.map(map_data, latitude="latitude", longitude="longitude", zoom=5, use_container_width=True)

st.markdown("### Selected area")
st.write(f"**Location:** {st.session_state.aoi_name}")
st.write("**Bounding box** (min longitude, min latitude, max longitude, max latitude):")
st.code(f"{min_lon:.5f}, {min_lat:.5f}, {max_lon:.5f}, {max_lat:.5f}")

st.caption(
    "The map shows the selected city marker. Adjust the area size sliders to "
    "change the bounding box used for Sentinel-1 data retrieval."
)

st.sidebar.subheader("Acquisition dates")
today = date.today()
start_date = st.sidebar.date_input(
    "Start date", value=today - timedelta(days=30), max_value=today
)
end_date = st.sidebar.date_input(
    "End date", value=today, min_value=start_date, max_value=today
)
image_size = st.sidebar.selectbox("Image resolution", [256, 512, 768, 1024], index=1)
run_button = st.sidebar.button(
    "Fetch data and run screening", type="primary", use_container_width=True
)

st.markdown("**Satellite data:** Copernicus Sentinel-1 GRD  \n**Model checkpoint:** `models/best.pt`")

if run_button:
    if min_lon >= max_lon or min_lat >= max_lat:
        st.error("Invalid area of interest. Adjust the area-size sliders.")
    elif start_date > end_date:
        st.error("Start date must not be after end date.")
    else:
        client_id, client_secret = get_credentials()
        if not client_id or not client_secret:
            st.error("CDSE credentials missing. Check Streamlit Cloud → Settings → Secrets.")
            st.code(
                'CDSE_CLIENT_ID = "your-client-id"\n'
                'CDSE_CLIENT_SECRET = "your-client-secret"',
                language="toml",
            )
        else:
            try:
                with st.spinner("Authenticating with CDSE..."):
                    token = get_access_token(client_id, client_secret)
                st.success("CDSE authentication successful.")

                with st.spinner("Fetching Sentinel-1 imagery..."):
                    image = fetch_sentinel1_image(
                        token, bbox, start_date.isoformat(),
                        end_date.isoformat(), image_size
                    )

                st.markdown("### Retrieved SAR visualisation")
                st.image(
                    image,
                    caption="Sentinel-1 VV/VH-derived pseudo-RGB",
                    use_container_width=True,
                )

                with st.spinner("Loading trained model..."):
                    model = cached_model()
                with st.spinner("Running experimental segmentation..."):
                    mask, probability = run_inference(model, image)

                mask_image = Image.fromarray((mask * 255).astype(np.uint8))
                overlay = make_overlay(image, mask)
                probability_image = Image.fromarray(
                    np.clip(probability * 255, 0, 255).astype(np.uint8)
                )

                target_pixels = int(mask.sum())
                total_pixels = int(mask.size)
                flagged_percentage = 100.0 * target_pixels / total_pixels if total_pixels else 0.0

                st.markdown("### Screening results")
                metric1, metric2, metric3 = st.columns(3)
                metric1.metric("Predicted target pixels", f"{target_pixels:,}")
                metric2.metric("Image area flagged", f"{flagged_percentage:.2f}%")
                metric3.metric("Mean class-1 probability", f"{float(probability.mean()):.3f}")

                st.warning(
                    "Experimental predictions only. Dark or smooth SAR regions can have "
                    "causes other than oil; independent verification is required."
                )

                col1, col2 = st.columns(2)
                with col1:
                    st.image(overlay, caption="Predicted regions overlay", use_container_width=True)
                with col2:
                    st.image(mask_image, caption="Binary segmentation mask", use_container_width=True)

                st.markdown("### Target-class probability map")
                st.image(probability_image, use_container_width=True)

                d1, d2, d3 = st.columns(3)
                d1.download_button(
                    "Download SAR image", png_bytes(image),
                    "veyora_sar_image.png", "image/png", use_container_width=True
                )
                d2.download_button(
                    "Download AI mask", png_bytes(mask_image),
                    "veyora_ai_mask.png", "image/png", use_container_width=True
                )
                d3.download_button(
                    "Download overlay", png_bytes(overlay),
                    "veyora_ai_overlay.png", "image/png", use_container_width=True
                )

            except Exception as exc:
                st.error("VEYORA could not complete the screening.")
                st.code(f"{type(exc).__name__}: {exc}")
                st.markdown("""
                **Troubleshooting**
                - `401 invalid_client`: verify the CDSE OAuth client ID and matching secret.
                - `UnpicklingError: invalid load key`: `models/best.pt` may not be the actual PyTorch checkpoint.
                - `Missing key(s)` / `Unexpected key(s)`: `tiny_unet.py` may not match the training architecture.
                """)

st.divider()
st.caption(
    "VEYORA | Experimental satellite-based SAR oil-spill screening. "
    "AI outputs require validation and expert interpretation."
)
