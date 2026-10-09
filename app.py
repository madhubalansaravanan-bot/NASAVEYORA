
from pathlib import Path
from io import BytesIO
from datetime import date, timedelta

import numpy as np
import requests
import streamlit as st
import torch
from PIL import Image

from tiny_unet import TinyUNet


# ============================================================
# VEYORA CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "best.pt"
DEVICE = torch.device("cpu")

# Official Copernicus Data Space Ecosystem endpoints
TOKEN_URL = (
    "https://identity.dataspace.copernicus.eu/"
    "auth/realms/CDSE/protocol/openid-connect/token"
)

PROCESS_URL = (
    "https://sh.dataspace.copernicus.eu/api/v1/process"
)

st.set_page_config(
    page_title="VEYORA | SAR Oil Spill Screening",
    page_icon="🌊",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp {
        background-color: #071521;
        color: #e8f1f8;
    }
    h1, h2, h3 {
        color: #62d9e8 !important;
    }
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


# ============================================================
# CREDENTIALS
# ============================================================

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
    client_id = get_secret(
        "CDSE_CLIENT_ID",
        "SH_CLIENT_ID",
        "SENTINEL_HUB_CLIENT_ID",
        "CLIENT_ID",
    )

    client_secret = get_secret(
        "CDSE_CLIENT_SECRET",
        "SH_CLIENT_SECRET",
        "SENTINEL_HUB_CLIENT_SECRET",
        "CLIENT_SECRET",
    )

    return client_id, client_secret


# ============================================================
# AUTHENTICATION
# ============================================================

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
        raise RuntimeError(
            "CDSE authentication response did not contain an access token."
        )

    return token


# ============================================================
# SENTINEL-1 EVALSCRIPT
# ============================================================

def make_evalscript():
    return """
    //VERSION=3

    function setup() {
        return {
            input: [{
                bands: ["VV", "VH"],
                units: "LINEAR_POWER"
            }],
            output: {
                bands: 3,
                sampleType: "UINT8"
            }
        };
    }

    function clamp01(x) {
        return Math.max(0.0, Math.min(1.0, x));
    }

    function toDb(x) {
        return 10.0 * Math.log(Math.max(x, 0.00000001)) / Math.LN10;
    }

    function evaluatePixel(sample) {
        let vvDb = toDb(sample.VV);
        let vhDb = toDb(sample.VH);

        let vv = clamp01((vvDb + 30.0) / 30.0);
        let vh = clamp01((vhDb + 35.0) / 30.0);
        let ratio = clamp01(((vvDb - vhDb) + 5.0) / 25.0);

        return [
            Math.round(vv * 255.0),
            Math.round(vh * 255.0),
            Math.round(ratio * 255.0)
        ];
    }
    """


# ============================================================
# FETCH SENTINEL-1 IMAGE
# ============================================================

def fetch_sentinel1_image(
    token,
    bbox,
    start_date,
    end_date,
    width=512,
    height=512,
):
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
            "width": width,
            "height": height,
            "responses": [
                {
                    "identifier": "default",
                    "format": {"type": "image/png"},
                }
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
        raise RuntimeError(
            f"Could not decode the Sentinel Hub image response: {exc}"
        ) from exc


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

def extract_state_dict(checkpoint):
    if not isinstance(checkpoint, dict):
        raise RuntimeError("The checkpoint is not a PyTorch dictionary.")

    for key in (
        "model_state",
        "model_state_dict",
        "state_dict",
        "model",
    ):
        value = checkpoint.get(key)

        if isinstance(value, dict) and value:
            if all(torch.is_tensor(v) for v in value.values()):
                return value

    if checkpoint and all(
        torch.is_tensor(v) for v in checkpoint.values()
    ):
        return checkpoint

    raise RuntimeError(
        "Could not locate the trained model weights in best.pt."
    )


@st.cache_resource(show_spinner=False)
def load_model():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}. "
            "Check that models/best.pt exists in your GitHub repository."
        )

    # Use only for your own trusted training checkpoint.
    checkpoint = torch.load(
        str(MODEL_PATH),
        map_location=DEVICE,
        weights_only=False,
    )

    state_dict = extract_state_dict(checkpoint)

    cleaned_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[len("module."):]
        cleaned_state_dict[key] = value

    model = TinyUNet(
        in_channels=3,
        num_classes=2,
        base_ch=32,
    )

    model.load_state_dict(cleaned_state_dict, strict=True)
    model.to(DEVICE)
    model.eval()

    return model


# ============================================================
# AI INFERENCE
# ============================================================

def run_inference(model, image):
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0

    tensor = torch.from_numpy(
        rgb.transpose(2, 0, 1)
    ).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(tensor)

        if logits.ndim != 4 or logits.shape[1] != 2:
            raise RuntimeError(
                "Expected model output [1, 2, height, width], "
                f"received {tuple(logits.shape)}."
            )

        probabilities = torch.softmax(logits, dim=1)
        target_probability = probabilities[0, 1].cpu().numpy()

        mask = (
            probabilities.argmax(dim=1)[0].cpu().numpy() == 1
        ).astype(np.uint8)

    return mask, target_probability


# ============================================================
# OUTPUT VISUALISATIONS
# ============================================================

def make_overlay(image, mask, alpha=0.45):
    base = np.asarray(image.convert("RGB"), dtype=np.float32)
    overlay = base.copy()

    highlight = np.zeros_like(base)
    highlight[:, :, 1] = 255
    highlight[:, :, 2] = 255

    selected = mask.astype(bool)
    overlay[selected] = (
        (1 - alpha) * base[selected]
        + alpha * highlight[selected]
    )

    return Image.fromarray(
        np.clip(overlay, 0, 255).astype(np.uint8)
    )


def image_to_bytes(image):
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🌊 VEYORA")
st.sidebar.caption("Satellite SAR Oil Spill Screening")

st.sidebar.subheader("Area of interest")

min_lon = st.sidebar.number_input(
    "Minimum longitude",
    min_value=-180.0,
    max_value=180.0,
    value=-91.0,
    step=0.1,
    format="%.4f",
)

min_lat = st.sidebar.number_input(
    "Minimum latitude",
    min_value=-90.0,
    max_value=90.0,
    value=27.0,
    step=0.1,
    format="%.4f",
)

max_lon = st.sidebar.number_input(
    "Maximum longitude",
    min_value=-180.0,
    max_value=180.0,
    value=-90.0,
    step=0.1,
    format="%.4f",
)

max_lat = st.sidebar.number_input(
    "Maximum latitude",
    min_value=-90.0,
    max_value=90.0,
    value=28.0,
    step=0.1,
    format="%.4f",
)

st.sidebar.subheader("Acquisition dates")

today = date.today()
start_default = today - timedelta(days=30)

start_date = st.sidebar.date_input(
    "Start date",
    value=start_default,
    max_value=today,
)

end_date = st.sidebar.date_input(
    "End date",
    value=today,
    min_value=start_date,
    max_value=today,
)

image_size = st.sidebar.selectbox(
    "Image resolution",
    [256, 512, 768, 1024],
    index=1,
)

run_button = st.sidebar.button(
    "Fetch data and run screening",
    type="primary",
    use_container_width=True,
)


# ============================================================
# MAIN INTERFACE
# ============================================================

st.title("VEYORA")
st.subheader("Satellite-based SAR Oil Spill Screening")

st.info(
    "VEYORA retrieves Sentinel-1 radar imagery and runs an experimental "
    "segmentation model to identify possible target regions. Outputs are "
    "not confirmed oil-spill detections."
)

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Satellite data**")
    st.write("Copernicus Sentinel-1 GRD")

with col2:
    st.markdown("**AI checkpoint**")
    st.write("`models/best.pt`")


# ============================================================
# RUN
# ============================================================

if run_button:
    valid_bbox = min_lon < max_lon and min_lat < max_lat

    if not valid_bbox:
        st.error(
            "Invalid bounding box: minimum coordinates must be "
            "less than maximum coordinates."
        )

    elif start_date > end_date:
        st.error("The start date must not be after the end date.")

    else:
        client_id, client_secret = get_credentials()

        if not client_id or not client_secret:
            st.error(
                "CDSE credentials are missing. Check Streamlit Cloud "
                "Settings → Secrets."
            )

            st.code(
                'CDSE_CLIENT_ID = "your-client-id"\n'
                'CDSE_CLIENT_SECRET = "your-client-secret"',
                language="toml",
            )

        else:
            try:
                with st.spinner("Authenticating with CDSE..."):
                    token = get_access_token(
                        client_id,
                        client_secret,
                    )

                st.success("CDSE authentication successful.")

                bbox = [
                    min_lon,
                    min_lat,
                    max_lon,
                    max_lat,
                ]

                with st.spinner("Fetching Sentinel-1 imagery..."):
                    image = fetch_sentinel1_image(
                        token=token,
                        bbox=bbox,
                        start_date=start_date.isoformat(),
                        end_date=end_date.isoformat(),
                        width=image_size,
                        height=image_size,
                    )

                st.success("Sentinel-1 imagery retrieved.")

                st.markdown("### Retrieved SAR visualisation")
                st.image(
                    image,
                    caption="Sentinel-1 VV/VH-derived pseudo-RGB",
                    use_container_width=True,
                )

                with st.spinner("Loading the trained AI model..."):
                    model = load_model()

                with st.spinner("Running AI segmentation..."):
                    mask, probability = run_inference(model, image)

                overlay = make_overlay(image, mask)

                mask_image = Image.fromarray(
                    (mask * 255).astype(np.uint8)
                )

                probability_image = Image.fromarray(
                    np.clip(
                        probability * 255,
                        0,
                        255,
                    ).astype(np.uint8)
                )

                target_pixels = int(mask.sum())
                total_pixels = int(mask.size)
                flagged_percentage = (
                    target_pixels / total_pixels * 100
                    if total_pixels
                    else 0.0
                )

                st.markdown("### Screening results")

                m1, m2, m3 = st.columns(3)

                m1.metric(
                    "Predicted target pixels",
                    f"{target_pixels:,}",
                )

                m2.metric(
                    "Image area flagged",
                    f"{flagged_percentage:.2f}%",
                )

                m3.metric(
                    "Mean target probability",
                    f"{float(probability.mean()):.3f}",
                )

                st.warning(
                    "Predictions are experimental. Dark or smooth SAR "
                    "regions can arise from several causes other than oil. "
                    "Independent verification is necessary."
                )

                st.markdown("### AI segmentation")

                c1, c2 = st.columns(2)

                with c1:
                    st.image(
                        overlay,
                        caption="Predicted target regions",
                        use_container_width=True,
                    )

                with c2:
                    st.image(
                        mask_image,
                        caption="Binary prediction mask",
                        use_container_width=True,
                    )

                st.markdown("### Target-class probability map")

                st.image(
                    probability_image,
                    caption="Brighter pixels represent higher class-1 probability",
                    use_container_width=True,
                )

                st.markdown("### Download results")

                d1, d2, d3 = st.columns(3)

                with d1:
                    st.download_button(
                        "Download SAR image",
                        data=image_to_bytes(image),
                        file_name="veyora_sar_image.png",
                        mime="image/png",
                        use_container_width=True,
                    )

                with d2:
                    st.download_button(
                        "Download AI mask",
                        data=image_to_bytes(mask_image),
                        file_name="veyora_ai_mask.png",
                        mime="image/png",
                        use_container_width=True,
                    )

                with d3:
                    st.download_button(
                        "Download overlay",
                        data=image_to_bytes(overlay),
                        file_name="veyora_ai_overlay.png",
                        mime="image/png",
                        use_container_width=True,
                    )

                st.caption(
                    f"Date range: {start_date} to {end_date} | "
                    f"Bounding box: {bbox} | Device: {DEVICE}"
                )

            except Exception as exc:
                st.error("VEYORA could not complete the screening.")
                st.code(f"{type(exc).__name__}: {exc}")

                st.markdown(
                    """
                    **Troubleshooting**
                    - `401 invalid_client`: verify the CDSE OAuth client ID,
                      matching client secret, and credentials configuration.
                    - `403`: verify service access and permissions.
                    - `400`: inspect the Processing API error for invalid
                      parameters or unavailable data.
                    - `FileNotFoundError`: check `models/best.pt`.
                    - `Missing key(s)` / `Unexpected key(s)`: the model
                      architecture in `tiny_unet.py` may not match training.
                    """
                )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "VEYORA | Experimental satellite-based SAR oil-spill screening. "
    "AI predictions require validation and expert interpretation."
)
