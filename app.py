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
# VEYORA — SAR Oil Spill Screening
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "best.pt"

TOKEN_URL = (
    "https://services.sentinel-hub.com/"
    "auth/realms/main/protocol/openid-connect/token"
)
PROCESS_URL = "https://services.sentinel-hub.com/api/v1/process"

st.set_page_config(
    page_title="VEYORA | SAR Oil Spill Screening",
    page_icon="🌊",
    layout="wide",
)

DEVICE = torch.device("cpu")


# ============================================================
# PAGE STYLE
# ============================================================

st.markdown(
    """
    <style>
    .main {
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
    .veyora-note {
        padding: 12px;
        border-radius: 8px;
        background: #102736;
        border-left: 4px solid #62d9e8;
        margin-bottom: 15px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SECRETS
# ============================================================

def get_secret(*names):
    """Return the first configured Streamlit secret from the names."""
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
        "SH_CLIENT_ID",
        "SENTINEL_HUB_CLIENT_ID",
        "CDSE_CLIENT_ID",
        "CLIENT_ID",
    )

    client_secret = get_secret(
        "SH_CLIENT_SECRET",
        "SENTINEL_HUB_CLIENT_SECRET",
        "CDSE_CLIENT_SECRET",
        "CLIENT_SECRET",
    )

    return client_id, client_secret



# ============================================================
# SENTINEL HUB AUTHENTICATION
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
            f"Authentication failed ({response.status_code}): "
            f"{response.text[:500]}"
        )

    token = response.json().get("access_token")

    if not token:
        raise RuntimeError("Authentication response did not contain a token.")

    return token


# ============================================================
# SENTINEL-1 DATA REQUEST
# ============================================================

def make_evalscript():
    """
    Produce a 3-channel visualisation from Sentinel-1 VV and VH.

    These channels are SAR-derived pseudo-RGB, not true optical RGB.
    The trained model's performance on this input must be validated.
    """
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


def fetch_sentinel1_image(
    token,
    bbox,
    start_date,
    end_date,
    width=512,
    height=512,
):
    """
    Fetch Sentinel-1 GRD VV/VH data for a WGS84 bounding box.

    bbox format:
        [minimum longitude, minimum latitude,
         maximum longitude, maximum latitude]
    """
    evalscript = make_evalscript()

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
        "evalscript": evalscript,
    }

    response = requests.post(
        PROCESS_URL,
        headers={"Authorization": f"Bearer {token}"},
        json=payload,
        timeout=120,
    )

    if not response.ok:
        raise RuntimeError(
            f"Sentinel Hub request failed ({response.status_code}): "
            f"{response.text[:1000]}"
        )

    try:
        image = Image.open(BytesIO(response.content)).convert("RGB")
        return image
    except Exception as exc:
        raise RuntimeError(
            "Sentinel Hub returned a response that could not be read as an image. "
            f"Details: {exc}"
        ) from exc


# ============================================================
# MODEL LOADING
# ============================================================

def extract_state_dict(checkpoint):
    """Support common PyTorch checkpoint dictionary formats."""
    if not isinstance(checkpoint, dict):
        raise RuntimeError(
            "Unexpected checkpoint format. Expected a PyTorch state dictionary "
            "or a checkpoint dictionary."
        )

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

    if checkpoint and all(torch.is_tensor(v) for v in checkpoint.values()):
        return checkpoint

    raise RuntimeError(
        "Could not find model weights in the checkpoint. "
        "Expected model_state, model_state_dict, state_dict, or raw weights."
    )


@st.cache_resource(show_spinner=False)
def load_model():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Model checkpoint not found: {MODEL_PATH}\n"
            "Confirm that models/best.pt is committed to your repository."
        )

    # weights_only=False is required for some checkpoint formats.
    # Only use this with a checkpoint from your own trusted training run.
    checkpoint = torch.load(
        str(MODEL_PATH),
        map_location=DEVICE,
        weights_only=False,
    )

    state_dict = extract_state_dict(checkpoint)

    # Remove DataParallel prefixes if present.
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
# MODEL INFERENCE
# ============================================================

def run_inference(model, image):
    """
    Return a predicted class mask and target-class probability map.

    This assumes class index 1 is the oil-spill class. Confirm this
    against the labels used to train the checkpoint.
    """
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0

    # HWC -> CHW -> NCHW
    tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0)
    tensor = tensor.to(DEVICE)

    with torch.no_grad():
        logits = model(tensor)

        if logits.ndim != 4 or logits.shape[1] != 2:
            raise RuntimeError(
                "Unexpected model output. Expected shape [1, 2, height, width]. "
                f"Received {tuple(logits.shape)}."
            )

        probabilities = torch.softmax(logits, dim=1)
        target_probability = probabilities[0, 1].cpu().numpy()
        predicted_mask = (
            probabilities.argmax(dim=1)[0].cpu().numpy() == 1
        ).astype(np.uint8)

    return predicted_mask, target_probability


# ============================================================
# VISUALISATION HELPERS
# ============================================================

def make_mask_image(mask):
    """Convert a binary mask into a visible grayscale PNG."""
    mask_image = Image.fromarray((mask * 255).astype(np.uint8))
    return mask_image


def make_overlay(image, mask, alpha=0.45):
    """Create an RGB overlay highlighting predicted target pixels."""
    base = np.asarray(image.convert("RGB"), dtype=np.float32)

    # Cyan highlight for predicted target pixels.
    highlight = np.zeros_like(base)
    highlight[:, :, 0] = 0
    highlight[:, :, 1] = 255
    highlight[:, :, 2] = 255

    overlay = base.copy()
    selected = mask.astype(bool)
    overlay[selected] = (
        (1 - alpha) * base[selected] + alpha * highlight[selected]
    )

    return Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8))


def make_probability_image(probability):
    """Convert target-class probabilities to an 8-bit grayscale image."""
    return Image.fromarray(
        np.clip(probability * 255, 0, 255).astype(np.uint8)
    )


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🌊 VEYORA")
st.sidebar.caption("Satellite SAR oil-spill screening")

st.sidebar.markdown("### Area of interest")

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

st.sidebar.markdown("### Acquisition dates")

today = date.today()
default_start = today - timedelta(days=30)

start_date = st.sidebar.date_input(
    "Start date",
    value=default_start,
    max_value=today,
)

end_date = st.sidebar.date_input(
    "End date",
    value=today,
    min_value=start_date,
    max_value=today,
)

st.sidebar.markdown("### Image settings")

image_size = st.sidebar.selectbox(
    "Output image size",
    options=[256, 512, 768, 1024],
    index=1,
)

run_button = st.sidebar.button(
    "Fetch data and run screening",
    type="primary",
    use_container_width=True,
)


# ============================================================
# MAIN PAGE
# ============================================================

st.title("VEYORA")
st.subheader("Satellite-based SAR Oil Spill Screening")

st.markdown(
    """
    <div class="veyora-note">
    <b>Purpose:</b> Retrieve Sentinel-1 VV/VH radar data and run a
    segmentation model to screen for possible oil-spill-like regions.
    Model outputs are experimental and require validation.
    </div>
    """,
    unsafe_allow_html=True,
)

left, right = st.columns(2)

with left:
    st.markdown("#### Data source")
    st.write("Copernicus Sentinel-1 GRD via Sentinel Hub Processing API")

with right:
    st.markdown("#### AI model")
    st.write("Two-class segmentation model loaded from `models/best.pt`")


# ============================================================
# VALIDATION
# ============================================================

bbox = [min_lon, min_lat, max_lon, max_lat]

bbox_is_valid = (
    min_lon < max_lon
    and min_lat < max_lat
    and -180 <= min_lon <= 180
    and -180 <= max_lon <= 180
    and -90 <= min_lat <= 90
    and -90 <= max_lat <= 90
)

if not bbox_is_valid:
    st.warning(
        "Please enter a valid bounding box. Minimum coordinates must be "
        "smaller than maximum coordinates."
    )

if start_date > end_date:
    st.warning("The start date must be on or before the end date.")


# ============================================================
# EXECUTION
# ============================================================

if run_button:
    if not bbox_is_valid:
        st.error("Cannot continue: the bounding box is invalid.")

    elif start_date > end_date:
        st.error("Cannot continue: the date range is invalid.")

    else:
        client_id, client_secret = get_credentials()

        if not client_id or not client_secret:
            st.error(
                "Sentinel Hub credentials are missing. Add your client ID "
                "and client secret to Streamlit Cloud → App settings → Secrets."
            )
            st.code(
                """
SH_CLIENT_ID = "your-client-id"
SH_CLIENT_SECRET = "your-client-secret"
                """.strip(),
                language="toml",
            )

        else:
            try:
                with st.spinner("Authenticating with Sentinel Hub..."):
                    token = get_access_token(client_id, client_secret)

                with st.spinner("Retrieving Sentinel-1 VV/VH imagery..."):
                    image = fetch_sentinel1_image(
                        token=token,
                        bbox=bbox,
                        start_date=start_date.isoformat(),
                        end_date=end_date.isoformat(),
                        width=image_size,
                        height=image_size,
                    )

                st.success("Sentinel-1 imagery retrieved successfully.")

                st.markdown("### Retrieved SAR visualisation")
                st.image(
                    image,
                    caption=(
                        f"SAR-derived pseudo-RGB | "
                        f"{start_date} to {end_date}"
                    ),
                    use_container_width=True,
                )

                with st.spinner("Loading trained segmentation model..."):
                    model = load_model()

                with st.spinner("Running experimental AI segmentation..."):
                    mask, probability = run_inference(model, image)

                overlay = make_overlay(image, mask)
                mask_image = make_mask_image(mask)
                probability_image = make_probability_image(probability)

                predicted_pixels = int(mask.sum())
                total_pixels = int(mask.size)
                predicted_percentage = (
                    100.0 * predicted_pixels / total_pixels
                    if total_pixels
                    else 0.0
                )

                st.markdown("### Screening results")

                metric1, metric2, metric3 = st.columns(3)

                metric1.metric(
                    "Predicted target pixels",
                    f"{predicted_pixels:,}",
                )

                metric2.metric(
                    "Image area flagged",
                    f"{predicted_percentage:.2f}%",
                )

                metric3.metric(
                    "Mean target probability",
                    f"{float(probability.mean()):.3f}",
                )

                st.warning(
                    "These values describe model predictions, not confirmed "
                    "oil spills. Dark SAR areas can also result from low wind, "
                    "look-alikes, sensor conditions, or other surface effects."
                )

                st.markdown("### AI output")

                col1, col2 = st.columns(2)

                with col1:
                    st.image(
                        overlay,
                        caption="Predicted target regions overlaid on SAR imagery",
                        use_container_width=True,
                    )

                with col2:
                    st.image(
                        mask_image,
                        caption="Binary segmentation mask",
                        use_container_width=True,
                    )

                st.markdown("### Target-class probability")

                st.image(
                    probability_image,
                    caption=(
                        "Brighter pixels indicate higher model probability "
                        "for class index 1."
                    ),
                    use_container_width=True,
                )

                st.markdown("### Download outputs")

                download_col1, download_col2, download_col3 = st.columns(3)

                with download_col1:
                    st.download_button(
                        "Download SAR image",
                        data=BytesIO(
                            _image_bytes := (
                                lambda buffer: (
                                    image.save(buffer, format="PNG"),
                                    buffer.getvalue(),
                                )[1]
                            )(BytesIO())
                        ).getvalue(),
                        file_name="veyora_sar_image.png",
                        mime="image/png",
                        use_container_width=True,
                    )

                with download_col2:
                    mask_buffer = BytesIO()
                    mask_image.save(mask_buffer, format="PNG")

                    st.download_button(
                        "Download AI mask",
                        data=mask_buffer.getvalue(),
                        file_name="veyora_ai_mask.png",
                        mime="image/png",
                        use_container_width=True,
                    )

                with download_col3:
                    overlay_buffer = BytesIO()
                    overlay.save(overlay_buffer, format="PNG")

                    st.download_button(
                        "Download overlay",
                        data=overlay_buffer.getvalue(),
                        file_name="veyora_ai_overlay.png",
                        mime="image/png",
                        use_container_width=True,
                    )

                st.caption(
                    f"Bounding box: {bbox} | "
                    f"Dates: {start_date} to {end_date} | "
                    f"Device: {DEVICE}"
                )

            except Exception as exc:
                st.error("The screening process could not be completed.")
                st.code(f"{type(exc).__name__}: {exc}")

                st.markdown(
                    """
                    **Troubleshooting**
                    - Check that Sentinel Hub credentials are correct.
                    - Check that `models/best.pt` exists in your repository.
                    - If the error mentions missing or unexpected model keys,
                      the `TinyUNet` architecture in `tiny_unet.py` must match
                      the architecture used during training.
                    - If Sentinel Hub reports no data, try another date range
                      or a different area of interest.
                    """
                )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "VEYORA | Experimental satellite-based oil-spill screening. "
    "AI outputs are not a substitute for expert interpretation or "
    "independent confirmation."
)
