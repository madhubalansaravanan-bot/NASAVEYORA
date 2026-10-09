import streamlit as st
import numpy as np
import cv2
from PIL import Image
import folium
from streamlit_folium import st_folium
import io

try:
    import rasterio
    from rasterio.io import MemoryFile
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False


st.set_page_config(
    page_title="VEYORA | SAR Intelligence",
    page_icon="🛰️",
    layout="wide"
)

st.markdown("""
<style>
.stApp {background-color: #07111f; color: #e5eefb;}
[data-testid="stMetric"] {
    background: #101f32;
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #233b55;
}
h1, h2, h3 {color: #65d9ff;}
</style>
""", unsafe_allow_html=True)

st.title("🛰️ VEYORA")
st.subheader("SAR-Based Marine Oil Spill Screening")
st.caption(
    "Earth observation • Image analysis • Environmental awareness"
)

st.warning(
    "Research prototype: dark SAR features may be oil, but can also "
    "be natural look-alikes. This screening algorithm is not a trained "
    "or validated oil-spill AI model."
)

with st.sidebar:
    st.header("Mission settings")
    latitude = st.number_input(
        "Approximate latitude",
        min_value=-90.0, max_value=90.0,
        value=13.05, step=0.01
    )
    longitude = st.number_input(
        "Approximate longitude",
        min_value=-180.0, max_value=180.0,
        value=80.32, step=0.01
    )
    threshold = st.slider(
        "Dark-pixel threshold",
        min_value=5, max_value=120, value=45
    )
    min_region = st.slider(
        "Minimum region size (pixels)",
        min_value=10, max_value=2000, value=100
    )

st.markdown("### 1. Upload satellite imagery")

uploaded = st.file_uploader(
    "Upload a Sentinel-1 SAR image",
    type=["png", "jpg", "jpeg", "tif", "tiff"]
)

if uploaded is None:
    st.info(
        "Upload a SAR image to begin. For best results, use a "
        "single-band grayscale SAR image or a georeferenced GeoTIFF."
    )
    st.markdown("""
    **Prototype workflow**

    1. Upload SAR imagery.
    2. Inspect and preprocess the image.
    3. Screen for dark regions.
    4. Review candidate regions and their locations.
    5. Export the screening result for further analysis.
    """)
    st.stop()

raw_bytes = uploaded.getvalue()
geotiff = uploaded.name.lower().endswith((".tif", ".tiff"))
transform = None
crs = None
pixel_area_m2 = None

try:
    if geotiff and RASTERIO_AVAILABLE:
        with MemoryFile(raw_bytes) as memfile:
            with memfile.open() as src:
                band = src.read(1).astype(np.float32)
                transform = src.transform
                crs = src.crs
                if src.crs and src.crs.is_projected:
                    pixel_area_m2 = abs(
                        src.transform.a * src.transform.e
                        - src.transform.b * src.transform.d
                    )
                nodata = src.nodata
                if nodata is not None:
                    band[band == nodata] = np.nan

        valid = np.isfinite(band)
        if not valid.any():
            st.error("The GeoTIFF contains no valid pixels.")
            st.stop()

        low, high = np.nanpercentile(band, [2, 98])
        if high <= low:
            st.error("Image has insufficient intensity variation.")
            st.stop()

        scaled = np.nan_to_num(
            (band - low) / (high - low) * 255,
            nan=0
        )
        gray = np.clip(scaled, 0, 255).astype(np.uint8)
        gray[~valid] = 0

    else:
        image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        rgb = np.array(image)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        valid = np.ones(gray.shape, dtype=bool)

except Exception as exc:
    st.error(f"Unable to read this image: {exc}")
    st.stop()

# Light smoothing reduces isolated pixel noise.
smoothed = cv2.GaussianBlur(gray, (5, 5), 0)

# Simple dark-region screening, NOT a trained AI model.
candidate = ((smoothed < threshold) & valid).astype(np.uint8)

count, labels, stats, centroids = cv2.connectedComponentsWithStats(
    candidate, connectivity=8
)

mask = np.zeros_like(candidate, dtype=np.uint8)
regions = []

for i in range(1, count):
    area_px = int(stats[i, cv2.CC_STAT_AREA])
    if area_px >= min_region:
        mask[labels == i] = 1
        regions.append({
            "id": len(regions) + 1,
            "pixels": area_px,
            "cx": float(centroids[i][0]),
            "cy": float(centroids[i][1])
        })

mask_pixels = int(mask.sum())
valid_pixels = int(valid.sum())
coverage = 100 * mask_pixels / max(valid_pixels, 1)

# Create a visual overlay.
rgb_display = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
overlay = rgb_display.copy()
overlay[mask == 1] = [255, 55, 55]
visual = cv2.addWeighted(rgb_display, 0.70, overlay, 0.30, 0)

st.markdown("### 2. Image analysis")

left, right = st.columns(2)

with left:
    st.image(gray, caption="Processed SAR intensity", use_container_width=True)

with right:
    st.image(visual, caption="Dark-region candidates in red", use_container_width=True)

m1, m2, m3 = st.columns(3)
m1.metric("Candidate regions", len(regions))
m2.metric("Candidate pixels", f"{mask_pixels:,}")
m3.metric("Image coverage", f"{coverage:.2f}%")

if len(regions):
    st.warning(
        f"{len(regions)} dark region(s) passed the selected pixel-size "
        "filter. These are candidates, not confirmed oil spills."
    )
else:
    st.info("No dark regions passed the current screening settings.")

# Only calculate area when a projected GeoTIFF supplies usable pixel scale.
st.markdown("### 3. Area estimate")

if pixel_area_m2 is not None:
    estimated_area_km2 = mask_pixels * pixel_area_m2 / 1_000_000
    st.metric("Candidate area", f"{estimated_area_km2:.4f} km²")
    st.caption(
        "This is the area of threshold-selected pixels, not a validated "
        "oil-spill area. Confirm the CRS, pixel scale, calibration and mask."
    )
else:
    st.info(
        "Physical area is unavailable because the image has no usable "
        "projected pixel scale. Upload a correctly georeferenced GeoTIFF "
        "to estimate area."
    )

# Export binary mask and overlay.
mask_png = Image.fromarray(mask * 255)
buffer = io.BytesIO()
mask_png.save(buffer, format="PNG")

st.download_button(
    "Download candidate mask (PNG)",
    data=buffer.getvalue(),
    file_name="veyora_candidate_mask.png",
    mime="image/png"
)

overlay_buffer = io.BytesIO()
Image.fromarray(visual).save(overlay_buffer, format="PNG")

st.download_button(
    "Download analysis overlay (PNG)",
    data=overlay_buffer.getvalue(),
    file_name="veyora_analysis_overlay.png",
    mime="image/png"
)

st.markdown("### 4. Location reference")

st.caption(
    "The pin below is the approximate coordinate entered in the sidebar. "
    "It is not automatically derived from the uploaded image."
)

map_object = folium.Map(
    location=[latitude, longitude],
    zoom_start=7,
    tiles="OpenStreetMap"
)

folium.Marker(
    [latitude, longitude],
    tooltip="User-supplied reference coordinate",
    popup=f"Reference: {latitude:.4f}, {longitude:.4f}",
    icon=folium.Icon(color="blue", icon="info-sign")
).add_to(map_object)

st_folium(map_object, width=None, height=420)

st.markdown("---")
st.caption(
    "VEYORA prototype • SAR dark-target screening • "
    "Human review and independent validation required"
)
