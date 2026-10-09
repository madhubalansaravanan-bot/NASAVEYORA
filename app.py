from pathlib import Path
from io import BytesIO
from datetime import date, timedelta

import folium
from folium.plugins import Draw
import numpy as np
import requests
import streamlit as st
import torch
from PIL import Image
from streamlit_folium import st_folium
from tiny_unet import TinyUNet

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "best.pt"
DEVICE = torch.device("cpu")
TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
PROCESS_URL = "https://sh.dataspace.copernicus.eu/api/v1/process"

st.set_page_config(page_title="VEYORA | SAR Oil Spill Screening", page_icon="🌊", layout="wide")
st.markdown("""
<style>
.stApp {background:#071521;color:#e8f1f8}
h1,h2,h3 {color:#62d9e8!important}
div[data-testid="stMetric"] {background:#102736;border:1px solid #245064;padding:12px;border-radius:10px}
</style>
""", unsafe_allow_html=True)


def secret(*names):
    for name in names:
        try:
            value = st.secrets.get(name)
            if value:
                return str(value).strip()
        except Exception:
            pass
    return None


def credentials():
    return (
        secret("CDSE_CLIENT_ID", "SH_CLIENT_ID", "SENTINEL_HUB_CLIENT_ID", "CLIENT_ID"),
        secret("CDSE_CLIENT_SECRET", "SH_CLIENT_SECRET", "SENTINEL_HUB_CLIENT_SECRET", "CLIENT_SECRET"),
    )


@st.cache_data(ttl=3000, show_spinner=False)
def get_token(client_id, client_secret):
    r = requests.post(TOKEN_URL, data={
        "grant_type": "client_credentials",
        "client_id": client_id,
        "client_secret": client_secret,
    }, timeout=30)
    if not r.ok:
        raise RuntimeError(f"CDSE authentication failed ({r.status_code}): {r.text[:500]}")
    token = r.json().get("access_token")
    if not token:
        raise RuntimeError("Authentication response did not contain an access token.")
    return token


def evalscript():
    return """
    //VERSION=3
    function setup(){return {input:[{bands:["VV","VH"],units:"LINEAR_POWER"}],
      output:{bands:3,sampleType:"UINT8"}};}
    function clamp01(x){return Math.max(0,Math.min(1,x));}
    function toDb(x){return 10*Math.log(Math.max(x,0.00000001))/Math.LN10;}
    function evaluatePixel(s){
      let vvDb=toDb(s.VV), vhDb=toDb(s.VH);
      return [
        Math.round(clamp01((vvDb+30)/30)*255),
        Math.round(clamp01((vhDb+35)/30)*255),
        Math.round(clamp01(((vvDb-vhDb)+5)/25)*255)
      ];
    }
    """


def fetch_image(token, bbox, start, end, size):
    payload = {
        "input": {
            "bounds": {"bbox": bbox, "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"}},
            "data": [{"type": "sentinel-1-grd", "dataFilter": {
                "timeRange": {"from": f"{start}T00:00:00Z", "to": f"{end}T23:59:59Z"},
                "resolution": "HIGH", "acquisitionMode": "IW",
                "polarization": "DV", "orthorectify": True
            }}]
        },
        "output": {"width": size, "height": size, "responses": [
            {"identifier": "default", "format": {"type": "image/png"}}
        ]},
        "evalscript": evalscript()
    }
