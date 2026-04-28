import os
import io
import base64
import math
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from contextlib import nullcontext
from functools import lru_cache

import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

# Geo / raster
import rasterio
from rasterio.windows import from_bounds, Window
from rasterio.windows import transform as win_transform, bounds as win_bounds
from rasterio.enums import Resampling
from rasterio.features import geometry_mask
from rasterio.warp import reproject
from affine import Affine
from shapely.geometry import shape, mapping
from shapely.ops import transform as shp_transform
from pyproj import Transformer
from scipy.ndimage import gaussian_filter, zoom

# STAC & providers
from pystac_client import Client
import planetary_computer

# plotting / image
from PIL import Image

# ---------- Environment / tuning ----------
os.environ.setdefault("CPL_VSIL_CURL_USE_HEAD", "FALSE")
os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif,.tiff,.jp2,.JP2,.TIF,.TIFF,.JP2")
os.environ.setdefault("AWS_NO_SIGN_REQUEST", "YES")
os.environ.setdefault("GDAL_HTTP_MULTIRANGE", "YES")
os.environ.setdefault("GDAL_CACHEMAX", "512")

EARTH_SEARCH_AWS = "https://earth-search.aws.element84.com/v1"
PLANETARY_STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"
SENTINEL1_COLLECTION = "sentinel-1-grd"
SENTINEL2_COLLECTIONS = ["sentinel-2-l2a", "sentinel-2-l1c"]

# default threads (reduce if network-bound)
THREADS = min(4, (os.cpu_count() or 4))

# Best-effort diagnostics to explain provider fallback failures.
_PROVIDER_ERRORS: Dict[str, str] = {}

def _set_provider_error(provider: str, err: Exception) -> None:
    _PROVIDER_ERRORS[provider] = str(err)

def _clear_provider_error(provider: str) -> None:
    _PROVIDER_ERRORS.pop(provider, None)

def get_provider_error_summary() -> str:
    if not _PROVIDER_ERRORS:
        return ""
    ordered = []
    for p in ("planetary", "aws"):
        if p in _PROVIDER_ERRORS:
            ordered.append(f"{p}: {_PROVIDER_ERRORS[p]}")
    for p, msg in _PROVIDER_ERRORS.items():
        if p not in {"planetary", "aws"}:
            ordered.append(f"{p}: {msg}")
    return " | ".join(ordered)


@lru_cache(maxsize=2)
def get_planetary_client():
    return Client.open(PLANETARY_STAC)


@lru_cache(maxsize=2)
def get_aws_client():
    return Client.open(EARTH_SEARCH_AWS)


def get_collections_for_satellite(satellite: str) -> List[str]:
    sat = (satellite or "s2").lower()
    if sat.startswith("s1"):
        return [SENTINEL1_COLLECTION]
    return SENTINEL2_COLLECTIONS


def get_provider_search_order(provider: Optional[str], prefer_pc_default: bool = True) -> List[str]:
    mode = (provider or "both").lower()
    if mode in {"pc", "planetary", "planetary-computer"}:
        return ["planetary"]
    if mode == "aws":
        return ["aws"]
    if prefer_pc_default:
        return ["planetary", "aws"]
    return ["aws", "planetary"]

# ---------- Palettes & labels ----------
# Default palette and labels (using NDVI as fallback)
PALETTE = [
    '#ffffff', '#d73027', '#f46d43', '#fdae61', '#fee08b', '#d9ef8b',
    '#a6d96a', '#66bd63', '#1a9850', '#006837', '#004529'
]
LABELS = [
    'Clouds', 'Very Poor', 'Poor', 'Fair', 'Moderate', 'Good',
    'Very Good', 'Excellent', 'Dense', 'Very Dense', 'Extreme'
]
EDGES = np.array([-0.2,0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.01], dtype="float32")
# Per-index bin edges (12 values -> 11 bins including "clouds"/nodata bucket handling in caller).
# CCC can exceed 1.0 in healthy crop canopies, so it needs wider edges than NDVI-like indices.
INDEX_EDGES = {
    "CCC": np.array([0.2, 0.5, 0.8, 1.0, 1.2, 1.5, 1.8, 2.2, 2.8, 3.5, 5.0, 8.0], dtype="float32"),
}

def get_index_edges(index_name: str) -> np.ndarray:
    return INDEX_EDGES.get((index_name or "").upper(), EDGES)

index_palettes_labels = {
    'NDVI': {
        'palette': [
            '#ffffff', '#d73027', '#f46d43', '#fdae61', '#fee08b', '#d9ef8b',
            '#a6d96a', '#66bd63', '#1a9850', '#006837', '#004529'
        ],
        'labels': [
            'Clouds', 'Very Poor', 'Poor', 'Fair', 'Moderate', 'Good',
            'Very Good', 'Excellent', 'Dense', 'Very Dense', 'Extreme'
        ]
    },
    'EVI': {
        'palette': [
            '#ffffff',  # Clouds - white
            '#a50026',  # Very Low - dark red
            '#d73027',
            '#f46d43',
            '#fdae61',
            '#d9ef8b',
            '#d9ef8b',
            '#a6d96a',
            '#66bd63',
            '#1a9850',
            '#006837'   # Extreme - dark green
        ],
        'labels': [
            'Clouds', 'Very Low', 'Low', 'Fair', 'Moderate',
            'Good', 'Very Good', 'Excellent', 'Dense', 'Very Dense', 'Extreme'
        ]
    },
    'EVI2': {
        'palette': [
            '#ffffff',
            '#a50026', '#d73027', '#f46d43', '#fdae61', '#d9ef8b',
            '#d9ef8b', '#a6d96a', '#66bd63', '#1a9850', '#006837'
        ],
        'labels': [
            'Clouds', 'Very Low', 'Low', 'Fair', 'Moderate',
            'Good', 'Very Good', 'Excellent', 'Dense', 'Very Dense', 'Extreme'
        ]
    },
    'SAVI': {
        'palette': [
            '#ffffff',
            '#a50026', '#d73027', '#f46d43', '#fdae61', '#fee08b',
            '#d9ef8b', '#a6d96a', '#66bd63', '#1a9850', '#006837'
        ],
        'labels': [
            'Clouds', 'Very Low', 'Low', 'Fair', 'Moderate',
            'Good', 'Very Good', 'Excellent', 'Dense', 'Very Dense', 'Extreme'
        ]
    },
    'MSAVI': {
        'palette': [
            '#ffffff',
            '#a50026', '#d73027', '#f46d43', '#fdae61', '#fee08b',
            '#d9ef8b', '#a6d96a', '#66bd63', '#1a9850', '#006837'
        ],
        'labels': [
            'Clouds', 'Very Low', 'Low', 'Fair', 'Moderate',
            'Good', 'Very Good', 'Excellent', 'Dense', 'Very Dense', 'Extreme'
        ]
    },
    'NDRE': {
        'palette': [
            '#ffffff',
            '#a50026', '#d73027', '#f46d43', '#fdae61', '#fee08b',
            '#d9ef8b', '#a6d96a', '#66bd63', '#1a9850', '#006837'
        ],
        'labels': [
            'Clouds', 'Very Low', 'Low', 'Fair', 'Moderate',
            'Good', 'Very Good', 'Excellent', 'Dense', 'Very Dense', 'Extreme'
        ]
    },
    'CCC': {
        'palette': [
            '#ffffff',
            '#a50026', '#d73027', '#f46d43', '#fdae61', '#fee08b',
            '#d9ef8b', '#a6d96a', '#66bd63', '#1a9850', '#006837'
        ],
        'labels': [
            'Clouds', 'Very Low', 'Low', 'Fair', 'Moderate',
            'Good', 'Very Good', 'Excellent', 'Rich', 'Very Rich', 'Excessive'
        ]
    },
    'NITROGEN': {
        'palette': [
            '#ffffff',
            '#a50026', '#d73027', '#f46d43', '#fdae61', '#fee08b',
            '#d9ef8b', '#a6d96a', '#66bd63', '#1a9850', '#006837'
        ],
        'labels': [
            'Clouds', 'Very Low', 'Low', 'Fair', 'Moderate',
            'Good', 'Very Good', 'Excellent', 'Rich', 'Very Rich', 'Excessive'
        ]
    },
    'SOC': {
        'palette': [
            '#ffffe5', '#f7fcb9', '#d9f0a3', '#addd8e', '#78c679',
            '#41ab5d', '#238443', '#006837', '#004529', '#002d1d'
        ],
        'labels': [
            'Very Low', 'Low', 'Fair', 'Moderate',
            'Good', 'Very Good', 'Excellent', 'Rich', 'Very Rich', 'Excessive'
        ]
    },
    'RECI': {
        'palette': [
            '#ffffff',
            '#a50026', '#d73027', '#f46d43', '#fdae61', '#fee08b',
            '#d9ef8b', '#a6d96a', '#66bd63', '#1a9850', '#006837'
        ],
        'labels': [
            'Clouds', 'Very Low', 'Low', 'Fair', 'Moderate',
            'Good', 'Very Good', 'Excellent', 'Rich', 'Very Rich', 'Excessive'
        ]
    },
    'NDMI': {
        'palette': [
            '#ffffff',  # Clouds
            '#deebf7', '#c6dbef', '#9ecae1', '#6baed6', '#4292c6',
            '#2171b5', '#08519c', '#08306b', '#041b3d', '#021122'
        ],
        'labels': [
            'Clouds', 'Very Dry', 'Dry', 'Low Moisture', 'Moderate Moisture',
            'Moist', 'Good Moisture', 'High Moisture', 'Very Moist', 'Wet', 'Waterlogged'
        ]
    },
    'NDWI': {
        'palette': [
            '#ffffff',
            '#eff3ff', '#bdd7e7', '#6baed6', '#3182bd', '#08519c',
            '#084081', '#042340', '#021720', '#010d10', '#000500'
        ],
        'labels': [
            'Clouds', 'Very Dry', 'Dry', 'Low Water', 'Moderate Water',
            'Moist', 'High Moisture', 'Very Moist', 'Wet', 'Water Saturated', 'Waterlogged'
        ]
    },
    'SMI': {
        'palette': [
            '#ffffff',
            '#deebf7', '#c6dbef', '#9ecae1', '#6baed6', '#4292c6',
            '#2171b5', '#08519c', '#08306b', '#041b3d', '#041A33'
        ],
        'labels': [
            'Clouds', 'Very Low', 'Low', 'Fair', 'Moderate',
            'Good', 'Very Good', 'Excellent', 'Wet', 'Very Wet', 'Flooded'
        ]
    }
}

PALETTE_MAP = {
    'NDVI': (index_palettes_labels['NDVI']['palette'], index_palettes_labels['NDVI']['labels']),
    'EVI': (index_palettes_labels['EVI']['palette'], index_palettes_labels['EVI']['labels']),
    'EVI2': (index_palettes_labels['EVI2']['palette'], index_palettes_labels['EVI2']['labels']),
    'SAVI': (index_palettes_labels['SAVI']['palette'], index_palettes_labels['SAVI']['labels']),
    'MSAVI': (index_palettes_labels['MSAVI']['palette'], index_palettes_labels['MSAVI']['labels']),
    'NDRE': (index_palettes_labels['NDRE']['palette'], index_palettes_labels['NDRE']['labels']),
    'CCC': (index_palettes_labels['CCC']['palette'], index_palettes_labels['CCC']['labels']),
    'NITROGEN': (index_palettes_labels['NITROGEN']['palette'], index_palettes_labels['NITROGEN']['labels']),
    'SOC': (index_palettes_labels['SOC']['palette'], index_palettes_labels['SOC']['labels']),
    'RECI': (index_palettes_labels['RECI']['palette'], index_palettes_labels['RECI']['labels']),
    'NDMI': (index_palettes_labels['NDMI']['palette'], index_palettes_labels['NDMI']['labels']),
    'NDWI': (index_palettes_labels['NDWI']['palette'], index_palettes_labels['NDWI']['labels']),
    'SMI': (index_palettes_labels['SMI']['palette'], index_palettes_labels['SMI']['labels']),
    'TRUE_COLOR': (["#000000"], ["True Color"]),
}

# ---------- Utility / index math ----------
def sign_href_if_pc(href: Optional[str]) -> Optional[str]:
    if not href:
        return None
    try:
        return planetary_computer.sign(href)
    except Exception:
        return href

def s3_to_https(href: str) -> str:
    if href and href.startswith("s3://"):
        parts = href[5:].split("/", 1)
        bucket = parts[0]; key = parts[1] if len(parts) > 1 else ""
        return f"https://{bucket}.s3.amazonaws.com/{key}"
    return href

def prefer_http_from_asset(asset) -> Optional[str]:
    if asset is None:
        return None
    href = getattr(asset, "href", "") or ""
    alt = getattr(asset, "extra_fields", {}).get("alternate", {}) if hasattr(asset, "extra_fields") else {}
    for k in ("https", "http", "self"):
        v = alt.get(k)
        if isinstance(v, dict):
            url = v.get("href")
            if url and url.startswith("http"):
                return url
        elif isinstance(v, str) and v.startswith("http"):
            return v
    if href.startswith("http"):
        return href
    return s3_to_https(href) if href else None

def compute_index_array_by_name(index_name: str, bands: Dict[str, np.ndarray]) -> np.ndarray:
    for k in list(bands.keys()):
        if bands[k] is not None:
            bands[k] = bands[k].astype("float32")
    name = index_name.upper()
    eps = 1e-6
    def missing(*args):
        return any(arg is None for arg in args)

    if name == "NDVI":
        NIR, RED = bands.get("B08"), bands.get("B04")
        if missing(NIR, RED): raise ValueError("NDVI requires B08 and B04")
        return (NIR - RED) / (NIR + RED + eps)
    if name == "EVI":
        NIR, RED, BLUE = bands.get("B08"), bands.get("B04"), bands.get("B02")
        if missing(NIR, RED, BLUE): raise ValueError("EVI requires B08,B04,B02")
        return 2.5 * (NIR - RED) / (NIR + 6*RED - 7.5*BLUE + 1.0 + eps)
    if name == "EVI2":
        NIR, RED = bands.get("B08"), bands.get("B04")
        if missing(NIR, RED): raise ValueError("EVI2 requires B08,B04")
        return 2.5 * (NIR - RED) / (NIR + 2.4*RED + 1.0 + eps)
    if name == "SAVI":
        NIR, RED = bands.get("B08"), bands.get("B04")
        if missing(NIR, RED): raise ValueError("SAVI requires B08,B04")
        L = 0.5
        return ((NIR - RED) / (NIR + RED + L + eps)) * (1.0 + L)
    if name == "MSAVI":
        NIR, RED = bands.get("B08"), bands.get("B04")
        if missing(NIR, RED): raise ValueError("MSAVI requires B08,B04")
        a = 2*NIR + 1.0
        inside = np.maximum((a*a) - 8*(NIR - RED), 0.0)
        return 0.5 * (a - np.sqrt(inside))
    if name == "NDMI":
        NIR, SWIR = bands.get("B08"), bands.get("B11")
        if missing(NIR, SWIR): raise ValueError("NDMI requires B08,B11")
        return (NIR - SWIR) / (NIR + SWIR + eps)
    if name == "NDWI":
        GREEN, SWIR = bands.get("B03"), bands.get("B11")
        if missing(GREEN, SWIR): raise ValueError("NDWI requires B03,B11")
        return (GREEN - SWIR) / (GREEN + SWIR + eps)
    if name == "SMI":
        NIR, SWIR = bands.get("B08"), bands.get("B11")
        if missing(NIR, SWIR): raise ValueError("SMI requires B08,B11")
        return (NIR - SWIR) / (NIR + SWIR + eps)
    if name == "CCC":
        B3, B4, B5 = bands.get("B03"), bands.get("B04"), bands.get("B05")
        if missing(B3, B4, B5): raise ValueError("CCC requires B03,B04,B05")
        return (B5 * B5) / (B4 * B3 + eps)
    if name == "NITROGEN":
        B4, B5 = bands.get("B04"), bands.get("B05")
        if missing(B4, B5): raise ValueError("NITROGEN requires B04,B05")
        return (B5 - B4) / (B5 + B4 + eps)
    if name == "SOC":
        B3, B4, B11, B12 = bands.get("B03"), bands.get("B04"), bands.get("B11"), bands.get("B12")
        if missing(B3, B4, B11, B12): raise ValueError("SOC requires B03,B04,B11,B12")
        return (B3 + B4) / (B11 + B12 + eps)
    if name == "NDRE":
        B8A, B5 = bands.get("B8A"), bands.get("B05")
        if missing(B8A, B5): raise ValueError("NDRE requires B8A,B05")
        return (B8A - B5) / (B8A + B5 + eps)
    if name == "RECI":
        B8, B5 = bands.get("B08"), bands.get("B05")
        if missing(B8, B5): raise ValueError("RECI requires B08,B05")
        return (B8 - B5) / (B5 + eps)
    if name == "TRUE_COLOR":
        raise ValueError("TRUE_COLOR is a special rendering path (RGB)")

    raise ValueError(f"Unsupported index: {index_name}")

# ---------- Grid / read helpers ----------
def aoi_to_scene(aoi_ll_geojson, crs_str):
    t = Transformer.from_crs("EPSG:4326", crs_str, always_xy=True)
    return shp_transform(lambda x, y, z=None: t.transform(x, y), shape(aoi_ll_geojson))

def build_adaptive_grid(crs, aoi_ll_geojson, native_res_m=10.0,
                        MIN_PX_LONG=600, MAX_PX_LONG=1200, MIN_RES_M=0.5, MAX_RES_M=40.0):
    aoi_sc = aoi_to_scene(aoi_ll_geojson, crs.to_string())
    minx, miny, maxx, maxy = aoi_sc.bounds
    dx = max(maxx - minx, 1e-6)
    dy = max(maxy - miny, 1e-6)
    long = max(dx, dy)
    res_for_min = long / MAX_PX_LONG
    res_for_max = long / MIN_PX_LONG
    res_m = min(max(res_for_min, native_res_m / 2.0), res_for_max)
    res_m = float(np.clip(res_m, MIN_RES_M, MAX_RES_M))
    width = max(1, int(math.ceil(dx / res_m)))
    height = max(1, int(math.ceil(dy / res_m)))
    transform = Affine.translation(minx, maxy) * Affine.scale(res_m, -res_m)
    return aoi_sc, transform, height, width, res_m

def read_band_window(src, geom_sc, out_h, out_w, transform, resampling):
    win = from_bounds(*geom_sc.bounds, src.transform).round_offsets().round_lengths()
    if win.width <= 0 or win.height <= 0:
        return np.full((out_h, out_w), np.nan, dtype="float32")
    try:
        arr = src.read(1, window=win, out_shape=(out_h, out_w), resampling=resampling, masked=True).filled(0).astype("float32")
        return arr
    except Exception:
        arr = src.read(1, window=win, masked=True).filled(0).astype("float32")
    src_tr = win_transform(win, src.transform)
    dst = np.full((out_h, out_w), np.nan, dtype="float32")
    reproject(arr, dst,
              src_transform=src_tr, src_crs=src.crs,
              dst_transform=transform, dst_crs=src.crs,
              src_nodata=0.0, dst_nodata=np.nan,
              resampling=resampling)
    return dst

def read_scl_window(src, geom_sc, out_h, out_w, transform):
    win = from_bounds(*geom_sc.bounds, src.transform).round_offsets().round_lengths()
    if win.width <= 0 or win.height <= 0:
        return np.zeros((out_h, out_w), dtype="int16")
    try:
        arr = src.read(1, window=win, out_shape=(out_h, out_w), resampling=Resampling.nearest, masked=True).filled(0).astype("int16")
        return arr
    except Exception:
        arr = src.read(1, window=win, masked=True).filled(0).astype("int16")
    src_tr = win_transform(win, src.transform)
    dst = np.zeros((out_h, out_w), dtype="int16")
    reproject(arr, dst,
              src_transform=src_tr, src_crs=src.crs,
              dst_transform=transform, dst_crs=src.crs,
              src_nodata=0, dst_nodata=0,
              resampling=Resampling.nearest)
    return dst

# ---------- STAC helpers ----------
def search_planetary(collections, intersects, dt, limit=50):
    try:
        cat = get_planetary_client()
        items = list(cat.search(collections=collections, intersects=intersects, datetime=dt, limit=limit).items())
        _clear_provider_error("planetary")
        return items
    except Exception as e:
        _set_provider_error("planetary", e)
        # Fail-soft so callers can fall back to AWS when Planetary is slow/unavailable.
        try:
            get_planetary_client.cache_clear()
            cat = get_planetary_client()
            items = list(cat.search(collections=collections, intersects=intersects, datetime=dt, limit=limit).items())
            _clear_provider_error("planetary")
            return items
        except Exception as e2:
            _set_provider_error("planetary", e2)
            return []

def search_aws(collections, intersects, dt, limit=50):
    try:
        cat = get_aws_client()
        items = list(cat.search(collections=collections, intersects=intersects, datetime=dt, limit=limit).items())
        _clear_provider_error("aws")
        return items
    except Exception as e:
        _set_provider_error("aws", e)
        # Fail-soft for consistency with Planetary search helper.
        try:
            get_aws_client.cache_clear()
            cat = get_aws_client()
            items = list(cat.search(collections=collections, intersects=intersects, datetime=dt, limit=limit).items())
            _clear_provider_error("aws")
            return items
        except Exception as e2:
            _set_provider_error("aws", e2)
            return []

def items_for_date(aoi_geojson, iso_dt, collection, prefer_pc=True, limit=6):
    day = iso_dt[:10]
    dt = f"{day}/{day}"
    if prefer_pc:
        items = search_planetary([collection], aoi_geojson, dt, limit=limit)
        if items:
            return items
        return search_aws([collection], aoi_geojson, dt, limit=limit)
    else:
        items = search_aws([collection], aoi_geojson, dt, limit=limit)
        if items:
            return items
        return search_planetary([collection], aoi_geojson, dt, limit=limit)

# quick keep pct
def quick_keep_pct(item, aoi_geojson):
    assets = item.assets
    red = prefer_http_from_asset(assets.get("red") or assets.get("B04"))
    nir = prefer_http_from_asset(assets.get("nir") or assets.get("B08"))
    scl = assets.get("scl") or assets.get("SCL")
    scl_url = prefer_http_from_asset(scl) if scl else None
    if not (red and nir):
        return 0.0, False
    try:
        with rasterio.Env():
            with rasterio.open(sign_href_if_pc(red)) as rsrc, rasterio.open(sign_href_if_pc(nir)) as nsrc, \
                 (rasterio.open(sign_href_if_pc(scl_url)) if scl_url else nullcontext()) as sds:
                crs = rsrc.crs or nsrc.crs
                aoi_sc = aoi_to_scene(aoi_geojson, crs.to_string())
                win = from_bounds(*aoi_sc.bounds, rsrc.transform).round_offsets().round_lengths()
                if win.width <= 0 or win.height <= 0:
                    return 0.0, bool(sds)
                th = max(1, min(64, int(win.height))); tw = max(1, min(64, int(win.width)))
                sub = Window(win.col_off, win.row_off, win.width, win.height)
                R = rsrc.read(1, window=sub, out_shape=(th, tw), resampling=Resampling.nearest, masked=True).filled(0).astype("float32")
                N = nsrc.read(1, window=sub, out_shape=(th, tw), resampling=Resampling.nearest, masked=True).filled(0).astype("float32")
                if R.max() > 1.5 or N.max() > 1.5:
                    R *= 1/10000.0; N *= 1/10000.0
                den = (N + R); den[den == 0] = np.nan
                nd = (N - R) / den
                tr = win_transform(sub, rsrc.transform) * Affine.scale(win.width/float(tw), win.height/float(th))
                mask = geometry_mask([mapping(aoi_sc)], out_shape=(th, tw), transform=tr, invert=True)
                if sds:
                    tb = win_bounds(sub, rsrc.transform)
                    sw = from_bounds(*tb, transform=sds.transform).round_offsets().round_lengths()
                    sw = sw.intersection(Window(0, 0, sds.width, sds.height)).round_offsets().round_lengths()
                    if sw.width > 0 and sw.height > 0:
                        S = sds.read(1, window=sw, out_shape=(th, tw), resampling=Resampling.nearest, masked=True).filled(0).astype("int16")
                        classes = [8,9,10,11]
                        nd[np.isin(S, classes)] = np.nan
                kept = np.count_nonzero(np.isfinite(nd) & mask)
                total = np.count_nonzero(mask)
                return (kept / max(total, 1)) * 100.0, bool(sds)
    except Exception:
        return 0.0, False

# pick best item
def pick_best_item(aoi_geojson, start, end, prefer_pc=True, satellite="s2"):
    collections_try = get_collections_for_satellite(satellite)
    items = []
    if prefer_pc:
        items = search_planetary(collections_try, aoi_geojson, f"{start}/{end}", limit=12)
        if not items:
            items = search_aws(collections_try, aoi_geojson, f"{start}/{end}", limit=12)
    else:
        items = search_aws(collections_try, aoi_geojson, f"{start}/{end}", limit=12)
        if not items:
            items = search_planetary(collections_try, aoi_geojson, f"{start}/{end}", limit=12)
    if not items:
        fmt = "%Y-%m-%d"
        s = datetime.strptime(start, fmt) - timedelta(days=14)
        e = datetime.strptime(end, fmt) + timedelta(days=14)
        items = search_planetary(collections_try, aoi_geojson, f"{s.strftime(fmt)}/{e.strftime(fmt)}", limit=24)
        if not items:
            items = search_aws(collections_try, aoi_geojson, f"{s.strftime(fmt)}/{e.strftime(fmt)}", limit=24)
    if not items:
        return None, False, None

    scored = []
    with ThreadPoolExecutor(max_workers=min(6, len(items))) as ex:
        futs = {ex.submit(quick_keep_pct, it, aoi_geojson): it for it in items}
        for f in as_completed(futs):
            try:
                pct, has_scl = f.result()
            except Exception:
                pct, has_scl = 0.0, False
            scored.append((pct, futs[f], has_scl))
    scored.sort(key=lambda x: x[0], reverse=True)
    best = scored[0]
    def get_collection_id(it):
        try:
            col = getattr(it, "collection", None)
            if isinstance(col, str):
                return col
            if hasattr(col, "id"):
                return col.id
        except Exception:
            pass
        try:
            props = getattr(it, "properties", {}) or {}
            for k in ("collection", "collection_id"):
                if k in props:
                    return props[k]
        except Exception:
            pass
        return None
    return best[1], best[2], get_collection_id(best[1])

# read tile
def _read_tile_into_stack(item, aoi_geojson, dst_transform, H, W, want_scl, required_bands=None):
    out = {"used": False, "bands": {}, "S": None, "id": getattr(item, "id", None)}
    assets = item.assets or {}
    required_bands = list(dict.fromkeys(required_bands or ["B04", "B08"]))
    # pick best asset href heuristically
    def first_asset_href(assets_dict):
        # try some known names then fall back to first valid href
        for k in ("red","B04","B04.jp2","B04.tif","B04.TIF"):
            a = assets_dict.get(k)
            if a:
                h = prefer_http_from_asset(a)
                if h:
                    return h
        # fallback any asset with href
        for a in assets_dict.values():
            h = prefer_http_from_asset(a)
            if h:
                return h
        return None

    primary_asset = prefer_http_from_asset(
        assets.get("red") or assets.get("B04") or assets.get("RED") or first_asset_href(assets)
    )
    if not primary_asset:
        return out
    scl_ref = assets.get("scl") or assets.get("SCL")
    scl_url = prefer_http_from_asset(scl_ref) if (want_scl and scl_ref) else None
    scl_url = sign_href_if_pc(scl_url) if scl_url else None
    try:
        with rasterio.open(sign_href_if_pc(primary_asset)) as ref_ds, \
             (rasterio.open(scl_url) if scl_url else nullcontext()) as scl:
            crs = ref_ds.crs
            aoi_sc = aoi_to_scene(aoi_geojson, crs.to_string())
            S = None
            if scl:
                S = read_scl_window(scl, aoi_sc, H, W, dst_transform)
            out["S"] = S

            loaded_required = 0
            for bkey in required_bands:
                a = assets.get(bkey) or assets.get(bkey.lower())
                if not a:
                    continue
                url = prefer_http_from_asset(a)
                if not url:
                    continue
                url = sign_href_if_pc(url)
                try:
                    with rasterio.open(url) as ds:
                        arr = read_band_window(ds, aoi_sc, H, W, dst_transform, Resampling.bilinear)
                        if np.isfinite(arr).any() and np.nanmax(arr) > 1.5:
                            arr *= 1 / 10000.0
                        out["bands"][bkey] = arr
                        loaded_required += 1
                except Exception:
                    out["bands"][bkey] = None

            out["used"] = loaded_required > 0
            return out
    except Exception:
        return out

# rendering helpers
def ndvi_to_bins(arr: np.ndarray, max_bin: int = 10, edges: Optional[np.ndarray] = None) -> np.ndarray:
    edges_arr = np.asarray(edges if edges is not None else EDGES, dtype="float32")
    arr = np.clip(arr, float(edges_arr[0]), float(edges_arr[-1]) - 1e-6)
    bins = np.digitize(arr, edges_arr).astype("uint8")
    # Bin 0 is reserved for nodata/cloud in rendering. Any finite value should be
    # mapped to at least bin 1, then capped to the available data-bin range.
    bins = np.clip(bins, 1, max(1, int(max_bin))).astype("uint8")
    return np.where(np.isfinite(arr), bins, 0)

def compute_bounds_wgs84(transform: Affine, width: int, height: int, crs) -> List[float]:
    tl = transform * (0, 0)
    tr = transform * (width, 0)
    br = transform * (width, height)
    bl = transform * (0, height)
    xs = [tl[0], tr[0], br[0], bl[0]]
    ys = [tl[1], tr[1], br[1], bl[1]]
    transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    lons, lats = transformer.transform(xs, ys)
    return [float(min(lons)), float(min(lats)), float(max(lons)), float(max(lats))]

def hex_to_rgba_tuple(hexcolor: str) -> Tuple[int,int,int,int]:
    h = hexcolor.lstrip('#')
    if len(h) == 6:
        r = int(h[0:2], 16); g = int(h[2:4], 16); b = int(h[4:6], 16)
        return (r,g,b,255)
    return (0,0,0,255)

def darken_rgba(rgba: Tuple[int, int, int, int], factor: float = 0.82) -> Tuple[int, int, int, int]:
    r, g, b, a = rgba
    f = float(np.clip(factor, 0.0, 1.0))
    return (int(r * f), int(g * f), int(b * f), a)

def render_spread_png_fast(bins_canvas: np.ndarray, NDVI_canvas: np.ndarray, res_m: float,
                           supersample: int, smooth: bool, gaussian_sigma: float,
                           out_w: int, out_h: int, palette: Optional[List[str]] = None,
                           labels: Optional[List[str]] = None, nodata_transparent: bool = True,
                           aoi_ll_geojson: Optional[dict] = None, transform: Optional[Affine] = None,
                           crs = None, max_bin: int = 10, edges: Optional[np.ndarray] = None) -> str:
    if palette is None:
        palette = PALETTE
    if labels is None:
        labels = LABELS

    # Create geometry mask BEFORE upsampling to ensure proper clipping
    Hs_orig, Ws_orig = bins_canvas.shape
    aoi_mask_orig = None
    if aoi_ll_geojson is not None and transform is not None:
        try:
            if crs is not None:
                crs_str = crs.to_string() if hasattr(crs, 'to_string') else str(crs)
            else:
                crs_str = "EPSG:4326"
            aoi_sc = aoi_to_scene(aoi_ll_geojson, crs_str)
            aoi_mask_orig = geometry_mask([mapping(aoi_sc)], out_shape=(Hs_orig, Ws_orig), transform=transform, invert=True)
        except Exception:
            aoi_mask_orig = None

    z = max(1, int(supersample))
    if (z > 1 or smooth) and NDVI_canvas is not None:
        V = np.where(np.isfinite(NDVI_canvas), NDVI_canvas, 0.0).astype("float32")
        M = np.isfinite(NDVI_canvas).astype("float32")
        
        # Apply AOI mask to the data BEFORE upsampling
        if aoi_mask_orig is not None:
            V = np.where(aoi_mask_orig, V, 0.0)
            M = np.where(aoi_mask_orig, M, 0.0)
        
        if z > 1:
            Vz = zoom(V, z, order=0)
            Mz = zoom(M, z, order=0)
            NDVI_up = np.where(Mz > 1e-6, Vz / Mz, np.nan)
        else:
            NDVI_up = NDVI_canvas
        if smooth:
            sigma = max(0.1, float(gaussian_sigma))
            inside = np.isfinite(NDVI_up).astype("float32")
            vals = np.where(np.isfinite(NDVI_up), NDVI_up, 0.0).astype("float32")
            num = gaussian_filter(vals, sigma=sigma)
            den = gaussian_filter(inside, sigma=sigma)
            NDVI_up = np.where(den > 1e-6, num / den, np.nan)
        edges_arr = np.asarray(edges if edges is not None else EDGES, dtype="float32")
        bins_up = np.digitize(
            np.clip(NDVI_up, float(edges_arr[0]), float(edges_arr[-1]) - 1e-6),
            edges_arr
        ).astype("uint8")
        bins_up = np.clip(bins_up, 1, max(1, int(max_bin))).astype("uint8")
        bins_up = np.where(np.isfinite(NDVI_up), bins_up, 0)
    else:
        bins_up = bins_canvas.astype("uint8")
        # Apply mask to bins directly if no upsampling
        if aoi_mask_orig is not None:
            bins_up = np.where(aoi_mask_orig, bins_up, 0)

    Hs, Ws = bins_up.shape
    palette_rgba = [hex_to_rgba_tuple(c) for c in palette]
    # Darken index classes slightly so map colors are less faint and easier to read.
    # Keep the first palette entry unchanged (typically "Clouds"/NoData color).
    if len(palette_rgba) > 1:
        palette_rgba = [palette_rgba[0]] + [darken_rgba(c, factor=0.82) for c in palette_rgba[1:]]
    if len(palette_rgba) < 2:
        palette_rgba = [(0,0,0,0), (0,255,0,255)]

    # Upscale the mask if we upsampled the data
    aoi_mask = None
    if aoi_mask_orig is not None:
        if z > 1:
            # Upscale the mask using nearest neighbor to match bins_up dimensions
            aoi_mask = zoom(aoi_mask_orig.astype("float32"), z, order=0) > 0.5
        else:
            aoi_mask = aoi_mask_orig

    rgba = np.zeros((Hs, Ws, 4), dtype=np.uint8)
    
    # Determine valid pixels - combine data validity with geometry mask
    mask_valid = (bins_up > 0)
    if aoi_mask is not None:
        mask_valid = mask_valid & aoi_mask

    max_palette_idx = len(palette_rgba) - 1
    bins_idx = np.where(mask_valid, np.minimum(bins_up, max_palette_idx), 0).astype(np.int32)
    lut = np.array(palette_rgba, dtype=np.uint8)
    rgba[mask_valid, :] = lut[bins_idx[mask_valid]]
    
    # CRITICAL: Set fully transparent for invalid pixels (outside polygon or no data)
    # Alpha channel must be 0 for transparency
    rgba[~mask_valid, :] = [0, 0, 0, 0]  # Fully transparent black

    pil = Image.fromarray(rgba, mode="RGBA")
    # Use LANCZOS for better quality while preserving transparency
    pil = pil.resize((out_w, out_h), resample=Image.LANCZOS)
    buf = io.BytesIO()
    # RGBA mode automatically handles transparency via alpha channel
    pil.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")

def temporal_fill_median(band_key: str, items: List[Any], aoi_geojson, dst_transform, H, W, want_scl=False, max_items=6):
    stacks = []
    used = 0
    for it in items[:max_items]:
        assets = it.assets or {}
        a = assets.get(band_key) or assets.get(band_key.lower())
        url = prefer_http_from_asset(a) if a else None
        if not url:
            continue
        url = sign_href_if_pc(url)
        try:
            with rasterio.open(url) as ds:
                arr = read_band_window(ds, aoi_to_scene(aoi_geojson, ds.crs.to_string()), H, W, dst_transform, Resampling.bilinear)
                if np.nanmax(arr) > 1.5:
                    arr *= 1/10000.0
                stacks.append(np.where(np.isfinite(arr), arr, np.nan))
                used += 1
        except Exception:
            continue
    if used == 0:
        return None
    stacked = np.stack(stacks, axis=0)
    median = np.nanmedian(stacked, axis=0)
    return median