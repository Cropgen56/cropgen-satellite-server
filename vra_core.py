"""SOC spatial analysis and VRA zoning pipeline (Sentinel-2 L2A).

VRA zoning is field-relative (quantile / k-means) on the native 10 m
Sentinel-2 grid, then cleaned and vectorized into zone patches.
"""

from __future__ import annotations

import io
import math
import base64
from datetime import datetime
from typing import Dict, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable

import numpy as np
import rasterio
from affine import Affine
from rasterio.features import geometry_mask, shapes as rio_shapes
from shapely.geometry import mapping, shape as shp_shape
from shapely.ops import transform as shp_transform
from pyproj import Transformer
from scipy.ndimage import gaussian_filter, median_filter, label as cc_label

import utils

DPI = 150
SMOOTH_SIGMA = 1.0
USE_SHADOW = True
NATIVE_RES_M = 10.0
MAX_GRID_PX_HARD = 2000

CROP_DEMAND: dict[str, dict[str, float]] = {
    "wheat": {"N": 12.5, "P": 5.0, "K": 10.5},
    "rice": {"N": 15.0, "P": 5.5, "K": 12.0},
    "maize": {"N": 18.0, "P": 6.0, "K": 14.0},
    "soybean": {"N": 8.0, "P": 8.0, "K": 10.0},
    "sugarcane": {"N": 22.0, "P": 9.0, "K": 20.0},
    "cotton": {"N": 18.0, "P": 7.0, "K": 14.0},
    "onion": {"N": 14.0, "P": 7.0, "K": 14.0},
    "potato": {"N": 20.0, "P": 10.0, "K": 22.0},
    "tomato": {"N": 16.0, "P": 8.0, "K": 18.0},
    "banana": {"N": 22.0, "P": 10.0, "K": 28.0},
    "groundnut": {"N": 8.5, "P": 7.0, "K": 8.5},
    "jowar": {"N": 17.0, "P": 5.5, "K": 13.0},
    "bajra": {"N": 14.5, "P": 4.5, "K": 11.0},
    "chili": {"N": 18.0, "P": 8.0, "K": 15.0},
    "turmeric": {"N": 22.0, "P": 10.0, "K": 24.0},
    "ginger": {"N": 22.0, "P": 10.0, "K": 24.0},
    "mustard": {"N": 14.0, "P": 7.0, "K": 11.0},
    "lentil": {"N": 7.5, "P": 7.5, "K": 9.0},
    "gram": {"N": 7.0, "P": 7.0, "K": 9.0},
    "default": {"N": 14.0, "P": 7.0, "K": 12.0},
}

FERTILISER_MAX_DOSE = {"N": 150, "P": 60, "K": 100}

FERTILISER_PRODUCTS = {
    "N": {"name": "Urea", "nutrient_pct": 46.0},
    "P": {"name": "DAP (18-46-0)", "nutrient_pct": 46.0},
    "K": {"name": "MOP (0-0-60)", "nutrient_pct": 60.0},
}

MAX_DOSE_FRACTION = 1.00
MIN_DOSE_FRACTION = 0.25

_SOC_COLORS = [
    "#3d1c00", "#7a3b00", "#c17f00", "#e8c840", "#9ecb3c",
    "#4caf50", "#1b5e20",
]
CMAP_SOC = LinearSegmentedColormap.from_list("soc", _SOC_COLORS, N=256)

_REQUIRED_BANDS = ["B02", "B04", "B08", "B11", "B12"]

M2_HA = 10000.0
HA_ACRE = 2.47105


def _validate_dates(start_date: str, end_date: str) -> None:
    fmt = "%Y-%m-%d"
    try:
        start = datetime.strptime(start_date, fmt)
        end = datetime.strptime(end_date, fmt)
    except ValueError as exc:
        raise ValueError("start_date and end_date must be YYYY-MM-DD") from exc
    if start > end:
        raise ValueError("start_date must be on or before end_date")


def _validate_vra_options(n_zones: int, zone_method: str) -> None:
    if not (2 <= n_zones <= 7):
        raise ValueError("n_zones must be between 2 and 7")
    if zone_method not in ("quantile", "kmeans"):
        raise ValueError("zone_method must be 'quantile' or 'kmeans'")


def zone_dose_fractions(n_zones: int) -> dict[int, float]:
    """Zone 1 (lowest soil nutrient) gets the highest dose fraction."""
    if n_zones == 1:
        return {1: MAX_DOSE_FRACTION}
    return {
        i: round(
            MAX_DOSE_FRACTION - (MAX_DOSE_FRACTION - MIN_DOSE_FRACTION) * (i - 1) / (n_zones - 1),
            3,
        )
        for i in range(1, n_zones + 1)
    }


def zone_labels_for(n_zones: int) -> dict[int, str]:
    presets = {
        2: {1: "Low", 2: "High"},
        3: {1: "Low", 2: "Medium", 3: "High"},
        4: {1: "Low", 2: "Medium-Low", 3: "Medium-High", 4: "High"},
        5: {1: "Very Low", 2: "Low", 3: "Medium", 4: "High", 5: "Very High"},
    }
    if n_zones in presets:
        return presets[n_zones]
    return {i: f"Zone {i}" for i in range(1, n_zones + 1)}


def zone_colors_for(n_zones: int) -> dict[str, str]:
    ramp = LinearSegmentedColormap.from_list("zone_ramp", ["#b71c1c", "#f9a825", "#2e7d32"], N=256)
    labels = zone_labels_for(n_zones)
    colors = {}
    for i in range(1, n_zones + 1):
        t = 0.0 if n_zones == 1 else (i - 1) / (n_zones - 1)
        colors[labels[i]] = mcolors.to_hex(ramp(t))
    return colors


def build_native_grid(crs, aoi_ll: dict, native_m: float = NATIVE_RES_M) -> tuple:
    """True 10 m Sentinel-2 grid; relax only if the AOI would exceed MAX_GRID_PX_HARD."""
    aoi_sc = utils.aoi_to_scene(aoi_ll, crs.to_string())
    minx, miny, maxx, maxy = aoi_sc.bounds
    dx = max(maxx - minx, 1e-6)
    dy = max(maxy - miny, 1e-6)
    long_side = max(dx, dy)

    res = native_m
    if long_side / native_m > MAX_GRID_PX_HARD:
        res = long_side / MAX_GRID_PX_HARD

    width = max(1, int(math.ceil(dx / res)))
    height = max(1, int(math.ceil(dy / res)))
    transform = Affine.translation(minx, maxy) * Affine.scale(res, -res)
    return aoi_sc, transform, height, width, res


def _safe_div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(np.abs(b) > 1e-9, a / b, np.nan).astype("float32")


def _smooth(arr: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return arr
    valid = np.isfinite(arr)
    if not np.any(valid):
        return arr
    num = gaussian_filter(np.where(valid, arr, 0.0).astype("float32"), sigma)
    wgt = gaussian_filter(valid.astype("float32"), sigma)
    return np.where(wgt > 1e-6, num / wgt, np.nan).astype("float32")


def _clamp_arr(a: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.clip(a, lo, hi).astype("float32")


def _fetch_bands(
    item,
    geometry: dict,
    dst_transform,
    height: int,
    width: int,
) -> Optional[dict]:
    tile = utils._read_tile_into_stack(
        item,
        geometry,
        dst_transform,
        height,
        width,
        want_scl=True,
        required_bands=_REQUIRED_BANDS,
    )
    if not tile.get("used"):
        return None
    bands = tile.get("bands") or {}
    if any(bands.get(b) is None for b in _REQUIRED_BANDS):
        return None
    bands["SCL"] = tile.get("S")
    return bands


def compute_index_maps(bands: dict, smooth_sigma: float = SMOOTH_SIGMA) -> dict:
    b02 = bands["B02"]
    b04 = bands["B04"]
    b08 = bands["B08"]
    b11 = bands["B11"]
    b12 = bands["B12"]
    scl = bands.get("SCL")

    ndvi = _safe_div(b08 - b04, b08 + b04)
    ndwi = _safe_div(b08 - b12, b08 + b12)
    l_factor = 0.5
    savi = ((b08 - b04) / np.where((b08 + b04 + l_factor) != 0, b08 + b04 + l_factor, np.nan)) * (
        1 + l_factor
    )
    savi = savi.astype("float32")
    bsi = _safe_div((b11 + b04) - (b08 + b02), (b11 + b04) + (b08 + b02))
    swir1 = b11.copy()
    clay = _safe_div(b11, b12)

    if scl is not None:
        bad = np.isin(scl, [8, 9, 10, 11] + ([3] if USE_SHADOW else []))
        for arr in (ndvi, ndwi, savi, bsi, swir1, clay):
            arr[bad] = np.nan

    return {
        "NDVI": _smooth(ndvi, smooth_sigma),
        "NDWI": _smooth(ndwi, smooth_sigma),
        "SAVI": _smooth(savi, smooth_sigma),
        "BSI": _smooth(bsi, smooth_sigma),
        "SWIR1": _smooth(swir1, smooth_sigma),
        "CLAY": _smooth(clay, smooth_sigma),
    }


def compute_soil_maps(idx: dict) -> dict:
    ndvi = idx["NDVI"]
    savi = idx["SAVI"]
    bsi = idx["BSI"]
    clay = idx["CLAY"]

    soc = _clamp_arr(0.8 + 1.5 * ndvi - 0.6 * bsi, 0.1, 5.0)
    n_map = _clamp_arr(soc * 140.0, 50.0, 600.0)
    p_map = _clamp_arr(15.0 + 40.0 * (savi + 0.2), 10.0, 80.0)
    k_map = _clamp_arr(80.0 + 120.0 * (clay - 0.7), 50.0, 350.0)

    return {"SOC": soc, "N": n_map, "P": p_map, "K": k_map}


def classify_zones_relative(arr: np.ndarray, n_zones: int = 5, method: str = "quantile") -> np.ndarray:
    """Field-relative zones: 1 = lowest nutrient (highest dose), n_zones = richest. 0 = nodata."""
    valid = np.isfinite(arr)
    out = np.zeros(arr.shape, dtype="int8")
    if not np.any(valid):
        return out
    vals = arr[valid]

    if method == "kmeans":
        edges = _kmeans_1d_edges(vals, n_zones)
    else:
        qs = np.linspace(0, 100, n_zones + 1)
        edges = np.unique(np.percentile(vals, qs))
        if len(edges) < 2:
            edges = np.array([vals.min(), vals.max() + 1e-6])

    idx = np.digitize(vals, edges[1:-1], right=True) + 1
    idx = np.clip(idx, 1, n_zones)
    out[valid] = idx.astype("int8")
    return out


def _kmeans_1d_edges(vals: np.ndarray, k: int, n_iter: int = 25) -> np.ndarray:
    v = np.sort(vals.astype("float64"))
    centroids = np.percentile(v, np.linspace(5, 95, k))
    for _ in range(n_iter):
        assign = np.argmin(np.abs(v[:, None] - centroids[None, :]), axis=1)
        new_c = np.array([
            v[assign == c].mean() if np.any(assign == c) else centroids[c]
            for c in range(k)
        ])
        if np.allclose(new_c, centroids, atol=1e-4):
            centroids = new_c
            break
        centroids = new_c
    centroids = np.sort(centroids)
    return np.concatenate(([v.min()], (centroids[:-1] + centroids[1:]) / 2.0, [v.max() + 1e-6]))


def majority_denoise(zone_map: np.ndarray, size: int = 3) -> np.ndarray:
    if zone_map.max() == 0:
        return zone_map
    filtered = median_filter(zone_map, size=size, mode="nearest")
    return np.clip(filtered, 0, zone_map.max()).astype(zone_map.dtype)


def dissolve_small_patches(zone_map: np.ndarray, res_m: float, min_area_ha: float = 0.05) -> np.ndarray:
    out = zone_map.copy()
    px_area_ha = (res_m * res_m) / M2_HA
    min_px = max(1, int(round(min_area_ha / px_area_ha)))

    zone_values = [z for z in np.unique(zone_map) if z != 0]
    for this_zone in zone_values:
        labeled, n_comp = cc_label(zone_map == this_zone, structure=np.ones((3, 3)))
        for comp_id in range(1, n_comp + 1):
            comp_mask = labeled == comp_id
            if int(comp_mask.sum()) >= min_px:
                continue
            ys, xs = np.where(comp_mask)
            y0, y1 = max(ys.min() - 1, 0), min(ys.max() + 2, zone_map.shape[0])
            x0, x1 = max(xs.min() - 1, 0), min(xs.max() + 2, zone_map.shape[1])
            neighborhood = zone_map[y0:y1, x0:x1]
            local_mask = comp_mask[y0:y1, x0:x1]
            border_vals = neighborhood[(~local_mask) & (neighborhood > 0) & (neighborhood != this_zone)]
            if border_vals.size == 0:
                continue
            vals, counts = np.unique(border_vals, return_counts=True)
            out[comp_mask] = vals[np.argmax(counts)]
    return out


def vectorize_zone_patches(
    zone_map: np.ndarray,
    transform: Affine,
    res_m: float,
    zone_labels: dict,
) -> list:
    px_area_ha = (res_m * res_m) / M2_HA
    height, width = zone_map.shape
    patches, pid = [], 0
    for geom, val in rio_shapes(zone_map.astype("int32"), mask=zone_map > 0, transform=transform):
        val = int(val)
        if val == 0:
            continue
        pid += 1
        poly = shp_shape(geom)
        area_ha = poly.area / M2_HA
        px_count = int(round(area_ha / px_area_ha))
        minx, miny, maxx, maxy = poly.bounds
        touches_edge = (
            minx <= transform.c
            or miny <= (transform.f + height * transform.e)
            or maxx >= (transform.c + width * transform.a)
            or maxy >= transform.f
        )
        ptype = "core" if area_ha >= 1.0 else ("edge" if touches_edge else "isolated")
        patches.append({
            "patch_id": pid,
            "zone_class": zone_labels.get(val, str(val)),
            "patch_type": ptype,
            "geometry": poly,
            "area_ha": round(area_ha, 4),
            "pixel_count": px_count,
        })
    return patches


def patches_to_geojson(patches: list, crs_epsg: int = 4326, transformer=None) -> dict:
    features = []
    for patch in patches:
        geom = patch["geometry"]
        if transformer is not None:
            geom = shp_transform(lambda x, y, z=None: transformer.transform(x, y), geom)
        features.append({
            "type": "Feature",
            "properties": {
                "patch_id": patch["patch_id"],
                "zone_class": patch["zone_class"],
                "patch_type": patch["patch_type"],
                "area_ha": patch["area_ha"],
                "pixel_count": patch["pixel_count"],
            },
            "geometry": mapping(geom),
        })
    return {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": f"EPSG:{crs_epsg}"}},
        "features": features,
    }


def build_grid_zones(
    nutrient_arr: np.ndarray,
    transform: Affine,
    res_m: float,
    n_zones: int = 5,
    method: str = "quantile",
    min_patch_ha: float = 0.05,
) -> dict:
    zone_labels = zone_labels_for(n_zones)
    raw = classify_zones_relative(nutrient_arr, n_zones=n_zones, method=method)
    denoised = majority_denoise(raw, size=3)
    cleaned = dissolve_small_patches(denoised, res_m, min_area_ha=min_patch_ha)
    patches = vectorize_zone_patches(cleaned, transform, res_m, zone_labels)
    return {
        "zone_grid": cleaned,
        "patches": patches,
        "zone_labels": zone_labels,
        "n_zones": n_zones,
        "method": method,
    }


def calc_vra_rates_from_patches(zres: dict, nutrient: str, crop: str) -> dict:
    n_zones = zres["n_zones"]
    fractions = zone_dose_fractions(n_zones)
    labels = zres["zone_labels"]
    max_dose = FERTILISER_MAX_DOSE[nutrient]
    prod = FERTILISER_PRODUCTS[nutrient]

    by_class = {
        labels[i]: {"pixel_count": 0, "area_ha": 0.0, "patch_ids": []}
        for i in range(1, n_zones + 1)
    }
    for patch in zres["patches"]:
        cls = by_class[patch["zone_class"]]
        cls["pixel_count"] += patch["pixel_count"]
        cls["area_ha"] += patch["area_ha"]
        cls["patch_ids"].append(patch["patch_id"])

    result = {}
    for i in range(1, n_zones + 1):
        lbl = labels[i]
        cls = by_class[lbl]
        frac = fractions[i]
        dose_nut = round(max_dose * frac, 1)
        dose_prod = round(dose_nut / (prod["nutrient_pct"] / 100.0), 1)
        total_kg = round(dose_prod * cls["area_ha"], 1)
        result[lbl] = {
            "pixel_count": cls["pixel_count"],
            "area_ha": round(cls["area_ha"], 3),
            "area_acres": round(cls["area_ha"] * HA_ACRE, 3),
            "nutrient_dose_kg_ha": dose_nut,
            "product": prod["name"],
            "product_dose_kg_ha": dose_prod,
            "total_product_kg": total_kg,
            "patch_count": len(cls["patch_ids"]),
        }
    return result


def compute_soc_stats(soc_arr: np.ndarray, res_m: float) -> dict:
    px_area = res_m * res_m
    valid_soc = np.isfinite(soc_arr)
    soc_classes_out = {}
    soc_class_defs = [
        ("Very Low  (<0.3%)", None, 0.3),
        ("Low  (0.3–0.7%)", 0.3, 0.7),
        ("Medium (0.7–1.2%)", 0.7, 1.2),
        ("High  (1.2–2.0%)", 1.2, 2.0),
        ("Very High (>2.0%)", 2.0, None),
    ]
    total_valid_px = int(np.count_nonzero(valid_soc))

    for lbl, lo, hi in soc_class_defs:
        if lo is None:
            mask = valid_soc & (soc_arr < hi)
        elif hi is None:
            mask = valid_soc & (soc_arr >= lo)
        else:
            mask = valid_soc & (soc_arr >= lo) & (soc_arr < hi)
        cnt = int(np.count_nonzero(mask))
        ha = cnt * px_area / M2_HA
        soc_classes_out[lbl] = {
            "pixels": cnt,
            "ha": round(ha, 3),
            "acres": round(ha * HA_ACRE, 3),
            "pct_area": round(cnt / max(total_valid_px, 1) * 100, 1),
        }

    total_ha = total_valid_px * px_area / M2_HA
    return {
        "mean_pct": round(float(np.nanmean(soc_arr)), 3),
        "min_pct": round(float(np.nanmin(soc_arr)), 3),
        "max_pct": round(float(np.nanmax(soc_arr)), 3),
        "std_pct": round(float(np.nanstd(soc_arr)), 3),
        "total_area_ha": round(total_ha, 3),
        "total_area_acres": round(total_ha * HA_ACRE, 3),
        "classes": soc_classes_out,
    }


def _arr_to_b64(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=DPI, bbox_inches="tight", pad_inches=0.05, transparent=True)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def render_soc_map(soc_arr: np.ndarray, res_m: float, capture_date: str) -> str:
    valid = np.isfinite(soc_arr)
    if not np.any(valid):
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        ax.text(0.5, 0.5, "No valid data", ha="center", va="center")
        ax.axis("off")
        return _arr_to_b64(fig)

    rows = np.where(valid.any(axis=1))[0]
    cols = np.where(valid.any(axis=0))[0]
    soc_crop = soc_arr[rows[0]: rows[-1] + 1, cols[0]: cols[-1] + 1]
    valid_crop = np.isfinite(soc_crop)

    labels_bins = [
        ("Very Low  (<0.3%)", 0.15, "#3d1c00"),
        ("Low  (0.3–0.7%)", 0.5, "#c17f00"),
        ("Medium (0.7–1.2%)", 0.95, "#e8c840"),
        ("High  (1.2–2.0%)", 1.6, "#4caf50"),
        ("Very High (>2.0%)", 2.5, "#1b5e20"),
    ]
    soc_bins = np.full_like(soc_crop, np.nan)
    soc_bins[valid_crop & (soc_crop < 0.3)] = 0.15
    soc_bins[valid_crop & (soc_crop >= 0.3) & (soc_crop < 0.7)] = 0.5
    soc_bins[valid_crop & (soc_crop >= 0.7) & (soc_crop < 1.2)] = 0.95
    soc_bins[valid_crop & (soc_crop >= 1.2) & (soc_crop < 2.0)] = 1.6
    soc_bins[valid_crop & (soc_crop >= 2.0)] = 2.5

    fig = plt.figure(figsize=(10, 7), facecolor="#1a1a2e")
    fig.suptitle(
        f"SOIL ORGANIC CARBON (SOC) MAP\nCapture: {capture_date}  |  {res_m:.0f}-m Sentinel-2",
        color="white",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    gs = fig.add_gridspec(1, 2, width_ratios=[3, 1], wspace=0.05)
    ax_map = fig.add_subplot(gs[0])
    ax_info = fig.add_subplot(gs[1])

    img = np.ma.masked_invalid(soc_crop)
    im = ax_map.imshow(img, cmap=CMAP_SOC, vmin=0.1, vmax=3.0, interpolation="bilinear")
    ax_map.set_axis_off()
    ax_map.set_title("SOC Spatial Distribution", color="white", fontsize=10, pad=6)

    divider = make_axes_locatable(ax_map)
    cax = divider.append_axes("bottom", size="4%", pad=0.08)
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_label("SOC (%)", color="white", fontsize=9)
    cb.ax.xaxis.set_tick_params(color="white")
    plt.setp(cb.ax.xaxis.get_ticklabels(), color="white", fontsize=8)
    cax.set_facecolor("#1a1a2e")

    px_area = res_m * res_m
    stats = {}
    for lbl, val, _color in labels_bins:
        cnt = int(np.count_nonzero(valid_crop & (soc_bins == val)))
        area_ha = cnt * px_area / M2_HA
        stats[lbl] = {"ha": round(area_ha, 3)}

    ax_info.set_facecolor("#1a1a2e")
    ax_info.set_axis_off()
    y = 0.95
    ax_info.text(0.05, y, "SOC CLASSES", color="white", fontsize=9, fontweight="bold", va="top", transform=ax_info.transAxes)
    y -= 0.07
    total_ha = max(sum(s["ha"] for s in stats.values()), 1e-9)
    for lbl, val, color in labels_bins:
        area_ha = stats[lbl]["ha"]
        pct_area = area_ha / total_ha * 100
        rect = mpatches.FancyBboxPatch(
            (0.04, y - 0.025), 0.08, 0.04,
            boxstyle="round,pad=0.005",
            linewidth=0,
            facecolor=color,
            transform=ax_info.transAxes,
            clip_on=False,
        )
        ax_info.add_patch(rect)
        ax_info.text(
            0.16, y - 0.005,
            f"{lbl}\n{area_ha:.2f} ha  ({pct_area:.0f}%)",
            color="white",
            fontsize=6.5,
            va="top",
            transform=ax_info.transAxes,
        )
        y -= 0.11

    return _arr_to_b64(fig)


def render_grid_patch_map(
    zone_grid: np.ndarray,
    zone_colors: dict,
    capture_date: str,
    res_m: float,
    nutrient_name: str,
    vra_rates: dict,
    n_zones: int,
    draw_grid_lines: bool = True,
) -> str:
    valid = zone_grid > 0
    if not np.any(valid):
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.text(0.5, 0.5, "No valid data", ha="center", va="center")
        ax.axis("off")
        return _arr_to_b64(fig)

    rows = np.where(valid.any(axis=1))[0]
    cols = np.where(valid.any(axis=0))[0]
    cropped = zone_grid[rows[0]: rows[-1] + 1, cols[0]: cols[-1] + 1]

    labels = zone_labels_for(n_zones)
    color_list = ["#00000000"] + [zone_colors[labels[i]] for i in range(1, n_zones + 1)]
    cmap = ListedColormap(color_list)

    fig = plt.figure(figsize=(11, 7), facecolor="#1a1a2e")
    gs = fig.add_gridspec(1, 2, width_ratios=[3, 1], wspace=0.04)
    ax_map = fig.add_subplot(gs[0])
    ax_info = fig.add_subplot(gs[1])

    ax_map.imshow(cropped, cmap=cmap, vmin=0, vmax=n_zones, interpolation="nearest")

    if draw_grid_lines:
        height, width = cropped.shape
        if height * width <= 150 * 150:
            for x in range(width + 1):
                ax_map.axvline(x - 0.5, color="#ffffff22", linewidth=0.4)
            for y in range(height + 1):
                ax_map.axhline(y - 0.5, color="#ffffff22", linewidth=0.4)

    ax_map.set_axis_off()
    ax_map.set_title(
        f"{nutrient_name} — Zone Patches  (native {res_m:.0f} m Sentinel-2 grid)\n"
        f"Capture: {capture_date}",
        color="white",
        fontsize=10,
        pad=6,
    )

    ax_info.set_facecolor("#1a1a2e")
    ax_info.set_axis_off()
    prod = FERTILISER_PRODUCTS.get(nutrient_name, {}).get("name", nutrient_name)
    y = 0.97
    ax_info.text(
        0.05, y, f"APPLICATION RATES\n{prod}",
        color="white", fontsize=8, fontweight="bold", va="top", transform=ax_info.transAxes,
    )
    y -= 0.10
    for i in range(1, n_zones + 1):
        lbl = labels[i]
        d = vra_rates.get(lbl, {})
        if d.get("pixel_count", 0) == 0:
            continue
        color = zone_colors[lbl]
        rect = mpatches.FancyBboxPatch(
            (0.03, y - 0.035), 0.09, 0.055,
            boxstyle="round,pad=0.005",
            linewidth=0,
            facecolor=color,
            transform=ax_info.transAxes,
            clip_on=False,
        )
        ax_info.add_patch(rect)
        lines = [
            f"Zone: {lbl}",
            f"{d['product_dose_kg_ha']} kg/ha",
            f"Area: {d['area_ha']:.2f} ha",
            f"Patches: {d.get('patch_count', 0)}",
            f"Total: {d['total_product_kg']:.0f} kg",
        ]
        for j, line in enumerate(lines):
            ax_info.text(0.16, y - j * 0.026, line, color="white", fontsize=6.2, va="top", transform=ax_info.transAxes)
        y -= 0.19

    return _arr_to_b64(fig)


def render_combined_4panel(soc_arr, zres_by_nutrient: dict, res_m: float, capture_date: str) -> str:
    def _crop(arr):
        v2 = np.isfinite(arr) if arr.dtype.kind == "f" else (arr != 0)
        if not np.any(v2):
            return arr
        rows = np.where(v2.any(axis=1))[0]
        cols = np.where(v2.any(axis=0))[0]
        if rows.size == 0 or cols.size == 0:
            return arr
        return arr[rows[0]: rows[-1] + 1, cols[0]: cols[-1] + 1]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), facecolor="#0f0f23")
    fig.suptitle(
        f"VARIABLE RATE APPLICATION — FIELD MANAGEMENT ZONES\n"
        f"Sentinel-2 L2A  |  Scene: {capture_date}  |  {res_m:.0f} m native grid",
        color="white",
        fontsize=13,
        fontweight="bold",
    )
    fig.subplots_adjust(hspace=0.15, wspace=0.1)

    panels = [("SOC", "Soil Organic Carbon (%)", soc_arr, None)] + [
        (nut, f"{nut} Zones (VRA)", zres_by_nutrient[nut]["zone_grid"], nut)
        for nut in ["N", "P", "K"]
    ]

    for ax, (key, title, arr, nut) in zip(axes.flat, panels):
        ax.set_facecolor("#0f0f23")
        cropped = _crop(arr)
        if key == "SOC":
            img = np.ma.masked_invalid(cropped)
            im = ax.imshow(img, cmap=CMAP_SOC, vmin=0.1, vmax=3.0, interpolation="bilinear")
            cb = plt.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
            cb.set_label("SOC %", color="white", fontsize=8)
            cb.ax.yaxis.set_tick_params(color="white")
            plt.setp(cb.ax.yaxis.get_ticklabels(), color="white", fontsize=7)
        else:
            n_zones = zres_by_nutrient[nut]["n_zones"]
            colors = zone_colors_for(n_zones)
            labels = zone_labels_for(n_zones)
            color_list = ["#00000000"] + [colors[labels[i]] for i in range(1, n_zones + 1)]
            cmap = ListedColormap(color_list)
            img = np.ma.masked_equal(cropped, 0)
            ax.imshow(img, cmap=cmap, vmin=0, vmax=n_zones, interpolation="nearest")
            handles = [mpatches.Patch(color=colors[labels[i]], label=labels[i]) for i in range(1, n_zones + 1)]
            ax.legend(handles=handles, loc="lower left", fontsize=6, framealpha=0.85, title=f"Soil {nut}", title_fontsize=7)
        ax.set_title(title, color="white", fontsize=9, pad=4)
        ax.set_axis_off()

    return _arr_to_b64(fig)


def build_text_report(result: dict) -> str:
    meta = result["metadata"]
    soc_stats = result["soc_stats"]
    vra = result["vra_rates"]
    crop = result["crop"]
    hr = "═" * 70
    lines = [
        hr,
        "  VARIABLE RATE APPLICATION (VRA) FIELD REPORT — Grid-Patch Edition",
        f"  Crop        : {crop}",
        f"  Scene date  : {meta['capture_date']}",
        f"  Cloud cover : {meta['cloud_cover']} %",
        f"  Data source : {meta['collection']}",
        f"  Grid        : {meta['grid_H']} × {meta['grid_W']} px  @  {meta['res_m']:.1f} m/px "
        f"(native Sentinel-2 resolution)",
        f"  Zones       : {meta['n_zones']} ({meta['zone_method']})",
        hr,
        "",
        "  ── SOIL ORGANIC CARBON (SOC) SUMMARY ─────────────────────────",
        f"  Mean  : {soc_stats['mean_pct']:.2f} %",
        f"  Min   : {soc_stats['min_pct']:.2f} %",
        f"  Max   : {soc_stats['max_pct']:.2f} %",
        f"  Field : {soc_stats['total_area_ha']:.2f} ha  ({soc_stats['total_area_acres']:.2f} acres)",
        "",
    ]
    for soc_class, data in soc_stats["classes"].items():
        lines.append(f"  {soc_class:<28}: {data['ha']:.3f} ha  ({data['acres']:.3f} ac)")

    lines += ["", hr, "  ── VRA NUTRIENT ZONES & APPLICATION RATES (patch-based) ──", ""]
    for nut in ["N", "P", "K"]:
        prod = FERTILISER_PRODUCTS[nut]
        lines.append(f"  ► {nut} — {prod['name']}  ({prod['nutrient_pct']:.0f}% nutrient)")
        lines.append(
            f"    {'Zone':<12} {'Patches':>8} {'Area (ha)':>10} {'Area (ac)':>10} "
            f"{'Dose kg/ha':>12} {'Total kg':>10}"
        )
        lines.append(f"    {'─'*12} {'─'*8} {'─'*10} {'─'*10} {'─'*12} {'─'*10}")
        for lbl, d in vra[nut].items():
            if d.get("pixel_count", 0) == 0:
                continue
            lines.append(
                f"    {lbl:<12} {d.get('patch_count', 0):>8} {d['area_ha']:>10.3f} "
                f"{d['area_acres']:>10.3f} {d['product_dose_kg_ha']:>12.1f} "
                f"{d['total_product_kg']:>10.1f}"
            )
        lines.append("")
    lines += [
        hr,
        "  ── INTERPRETATION ────────────────────────────────────────────",
        "  Lowest zone class  = Soil is nutrient-deficient → highest dose",
        "  Highest zone class = Soil is nutrient-rich      → lowest dose",
        "  Zones are RELATIVE to this field (quantile/k-means breaks), and each",
        "  zone is a vectorized patch on the native Sentinel-2 grid.",
        hr,
    ]
    return "\n".join(lines)


def _run_soil_pipeline(
    geometry: dict,
    start_date: str,
    end_date: str,
    provider: str = "both",
    satellite: str = "s2",
) -> dict:
    _validate_dates(start_date, end_date)
    search_order = utils.get_provider_search_order(provider, prefer_pc_default=True)
    prefer_pc = search_order[0] == "planetary"

    item, _has_scl, collection = utils.pick_best_item(
        geometry, start_date, end_date, prefer_pc=prefer_pc, satellite=satellite,
    )
    if item is None:
        diag = utils.get_provider_error_summary()
        msg = "No Sentinel-2 scene found for this AOI / date range."
        if diag:
            msg += f" Provider diagnostics: {diag}"
        raise RuntimeError(msg)

    cap_date = (item.properties.get("datetime") or "")[:10]
    cloud = item.properties.get("eo:cloud_cover")

    assets = item.assets or {}
    red_url = utils.prefer_http_from_asset(assets.get("red") or assets.get("B04"))
    if not red_url:
        raise RuntimeError("No red-band asset found — cannot derive CRS.")
    red_url = utils.sign_href_if_pc(red_url)

    with rasterio.open(red_url) as ref:
        crs = ref.crs

    aoi_sc, dst_tf, height, width, res_m = build_native_grid(crs, geometry, native_m=NATIVE_RES_M)
    bands = _fetch_bands(item, geometry, dst_tf, height, width)
    if bands is None:
        raise RuntimeError("Band fetch failed — check network / STAC availability.")

    aoi_mask = geometry_mask([mapping(aoi_sc)], out_shape=(height, width), transform=dst_tf, invert=True)
    for b in _REQUIRED_BANDS:
        bands[b][~aoi_mask] = np.nan

    idx_maps = compute_index_maps(bands)
    soil_maps = compute_soil_maps(idx_maps)

    return {
        "item": item,
        "crs": crs,
        "transform": dst_tf,
        "collection": collection or "sentinel-2-l2a",
        "capture_date": cap_date,
        "cloud_cover": cloud,
        "height": height,
        "width": width,
        "res_m": res_m,
        "idx_maps": idx_maps,
        "soil_maps": soil_maps,
        "soc_arr": soil_maps["SOC"],
    }


def run_soc_analysis(
    geometry: dict,
    start_date: str,
    end_date: str,
    provider: str = "both",
    satellite: str = "s2",
) -> dict:
    data = _run_soil_pipeline(geometry, start_date, end_date, provider, satellite)
    soc_arr = data["soc_arr"]
    res_m = data["res_m"]
    cap_date = data["capture_date"]

    soc_stats = compute_soc_stats(soc_arr, res_m)
    image_base64 = render_soc_map(soc_arr, res_m, cap_date)

    return {
        "date": cap_date,
        "cloud_cover": data["cloud_cover"],
        "image_base64": image_base64,
        "soc_stats": soc_stats,
        "metadata": {
            "capture_date": cap_date,
            "cloud_cover": data["cloud_cover"],
            "collection": data["collection"],
            "grid_H": data["height"],
            "grid_W": data["width"],
            "res_m": res_m,
            "mean_indices": {k: round(float(np.nanmean(v)), 5) for k, v in data["idx_maps"].items()},
        },
    }


def run_vra_analysis(
    geometry: dict,
    start_date: str,
    end_date: str,
    crop: str = "wheat",
    provider: str = "both",
    satellite: str = "s2",
    include_images: bool = True,
    n_zones: int = 5,
    zone_method: str = "quantile",
    min_patch_ha: float = 0.05,
) -> dict:
    crop_key = (crop or "wheat").lower()
    if crop_key not in CROP_DEMAND:
        crop_key = "default"
    _validate_vra_options(n_zones, zone_method)

    data = _run_soil_pipeline(geometry, start_date, end_date, provider, satellite)
    soc_arr = data["soc_arr"]
    res_m = data["res_m"]
    cap_date = data["capture_date"]
    soil_maps = data["soil_maps"]
    dst_tf = data["transform"]
    crs = data["crs"]

    zres_by_nutrient = {}
    vra_rates = {}
    for nut in ["N", "P", "K"]:
        zres = build_grid_zones(
            soil_maps[nut], dst_tf, res_m,
            n_zones=n_zones, method=zone_method, min_patch_ha=min_patch_ha,
        )
        zres_by_nutrient[nut] = zres
        vra_rates[nut] = calc_vra_rates_from_patches(zres, nut, crop_key)

    soc_stats = compute_soc_stats(soc_arr, res_m)

    to_wgs84 = Transformer.from_crs(crs.to_string(), "EPSG:4326", always_xy=True)
    zone_geojson = {
        nut: patches_to_geojson(zres_by_nutrient[nut]["patches"], transformer=to_wgs84)
        for nut in ["N", "P", "K"]
    }

    metadata = {
        "capture_date": cap_date,
        "cloud_cover": data["cloud_cover"],
        "collection": data["collection"],
        "grid_H": data["height"],
        "grid_W": data["width"],
        "res_m": res_m,
        "n_zones": n_zones,
        "zone_method": zone_method,
        "mean_indices": {k: round(float(np.nanmean(v)), 5) for k, v in data["idx_maps"].items()},
    }

    images = None
    if include_images:
        images = {
            "soc_map_b64": render_soc_map(soc_arr, res_m, cap_date),
            "vra_n_b64": render_grid_patch_map(
                zres_by_nutrient["N"]["zone_grid"],
                zone_colors_for(n_zones), cap_date, res_m, "N", vra_rates["N"], n_zones,
            ),
            "vra_p_b64": render_grid_patch_map(
                zres_by_nutrient["P"]["zone_grid"],
                zone_colors_for(n_zones), cap_date, res_m, "P", vra_rates["P"], n_zones,
            ),
            "vra_k_b64": render_grid_patch_map(
                zres_by_nutrient["K"]["zone_grid"],
                zone_colors_for(n_zones), cap_date, res_m, "K", vra_rates["K"], n_zones,
            ),
            "combined_b64": render_combined_4panel(soc_arr, zres_by_nutrient, res_m, cap_date),
        }

    result = {
        "date": cap_date,
        "crop": crop_key,
        "cloud_cover": data["cloud_cover"],
        "vra_rates": vra_rates,
        "zone_geojson": zone_geojson,
        "soc_stats": soc_stats,
        "images": images,
        "metadata": metadata,
    }
    result["text_report"] = build_text_report(result)
    return result
