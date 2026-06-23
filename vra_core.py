"""SOC spatial analysis and VRA zoning pipeline (Sentinel-2 L2A)."""

from __future__ import annotations

import io
import base64
from datetime import datetime
from typing import Any, Dict, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

import numpy as np
import rasterio
from rasterio.features import geometry_mask
from shapely.geometry import mapping
from scipy.ndimage import gaussian_filter

import utils

DPI = 150
SMOOTH_SIGMA = 2.0
USE_SHADOW = True

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

ZONE_DOSE_FRACTION = {"High": 0.30, "Medium": 0.65, "Low": 1.00}

SOC_THRESHOLDS = (0.5, 1.5)
N_THRESHOLDS = (100, 280)
P_THRESHOLDS = (15, 40)
K_THRESHOLDS = (100, 220)

_THRESH = {"SOC": SOC_THRESHOLDS, "N": N_THRESHOLDS, "P": P_THRESHOLDS, "K": K_THRESHOLDS}

_SOC_COLORS = [
    "#3d1c00", "#7a3b00", "#c17f00", "#e8c840", "#9ecb3c",
    "#4caf50", "#1b5e20",
]
CMAP_SOC = LinearSegmentedColormap.from_list("soc", _SOC_COLORS, N=256)

ZONE_COLORS = {
    "Low": "#d32f2f",
    "Medium": "#f9a825",
    "High": "#388e3c",
}
ZONE_INT = {"Low": 1, "Medium": 2, "High": 3}

_ZONE_CMAP = mcolors.ListedColormap(
    ["#000000", ZONE_COLORS["Low"], ZONE_COLORS["Medium"], ZONE_COLORS["High"]]
)
_ZONE_NORM = BoundaryNorm([0, 1, 2, 3, 4], _ZONE_CMAP.N)

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


def _safe_div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(np.abs(b) > 1e-9, a / b, np.nan).astype("float32")


def _smooth(arr: np.ndarray, sigma: float) -> np.ndarray:
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


def compute_index_maps(bands: dict) -> dict:
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
        "NDVI": _smooth(ndvi, SMOOTH_SIGMA),
        "NDWI": _smooth(ndwi, SMOOTH_SIGMA),
        "SAVI": _smooth(savi, SMOOTH_SIGMA),
        "BSI": _smooth(bsi, SMOOTH_SIGMA),
        "SWIR1": _smooth(swir1, SMOOTH_SIGMA),
        "CLAY": _smooth(clay, SMOOTH_SIGMA),
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


def classify_map(arr: np.ndarray, nutrient: str) -> np.ndarray:
    lo, hi = _THRESH[nutrient]
    out = np.zeros_like(arr, dtype="int8")
    valid = np.isfinite(arr)
    out[valid & (arr < lo)] = 1
    out[valid & (arr >= lo) & (arr <= hi)] = 2
    out[valid & (arr > hi)] = 3
    return out


def calc_vra_rates(zone_map: np.ndarray, nutrient: str, crop: str, res_m: float) -> dict:
    prod = FERTILISER_PRODUCTS[nutrient]
    max_dose = FERTILISER_MAX_DOSE[nutrient]
    px_area_m2 = res_m * res_m
    result = {}

    for zone_label, zone_int in ZONE_INT.items():
        cnt_px = int(np.count_nonzero(zone_map == zone_int))
        area_ha = (cnt_px * px_area_m2) / M2_HA
        area_ac = area_ha * HA_ACRE
        frac = ZONE_DOSE_FRACTION[zone_label]
        dose_nut = round(max_dose * frac, 1)
        dose_prod = round(dose_nut / (prod["nutrient_pct"] / 100.0), 1)
        total_kg = round(dose_prod * area_ha, 1)

        result[zone_label] = {
            "pixel_count": cnt_px,
            "area_ha": round(area_ha, 3),
            "area_acres": round(area_ac, 3),
            "nutrient_dose_kg_ha": dose_nut,
            "product": prod["name"],
            "product_dose_kg_ha": dose_prod,
            "total_product_kg": total_kg,
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


def _add_zone_legend(ax, title: str = "Soil Level") -> None:
    patches = [
        mpatches.Patch(color=ZONE_COLORS["Low"], label="Low  (High dose needed)"),
        mpatches.Patch(color=ZONE_COLORS["Medium"], label="Medium (Mod. dose)"),
        mpatches.Patch(color=ZONE_COLORS["High"], label="High  (Low dose needed)"),
    ]
    ax.legend(handles=patches, loc="lower left", fontsize=6, framealpha=0.85, title=title, title_fontsize=7)


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
        f"SOIL ORGANIC CARBON (SOC) MAP\nCapture: {capture_date}  |  10-m Sentinel-2",
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


def render_zone_map_single(
    zone_arr: np.ndarray,
    nutrient: str,
    vra_rates: dict,
    capture_date: str,
) -> str:
    valid = np.isfinite(zone_arr.astype(float)) & (zone_arr > 0)
    if not np.any(valid):
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        ax.text(0.5, 0.5, "No valid data", ha="center", va="center")
        ax.axis("off")
        return _arr_to_b64(fig)

    rows = np.where(valid.any(axis=1))[0]
    cols = np.where(valid.any(axis=0))[0]
    crop = zone_arr[rows[0]: rows[-1] + 1, cols[0]: cols[-1] + 1]
    img = np.ma.masked_equal(crop, 0)
    prod = FERTILISER_PRODUCTS[nutrient]["name"]

    fig = plt.figure(figsize=(8, 6), facecolor="#1a1a2e")
    fig.suptitle(
        f"VRA ZONE MAP — {nutrient}  |  {prod}\nCapture: {capture_date}",
        color="white",
        fontsize=11,
        fontweight="bold",
        y=0.99,
    )
    gs = fig.add_gridspec(1, 2, width_ratios=[3, 1], wspace=0.04)
    ax_map = fig.add_subplot(gs[0])
    ax_info = fig.add_subplot(gs[1])

    ax_map.imshow(img, cmap=_ZONE_CMAP, norm=_ZONE_NORM, interpolation="nearest")
    ax_map.set_axis_off()
    _add_zone_legend(ax_map, title=f"Soil {nutrient} Level")

    ax_info.set_facecolor("#1a1a2e")
    ax_info.set_axis_off()
    y = 0.97
    ax_info.text(
        0.05, y, f"APPLICATION RATES\n{prod}",
        color="white", fontsize=8, fontweight="bold", va="top", transform=ax_info.transAxes,
    )
    y -= 0.14

    for zone_lbl, dose_label in [("Low", "HIGH dose"), ("Medium", "MED dose"), ("High", "LOW dose")]:
        d = vra_rates.get(zone_lbl, {})
        if d.get("pixel_count", 0) == 0:
            continue
        color = ZONE_COLORS[zone_lbl]
        rect = mpatches.FancyBboxPatch(
            (0.03, y - 0.04), 0.10, 0.065,
            boxstyle="round,pad=0.005",
            linewidth=0,
            facecolor=color,
            transform=ax_info.transAxes,
            clip_on=False,
        )
        ax_info.add_patch(rect)
        lines = [
            f"Zone: {zone_lbl} soil",
            f"({dose_label})",
            f"{d['product_dose_kg_ha']} kg/ha",
            f"Area: {d['area_ha']:.2f} ha",
            f"Total: {d['total_product_kg']:.0f} kg",
        ]
        for i, line in enumerate(lines):
            ax_info.text(0.18, y - 0.005 - i * 0.028, line, color="white", fontsize=6, va="top", transform=ax_info.transAxes)
        y -= 0.24

    return _arr_to_b64(fig)


def render_combined_4panel(zone_maps: dict, soc_arr: np.ndarray, capture_date: str) -> str:
    def _crop(arr):
        v2 = np.isfinite(arr)
        if not np.any(v2):
            return arr
        rows = np.where(v2.any(axis=1))[0]
        cols = np.where(v2.any(axis=0))[0]
        if rows.size == 0 or cols.size == 0:
            return arr
        return arr[rows[0]: rows[-1] + 1, cols[0]: cols[-1] + 1]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), facecolor="#0f0f23")
    fig.suptitle(
        f"VARIABLE RATE APPLICATION  —  FIELD MANAGEMENT ZONES\n"
        f"Sentinel-2 L2A  |  Scene: {capture_date}",
        color="white",
        fontsize=13,
        fontweight="bold",
    )
    fig.subplots_adjust(hspace=0.15, wspace=0.1)

    panels = [
        ("SOC", "Soil Organic Carbon (%)", soc_arr, None, CMAP_SOC, (0.1, 3.0)),
        ("N", "Nitrogen Zones (VRA)", zone_maps["N"], "N", None, None),
        ("P", "Phosphorus Zones (VRA)", zone_maps["P"], "P", None, None),
        ("K", "Potassium Zones (VRA)", zone_maps["K"], "K", None, None),
    ]

    for ax, (key, title, arr, nut, cmap, vrange) in zip(axes.flat, panels):
        ax.set_facecolor("#0f0f23")
        cropped = _crop(arr)
        if key == "SOC":
            img = np.ma.masked_invalid(cropped)
            im = ax.imshow(img, cmap=cmap, vmin=vrange[0], vmax=vrange[1], interpolation="bilinear")
            cb = plt.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
            cb.set_label("SOC %", color="white", fontsize=8)
            cb.ax.yaxis.set_tick_params(color="white")
            plt.setp(cb.ax.yaxis.get_ticklabels(), color="white", fontsize=7)
        else:
            img = np.ma.masked_equal(cropped, 0)
            ax.imshow(img, cmap=_ZONE_CMAP, norm=_ZONE_NORM, interpolation="nearest")
            _add_zone_legend(ax, title=f"Soil {nut}")
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
        "  VARIABLE RATE APPLICATION (VRA) FIELD REPORT",
        f"  Crop        : {crop}",
        f"  Scene date  : {meta['capture_date']}",
        f"  Cloud cover : {meta['cloud_cover']} %",
        f"  Data source : {meta['collection']}",
        f"  Grid        : {meta['grid_H']} × {meta['grid_W']} px  @  {meta['res_m']:.1f} m/px",
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

    lines += ["", hr, "  ── VRA NUTRIENT ZONES & APPLICATION RATES ────────────────────", ""]
    for nut in ["N", "P", "K"]:
        prod = FERTILISER_PRODUCTS[nut]
        lines.append(f"  ► {nut} — {prod['name']}  ({prod['nutrient_pct']:.0f}% nutrient)")
        lines.append(
            f"    {'Zone':<10} {'Area (ha)':>10} {'Area (ac)':>10} "
            f"{'Dose kg/ha':>12} {'Total kg':>10}"
        )
        lines.append(f"    {'─'*10} {'─'*10} {'─'*10} {'─'*12} {'─'*10}")
        for zone_lbl in ["Low", "Medium", "High"]:
            d = vra[nut].get(zone_lbl, {})
            if d.get("pixel_count", 0) == 0:
                continue
            lines.append(
                f"    {zone_lbl:<10} {d['area_ha']:>10.3f} {d['area_acres']:>10.3f} "
                f"{d['product_dose_kg_ha']:>12.1f} {d['total_product_kg']:>10.1f}"
            )
        lines.append("")
    lines += [
        hr,
        "  ── INTERPRETATION ────────────────────────────────────────────",
        "  Zone 'Low'   = Soil is nutrient-deficient  → Apply FULL dose",
        "  Zone 'Medium'= Moderate soil content       → Apply 65% dose",
        "  Zone 'High'  = Soil is nutrient-rich       → Apply only 30% dose",
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

    aoi_sc, dst_tf, height, width, res_m = utils.build_adaptive_grid(crs, geometry, native_res_m=10.0)
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
) -> dict:
    crop_key = (crop or "wheat").lower()
    if crop_key not in CROP_DEMAND:
        crop_key = "default"

    data = _run_soil_pipeline(geometry, start_date, end_date, provider, satellite)
    soc_arr = data["soc_arr"]
    res_m = data["res_m"]
    cap_date = data["capture_date"]
    soil_maps = data["soil_maps"]

    zone_maps = {
        "N": classify_map(soil_maps["N"], "N"),
        "P": classify_map(soil_maps["P"], "P"),
        "K": classify_map(soil_maps["K"], "K"),
    }
    vra_rates = {nut: calc_vra_rates(z_map, nut, crop_key, res_m) for nut, z_map in zone_maps.items()}
    soc_stats = compute_soc_stats(soc_arr, res_m)

    metadata = {
        "capture_date": cap_date,
        "cloud_cover": data["cloud_cover"],
        "collection": data["collection"],
        "grid_H": data["height"],
        "grid_W": data["width"],
        "res_m": res_m,
        "mean_indices": {k: round(float(np.nanmean(v)), 5) for k, v in data["idx_maps"].items()},
    }

    images = None
    if include_images:
        images = {
            "soc_map_b64": render_soc_map(soc_arr, res_m, cap_date),
            "vra_n_b64": render_zone_map_single(zone_maps["N"], "N", vra_rates["N"], cap_date),
            "vra_p_b64": render_zone_map_single(zone_maps["P"], "P", vra_rates["P"], cap_date),
            "vra_k_b64": render_zone_map_single(zone_maps["K"], "K", vra_rates["K"], cap_date),
            "combined_b64": render_combined_4panel(zone_maps, soc_arr, cap_date),
        }

    result = {
        "date": cap_date,
        "crop": crop_key,
        "cloud_cover": data["cloud_cover"],
        "vra_rates": vra_rates,
        "soc_stats": soc_stats,
        "images": images,
        "metadata": metadata,
    }
    result["text_report"] = build_text_report(result)
    return result
