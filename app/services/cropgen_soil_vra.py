"""
CROPGEN SOIL + VRA ENGINE — v6 (single file)
=============================================
Sentinel-2 L2A, 10 m native, via STAC (AWS Element84 Earth-Search and
Microsoft Planetary Computer). No Google Earth Engine.

Everything in one module:
    1. Time-series scene discovery and cloud/shadow masking
    2. 13 spectral indices per scene
    3. Published exposed-soil-composite SOC retrieval (NDVI + NBR2 +
       spectral-shape cascade, multi-year median, PLSR)
    4. Empirical regional priors from 758 laboratory soil records
    5. Automatic region detection from the AOI coordinates
    6. Multivariate PCA + k-means management zones, vectorised to patches
    7. VRA prescription per nutrient, per zone, per patch
    8. Machine-executable prescription grid (GeoJSON + CSV)
    9. Maps, correlation matrix, time-series chart, text report

WHAT THIS ENGINE CLAIMS, AND WHAT IT DOES NOT
----------------------------------------------
A satellite measures reflected light. It does not see into the soil.
Every soil value here is inferred from how the bare surface and the crop
look, so the engine reports a calibration status on every map, every
cell of the prescription grid, and every JSON record:

  UNCALIBRATED    literature constants, not tuned to these soils.
                  Read as a relative pattern only.
  RANGE-ANCHORED  absolute scale fitted to measured lab distributions for
                  the detected region. The within-field pattern is still
                  unvalidated.
  CALIBRATED      fitted to geolocated samples matched pixel by pixel.

Variable-rate zoning does not need absolute values - it only needs to
know where the field is richer and poorer - so VRA output is sound at
every status level. Absolute numbers are not, and the stamp says so.

HONEST ACCURACY CEILING
------------------------
Published studies using this exact exposed-soil method with real
calibration samples report:

  Vaudour et al. 2021, Versailles Plain   R2 0.53, RMSE 3.2 g C/kg, RPD 1.46
  Belgium / Netherlands composite         R2 0.48 +/- 0.07, RPD 1.4 +/- 0.1
  Dvorakova, greening-up + NBR2 < 0.07    R2 0.54 +/- 0.12

R2 near 0.5 is what a correct implementation achieves for satellite SOC
over croplands. Any claim above 0.9 is a fit being reported as a
validation.

Install
-------
pip install pystac-client rasterio pyproj shapely scipy numpy matplotlib affine

Usage
-----
python cropgen_soil_vra.py
"""

import os, io, base64, math, json, csv, time, threading, warnings
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable

import rasterio
from rasterio.windows import from_bounds, Window
from rasterio.windows import transform as win_transform
from rasterio.enums import Resampling
from rasterio.features import geometry_mask, shapes as rio_shapes
from rasterio.warp import reproject
from affine import Affine

from pystac_client import Client
from rasterio.crs import CRS

try:
    import planetary_computer
except ImportError:  # AWS-only deployments
    planetary_computer = None
from shapely.geometry import shape, mapping, MultiPolygon
from shapely.ops import transform as shp_transform
from pyproj import Transformer
from scipy.ndimage import (gaussian_filter, median_filter, label as cc_label,
                            distance_transform_edt)

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════
# PART 1 — EMPIRICAL REGIONAL PRIORS (758 laboratory records)
# ══════════════════════════════════════════════════════════════════════════

SOIL_PRIORS = {
    "_ALL": {"Cu": (0.264, 1.314, 4.093, 758), "EC": (0.14, 0.42, 1.16, 758), "Fe": (2.114, 12.025, 22.469, 758), "K": (31.562, 77.44, 436.167, 758), "Mn": (3.356, 8.793, 15.37, 758), "OC": (0.14, 0.61, 1.449, 758), "P": (3.002, 24.881, 120.55, 758), "Zn": (0.181, 1.521, 3.814, 758), "pH": (6.6, 7.5, 8.4, 758)},
    "CHHATTISGARH": {"Cu": (1.3, 3.082, 4.563, 99), "EC": (0.318, 0.58, 1.045, 99), "Fe": (5.219, 14.37, 23.078, 99), "K": (97.139, 302.46, 601.155, 99), "Mn": (9.911, 12.06, 17.008, 99), "OC": (0.591, 1.29, 2.07, 99), "P": (2.509, 12.82, 65.092, 99), "Zn": (0.498, 1.478, 2.897, 99), "pH": (6.5, 6.9, 7.2, 99)},
    "D:BARMER": {"Cu": (0.044, 0.222, 0.498, 41), "EC": (0.14, 0.18, 0.36, 41), "Fe": (0.43, 1.202, 3.134, 41), "K": (45.48, 81.06, 217.33, 41), "Mn": (2.41, 4.92, 12.29, 41), "OC": (0.1, 0.14, 0.31, 41), "P": (5.13, 19.49, 38.98, 41), "Zn": (0.1, 0.14, 0.67, 41), "pH": (7.4, 7.6, 7.8, 41)},
    "D:BILASPUR": {"Cu": (2.852, 3.474, 4.569, 50), "EC": (0.474, 0.58, 0.69, 50), "Fe": (5.99, 19.27, 23.159, 50), "K": (193.703, 408.11, 606.442, 50), "Mn": (9.891, 11.745, 12.106, 50), "OC": (1.171, 1.7, 2.108, 50), "P": (3.309, 12.565, 25.981, 50), "Zn": (0.864, 1.63, 2.294, 50), "pH": (6.545, 6.8, 7.1, 50)},
    "D:BULDHANA": {"Cu": (2.991, 3.828, 5.811, 40), "EC": (0.367, 0.63, 0.98, 40), "Fe": (4.354, 8.666, 19.96, 40), "K": (90.174, 193.445, 1011.84, 40), "Mn": (9.063, 11.115, 12.121, 40), "OC": (0.427, 0.705, 1.291, 40), "P": (2.05, 3.59, 26.416, 40), "Zn": (0.711, 1.393, 2.498, 40), "pH": (7.395, 7.55, 8.0, 40)},
    "D:ETAH": {"Cu": (0.542, 1.006, 3.027, 97), "EC": (0.13, 0.22, 0.762, 97), "Fe": (6.142, 13.69, 23.572, 97), "K": (47.36, 65.55, 86.302, 97), "Mn": (5.077, 8.962, 13.278, 97), "OC": (0.26, 0.43, 0.654, 97), "P": (5.13, 61.56, 114.492, 97), "Zn": (0.853, 2.108, 3.11, 97), "pH": (6.6, 7.1, 7.6, 97)},
    "D:FIROZABAD": {"Cu": (0.658, 1.418, 2.595, 59), "EC": (0.35, 0.52, 0.752, 59), "Fe": (3.556, 8.368, 21.448, 59), "K": (45.024, 88.24, 235.267, 59), "Mn": (3.999, 6.498, 9.795, 59), "OC": (0.13, 0.43, 0.98, 59), "P": (3.946, 35.91, 96.228, 59), "Zn": (0.283, 1.044, 3.989, 59), "pH": (7.3, 7.7, 8.31, 59)},
    "D:HAPUR": {"Cu": (0.633, 0.98, 2.204, 53), "EC": (0.206, 0.33, 0.528, 53), "Fe": (5.3, 13.5, 21.776, 53), "K": (55.64, 69.088, 120.768, 53), "Mn": (3.931, 6.75, 9.342, 53), "OC": (0.37, 0.73, 0.99, 53), "P": (16.62, 92.33, 192.874, 53), "Zn": (0.896, 2.474, 3.996, 53), "pH": (7.7, 8.1, 8.62, 53)},
    "D:HARDOI": {"Cu": (0.863, 0.98, 1.117, 30), "EC": (0.145, 0.16, 0.216, 30), "Fe": (13.418, 14.41, 15.337, 30), "K": (81.141, 102.22, 109.951, 30), "Mn": (8.524, 9.18, 9.802, 30), "OC": (0.572, 0.715, 0.82, 30), "P": (72.378, 105.925, 135.242, 30), "Zn": (1.232, 1.414, 1.622, 30), "pH": (7.3, 7.4, 7.5, 30)},
    "D:HARIDWAR": {"Cu": (1.032, 1.408, 2.241, 24), "EC": (0.145, 0.22, 0.325, 24), "Fe": (8.099, 13.705, 22.983, 24), "K": (15.908, 27.88, 53.859, 24), "Mn": (5.272, 10.064, 13.543, 24), "OC": (0.193, 0.64, 0.96, 24), "P": (5.207, 9.233, 35.522, 24), "Zn": (0.91, 2.579, 3.936, 24), "pH": (7.815, 8.1, 8.3, 24)},
    "D:HATHRAS": {"Cu": (0.504, 0.674, 0.83, 45), "EC": (0.272, 0.47, 1.222, 45), "Fe": (5.93, 7.668, 8.946, 45), "K": (75.878, 118.27, 169.814, 45), "Mn": (7.266, 11.58, 14.764, 45), "OC": (0.244, 0.52, 0.91, 45), "P": (13.232, 60.02, 90.178, 45), "Zn": (0.789, 1.288, 1.924, 45), "pH": (7.2, 7.6, 8.1, 45)},
    "D:MAINPURI": {"Cu": (2.244, 2.787, 4.063, 50), "EC": (0.35, 0.48, 1.883, 50), "Fe": (11.361, 19.935, 22.646, 50), "K": (44.385, 51.57, 389.294, 50), "Mn": (4.961, 6.144, 7.34, 50), "OC": (0.49, 0.6, 0.828, 50), "P": (19.209, 60.785, 129.19, 50), "Zn": (1.243, 2.134, 4.042, 50), "pH": (7.045, 7.3, 7.5, 50)},
    "D:MUKTSAR": {"Cu": (0.739, 1.314, 1.987, 44), "EC": (0.5, 0.945, 1.616, 44), "Fe": (4.845, 13.9, 18.799, 44), "K": (36.616, 70.015, 169.291, 44), "Mn": (1.376, 5.59, 11.539, 44), "OC": (0.2, 0.545, 0.979, 44), "P": (2.56, 15.645, 71.868, 44), "Zn": (0.751, 1.812, 3.25, 44), "pH": (7.215, 7.5, 7.785, 44)},
    "D:RAIPUR": {"Cu": (1.21, 2.18, 4.345, 49), "EC": (0.288, 0.57, 1.38, 49), "Fe": (5.092, 11.37, 21.872, 49), "K": (81.36, 173.16, 449.696, 49), "Mn": (10.662, 15.8, 17.166, 49), "OC": (0.51, 0.97, 1.488, 49), "P": (2.05, 15.39, 72.946, 49), "Zn": (0.367, 1.18, 3.278, 49), "pH": (6.4, 6.9, 7.2, 49)},
    "D:SABARKANTHA": {"Cu": (0.651, 0.856, 1.135, 22), "EC": (0.261, 0.475, 0.698, 22), "Fe": (3.774, 4.807, 6.48, 22), "K": (40.048, 55.54, 360.95, 22), "Mn": (9.114, 11.125, 14.589, 22), "OC": (0.341, 0.5, 0.688, 22), "P": (23.78, 40.27, 125.008, 22), "Zn": (1.104, 2.026, 2.945, 22), "pH": (6.63, 7.4, 7.7, 22)},
    "D:SANGRUR": {"Cu": (0.616, 1.221, 2.002, 50), "EC": (0.34, 0.56, 1.146, 50), "Fe": (4.699, 14.81, 21.788, 50), "K": (39.517, 61.795, 173.535, 50), "Mn": (4.593, 7.34, 10.524, 50), "OC": (0.515, 0.945, 0.99, 50), "P": (13.055, 17.697, 71.455, 50), "Zn": (1.009, 1.939, 4.633, 50), "pH": (7.9, 8.2, 8.555, 50)},
    "D:WASHIM": {"Cu": (2.48, 3.72, 4.451, 23), "EC": (0.146, 0.27, 0.469, 23), "Fe": (5.536, 10.35, 15.403, 23), "K": (161.404, 305.24, 854.265, 23), "Mn": (11.007, 13.17, 14.843, 23), "OC": (0.65, 0.85, 0.99, 23), "P": (3.182, 14.36, 39.497, 23), "Zn": (0.832, 1.174, 2.311, 23), "pH": (6.92, 7.3, 7.4, 23)},
    "GUJARAT": {"Cu": (0.662, 1.359, 6.859, 46), "EC": (0.21, 0.405, 0.845, 46), "Fe": (3.56, 4.686, 20.707, 46), "K": (21.043, 53.37, 346.858, 46), "Mn": (5.349, 9.979, 14.545, 46), "OC": (0.333, 0.49, 0.885, 46), "P": (10.259, 27.445, 62.328, 46), "Zn": (0.643, 1.585, 2.934, 46), "pH": (7.2, 8.3, 8.4, 46)},
    "MAHARASHTRA": {"Cu": (2.504, 3.744, 5.163, 63), "EC": (0.23, 0.54, 0.918, 63), "Fe": (4.912, 8.938, 19.141, 63), "K": (90.725, 216.51, 895.968, 63), "Mn": (9.253, 11.52, 14.466, 63), "OC": (0.531, 0.76, 1.255, 63), "P": (2.101, 6.67, 34.521, 63), "Zn": (0.716, 1.306, 2.493, 63), "pH": (7.1, 7.5, 7.88, 63)},
    "PUNJAB": {"Cu": (0.688, 1.24, 2.038, 96), "EC": (0.34, 0.715, 1.51, 96), "Fe": (4.638, 14.81, 21.0, 96), "K": (39.367, 66.045, 177.97, 96), "Mn": (2.532, 6.708, 11.077, 96), "OC": (0.265, 0.8, 0.99, 96), "P": (3.972, 16.93, 78.996, 96), "Zn": (0.937, 1.939, 4.472, 96), "pH": (7.3, 7.85, 8.5, 96)},
    "RAJASTHAN": {"Cu": (0.044, 0.237, 0.494, 44), "EC": (0.14, 0.19, 0.36, 44), "Fe": (0.43, 1.246, 3.127, 44), "K": (45.594, 83.42, 213.561, 44), "Mn": (2.42, 5.171, 12.178, 44), "OC": (0.1, 0.14, 0.456, 44), "P": (5.207, 18.98, 38.827, 44), "Zn": (0.1, 0.148, 0.666, 44), "pH": (7.4, 7.6, 7.8, 44)},
    "U.K.": {"Cu": (0.683, 1.403, 2.486, 34), "EC": (0.14, 0.225, 0.454, 34), "Fe": (8.309, 16.11, 26.14, 34), "K": (17.433, 33.185, 56.383, 34), "Mn": (5.786, 10.125, 13.879, 34), "OC": (0.16, 0.605, 0.97, 34), "P": (3.08, 8.977, 67.995, 34), "Zn": (0.513, 2.136, 3.908, 34), "pH": (6.295, 8.05, 8.3, 34)},
    "U.P.": {"Cu": (0.485, 1.038, 3.303, 371), "EC": (0.13, 0.37, 0.985, 371), "Fe": (4.713, 12.84, 22.33, 371), "K": (33.565, 70.67, 215.996, 371), "Mn": (3.149, 7.836, 13.615, 371), "OC": (0.205, 0.52, 0.95, 371), "P": (5.08, 60.02, 134.655, 371), "Zn": (0.31, 1.592, 3.793, 371), "pH": (6.9, 7.4, 8.2, 371)},
}

# District keys are prefixed "D:". Lookup helper below handles that.
_PARAM_ALIASES = {
    "SOC": "OC", "PH": "pH", "EC": "EC", "K": "K", "P": "P",
    "Zn": "Zn", "Cu": "Cu", "Fe": "Fe", "Mn": "Mn",
}


def _norm(s):
    return str(s).strip().upper() if s is not None else None


def lookup_prior(param, district=None, state=None):
    """
    Return (p5, p50, p95, n) for a parameter, most specific region first.

    param accepts either the register's own name ("OC", "pH") or the
    engine parameter key ("SOC", "PH"). Returns None when no region has
    enough records for that parameter, which callers must treat as "no
    prior available" rather than substituting a guess.
    """
    key = _PARAM_ALIASES.get(param, param)
    for region in (f"D:{_norm(district)}" if district else None,
                   _norm(state) if state else None,
                   "_ALL"):
        if region and region in SOIL_PRIORS:
            hit = SOIL_PRIORS[region].get(key)
            if hit:
                return hit
    return None


def prior_range(param, district=None, state=None, spread="p5_p95"):
    """
    (low, high) bounds for anchoring a 0-1 index into physical units.

    spread="p5_p95" uses the observed 5th-95th percentile, which is the
    right default: it ignores lab outliers while still covering the range
    a field will actually contain.
    """
    hit = lookup_prior(param, district, state)
    if hit is None:
        return None
    p5, p50, p95, _n = hit
    if spread == "p5_p95":
        return (p5, p95)
    if spread == "median_centred":
        half = max(p95 - p50, p50 - p5)
        return (p50 - half, p50 + half)
    raise ValueError(f"unknown spread {spread!r}")


def region_support(district=None, state=None):
    """How many records back this region, so callers can judge trust."""
    for region in (f"D:{_norm(district)}" if district else None,
                   _norm(state) if state else None,
                   "_ALL"):
        if region and region in SOIL_PRIORS:
            any_param = next(iter(SOIL_PRIORS[region].values()), None)
            if any_param:
                return region, any_param[3]
    return None, 0


def available_regions():
    states = sorted(k for k in SOIL_PRIORS if not k.startswith("D:") and k != "_ALL")
    districts = sorted(k[2:] for k in SOIL_PRIORS if k.startswith("D:"))
    return {"states": states, "districts": districts}

# ══════════════════════════════════════════════════════════════════════════
# PART 2 — PUBLISHED EXPOSED-SOIL SOC ENGINE
# ══════════════════════════════════════════════════════════════════════════

# ──────────────────────────────────────────────────────────────────────────
# PUBLISHED THRESHOLDS
# ──────────────────────────────────────────────────────────────────────────
NDVI_BARE_MAX = 0.30    # Vaudour 2021 / Urbina-Salazar 2021 use 0.25-0.30
NBR2_MAX      = 0.075   # Castaldi 2019a; stricter 0.05 available via arg
USE_SHAPE_FILTER = True # Dematte 2018a visible-slope test

# Sentinel-2 bands used for the soil spectrum, in wavelength order.
SOIL_BANDS = ["B02", "B03", "B04", "B08", "B11", "B12"]


def nbr2(b11, b12):
    """Normalized Burn Ratio 2. Low values indicate dry, residue-free soil."""
    with np.errstate(invalid="ignore", divide="ignore"):
        d = b11 + b12
        return np.where(np.abs(d) > 1e-9, (b11 - b12) / d, np.nan).astype("float32")


def ndvi(b08, b04):
    with np.errstate(invalid="ignore", divide="ignore"):
        d = b08 + b04
        return np.where(np.abs(d) > 1e-9, (b08 - b04) / d, np.nan).astype("float32")


def exposed_soil_mask(bands, ndvi_max=NDVI_BARE_MAX, nbr2_max=NBR2_MAX,
                       shape_filter=USE_SHAPE_FILTER):
    """
    Per-pixel "this is genuinely exposed, dry, residue-free soil" mask for
    one scene, following the published cascade.

    Returns (mask, diagnostics) so a caller can report why pixels were
    dropped rather than silently losing them.
    """
    B02, B03, B04 = bands["B02"], bands["B03"], bands["B04"]
    B08, B11, B12 = bands["B08"], bands["B11"], bands["B12"]

    finite = np.ones(B04.shape, bool)
    for b in SOIL_BANDS:
        finite &= np.isfinite(bands[b])

    nd = ndvi(B08, B04)
    nb = nbr2(B11, B12)

    m_veg = finite & (nd < ndvi_max)                 # growing crop removed
    m_res = m_veg & (nb < nbr2_max)                  # residue + wet soil removed
    if shape_filter:
        m_all = m_res & (B03 > B02) & (B04 > B03)    # soil spectral shape
    else:
        m_all = m_res

    diag = {
        "n_finite": int(finite.sum()),
        "after_ndvi": int(m_veg.sum()),
        "after_nbr2": int(m_res.sum()),
        "after_shape": int(m_all.sum()),
    }
    return m_all, diag


def build_soil_reflectance_composite(scene_bands, masks, percentile=50,
                                      min_obs=3):
    """
    Multi-year exposed-soil reflectance composite (SRC).

    scene_bands : list of per-scene band dicts
    masks       : list of per-scene exposed-soil masks
    percentile  : 50 = median (standard). Lower values bias toward darker,
                  drier surfaces; Vaudour 2021 compares such variants.
    min_obs     : a pixel needs at least this many exposed-soil looks
                  before a composite value is trusted. Multi-year archives
                  are what make this achievable under strict masking.

    Returns (composite, n_obs) where composite maps band -> (H,W).
    """
    if not scene_bands:
        return None, None
    H, W = scene_bands[0]["B04"].shape
    S = len(scene_bands)

    stack = {b: np.full((S, H, W), np.nan, "float32") for b in SOIL_BANDS}
    for i, (bd, mk) in enumerate(zip(scene_bands, masks)):
        for b in SOIL_BANDS:
            stack[b][i] = np.where(mk, bd[b], np.nan)

    n_obs = np.isfinite(stack["B04"]).sum(axis=0).astype("int16")
    enough = n_obs >= min_obs

    comp = {}
    with np.errstate(all="ignore"):
        for b in SOIL_BANDS:
            v = np.nanpercentile(stack[b], percentile, axis=0)
            comp[b] = np.where(enough, v, np.nan).astype("float32")
    return comp, n_obs


def normalise_spectra(comp):
    """
    Divide each band by the mean reflectance across all bands.

    Cancels the albedo shift between soil crusts and smooth seed-bed
    surfaces, which is a surface-condition artefact rather than a soil
    property and otherwise dominates the composite.
    """
    arrs = np.stack([comp[b] for b in SOIL_BANDS], axis=0)
    with np.errstate(all="ignore"):
        mean = np.nanmean(arrs, axis=0)
        out = {b: np.where(np.abs(mean) > 1e-9, comp[b] / mean, np.nan).astype("float32")
               for b in SOIL_BANDS}
    return out, mean.astype("float32")


# ══════════════════════════════════════════════════════════════════════════
# PLSR  (NIPALS) — the standard regression for soil reflectance spectra
# ══════════════════════════════════════════════════════════════════════════
class PLSR:
    """
    Partial Least Squares Regression, NIPALS algorithm (Wold et al. 2001).

    Ordinary least squares fails on soil spectra because the bands are
    strongly collinear. PLSR projects onto a small number of latent
    variables built to covary with the target, which is why the soil
    literature standardises on it. Four latent variables was optimal in
    71% of runs in the Belgium/Netherlands study.
    """

    def __init__(self, n_components=4):
        self.n_components = n_components
        self.x_mean_ = self.x_std_ = self.y_mean_ = None
        self.coef_ = self.intercept_ = None

    def fit(self, X, y):
        X = np.asarray(X, float)
        y = np.asarray(y, float).ravel()
        n, p = X.shape
        k = max(1, min(self.n_components, p, n - 1))

        self.x_mean_ = X.mean(0)
        self.x_std_ = X.std(0)
        self.x_std_[self.x_std_ < 1e-12] = 1.0
        self.y_mean_ = y.mean()

        Xc = (X - self.x_mean_) / self.x_std_
        yc = y - self.y_mean_

        P = np.zeros((p, k))
        W = np.zeros((p, k))
        Q = np.zeros(k)
        E, f = Xc.copy(), yc.copy()

        for a in range(k):
            w = E.T @ f
            nw = np.linalg.norm(w)
            if nw < 1e-12:
                k = a
                P, W, Q = P[:, :k], W[:, :k], Q[:k]
                break
            w /= nw
            t = E @ w
            tt = t @ t
            if tt < 1e-12:
                k = a
                P, W, Q = P[:, :k], W[:, :k], Q[:k]
                break
            p_load = (E.T @ t) / tt
            q = (f @ t) / tt
            E = E - np.outer(t, p_load)
            f = f - q * t
            W[:, a], P[:, a], Q[a] = w, p_load, q

        if k == 0:
            self.coef_ = np.zeros(p)
            self.intercept_ = self.y_mean_
            return self

        Rstar = W @ np.linalg.pinv(P.T @ W)
        beta_scaled = Rstar @ Q
        self.coef_ = beta_scaled / self.x_std_
        self.intercept_ = self.y_mean_ - self.x_mean_ @ self.coef_
        self.n_components_used_ = k
        return self

    def predict(self, X):
        return np.asarray(X, float) @ self.coef_ + self.intercept_


def regression_metrics(y_true, y_pred):
    """
    R2, RMSE, RPD, RPIQ — the metric set used across this literature, so
    results can be compared to published values instead of an internal one.

    RPD  = SD(observed) / RMSE.  < 1.4 not usable beyond ranking,
           1.4-2.0 usable for spatial patterns, > 2.0 rare at field scale.
    RPIQ = IQR(observed) / RMSE. More robust than RPD on skewed SOC data.
    """
    y_true = np.asarray(y_true, float).ravel()
    y_pred = np.asarray(y_pred, float).ravel()
    n = len(y_true)
    res = y_true - y_pred
    ss_res = float((res ** 2).sum())
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    rmse = float(np.sqrt(ss_res / n))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan")
    sd = float(y_true.std(ddof=1)) if n > 1 else float("nan")
    q1, q3 = np.percentile(y_true, [25, 75])
    return {
        "n": n,
        "r2": round(r2, 4),
        "rmse": round(rmse, 4),
        "rpd": round(sd / rmse, 3) if rmse > 1e-12 else None,
        "rpiq": round(float(q3 - q1) / rmse, 3) if rmse > 1e-12 else None,
        "bias": round(float(res.mean()), 4),
    }


def cross_validate_plsr(X, y, max_components=8, n_folds=10, seed=0):
    """
    k-fold CV over latent-variable counts; returns the best model plus its
    cross-validated metrics.

    The reported metrics come from held-out folds, never from the fit, so
    they cannot flatter the way an in-sample R2 does on small samples.
    """
    X = np.asarray(X, float)
    y = np.asarray(y, float).ravel()
    n = len(y)
    n_folds = max(2, min(n_folds, n))
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    folds = np.array_split(order, n_folds)

    max_components = max(1, min(max_components, X.shape[1], n - max(len(f) for f in folds) - 1))

    results = {}
    for k in range(1, max_components + 1):
        pred = np.zeros(n)
        for f in folds:
            tr = np.setdiff1d(order, f)
            if len(tr) <= k:
                pred[f] = y[tr].mean() if len(tr) else y.mean()
                continue
            m = PLSR(k).fit(X[tr], y[tr])
            pred[f] = m.predict(X[f])
        results[k] = regression_metrics(y, pred)

    best_k = min(results, key=lambda k: results[k]["rmse"])
    final = PLSR(best_k).fit(X, y)
    return {
        "model": final,
        "n_components": best_k,
        "cv_metrics": results[best_k],
        "cv_by_component": {k: v for k, v in results.items()},
    }


def interpret_rpd(rpd):
    """Plain reading of the RPD ratio, so a number is never quoted alone."""
    if rpd is None or not np.isfinite(rpd):
        return "not computable"
    if rpd < 1.4:
        return ("below 1.4 — not usable for absolute values; treat the output "
                "as a ranking of high/low areas only")
    if rpd < 2.0:
        return ("1.4-2.0 — usable for spatial patterns and variable-rate zoning, "
                "which is the normal result for satellite SOC over croplands")
    return "above 2.0 — strong for this scale; verify it is not an overfit"


def build_feature_matrix(comp_norm, brightness, extra=None):
    """
    Feature matrix for PLSR: normalised band reflectances plus overall
    brightness, optionally with extra covariates (terrain, climate).

    Returns (X, names, valid_mask) flattened over the raster.
    """
    layers = [comp_norm[b] for b in SOIL_BANDS] + [brightness]
    names = [f"{b}_norm" for b in SOIL_BANDS] + ["brightness"]
    if extra:
        for k, v in extra.items():
            layers.append(v)
            names.append(k)
    stack = np.stack(layers, axis=0)
    valid = np.all(np.isfinite(stack), axis=0)
    X = stack[:, valid].T
    return X, names, valid

# ══════════════════════════════════════════════════════════════════════════
# PART 3 — PRESCRIPTION GRID
# ══════════════════════════════════════════════════════════════════════════

DEFAULT_CELL_M = 20.0      # a common spreader working width
MIN_COVERAGE = 0.30        # drop cells less than 30% inside the field
UNIFORM_CV_MAX = 0.15      # within-cell CV below this counts as uniform


def _cell_blocks(H, W, res_m, cell_m):
    """Yield (row, col, slice_y, slice_x) for each grid cell."""
    step = max(1, int(round(cell_m / res_m)))
    for r, y0 in enumerate(range(0, H, step)):
        for c, x0 in enumerate(range(0, W, step)):
            yield r, c, slice(y0, min(y0 + step, H)), slice(x0, min(x0 + step, W))


def build_prescription_grid(aoi_mask, transform, res_m, soc_map,
                             zone_grid=None, zone_labels=None,
                             vra_rates=None, nutrient_maps=None,
                             cell_m=DEFAULT_CELL_M,
                             min_coverage=MIN_COVERAGE,
                             to_wgs84=None, confidence=None):
    """
    Aggregate pixel rasters into machine-executable grid cells.

    aoi_mask     (H,W) bool, True inside the field
    transform    affine of the pixel grid, in the scene CRS
    soc_map      (H,W) float SOC values
    zone_grid    (H,W) int zone ids, 0 = outside
    vra_rates    {nutrient: {zone_class: {...dose fields...}}}
    to_wgs84     pyproj Transformer scene CRS -> EPSG:4326; if None,
                 coordinates are emitted in the scene CRS and flagged
    confidence   {param: status} carried onto every cell

    Returns a dict with "cells" (list) and "summary".
    """
    H, W = aoi_mask.shape
    a, e, cx, fy = transform.a, transform.e, transform.c, transform.f
    px_area_ha = (res_m * res_m) / 10000.0
    step = max(1, int(round(cell_m / res_m)))
    cells = []
    cid = 0

    for r, c, sy, sx in _cell_blocks(H, W, res_m, cell_m):
        m = aoi_mask[sy, sx]
        n_tot = m.size
        n_in = int(m.sum())
        if n_in == 0:
            continue
        cov = n_in / n_tot
        if cov < min_coverage:
            continue

        cid += 1
        soc_vals = soc_map[sy, sx][m]
        soc_vals = soc_vals[np.isfinite(soc_vals)]
        if soc_vals.size == 0:
            continue
        soc_mean = float(soc_vals.mean())
        soc_std = float(soc_vals.std())
        soc_cv = soc_std / abs(soc_mean) if abs(soc_mean) > 1e-9 else float("nan")

        # cell polygon corners in scene CRS
        y0, x0 = sy.start, sx.start
        y1, x1 = sy.stop, sx.stop
        corners = [(cx + x0 * a, fy + y0 * e), (cx + x1 * a, fy + y0 * e),
                   (cx + x1 * a, fy + y1 * e), (cx + x0 * a, fy + y1 * e)]
        cen = (cx + (x0 + x1) / 2 * a, fy + (y0 + y1) / 2 * e)

        if to_wgs84 is not None:
            corners_ll = [to_wgs84.transform(x, y) for x, y in corners]
            cen_ll = to_wgs84.transform(*cen)
            crs_note = "EPSG:4326"
        else:
            corners_ll = corners
            cen_ll = cen
            crs_note = "scene CRS (no transformer supplied)"

        rec = {
            "cell_id": cid, "row": r, "col": c,
            "lon": round(cen_ll[0], 8), "lat": round(cen_ll[1], 8),
            "polygon": [[round(x, 8), round(y, 8)] for x, y in corners_ll] +
                       [[round(corners_ll[0][0], 8), round(corners_ll[0][1], 8)]],
            "crs": crs_note,
            "pixel_count": n_in,
            "area_ha": round(n_in * px_area_ha, 5),
            "coverage_frac": round(cov, 3),
            "soc_mean": round(soc_mean, 4),
            "soc_std": round(soc_std, 4),
            "soc_cv": round(soc_cv, 4) if np.isfinite(soc_cv) else None,
            "uniform": bool(np.isfinite(soc_cv) and soc_cv <= UNIFORM_CV_MAX),
        }

        if zone_grid is not None:
            zv = zone_grid[sy, sx][m]
            zv = zv[zv > 0]
            if zv.size:
                vals, counts = np.unique(zv, return_counts=True)
                dom = int(vals[np.argmax(counts)])
                rec["zone"] = dom
                rec["zone_purity"] = round(float(counts.max() / zv.size), 3)
                if zone_labels:
                    rec["zone_class"] = zone_labels.get(dom, str(dom))

        if nutrient_maps:
            for nut, arr in nutrient_maps.items():
                v = arr[sy, sx][m]
                v = v[np.isfinite(v)]
                if v.size:
                    rec[f"{nut.lower()}_soil_mean"] = round(float(v.mean()), 3)

        if vra_rates and "zone_class" in rec:
            for nut, table in vra_rates.items():
                d = table.get(rec["zone_class"])
                if not d:
                    continue
                dose = d.get("product_dose_kg_ha")
                if dose is None:
                    continue
                rec[f"{nut.lower()}_product"] = d.get("product")
                rec[f"{nut.lower()}_dose_kg_ha"] = dose
                rec[f"{nut.lower()}_total_kg"] = round(dose * rec["area_ha"], 3)

        # A prescription file must never leave the calibration status
        # implicit. If none was supplied, say so explicitly rather than
        # emitting a file that looks authoritative.
        rec["confidence"] = dict(confidence) if confidence else {"_": "UNKNOWN"}
        cells.append(rec)

    n_uniform = sum(1 for c in cells if c["uniform"])
    total_ha = sum(c["area_ha"] for c in cells)
    summary = {
        "cell_size_m": cell_m,
        "pixels_per_cell": step * step,
        "n_cells": len(cells),
        "total_area_ha": round(total_ha, 4),
        "uniform_cells": n_uniform,
        "mixed_cells": len(cells) - n_uniform,
        "uniform_pct": round(n_uniform / max(len(cells), 1) * 100, 1),
        "min_coverage_applied": min_coverage,
    }
    if vra_rates:
        for nut in vra_rates:
            tot = sum(c.get(f"{nut.lower()}_total_kg", 0.0) for c in cells)
            summary[f"{nut.lower()}_total_kg"] = round(tot, 2)
            if total_ha > 0:
                summary[f"{nut.lower()}_avg_kg_ha"] = round(tot / total_ha, 2)
    return {"cells": cells, "summary": summary}


def grid_to_geojson(grid):
    """FeatureCollection of cell polygons — loads into QGIS, most farm
    software, and machinery import tools."""
    feats = []
    for c in grid["cells"]:
        props = {k: v for k, v in c.items() if k != "polygon"}
        feats.append({"type": "Feature", "properties": props,
                       "geometry": {"type": "Polygon", "coordinates": [c["polygon"]]}})
    return {"type": "FeatureCollection",
            "crs": {"type": "name", "properties": {"name": "EPSG:4326"}},
            "metadata": grid["summary"], "features": feats}


def grid_to_csv(grid, path):
    """Flat CSV — one row per cell. Nested confidence is flattened so the
    file opens cleanly in Excel and in controller import tools."""
    rows = []
    for c in grid["cells"]:
        r = {k: v for k, v in c.items() if k not in ("polygon", "confidence")}
        conf = c.get("confidence") or {"_": "UNKNOWN"}
        for pk, pv in conf.items():
            r[f"conf_{pk}"] = pv
        rows.append(r)
    if not rows:
        return 0
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    return len(rows)


def grid_report(grid):
    s = grid["summary"]
    L = ["=" * 68,
         "  PRESCRIPTION GRID",
         "=" * 68,
         f"  Cell size        : {s['cell_size_m']:.0f} m "
         f"({s['pixels_per_cell']} pixels per cell)",
         f"  Cells            : {s['n_cells']}",
         f"  Area             : {s['total_area_ha']:.3f} ha",
         f"  Uniform cells    : {s['uniform_cells']} ({s['uniform_pct']:.0f}%)",
         f"  Mixed cells      : {s['mixed_cells']}",
         ""]
    nuts = [k[:-9] for k in s if k.endswith("_total_kg")]
    if nuts:
        L.append(f"  {'Product':<10}{'Total kg':>12}{'Avg kg/ha':>12}")
        L.append("  " + "-" * 34)
        for n in nuts:
            L.append(f"  {n.upper():<10}{s[f'{n}_total_kg']:>12.1f}"
                     f"{s.get(f'{n}_avg_kg_ha', 0):>12.2f}")
        L.append("")
    L += ["  Cell size should match the spreader working width. A cell marked",
          "  'mixed' contains real variation the machine cannot resolve at this",
          "  width — narrow the cell or accept the averaged rate there.",
          "=" * 68]
    return "\n".join(L)

# ══════════════════════════════════════════════════════════════════════════
# PART 4 — PIPELINE
# ══════════════════════════════════════════════════════════════════════════



# ──────────────────────────────────────────────────────────────────────────
# GDAL / AWS env
# ──────────────────────────────────────────────────────────────────────────
for k, v in {
    "CPL_VSIL_CURL_USE_HEAD":           "FALSE",
    "GDAL_DISABLE_READDIR_ON_OPEN":     "EMPTY_DIR",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif,.tiff,.jp2,.TIF,.TIFF,.JP2",
    "AWS_NO_SIGN_REQUEST":              "YES",
    "AWS_REQUEST_PAYER":                "requester",
    "GDAL_HTTP_MULTIRANGE":             "YES",
    "GDAL_CACHEMAX":                    "512",
    "CPL_VSIL_CURL_CHUNK_SIZE":         "32768",
}.items():
    os.environ.setdefault(k, v)

# ──────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────
EARTH_SEARCH_URL = "https://earth-search.aws.element84.com/v1"
PLANETARY_URL    = "https://planetarycomputer.microsoft.com/api/stac/v1"

SEARCH_CLOUD_MAX = 80      # scene-level prefilter; per-pixel SCL does the real work
TRY_N            = 60
THREADS          = min(8, (os.cpu_count() or 4))
SCENE_WORKERS    = 8       # max scenes fetched concurrently (each uses THREADS band reads)
SCENE_INFLIGHT_MB = 400    # memory budget for scenes being fetched at once
STACK_CACHE_TTL  = 10 * 60 # VRA and SOC on the same field/window share one stack
STACK_CACHE_MAX  = 4
STACK_CACHE_MAX_MB = 256   # larger stacks are not cached
USE_SHADOW       = True

MAX_SCENES         = 20    # ceiling on scenes stacked per run
MIN_CLEAR_FRACTION = 0.55  # a scene is used only if >=55% of AOI is clear
MIN_OBS_PER_PIXEL  = 2     # pixels seen fewer times than this are dropped
STACK_MEMORY_MB    = 1200  # soft budget; scene count trimmed to fit

BARE_SOIL_FRACTION = 0.35  # adaptive bare/veg split point within NDVI range
MIN_SCENES_PER_GROUP = 2

SMOOTH_SIGMA      = 1.0
SMOOTH_MIN_WEIGHT = 0.50   # strict: stops values bleeding outside the AOI

NATIVE_RES_M     = 10.0
MAX_GRID_PX_HARD = 2000

DPI = 160

# ──────────────────────────────────────────────────────────────────────────
# AGRONOMIC TABLES
# ──────────────────────────────────────────────────────────────────────────
CROP_DEMAND = {
    "wheat":     {"N": 12.5, "P": 5.0, "K": 10.5},
    "rice":      {"N": 15.0, "P": 5.5, "K": 12.0},
    "maize":     {"N": 18.0, "P": 6.0, "K": 14.0},
    "soybean":   {"N":  8.0, "P": 8.0, "K": 10.0},
    "sugarcane": {"N": 22.0, "P": 9.0, "K": 20.0},
    "cotton":    {"N": 18.0, "P": 7.0, "K": 14.0},
    "onion":     {"N": 14.0, "P": 7.0, "K": 14.0},
    "potato":    {"N": 20.0, "P":10.0, "K": 22.0},
    "tomato":    {"N": 16.0, "P": 8.0, "K": 18.0},
    "banana":    {"N": 22.0, "P":10.0, "K": 28.0},
    "groundnut": {"N":  8.5, "P": 7.0, "K":  8.5},
    "jowar":     {"N": 17.0, "P": 5.5, "K": 13.0},
    "bajra":     {"N": 14.5, "P": 4.5, "K": 11.0},
    "chili":     {"N": 18.0, "P": 8.0, "K": 15.0},
    "turmeric":  {"N": 22.0, "P":10.0, "K": 24.0},
    "ginger":    {"N": 22.0, "P":10.0, "K": 24.0},
    "mustard":   {"N": 14.0, "P": 7.0, "K": 11.0},
    "lentil":    {"N":  7.5, "P": 7.5, "K":  9.0},
    "gram":      {"N":  7.0, "P": 7.0, "K":  9.0},
    "default":   {"N": 14.0, "P": 7.0, "K": 12.0},
}

FERTILISER_MAX_DOSE = {"N": 150, "P": 60, "K": 100}      # kg nutrient/ha, ICAR
FERTILISER_PRODUCTS = {
    "N": {"name": "Urea",          "nutrient_pct": 46.0},
    "P": {"name": "DAP (18-46-0)", "nutrient_pct": 46.0},
    "K": {"name": "MOP (0-0-60)",  "nutrient_pct": 60.0},
}

MAX_DOSE_FRACTION = 1.00
MIN_DOSE_FRACTION = 0.25


def zone_dose_fractions(n_zones):
    if n_zones <= 1:
        return {1: MAX_DOSE_FRACTION}
    return {i: round(MAX_DOSE_FRACTION -
                     (MAX_DOSE_FRACTION - MIN_DOSE_FRACTION) * (i - 1) / (n_zones - 1), 3)
            for i in range(1, n_zones + 1)}


def zone_labels_for(n_zones):
    presets = {
        2: {1: "Low", 2: "High"},
        3: {1: "Low", 2: "Medium", 3: "High"},
        4: {1: "Low", 2: "Medium-Low", 3: "Medium-High", 4: "High"},
        5: {1: "Very Low", 2: "Low", 3: "Medium", 4: "High", 5: "Very High"},
        6: {1: "Very Low", 2: "Low", 3: "Med-Low", 4: "Med-High", 5: "High", 6: "Very High"},
        7: {1: "Very Low", 2: "Low", 3: "Med-Low", 4: "Medium", 5: "Med-High",
            6: "High", 7: "Very High"},
    }
    return presets.get(n_zones, {i: f"Zone {i}" for i in range(1, n_zones + 1)})


def zone_colors_for(n_zones):
    ramp = LinearSegmentedColormap.from_list("zr", ["#b71c1c", "#f9a825", "#2e7d32"], N=256)
    labels = zone_labels_for(n_zones)
    return {labels[i]: mcolors.to_hex(ramp(0.0 if n_zones <= 1 else (i - 1) / (n_zones - 1)))
            for i in range(1, n_zones + 1)}


_HEAT_COLORS = ["#7f0000", "#c62828", "#ef5350", "#ff9800", "#ffd54f",
                "#dce775", "#9ccc65", "#4caf50", "#1b5e20"]
CMAP_HEAT = LinearSegmentedColormap.from_list("heat_rdylgn", _HEAT_COLORS, N=256)
CMAP_SOC = LinearSegmentedColormap.from_list(
    "soc", ["#3d1c00", "#7a3b00", "#c17f00", "#e8c840", "#9ecb3c", "#4caf50", "#1b5e20"], N=256)
CMAP_MOIST = LinearSegmentedColormap.from_list(
    "moist", ["#8d6e63", "#d7ccc8", "#b3e5fc", "#4fc3f7", "#0277bd"], N=256)


# ══════════════════════════════════════════════════════════════════════════
# PARAMETER REGISTRY
# Declares every output parameter, which composite it is derived from, and
# how confident we are in it. `confidence` is deliberately conservative and
# is printed on the map and in the report so nobody over-trusts a weak
# retrieval (pH in particular).
# ══════════════════════════════════════════════════════════════════════════
PARAMETERS = {
    # --- soil properties: retrieved from the BARE-SOIL composite ---
    "SOC":      {"label": "Soil Organic Carbon", "unit": "%",     "source": "bare",
                 "cmap": CMAP_SOC,   "confidence": "moderate", "higher_is_better": True},
    "CLAY":     {"label": "Clay Content",        "unit": "%",     "source": "bare",
                 "cmap": CMAP_HEAT,  "confidence": "moderate", "higher_is_better": True},
    "MOISTURE": {"label": "Soil Moisture Index", "unit": "index", "source": "bare",
                 "cmap": CMAP_MOIST, "confidence": "good",     "higher_is_better": True},
    "EC":       {"label": "Salinity (EC proxy)", "unit": "dS/m",  "source": "bare",
                 "cmap": CMAP_HEAT,  "confidence": "low",      "higher_is_better": False},
    "PH":       {"label": "Soil pH (indicative)", "unit": "pH",   "source": "bare",
                 "cmap": CMAP_HEAT,  "confidence": "very low", "higher_is_better": True},
    # --- nutrients: retrieved from the VEGETATION composite ---
    "N":        {"label": "Available Nitrogen",  "unit": "kg/ha", "source": "veg",
                 "cmap": CMAP_HEAT,  "confidence": "moderate", "higher_is_better": True},
    "P":        {"label": "Available Phosphorus","unit": "kg/ha", "source": "veg",
                 "cmap": CMAP_HEAT,  "confidence": "low",      "higher_is_better": True},
    "K":        {"label": "Available Potassium", "unit": "kg/ha", "source": "veg",
                 "cmap": CMAP_HEAT,  "confidence": "low",      "higher_is_better": True},
    # --- vigour: from the VEGETATION composite ---
    "VIGOUR":   {"label": "Crop Vigour (NDVI)",  "unit": "index", "source": "veg",
                 "cmap": CMAP_HEAT,  "confidence": "high",     "higher_is_better": True},
}

VRA_NUTRIENTS = ["N", "P", "K"]

# Calibration status ladder. Each map and JSON record carries one.
#   UNCALIBRATED   - literature constants, not tuned to these soils
#   RANGE-ANCHORED - absolute scale fitted to measured regional lab
#                    distributions; spatial pattern still unvalidated
#   CALIBRATED     - fitted to geolocated samples matched pixel-by-pixel
CALIB_UNCAL   = "UNCALIBRATED"
CALIB_ANCHOR  = "RANGE-ANCHORED"
CALIB_FULL    = "CALIBRATED"

CALIB_COLOR = {CALIB_UNCAL: "#c62828", CALIB_ANCHOR: "#ef6c00", CALIB_FULL: "#2e7d32"}

CONFIDENCE_NOTE = {
    "high":     "Direct optical measurement.",
    "good":     "Well-established spectral relationship.",
    "moderate": "Published proxy relationship; calibrate for absolute values.",
    "low":      "Weak proxy. Treat as RELATIVE pattern only.",
    "very low": "Not reliably retrievable from Sentinel-2. Pattern only, "
                "never quote as an absolute number.",
}




# ──────────────────────────────────────────────────────────────────────────
# AUTOMATIC REGION DETECTION
# ──────────────────────────────────────────────────────────────────────────
# Approximate bounding boxes for the regions the lab register covers, so a
# run can anchor itself from coordinates alone with no manual input.
# These are deliberately coarse: they only pick which measured
# distribution to use, and every result carries the region it resolved to
# so a wrong pick is visible rather than silent.
_STATE_BBOX = {
    # state: (lat_min, lat_max, lon_min, lon_max)
    "U.P.":         (23.8, 30.5, 77.0, 84.7),
    "PUNJAB":       (29.5, 32.6, 73.8, 76.9),
    "RAJASTHAN":    (23.0, 30.2, 69.4, 78.3),
    "GUJARAT":      (20.1, 24.7, 68.1, 74.5),
    "MAHARASHTRA":  (15.6, 22.1, 72.6, 80.9),
    "CHHATTISGARH": (17.7, 24.2, 80.2, 84.4),
    "U.K.":         (28.7, 31.5, 77.5, 81.1),
}

_DISTRICT_BBOX = {
    # tighter boxes for districts with their own measured distribution
    "WASHIM":      (19.9, 20.5, 76.7, 77.4),
    "BULDHANA":    (19.8, 21.1, 75.9, 76.9),
    "BARMER":      (24.7, 26.6, 70.2, 72.5),
    "HARIDWAR":    (29.6, 30.2, 77.7, 78.4),
    "MUKTSAR":     (30.1, 30.7, 74.2, 74.9),
    "SANGRUR":     (29.8, 30.7, 75.5, 76.4),
    "BILASPUR":    (21.7, 22.6, 81.6, 82.6),
    "RAIPUR":      (20.6, 21.6, 81.2, 82.2),
    "ETAH":        (27.3, 28.0, 78.4, 79.2),
    "FIROZABAD":   (26.9, 27.5, 78.1, 78.8),
    "HAPUR":       (28.5, 29.0, 77.6, 78.2),
    "HARDOI":      (26.8, 27.7, 79.6, 80.7),
    "HATHRAS":     (27.3, 27.9, 77.8, 78.4),
    "MAINPURI":    (26.9, 27.5, 78.7, 79.4),
    "SABARKANTHA": (23.0, 24.3, 72.6, 73.6),
}


def detect_region(lat, lon):
    """
    Resolve an AOI centroid to the most specific region that has measured
    lab data behind it.

    Returns a dict describing what was found and how confident the match
    is, never a bare string, so the caller can report the basis rather
    than presenting an anchored number as if it were universal.
    """
    hits_d = [d for d, (a, b, c, e) in _DISTRICT_BBOX.items()
              if a <= lat <= b and c <= lon <= e]
    hits_s = [s for s, (a, b, c, e) in _STATE_BBOX.items()
              if a <= lat <= b and c <= lon <= e]

    district = hits_d[0] if len(hits_d) == 1 else None
    state = hits_s[0] if len(hits_s) == 1 else (hits_s[0] if hits_s else None)

    region, n = region_support(district, state)
    if district:
        basis = "district bounding box"
    elif state and len(hits_s) == 1:
        basis = "state bounding box"
    elif state:
        basis = f"state bounding box (ambiguous, {len(hits_s)} overlap; took {state})"
    else:
        basis = "no measured region matches these coordinates"

    return {
        "lat": round(float(lat), 6), "lon": round(float(lon), 6),
        "district": district, "state": state,
        "resolved_region": region if (district or state) else None,
        "n_lab_records": n if (district or state) else 0,
        "basis": basis,
        "anchored": bool(district or state),
    }


def aoi_centroid_wgs84(aoi_geojson):
    """Area-weighted centroid of the AOI in WGS84, for region detection."""
    g = shape(aoi_geojson)
    c = g.centroid
    return float(c.y), float(c.x)


def soilgrids_prior(lat, lon, timeout=20):
    """
    Global fallback anchor from ISRIC SoilGrids (250 m, CC-BY 4.0), for
    AOIs outside the regions covered by local lab data.

    Requires network. ISRIC state that their REST API is a beta service
    with no uptime guarantee and it has been paused at times, so this
    function fails soft: on any error it returns None and the caller
    falls back to literature ranges and reports UNCALIBRATED rather than
    inventing a value.

    Returns {param: (low, high)} in the units this engine uses.
    """
    try:
        import urllib.request
        url = ("https://rest.isric.org/soilgrids/v2.0/properties/query"
               f"?lon={lon}&lat={lat}"
               "&property=soc&property=phh2o&property=clay&property=nitrogen"
               "&depth=0-5cm&depth=5-15cm&value=Q0.05&value=Q0.95")
        with urllib.request.urlopen(url, timeout=timeout) as r:
            data = json.loads(r.read().decode())
        out = {}
        for layer in data.get("properties", {}).get("layers", []):
            name = layer.get("name")
            for d in layer.get("depths", []):
                v = d.get("values", {})
                lo, hi = v.get("Q0.05"), v.get("Q0.95")
                if lo is None or hi is None:
                    continue
                # SoilGrids ships integers with documented conversion factors
                if name == "soc":        # dg/kg -> %
                    out["SOC"] = (lo / 100.0, hi / 100.0)
                elif name == "phh2o":    # pH*10
                    out["PH"] = (lo / 10.0, hi / 10.0)
                elif name == "clay":     # g/kg -> %
                    out["CLAY"] = (lo / 10.0, hi / 10.0)
                break                    # topmost depth only
        return out or None
    except Exception as exc:
        print(f"      [SoilGrids] unavailable ({exc}); falling back to "
              f"literature ranges")
        return None


# ──────────────────────────────────────────────────────────────────────────
# VALIDATION
# ──────────────────────────────────────────────────────────────────────────
def _validate_date(label, value):
    try:
        datetime.strptime(value, "%Y-%m-%d")
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{label}={value!r} invalid. Expected YYYY-MM-DD. ({exc})")
    return value


def _validate_inputs(aoi_geojson, start_date, end_date, crop, n_zones):
    if not isinstance(aoi_geojson, dict) or "type" not in aoi_geojson:
        raise ValueError("aoi_geojson must be a GeoJSON geometry dict.")
    geom = shape(aoi_geojson)
    if not geom.is_valid:
        raise ValueError("aoi_geojson is not a valid geometry (self-intersecting?).")
    if geom.area == 0:
        raise ValueError("aoi_geojson has zero area.")
    _validate_date("start_date", start_date)
    _validate_date("end_date", end_date)
    d0 = datetime.strptime(start_date, "%Y-%m-%d")
    d1 = datetime.strptime(end_date, "%Y-%m-%d")
    if d0 >= d1:
        raise ValueError(f"start_date ({start_date}) must be before end_date ({end_date}).")
    if (d1 - d0).days < 20:
        print(f"  [warn] window is only {(d1-d0).days} days. Sentinel-2 revisits every "
              f"~5 days, so few scenes will be available and the time-series "
              f"advantage shrinks. 60-120 days is recommended.")
    if crop not in CROP_DEMAND:
        print(f"  [warn] crop '{crop}' not in table — using 'default' demand.")
    if not (2 <= n_zones <= 7):
        raise ValueError(f"n_zones must be 2..7, got {n_zones}.")


# ──────────────────────────────────────────────────────────────────────────
# STAC HELPERS
# ──────────────────────────────────────────────────────────────────────────
def _s3_to_https(href):
    if href.startswith("s3://sentinel-cogs/"):
        return href.replace("s3://sentinel-cogs/", "https://sentinel-cogs.s3.amazonaws.com/")
    if href.startswith("s3://"):
        bucket, key = href[5:].split("/", 1) if "/" in href[5:] else (href[5:], "")
        return f"https://{bucket}.s3.amazonaws.com/{key}"
    return href


def _prefer_https(asset):
    if asset is None:
        return None
    href = getattr(asset, "href", "") or ""
    alt = getattr(asset, "extra_fields", {}).get("alternate", {})
    for k in ("https", "http", "self"):
        v = alt.get(k)
        if isinstance(v, dict) and v.get("href", "").startswith("http"):
            return v["href"]
        if isinstance(v, str) and v.startswith("http"):
            return v
    return href if href.startswith("http") else (_s3_to_https(href) if href else None)


def _sign(url):
    """Planetary Computer blobs need a SAS token; other URLs pass through."""
    if url and planetary_computer is not None and "blob.core.windows.net" in url:
        try:
            return planetary_computer.sign(url)
        except Exception:
            return url
    return url


def _pick_url(assets, *keys):
    for k in keys:
        a = assets.get(k)
        if a:
            url = _prefer_https(a)
            if url:
                return _sign(url)
    return None


def _item_crs(item):
    """Scene CRS from STAC metadata, so no COG has to be opened just for it."""
    props = item.properties or {}
    code = props.get("proj:code") or (f"EPSG:{props['proj:epsg']}" if props.get("proj:epsg") else None)
    if code:
        try:
            return CRS.from_user_input(code)
        except Exception:
            pass
    return None


def _tile_id(props):
    # Planetary: s2:mgrs_tile="43QDA"; Earth Search v1: grid:code="MGRS-43QDA"
    tile = (props.get("s2:mgrs_tile") or props.get("grid:code")
            or props.get("sentinel:grid_square") or "")
    return str(tile).replace("MGRS-", "")


def _aoi_scene(aoi_ll, crs_str):
    t = Transformer.from_crs("EPSG:4326", crs_str, always_xy=True)
    return shp_transform(lambda x, y, z=None: t.transform(x, y), shape(aoi_ll))


def _stac_search(catalog, aoi, start, end, max_cloud, n):
    try:
        cat = Client.open(catalog)
        return list(cat.search(
            collections=["sentinel-2-l2a"], intersects=aoi,
            datetime=f"{start}/{end}",
            sortby=[{"field": "properties.datetime", "direction": "desc"}],
            query={"eo:cloud_cover": {"lt": max_cloud}}, limit=n).items())
    except Exception as exc:
        print(f"  [STAC] {catalog} -> {exc}")
        return []


def find_all_scenes(aoi, start, end, max_scenes=MAX_SCENES):
    """
    Every usable scene in the window, deduplicated across both catalogs.

    The old find_best_scene() returned items[0] of a datetime-sorted search:
    it analysed one day, ignored the rest of the window, and despite its name
    never compared cloud cover at all.
    """
    seen, scenes = set(), []
    with ThreadPoolExecutor(max_workers=2) as ex:
        results = list(ex.map(
            lambda url: _stac_search(url, aoi, start, end, SEARCH_CLOUD_MAX, TRY_N),
            (EARTH_SEARCH_URL, PLANETARY_URL)))
    for found in results:
        for it in found:
            dt = (it.properties.get("datetime") or "")[:10]
            key = (dt, _tile_id(it.properties))
            if key in seen:
                continue
            seen.add(key)
            scenes.append({"item": it, "date": dt,
                            "cloud": float(it.properties.get("eo:cloud_cover", 100.0))})
    # least cloudy first, so the memory/scene cap keeps the best scenes
    scenes.sort(key=lambda s: s["cloud"])

    if not scenes:
        fmt = "%Y-%m-%d"
        s2 = (datetime.strptime(start, fmt) - timedelta(days=30)).strftime(fmt)
        e2 = (datetime.strptime(end, fmt) + timedelta(days=30)).strftime(fmt)
        print("  [scenes] none found — retrying with +/-30 day margin")
        seen2 = set()
        for cat_url in (EARTH_SEARCH_URL, PLANETARY_URL):
            for it in _stac_search(cat_url, aoi, s2, e2, SEARCH_CLOUD_MAX, TRY_N):
                dt = (it.properties.get("datetime") or "")[:10]
                if dt in seen2:
                    continue
                seen2.add(dt)
                scenes.append({"item": it, "date": dt,
                                "cloud": float(it.properties.get("eo:cloud_cover", 100.0))})
        scenes.sort(key=lambda s: s["cloud"])

    return scenes[:max_scenes]


# ──────────────────────────────────────────────────────────────────────────
# GRID
# ──────────────────────────────────────────────────────────────────────────
def build_grid(crs, aoi_ll, native_m=NATIVE_RES_M):
    aoi_sc = _aoi_scene(aoi_ll, crs.to_string())
    minx, miny, maxx, maxy = aoi_sc.bounds
    dx, dy = max(maxx - minx, 1e-6), max(maxy - miny, 1e-6)
    res = native_m
    if max(dx, dy) / native_m > MAX_GRID_PX_HARD:
        res = max(dx, dy) / MAX_GRID_PX_HARD
        print(f"  [grid] very large AOI — relaxing {native_m:.1f} m -> {res:.1f} m")
    W = max(1, int(math.ceil(dx / res)))
    H = max(1, int(math.ceil(dy / res)))
    tf = Affine.translation(minx, maxy) * Affine.scale(res, -res)
    return aoi_sc, tf, H, W, res


# ──────────────────────────────────────────────────────────────────────────
# BAND READING
# ──────────────────────────────────────────────────────────────────────────
_BAND_KEYS = {
    "B02": ("blue",     "B02"),
    "B03": ("green",    "B03"),
    "B04": ("red",      "B04"),
    "B05": ("rededge1", "B05"),
    "B08": ("nir",      "B08"),
    "B8A": ("nir08",    "B8A"),
    "B11": ("swir16",   "B11"),
    "B12": ("swir22",   "B12"),
}
_CORE_BANDS = ["B02", "B03", "B04", "B08", "B11", "B12"]   # must-have
_OPT_BANDS  = ["B05", "B8A"]                                # red-edge, optional


def _read_band(src, geom_sc, H, W, dst_tf, resamp=Resampling.bilinear):
    win = from_bounds(*geom_sc.bounds, src.transform).round_offsets().round_lengths()
    win = win.intersection(Window(0, 0, src.width, src.height)).round_offsets().round_lengths()
    if win.width <= 0 or win.height <= 0:
        return np.full((H, W), np.nan, "float32")
    arr = src.read(1, window=win, masked=True).filled(0).astype("float32")
    src_tf = win_transform(win, src.transform)
    dst = np.full((H, W), np.nan, "float32")
    reproject(arr, dst, src_transform=src_tf, src_crs=src.crs,
              dst_transform=dst_tf, dst_crs=src.crs,
              src_nodata=0.0, dst_nodata=np.nan, resampling=resamp)
    return dst


def _read_scl(src, geom_sc, H, W, dst_tf):
    win = from_bounds(*geom_sc.bounds, src.transform).round_offsets().round_lengths()
    win = win.intersection(Window(0, 0, src.width, src.height)).round_offsets().round_lengths()
    if win.width <= 0 or win.height <= 0:
        return np.zeros((H, W), "int16")
    arr = src.read(1, window=win, masked=True).filled(0).astype("int16")
    src_tf = win_transform(win, src.transform)
    dst = np.zeros((H, W), "int16")
    reproject(arr, dst, src_transform=src_tf, src_crs=src.crs,
              dst_transform=dst_tf, dst_crs=src.crs,
              src_nodata=0, dst_nodata=0, resampling=Resampling.nearest)
    return dst


def fetch_scene_bands(item, aoi, dst_tf, H, W):
    """One scene -> dict of band arrays + SCL. Missing optional bands are
    returned as None rather than failing the scene."""
    assets = item.assets
    urls = {}
    for band in _CORE_BANDS:
        url = _pick_url(assets, *_BAND_KEYS[band])
        if not url:
            return None
        urls[band] = url
    for band in _OPT_BANDS:
        url = _pick_url(assets, *_BAND_KEYS[band])
        if url:
            urls[band] = url

    scl_ref = assets.get("scl") or assets.get("SCL")
    scl_url = _sign(_prefer_https(scl_ref)) if scl_ref else None

    try:
        crs = _item_crs(item)
        if crs is None:
            with rasterio.open(urls["B04"]) as ref:
                crs = ref.crs
        geom_sc = _aoi_scene(aoi, crs.to_string())

        def _load(pair):
            band, url = pair
            with rasterio.open(url) as ds:
                arr = _read_band(ds, geom_sc, H, W, dst_tf)
            fin = arr[np.isfinite(arr)]
            if fin.size > 0 and fin.max() > 2.0:
                arr /= 10000.0
            return band, arr

        def _load_scl():
            with rasterio.open(scl_url) as ds:
                return _read_scl(ds, geom_sc, H, W, dst_tf)

        out = {}
        with ThreadPoolExecutor(max_workers=THREADS + 1) as ex:
            scl_fut = ex.submit(_load_scl) if scl_url else None
            for f in as_completed([ex.submit(_load, p) for p in urls.items()]):
                band, arr = f.result()
                out[band] = arr
            out["SCL"] = scl_fut.result() if scl_fut else None
        for band in _OPT_BANDS:
            out.setdefault(band, None)
        return out
    except Exception as exc:
        print(f"      fetch error: {exc}")
        return None


# ──────────────────────────────────────────────────────────────────────────
# CLOUD MASKING
# ──────────────────────────────────────────────────────────────────────────
_SCL_BAD    = [8, 9, 10, 11]   # cloud med/high prob, cirrus, snow
_SCL_SHADOW = [3]
_SCL_NODATA = [0, 1]


def scene_clear_mask(bands, aoi_mask):
    """Usable-pixel mask for one scene, from SCL plus finite core bands."""
    clear = np.ones(aoi_mask.shape, bool)
    for b in _CORE_BANDS:
        clear &= np.isfinite(bands[b])
    scl = bands.get("SCL")
    if scl is not None:
        bad = np.isin(scl, _SCL_BAD + _SCL_NODATA + (_SCL_SHADOW if USE_SHADOW else []))
        clear &= ~bad
    return clear & aoi_mask


# ──────────────────────────────────────────────────────────────────────────
# SPECTRAL INDICES  (13 indices, all literature-referenced)
# ──────────────────────────────────────────────────────────────────────────
def _sdiv(a, b):
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(np.abs(b) > 1e-9, a / b, np.nan).astype("float32")


def compute_indices(bands):
    """
    NDVI   Rouse 1973        vegetation vigour
    EVI    Huete 2002        vigour, resistant to soil/atmosphere
    SAVI   Huete 1988        vigour, soil-adjusted
    MSAVI  Qi 1994           vigour, best at low cover
    NDRE   Barnes 2000       chlorophyll / nitrogen (needs B05)
    RECI   Gitelson 2003     chlorophyll / nitrogen (needs B05)
    NDMI   Gao 1996          canopy + surface moisture
    NDWI   McFeeters 1996    open water / waterlogging
    BSI    Rikimaru 2002     bare soil exposure
    SOCI   Thaler 2019       soil organic carbon index
    CLAY   Drury 1987        clay mineral ratio
    BI     Escadafal 1989    surface brightness
    SI     Khan 2005         salinity index
    """
    B02, B03, B04 = bands["B02"], bands["B03"], bands["B04"]
    B08, B11, B12 = bands["B08"], bands["B11"], bands["B12"]
    B05 = bands.get("B05")

    idx = {}
    idx["NDVI"] = _sdiv(B08 - B04, B08 + B04)
    with np.errstate(all="ignore"):
        idx["EVI"] = np.clip(
            2.5 * (B08 - B04) / (B08 + 6.0 * B04 - 7.5 * B02 + 1.0), -1, 1).astype("float32")
    idx["SAVI"] = (1.5 * _sdiv(B08 - B04, B08 + B04 + 0.5)).astype("float32")
    with np.errstate(all="ignore"):
        t = 2.0 * B08 + 1.0
        idx["MSAVI"] = ((t - np.sqrt(np.maximum(t ** 2 - 8.0 * (B08 - B04), 0))) / 2.0
                        ).astype("float32")
    if B05 is not None:
        idx["NDRE"] = _sdiv(B08 - B05, B08 + B05)
        idx["RECI"] = (_sdiv(B08, B05) - 1.0).astype("float32")
    else:
        # red-edge unavailable: fall back to green-based chlorophyll proxy
        idx["NDRE"] = _sdiv(B08 - B03, B08 + B03)
        idx["RECI"] = (_sdiv(B08, B03) - 1.0).astype("float32")
    idx["NDMI"] = _sdiv(B08 - B11, B08 + B11)
    idx["NDWI"] = _sdiv(B03 - B08, B03 + B08)
    idx["BSI"]  = _sdiv((B11 + B04) - (B08 + B02), (B11 + B04) + (B08 + B02))
    idx["SOCI"] = _sdiv(B02, np.maximum(B03 * B04, 1e-6))
    idx["CLAY"] = _sdiv(B11, B12)
    with np.errstate(all="ignore"):
        idx["BI"] = np.sqrt(np.maximum((B04 ** 2 + B03 ** 2) / 2.0, 0)).astype("float32")
        idx["SI"] = np.sqrt(np.maximum(B04 * B03, 0)).astype("float32")
    return idx


INDEX_NAMES = ["NDVI", "EVI", "SAVI", "MSAVI", "NDRE", "RECI",
               "NDMI", "NDWI", "BSI", "SOCI", "CLAY", "BI", "SI"]


# ──────────────────────────────────────────────────────────────────────────
# TIME-SERIES STACK
# ──────────────────────────────────────────────────────────────────────────
def _max_scenes_for_memory(H, W, requested):
    per_scene_mb = (H * W * len(INDEX_NAMES) * 4) / (1024 ** 2)
    if per_scene_mb <= 0:
        return requested
    affordable = max(1, int(STACK_MEMORY_MB / per_scene_mb))
    if affordable < requested:
        print(f"  [stack] memory guard: {per_scene_mb:.1f} MB/scene -> "
              f"capping {requested} scenes to {affordable}")
    return min(requested, affordable)


_STACK_CACHE = {}
_STACK_CACHE_LOCK = threading.Lock()


def _copy_stack_result(res):
    stack, used, obs, raw_bands = res
    return ({k: v.copy() for k, v in stack.items()},
            [dict(u) for u in used],
            obs.copy(),
            [{b: a.copy() for b, a in rb.items()} for rb in raw_bands])


def build_index_timeseries(scenes, aoi, dst_tf, H, W, aoi_mask,
                            min_clear=MIN_CLEAR_FRACTION):
    """Cached wrapper around _build_index_timeseries (scene fetch dominates runtime)."""
    key = json.dumps([[sc["item"].id for sc in scenes], aoi, list(dst_tf)[:6],
                      H, W, min_clear], sort_keys=True, default=str)
    now = time.time()
    with _STACK_CACHE_LOCK:
        hit = _STACK_CACHE.get(key)
    if hit and now - hit[0] < STACK_CACHE_TTL:
        print("      reusing cached scene stack")
        return _copy_stack_result(hit[1])

    res = _build_index_timeseries(scenes, aoi, dst_tf, H, W, aoi_mask, min_clear)
    stack, _, obs, raw_bands = res
    if stack is not None:
        nbytes = (sum(v.nbytes for v in stack.values()) + obs.nbytes
                  + sum(a.nbytes for rb in raw_bands for a in rb.values()))
        if nbytes <= STACK_CACHE_MAX_MB * 1024 ** 2:
            with _STACK_CACHE_LOCK:
                for k in [k for k, (t, _) in _STACK_CACHE.items() if now - t >= STACK_CACHE_TTL]:
                    _STACK_CACHE.pop(k, None)
                while len(_STACK_CACHE) >= STACK_CACHE_MAX:
                    _STACK_CACHE.pop(next(iter(_STACK_CACHE)))
                _STACK_CACHE[key] = (now, _copy_stack_result(res))
    return res


def _build_index_timeseries(scenes, aoi, dst_tf, H, W, aoi_mask,
                             min_clear=MIN_CLEAR_FRACTION):
    """
    Fetch each scene, mask it with its own SCL, keep the clear-enough ones,
    and return a per-index time series stack.

    Returns (stack, used, obs_count) where
      stack     : index name -> (S,H,W) float32, NaN where that scene was cloudy
      used      : list of per-scene metadata dicts
      obs_count : (H,W) int16, clear looks per pixel
    """
    aoi_px = max(int(aoi_mask.sum()), 1)
    cap = _max_scenes_for_memory(H, W, len(scenes))
    scenes = scenes[:cap]

    per_scene_idx, used, raw_bands = [], [], []
    # Fetch scenes concurrently but keep at most SCENE_WORKERS in flight and
    # consume them in order, so memory stays bounded and output is deterministic.
    per_scene_mb = H * W * 4 * (len(_CORE_BANDS) + len(_OPT_BANDS) + 1) / (1024 ** 2)
    workers = max(2, min(SCENE_WORKERS, int(SCENE_INFLIGHT_MB / max(per_scene_mb, 1e-6))))
    ex = ThreadPoolExecutor(max_workers=workers)
    pending = {}

    def _fetch(j):
        return ex.submit(fetch_scene_bands, scenes[j]["item"], aoi, dst_tf, H, W)

    for j in range(min(workers, len(scenes))):
        pending[j] = _fetch(j)
    for i, sc in enumerate(scenes, 1):
        bands = pending.pop(i - 1).result()
        nxt = i - 1 + workers
        if nxt < len(scenes):
            pending[nxt] = _fetch(nxt)
        if bands is None:
            print(f"    [{i:2d}/{len(scenes)}] {sc['date']}  fetch failed — skipped")
            continue
        clear = scene_clear_mask(bands, aoi_mask)
        frac = float(clear.sum()) / aoi_px
        if frac < min_clear:
            print(f"    [{i:2d}/{len(scenes)}] {sc['date']}  clear={frac*100:5.1f}%  rejected")
            continue
        idx = compute_indices(bands)
        for k in idx:
            idx[k] = np.where(clear, idx[k], np.nan).astype("float32")
        per_scene_idx.append(idx)
        # keep cloud-masked raw reflectance: the published SOC method works
        # on reflectance spectra, not on indices
        raw_bands.append({b: np.where(clear, bands[b], np.nan).astype("float32")
                          for b in SOIL_BANDS})
        used.append({"date": sc["date"], "scene_cloud_pct": round(sc["cloud"], 2),
                      "clear_fraction": round(frac, 4)})
        print(f"    [{i:2d}/{len(scenes)}] {sc['date']}  clear={frac*100:5.1f}%  accepted")
    ex.shutdown(wait=True)

    if not per_scene_idx:
        return None, [], None, []

    stack = {k: np.stack([s[k] for s in per_scene_idx], axis=0) for k in INDEX_NAMES}
    obs = np.isfinite(stack["NDVI"]).sum(axis=0).astype("int16")
    return stack, used, obs, raw_bands


def split_bare_vegetation(stack, aoi_mask, bare_frac=BARE_SOIL_FRACTION,
                           min_each=MIN_SCENES_PER_GROUP):
    """
    Adaptive split of scenes into bare-soil and vegetation groups.

    A fixed NDVI cut-off (e.g. "<0.25 is bare") breaks across crops, regions
    and seasons, so the threshold is placed relative to THIS field's own
    observed NDVI range. Both groups are guaranteed non-empty via a
    rank-based fallback.
    """
    ndvi = stack["NDVI"]
    S = ndvi.shape[0]
    per_scene = np.array([float(np.nanmean(np.where(aoi_mask, ndvi[s], np.nan)))
                          for s in range(S)])
    finite = per_scene[np.isfinite(per_scene)]
    if finite.size == 0:
        allidx = np.arange(S)
        return allidx, allidx, float("nan"), per_scene
    lo, hi = float(finite.min()), float(finite.max())
    thr = lo + bare_frac * (hi - lo)
    safe = np.where(np.isfinite(per_scene), per_scene, np.inf)
    bare = np.where(safe <= thr)[0]
    veg  = np.where(safe > thr)[0]
    order = np.argsort(safe)
    if len(bare) < min_each:
        bare = order[:min(min_each, S)]
    if len(veg) < min_each:
        veg = order[-min(min_each, S):]
    return np.sort(bare), np.sort(veg), thr, per_scene


def temporal_composite(stack, scene_idx, index_names=None):
    """
    Per-pixel temporal MEDIAN over the selected scenes.

    Median rather than mean: validated to cut band RMSE ~3.6x versus a
    single scene and to stay robust when SCL misses thin haze on a minority
    of scenes. Mean/std/slope are computed separately as secondary features.
    """
    names = index_names or INDEX_NAMES
    out = {}
    with np.errstate(all="ignore"):
        for k in names:
            sel = stack[k][scene_idx]
            out[k] = np.nanmedian(sel, axis=0).astype("float32")
    return out


def temporal_features(stack, scene_idx, index_names=None):
    """
    Secondary temporal statistics.

    NOTE: min/max/amplitude are extreme-value statistics that noise inflates
    (validated: amplitude correlated 0.873 to the true driver versus 0.977
    for the mean), so they are exposed for diagnostics but are NOT used as
    primary zoning features.
    """
    names = index_names or INDEX_NAMES
    feats = {}
    with np.errstate(all="ignore"):
        for k in names:
            sel = stack[k][scene_idx]
            feats[f"{k}_mean"] = np.nanmean(sel, axis=0).astype("float32")
            feats[f"{k}_std"]  = np.nanstd(sel, axis=0).astype("float32")
    return feats


def correlation_matrix(feature_maps, aoi_mask, max_px=200000):
    """Pearson correlation between feature maps over valid AOI pixels.
    Reveals redundancy (e.g. NDVI vs BSI runs about -0.84), which is why
    zoning runs on PCA components rather than raw stacked indices."""
    names = list(feature_maps)
    cols = []
    for n in names:
        a = feature_maps[n]
        cols.append(np.where(aoi_mask, a, np.nan).ravel())
    M = np.column_stack(cols)
    ok = np.all(np.isfinite(M), axis=1)
    M = M[ok]
    if M.shape[0] > max_px:
        step = M.shape[0] // max_px + 1
        M = M[::step]
    if M.shape[0] < 10:
        return names, np.full((len(names), len(names)), np.nan)
    sd = M.std(axis=0)
    keep = sd > 1e-12
    C = np.full((len(names), len(names)), np.nan)
    if keep.sum() >= 2:
        sub = np.corrcoef(M[:, keep], rowvar=False)
        ki = np.where(keep)[0]
        for a, ia in enumerate(ki):
            for b, ib in enumerate(ki):
                C[ia, ib] = sub[a, b]
    return names, C


# ──────────────────────────────────────────────────────────────────────────
# SOIL / NUTRIENT RETRIEVAL
# ──────────────────────────────────────────────────────────────────────────
def _norm01(a, mask, lo_pct=2, hi_pct=98):
    """Scale to 0..1 using robust percentiles of the valid pixels."""
    v = a[mask & np.isfinite(a)]
    if v.size < 5:
        return np.full_like(a, np.nan, dtype="float32")
    lo, hi = np.percentile(v, [lo_pct, hi_pct])
    if hi - lo < 1e-12:
        return np.where(mask & np.isfinite(a), 0.5, np.nan).astype("float32")
    return np.clip((a - lo) / (hi - lo), 0, 1).astype("float32")


# Fallback ranges used only when no regional prior is available.
# These are literature defaults and are known to be wrong for Indian
# soils - the measured register showed 90% of samples below 1.0% OC
# against a 0.2-3.0 default. They exist so the pipeline still runs, and
# any parameter using them is reported as UNCALIBRATED.
_FALLBACK_RANGE = {
    "SOC": (0.20, 3.00), "CLAY": (10.0, 60.0), "EC": (0.10, 4.00),
    "PH": (5.70, 8.30), "N": (50.0, 500.0), "P": (10.0, 80.0), "K": (50.0, 350.0),
}


def _anchor(rel01, param, district=None, state=None, override=None):
    """
    Map a 0-1 relative index into physical units.

    Uses the measured p5-p95 range for the region when a prior exists, so
    the absolute scale matches soils that were actually tested nearby.
    Returns (array, status) so the caller can stamp each map honestly.
    """
    rng = None
    if override and param in override:
        rng = override[param]                 # e.g. SoilGrids, outside India
    if rng is None:
        rng = prior_range(param, district, state)
    if rng is not None:
        lo, hi = rng
        status = CALIB_ANCHOR
    else:
        lo, hi = _FALLBACK_RANGE.get(param, (0.0, 1.0))
        status = CALIB_UNCAL
    if hi <= lo:
        hi = lo + 1e-6
    return (lo + (hi - lo) * rel01).astype("float32"), status


def retrieve_parameters(bare_comp, veg_comp, aoi_mask, district=None, state=None,
                         override=None):
    """
    Derive every registered parameter, each from the composite that
    physically carries its signal.

    Soil properties come from the BARE-SOIL composite because a canopy
    obscures the soil surface. Nutrient and vigour parameters come from
    the VEGETATION composite, where the crop's response carries the
    signal.

    district/state anchor the output scale to measured lab distributions
    for that region (see soil_priors.py). Without them the pipeline falls
    back to literature ranges and says so.

    Returns (params, relative, status) - status maps each parameter to
    its calibration level.
    """
    P, status = {}, {}

    # ---- soil, from bare composite ----
    soci_n = _norm01(bare_comp["SOCI"], aoi_mask)
    bsi_n  = _norm01(bare_comp["BSI"],  aoi_mask)
    bi_n   = _norm01(bare_comp["BI"],   aoi_mask)
    # darker, less bright, lower-BSI bare surfaces indicate more organic carbon
    soc_rel = np.clip(0.55 * (1.0 - bi_n) + 0.25 * (1.0 - bsi_n) + 0.20 * soci_n, 0, 1)
    P["SOC"], status["SOC"] = _anchor(soc_rel, "SOC", district, state, override)

    clay_n = _norm01(bare_comp["CLAY"], aoi_mask)
    P["CLAY"], status["CLAY"] = _anchor(clay_n, "CLAY", district, state, override)

    P["MOISTURE"] = _norm01(bare_comp["NDMI"], aoi_mask).astype("float32")
    status["MOISTURE"] = CALIB_UNCAL   # an index, no absolute scale claimed

    si_n = _norm01(bare_comp["SI"], aoi_mask)
    P["EC"], status["EC"] = _anchor(si_n, "EC", district, state, override)

    # pH has no reliable Sentinel-2 signal. Kept because it was requested,
    # anchored so at least the range is regionally sane, flagged 'very low'.
    P["PH"], status["PH"] = _anchor(bi_n, "PH", district, state, override)

    # ---- vigour and nutrients, from vegetation composite ----
    ndvi_n = _norm01(veg_comp["NDVI"], aoi_mask)
    ndre_n = _norm01(veg_comp["NDRE"], aoi_mask)
    reci_n = _norm01(veg_comp["RECI"], aoi_mask)
    savi_n = _norm01(veg_comp["SAVI"], aoi_mask)
    evi_n  = _norm01(veg_comp["EVI"],  aoi_mask)
    ndmi_v = _norm01(veg_comp["NDMI"], aoi_mask)

    P["VIGOUR"] = veg_comp["NDVI"].astype("float32")
    status["VIGOUR"] = CALIB_FULL      # NDVI is a direct optical measurement

    # N tracks canopy chlorophyll; red-edge indices carry most of that signal
    n_rel = np.clip(0.45 * ndre_n + 0.30 * reci_n + 0.25 * ndvi_n, 0, 1)
    P["N"], status["N"] = _anchor(n_rel, "N", district, state, override)

    p_rel = np.clip(0.60 * savi_n + 0.40 * evi_n, 0, 1)
    P["P"], status["P"] = _anchor(p_rel, "P", district, state, override)

    # K associates with clay (the dominant K reservoir) plus moisture status
    k_rel = np.clip(0.60 * clay_n + 0.40 * ndmi_v, 0, 1)
    P["K"], status["K"] = _anchor(k_rel, "K", district, state, override)

    relative = {}
    for k, v in P.items():
        r = _norm01(v, aoi_mask) * 100.0
        relative[k] = (r if PARAMETERS[k]["higher_is_better"] else (100.0 - r)).astype("float32")

    for k in P:
        P[k] = np.where(aoi_mask, P[k], np.nan).astype("float32")
        relative[k] = np.where(aoi_mask, relative[k], np.nan).astype("float32")
    return P, relative, status


# ──────────────────────────────────────────────────────────────────────────
# CALIBRATION  —  the only route to defensible absolute values
# ──────────────────────────────────────────────────────────────────────────
def calibrate_parameters(params, ground_samples, aoi_sc, transform, aoi_mask,
                          crs_wgs_to_scene=None, base_status=None):
    """
    Fit y = a*proxy + b per parameter against user-supplied lab samples.

    ground_samples: [{"lat":.., "lon":.., "SOC":1.2, "N":180, ...}, ...]

    Returns (calibrated_params, report). Parameters with fewer than 5 usable
    samples are left uncalibrated and clearly marked as such. Leave-one-out
    RMSE is reported alongside fit R2, because R2 on a handful of points
    flatters badly and would otherwise oversell the result.
    """
    report = {}
    base_status = base_status or {}
    out = {k: v.copy() for k, v in params.items()}

    def _note_for(st):
        if st == CALIB_ANCHOR:
            return ("Absolute scale anchored to measured lab distributions for this "
                    "region. The spatial pattern within the field is NOT validated - "
                    "supply geolocated samples to reach CALIBRATED.")
        if st == CALIB_FULL:
            return "Direct optical measurement."
        return ("Literature constants, not tuned to these soils. Read as a relative "
                "pattern only.")

    if not ground_samples:
        for k in params:
            st = base_status.get(k, CALIB_UNCAL)
            report[k] = {"status": st, "note": _note_for(st)}
        return out, report

    H, W = aoi_mask.shape
    a_, e_, c_, f_ = transform.a, transform.e, transform.c, transform.f

    rows_cols = []
    for s in ground_samples:
        if crs_wgs_to_scene is not None:
            x, y = crs_wgs_to_scene.transform(s["lon"], s["lat"])
        else:
            x, y = s["lon"], s["lat"]
        col = int((x - c_) / a_)
        row = int((y - f_) / e_)
        rows_cols.append((row, col))

    for key in params:
        xs, ys = [], []
        for (row, col), s in zip(rows_cols, ground_samples):
            if key not in s:
                continue
            if not (0 <= row < H and 0 <= col < W) or not aoi_mask[row, col]:
                continue
            proxy = params[key][row, col]
            if not np.isfinite(proxy):
                continue
            xs.append(float(proxy))
            ys.append(float(s[key]))
        n = len(xs)
        if n < 5:
            st = base_status.get(key, CALIB_UNCAL)
            report[key] = {"status": st, "n_samples": n,
                            "note": f"Only {n} usable samples (need >=5), so the "
                                    f"status stays {st}. " + _note_for(st)}
            continue

        xs_a, ys_a = np.array(xs), np.array(ys)
        if xs_a.std() < 1e-9:
            st = base_status.get(key, CALIB_UNCAL)
            report[key] = {"status": st, "n_samples": n,
                            "note": "Proxy has no spatial variation at the sample "
                                    "points; cannot fit. " + _note_for(st)}
            continue

        A = np.column_stack([xs_a, np.ones_like(xs_a)])
        coef, *_ = np.linalg.lstsq(A, ys_a, rcond=None)
        pred = coef[0] * xs_a + coef[1]
        ss_res = float(((ys_a - pred) ** 2).sum())
        ss_tot = float(((ys_a - ys_a.mean()) ** 2).sum())
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan")
        rmse = float(np.sqrt(ss_res / n))

        # leave-one-out: the honest error estimate on small sample counts
        loo_err = []
        for i in range(n):
            m = np.ones(n, bool)
            m[i] = False
            Ai = np.column_stack([xs_a[m], np.ones(m.sum())])
            ci, *_ = np.linalg.lstsq(Ai, ys_a[m], rcond=None)
            loo_err.append(ys_a[i] - (ci[0] * xs_a[i] + ci[1]))
        loo_rmse = float(np.sqrt(np.mean(np.array(loo_err) ** 2)))

        out[key] = (coef[0] * params[key] + coef[1]).astype("float32")
        report[key] = {"status": CALIB_FULL, "n_samples": n,
                        "slope": round(float(coef[0]), 5),
                        "intercept": round(float(coef[1]), 5),
                        "r2": round(r2, 4), "rmse": round(rmse, 4),
                        "loo_rmse": round(loo_rmse, 4),
                        "note": "Absolute values fitted to local lab samples."}
    for k in params:
        if k not in report:
            st = base_status.get(k, CALIB_UNCAL)
            report[k] = {"status": st, "note": _note_for(st)}
    return out, report


# ══════════════════════════════════════════════════════════════════════════
# ZONING
# ══════════════════════════════════════════════════════════════════════════
def _kmeans(X, k, iters=100, seed=0, n_init=5):
    """k-means++ style init, best of n_init runs by inertia."""
    rng = np.random.default_rng(seed)
    best_lab, best_inertia, best_C = None, np.inf, None
    for run in range(n_init):
        C = [X[rng.integers(len(X))]]
        for _ in range(1, k):
            d = np.min(((X[:, None, :] - np.array(C)[None, :, :]) ** 2).sum(-1), axis=1)
            tot = d.sum()
            probs = d / tot if tot > 1e-12 else np.full(len(X), 1.0 / len(X))
            C.append(X[rng.choice(len(X), p=probs)])
        C = np.array(C)
        lab = np.zeros(len(X), int)
        for _ in range(iters):
            d = ((X[:, None, :] - C[None, :, :]) ** 2).sum(-1)
            lab = d.argmin(1)
            newC = np.array([X[lab == i].mean(0) if np.any(lab == i) else C[i]
                             for i in range(k)])
            if np.allclose(newC, C, atol=1e-6):
                C = newC
                break
            C = newC
        inertia = float(((X - C[lab]) ** 2).sum())
        if inertia < best_inertia:
            best_lab, best_inertia, best_C = lab, inertia, C
    return best_lab, best_C, best_inertia


def multivariate_zones(params, aoi_mask, n_zones=5, feature_keys=None,
                        variance_target=0.90, seed=0):
    """
    Management zones from the JOINT signature of several parameters, via
    standardisation -> PCA -> k-means.

    This is what makes zone shapes follow the field's actual variability
    instead of the contours of one variable. Redundant indices (NDVI and BSI
    correlate about -0.84) would otherwise be double-counted; PCA removes
    that before clustering.

    Zones are renumbered so zone 1 is always the least productive, which
    keeps the dose ramp meaningful across runs.
    """
    keys = feature_keys or ["SOC", "N", "P", "K", "MOISTURE", "CLAY"]
    keys = [k for k in keys if k in params]
    valid = aoi_mask.copy()
    for k in keys:
        valid &= np.isfinite(params[k])
    if valid.sum() < n_zones * 10:
        return np.zeros(aoi_mask.shape, "int8"), {"error": "too few valid pixels"}

    X = np.column_stack([params[k][valid] for k in keys]).astype("float64")
    mu, sd = X.mean(0), X.std(0)
    sd[sd < 1e-12] = 1.0
    Xs = (X - mu) / sd

    C = np.cov(Xs, rowvar=False)
    evals, evecs = np.linalg.eigh(C)
    order = np.argsort(evals)[::-1]
    evals, evecs = evals[order], evecs[:, order]
    evals = np.clip(evals, 0, None)
    evr = evals / max(evals.sum(), 1e-12)
    k_comp = int(np.searchsorted(np.cumsum(evr), variance_target) + 1)
    k_comp = max(1, min(k_comp, len(keys)))
    scores = Xs @ evecs[:, :k_comp]

    lab, cent, inertia = _kmeans(scores, n_zones, seed=seed)

    # order clusters by productivity so zone 1 = worst
    prod_key = "VIGOUR" if "VIGOUR" in params else keys[0]
    prod = params[prod_key][valid]
    means = [np.nanmean(prod[lab == i]) if np.any(lab == i) else np.nan
             for i in range(n_zones)]
    rank = np.argsort(np.argsort(np.nan_to_num(means, nan=-np.inf)))
    remap = {i: int(rank[i]) + 1 for i in range(n_zones)}

    zg = np.zeros(aoi_mask.shape, "int8")
    zg[valid] = np.array([remap[l] for l in lab], dtype="int8")

    info = {"features": keys, "n_components": k_comp,
            "explained_variance_ratio": [round(float(x), 4) for x in evr],
            "cumulative_variance": round(float(np.cumsum(evr)[k_comp - 1]), 4),
            "inertia": round(inertia, 3), "ordered_by": prod_key}
    return zg, info


def classify_zones_relative(arr, n_zones=5, method="quantile"):
    """Single-parameter zoning, relative to this field's own distribution."""
    valid = np.isfinite(arr)
    out = np.zeros(arr.shape, dtype="int8")
    if not np.any(valid):
        return out
    vals = arr[valid]
    if method == "kmeans":
        lab, _, _ = _kmeans(vals.reshape(-1, 1).astype("float64"), n_zones, seed=0, n_init=3)
        cent = np.array([vals[lab == i].mean() if np.any(lab == i) else np.inf
                         for i in range(n_zones)])
        rank = np.argsort(np.argsort(cent))
        out[valid] = np.array([rank[l] + 1 for l in lab], dtype="int8")
        return out
    edges = np.unique(np.percentile(vals, np.linspace(0, 100, n_zones + 1)))
    if len(edges) < 2:
        edges = np.array([vals.min(), vals.max() + 1e-6])
    out[valid] = np.clip(np.digitize(vals, edges[1:-1], right=True) + 1, 1, n_zones).astype("int8")
    return out


def majority_denoise(zone_map, size=3):
    if zone_map.max() == 0:
        return zone_map
    return np.clip(median_filter(zone_map, size=size, mode="nearest"),
                   0, zone_map.max()).astype(zone_map.dtype)


def dissolve_small_patches(zone_map, res_m, min_area_ha=0.05):
    """Components are computed PER ZONE VALUE. Computing them on the nonzero
    mask instead would make the whole field one component and no island
    would ever be dissolved."""
    out = zone_map.copy()
    px_area_ha = (res_m * res_m) / 10000.0
    min_px = max(1, int(round(min_area_ha / px_area_ha)))
    for zval in [z for z in np.unique(zone_map) if z != 0]:
        labeled, n_comp = cc_label(zone_map == zval, structure=np.ones((3, 3)))
        for cid in range(1, n_comp + 1):
            comp = labeled == cid
            if int(comp.sum()) >= min_px:
                continue
            ys, xs = np.where(comp)
            y0, y1 = max(ys.min() - 1, 0), min(ys.max() + 2, zone_map.shape[0])
            x0, x1 = max(xs.min() - 1, 0), min(xs.max() + 2, zone_map.shape[1])
            nb, lm = zone_map[y0:y1, x0:x1], comp[y0:y1, x0:x1]
            border = nb[(~lm) & (nb > 0) & (nb != zval)]
            if border.size == 0:
                continue
            v, cnts = np.unique(border, return_counts=True)
            out[comp] = v[np.argmax(cnts)]
    return out


def vectorize_zone_patches(zone_map, transform, res_m, zone_labels):
    px_area_ha = (res_m * res_m) / 10000.0
    H, W = zone_map.shape
    patches, pid = [], 0
    for geom, val in rio_shapes(zone_map.astype("int32"), mask=zone_map > 0,
                                 transform=transform):
        val = int(val)
        if val == 0:
            continue
        pid += 1
        poly = shape(geom)
        area_ha = poly.area / 10000.0
        minx, miny, maxx, maxy = poly.bounds
        touches = (minx <= transform.c or miny <= (transform.f + H * transform.e) or
                   maxx >= (transform.c + W * transform.a) or maxy >= transform.f)
        patches.append({
            "patch_id": pid, "zone_value": val,
            "zone_class": zone_labels.get(val, str(val)),
            "patch_type": "core" if area_ha >= 1.0 else ("edge" if touches else "isolated"),
            "geometry": poly, "area_ha": round(area_ha, 4),
            "pixel_count": int(round(area_ha / px_area_ha)),
        })
    return patches


def patches_to_geojson(patches, crs_epsg=4326, transformer=None):
    feats = []
    for p in patches:
        g = p["geometry"]
        if transformer is not None:
            g = shp_transform(lambda x, y, z=None: transformer.transform(x, y), g)
        feats.append({"type": "Feature",
                       "properties": {k: p[k] for k in
                                      ("patch_id", "zone_class", "zone_value",
                                       "patch_type", "area_ha", "pixel_count")},
                       "geometry": mapping(g)})
    return {"type": "FeatureCollection",
            "crs": {"type": "name", "properties": {"name": f"EPSG:{crs_epsg}"}},
            "features": feats}


def finalize_zones(zone_grid, transform, res_m, aoi_mask, n_zones, min_patch_ha=0.05):
    """Denoise, dissolve slivers, clip to AOI, vectorize."""
    labels = zone_labels_for(n_zones)
    z = majority_denoise(zone_grid, 3)
    z = dissolve_small_patches(z, res_m, min_patch_ha)
    z[~aoi_mask] = 0
    return {"zone_grid": z,
            "patches": vectorize_zone_patches(z, transform, res_m, labels),
            "zone_labels": labels, "n_zones": n_zones}


def calc_vra_rates(zres, nutrient, crop, param_map=None, aoi_mask=None):
    """Per-zone dose and totals, plus the crop's own uptake demand."""
    n_zones = zres["n_zones"]
    fr, labels = zone_dose_fractions(n_zones), zres["zone_labels"]
    max_dose, prod = FERTILISER_MAX_DOSE[nutrient], FERTILISER_PRODUCTS[nutrient]
    demand = CROP_DEMAND.get(crop, CROP_DEMAND["default"])[nutrient]

    by = {labels[i]: {"pixel_count": 0, "area_ha": 0.0, "patch_ids": []}
          for i in range(1, n_zones + 1)}
    for p in zres["patches"]:
        c = by[p["zone_class"]]
        c["pixel_count"] += p["pixel_count"]
        c["area_ha"] += p["area_ha"]
        c["patch_ids"].append(p["patch_id"])

    HA_ACRE = 2.47105
    out = {}
    for i in range(1, n_zones + 1):
        lbl, c = labels[i], by[labels[i]]
        dose_nut = round(max_dose * fr[i], 1)
        dose_prod = round(dose_nut / (prod["nutrient_pct"] / 100.0), 1)
        rec = {"pixel_count": c["pixel_count"], "area_ha": round(c["area_ha"], 3),
               "area_acres": round(c["area_ha"] * HA_ACRE, 3),
               "nutrient_dose_kg_ha": dose_nut, "product": prod["name"],
               "product_dose_kg_ha": dose_prod,
               "total_product_kg": round(dose_prod * c["area_ha"], 1),
               "patch_count": len(c["patch_ids"]), "dose_fraction": fr[i],
               "crop_uptake_kg_per_tonne": demand}
        if param_map is not None and aoi_mask is not None:
            m = (zres["zone_grid"] == i) & aoi_mask & np.isfinite(param_map)
            rec["soil_mean"] = round(float(np.nanmean(param_map[m])), 2) if m.any() else None
        out[lbl] = rec
    return out


# ══════════════════════════════════════════════════════════════════════════
# RENDERING
# ══════════════════════════════════════════════════════════════════════════
def _aoi_pixel_path(aoi_sc, transform, r0=0, c0=0):
    a, e, cx, fy = transform.a, transform.e, transform.c, transform.f

    def to_px(x, y):
        return ((x - cx) / a - c0 - 0.5, (y - fy) / e - r0 - 0.5)

    geoms = list(aoi_sc.geoms) if isinstance(aoi_sc, MultiPolygon) else [aoi_sc]
    verts, codes = [], []
    for poly in geoms:
        for ring in [poly.exterior] + list(poly.interiors):
            pts = [to_px(x, y) for x, y in ring.coords]
            if len(pts) < 3:
                continue
            verts.extend(pts)
            codes.extend([MplPath.MOVETO] + [MplPath.LINETO] * (len(pts) - 2)
                         + [MplPath.CLOSEPOLY])
    return MplPath(verts, codes)


def _crop_bounds(valid_mask):
    rows = np.where(valid_mask.any(axis=1))[0]
    cols = np.where(valid_mask.any(axis=0))[0]
    if rows.size == 0 or cols.size == 0:
        return 0, valid_mask.shape[0], 0, valid_mask.shape[1]
    return rows[0], rows[-1] + 1, cols[0], cols[-1] + 1


def _fill_for_display(arr, valid, dilate_px=3):
    """Bleed the raster a few pixels past the valid edge with nearest values,
    for display only, so the clip path lands on the true field boundary
    instead of a pixel staircase. Never used for any statistic."""
    if not np.any(valid):
        return arr, valid
    dist, (iy, ix) = distance_transform_edt(~valid, return_indices=True)
    filled = arr[iy, ix]
    show = valid | (dist <= dilate_px)
    return np.where(show, filled, np.nan).astype("float32"), show


def _contour_field(zone_grid, sigma=0.8, dilate_px=3):
    zf = zone_grid.astype("float32")
    m = zone_grid > 0
    if not np.any(m):
        return np.full_like(zf, np.nan)
    num = gaussian_filter(np.where(m, zf, 0.0), sigma)
    wgt = gaussian_filter(m.astype("float32"), sigma)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(wgt > 0.30, num / np.maximum(wgt, 1e-9), np.nan).astype("float32")
    out[~m] = np.nan
    if dilate_px > 0:
        out, _ = _fill_for_display(out, np.isfinite(out), dilate_px)
    return out


def _clip_contour(cs, clip_path, ax):
    colls = getattr(cs, "collections", None)
    if colls:
        for coll in colls:
            coll.set_clip_path(clip_path, transform=ax.transData)
    else:
        try:
            cs.set_clip_path(clip_path, transform=ax.transData)
        except Exception:
            pass


def _label_anchor(comp_mask):
    dist = distance_transform_edt(comp_mask)
    idx = int(np.argmax(dist))
    y, x = divmod(idx, comp_mask.shape[1])
    return x, y, float(dist.ravel()[idx])


def _sampling_points(zone_grid, n_per_zone=3, min_sep_px=6):
    pts = []
    for zval in [z for z in np.unique(zone_grid) if z != 0]:
        labeled, ncomp = cc_label(zone_grid == zval, structure=np.ones((3, 3)))
        for cid in range(1, ncomp + 1):
            comp = labeled == cid
            if comp.sum() < 6:
                continue
            dist = distance_transform_edt(comp)
            flat_d = dist.ravel()
            k = max(1, min(n_per_zone, int(comp.sum() // 40) or 1))
            chosen = []
            for fi in np.argsort(flat_d)[::-1]:
                if flat_d[fi] < 1.0:
                    break
                y, x = divmod(int(fi), comp.shape[1])
                if all((y - cy) ** 2 + (x - cx) ** 2 > min_sep_px ** 2 for cy, cx in chosen):
                    chosen.append((y, x))
                if len(chosen) >= k:
                    break
            for y, x in chosen:
                pts.append((x, y, int(zval)))
    return pts


def _fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=DPI, bbox_inches="tight",
                pad_inches=0.12, facecolor=fig.get_facecolor())
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def render_parameter_map(value_arr, zone_grid, aoi_sc, transform, res_m,
                          param_key, capture_info, vra_rates=None,
                          zone_labels=None, label_mode="percent",
                          calib_status="UNCALIBRATED", show_points=True):
    """Heatmap + white zone boundaries + per-patch badges, clipped to the AOI."""
    meta = PARAMETERS.get(param_key, {"label": param_key, "unit": "",
                                       "cmap": CMAP_HEAT, "confidence": "moderate"})
    cmap = meta["cmap"]
    valid = np.isfinite(value_arr) & (zone_grid > 0)
    if not np.any(valid):
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.text(0.5, 0.5, f"No valid data for {meta['label']}", ha="center", va="center")
        ax.axis("off")
        return _fig_to_b64(fig)

    r0, r1, c0, c1 = _crop_bounds(valid)
    vals, zgrid = value_arr[r0:r1, c0:c1], zone_grid[r0:r1, c0:c1]
    vmask = np.isfinite(vals) & (zgrid > 0)

    fig = plt.figure(figsize=(13.5, 8.5), facecolor="#ffffff")
    gs = fig.add_gridspec(1, 2, width_ratios=[3.1, 1], wspace=0.03)
    ax, ax_info = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])

    vmin = float(np.nanpercentile(vals[vmask], 2))
    vmax = float(np.nanpercentile(vals[vmask], 98))
    if vmax <= vmin:
        vmax = vmin + 1e-6

    disp, dmask = _fill_for_display(vals, vmask, 3)
    im = ax.imshow(np.ma.masked_where(~dmask, disp), cmap=cmap,
                   vmin=vmin, vmax=vmax, interpolation="bilinear")

    clip_path = _aoi_pixel_path(aoi_sc, transform, r0, c0)
    im.set_clip_path(clip_path, transform=ax.transData)
    ax.add_patch(PathPatch(clip_path, transform=ax.transData, facecolor="none",
                           edgecolor="white", linewidth=3.0, zorder=6))
    ax.add_patch(PathPatch(clip_path, transform=ax.transData, facecolor="none",
                           edgecolor="#37474f", linewidth=1.0, zorder=7))

    n_z = int(zgrid.max())
    if n_z >= 2:
        cs = ax.contour(_contour_field(zgrid), levels=[i + 0.5 for i in range(1, n_z)],
                        colors="white", linewidths=2.2, zorder=5)
        _clip_contour(cs, clip_path, ax)

    zone_labels = zone_labels or zone_labels_for(max(n_z, 2))
    px_area_ha = (res_m * res_m) / 10000.0

    patch_means = []
    for zval in [z for z in np.unique(zgrid) if z != 0]:
        lab, nc = cc_label((zgrid == zval) & vmask, structure=np.ones((3, 3)))
        for cid in range(1, nc + 1):
            cm = lab == cid
            if cm.sum() > 0:
                patch_means.append(float(np.nanmean(vals[cm])))
    best = max(patch_means) if patch_means else 1.0
    if abs(best) < 1e-9:
        best = 1e-9

    for zval in [z for z in np.unique(zgrid) if z != 0]:
        lab, nc = cc_label((zgrid == zval) & vmask, structure=np.ones((3, 3)))
        for cid in range(1, nc + 1):
            comp = lab == cid
            if comp.sum() * px_area_ha < 0.08:
                continue
            x, y, depth = _label_anchor(comp)
            if depth < 1.5:
                continue
            pm = float(np.nanmean(vals[comp]))
            if label_mode == "percent":
                txt = f"{pm / best * 100:.0f}%"
            elif label_mode == "dose" and vra_rates:
                lbl = zone_labels.get(int(zval), str(zval))
                txt = f"{vra_rates.get(lbl, {}).get('product_dose_kg_ha', 0):.0f}"
            else:
                txt = f"{pm:.1f}"
            ax.text(x, y, txt, ha="center", va="center", fontsize=9.5,
                    fontweight="bold", color="white", zorder=9,
                    bbox=dict(boxstyle="round,pad=0.32", facecolor="#1c1c1c",
                              edgecolor="white", linewidth=0.8, alpha=0.92))

    if show_points:
        for x, y, _z in _sampling_points(zgrid, 3):
            ax.plot(x, y, "o", markersize=5.5, markerfacecolor="#e0e0e0",
                    markeredgecolor="#424242", markeredgewidth=0.9, zorder=8)

    ax.set_xlim(-0.5, vals.shape[1] - 0.5)
    ax.set_ylim(vals.shape[0] - 0.5, -0.5)
    ax.set_axis_off()
    ax.set_title(f"{meta['label']} — Management Zones\n{capture_info}",
                 fontsize=12, fontweight="bold", color="#212121", pad=10)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("bottom", size="3.2%", pad=0.12)
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_label(f"{meta['label']} ({meta['unit']})", fontsize=9, color="#212121")
    cb.ax.tick_params(labelsize=8, colors="#212121")

    # status stamp so a calibrated and an uncalibrated map can never be confused
    ax.text(0.008, 0.012,
            f"{calib_status}  |  retrieval confidence: {meta['confidence']}",
            transform=ax.transAxes, fontsize=7.5, va="bottom", ha="left",
            color="#ffffff", zorder=12,
            bbox=dict(boxstyle="round,pad=0.35",
                      facecolor=CALIB_COLOR.get(calib_status, "#c62828"),
                      edgecolor="none", alpha=0.95))

    ax_info.set_axis_off()
    total_ha = float(vmask.sum()) * px_area_ha
    y = 0.985
    ax_info.text(0.02, y, meta["label"].upper(), fontsize=10, fontweight="bold",
                 va="top", color="#212121", transform=ax_info.transAxes)
    y -= 0.042
    ax_info.text(0.02, y, f"Field: {total_ha:.2f} ha ({total_ha*2.47105:.2f} ac)",
                 fontsize=7.5, va="top", color="#555555", transform=ax_info.transAxes)
    y -= 0.05

    zone_means = {}
    for zval in [z for z in np.unique(zgrid) if z != 0]:
        m = (zgrid == zval) & vmask
        if m.sum():
            zone_means[int(zval)] = float(np.nanmean(vals[m]))

    colors = zone_colors_for(max(n_z, 2))
    n_lines = 3 + (2 if vra_rates else 0)
    footnote_h = 0.13
    avail = y - footnote_h
    block_h = min(0.155, avail / max(len(zone_means), 1))
    line_h = min(0.028, (block_h - 0.016) / n_lines)
    fs = 6.6 if line_h >= 0.024 else 5.9

    for zval in sorted(zone_means):
        lbl = zone_labels.get(zval, str(zval))
        area_ha = float(((zgrid == zval) & vmask).sum()) * px_area_ha
        pct = area_ha / max(total_ha, 1e-9) * 100
        ax_info.add_patch(mpatches.FancyBboxPatch(
            (0.02, y - line_h * 1.1), 0.072, line_h * 1.5,
            boxstyle="round,pad=0.004", linewidth=0.6, edgecolor="#9e9e9e",
            facecolor=colors.get(lbl, "#9e9e9e"),
            transform=ax_info.transAxes, clip_on=False))
        lines = [lbl, f"mean {zone_means[zval]:.1f} {meta['unit']}",
                 f"{area_ha:.2f} ha ({pct:.0f}%)"]
        if vra_rates and lbl in vra_rates:
            d = vra_rates[lbl]
            lines += [f"{d['product']}: {d['product_dose_kg_ha']:.0f} kg/ha",
                      f"total {d['total_product_kg']:.0f} kg"]
        for i, ln in enumerate(lines):
            ax_info.text(0.112, y - i * line_h, ln, fontsize=fs, va="top",
                         color="#212121" if i == 0 else "#555555",
                         fontweight="bold" if i == 0 else "normal",
                         transform=ax_info.transAxes)
        y -= block_h

    note = "Badges = each patch's mean as %\nof the field's best patch."
    if show_points:
        note += "\nGrey dots = suggested sampling."
    note += f"\n\n{CONFIDENCE_NOTE.get(meta['confidence'], '')}"
    ax_info.text(0.02, max(y - 0.012, 0.012), note, fontsize=5.9, va="top",
                 color="#777777", transform=ax_info.transAxes)

    return _fig_to_b64(fig)


def render_correlation_matrix(names, C, title="Index correlation"):
    fig, ax = plt.subplots(figsize=(8.5, 7.5), facecolor="#ffffff")
    im = ax.imshow(C, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=90, fontsize=7)
    ax.set_yticklabels(names, fontsize=7)
    for i in range(len(names)):
        for j in range(len(names)):
            if np.isfinite(C[i, j]):
                ax.text(j, i, f"{C[i, j]:.2f}", ha="center", va="center",
                        fontsize=5.5,
                        color="white" if abs(C[i, j]) > 0.55 else "#212121")
    ax.set_title(f"{title}\nStrong pairs are redundant — PCA removes the "
                 f"double-counting before clustering",
                 fontsize=10, fontweight="bold", pad=10)
    fig.colorbar(im, ax=ax, shrink=0.75, label="Pearson r")
    return _fig_to_b64(fig)


def render_timeseries_chart(used_scenes, scene_ndvi, bare_idx, veg_idx, thr):
    fig, ax = plt.subplots(figsize=(11, 4.2), facecolor="#ffffff")
    dates = [s["date"] for s in used_scenes]
    x = np.arange(len(dates))
    colors = ["#8d6e63" if i in set(bare_idx.tolist()) else "#2e7d32"
              for i in range(len(dates))]
    ax.bar(x, scene_ndvi, color=colors, width=0.65)
    if np.isfinite(thr):
        ax.axhline(thr, color="#c62828", linestyle="--", linewidth=1.4,
                   label=f"adaptive bare/veg split = {thr:.3f}")
    ax.set_xticks(x)
    ax.set_xticklabels(dates, rotation=60, fontsize=7, ha="right")
    ax.set_ylabel("Field-mean NDVI", fontsize=9)
    ax.set_title(f"Scene time series — {len(dates)} clear scenes analysed "
                 f"({len(bare_idx)} bare-soil, {len(veg_idx)} vegetation)",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    handles = [mpatches.Patch(color="#8d6e63", label="bare-soil scene (soil params)"),
               mpatches.Patch(color="#2e7d32", label="vegetation scene (nutrients/vigour)")]
    ax.legend(handles=handles + [plt.Line2D([0], [0], color="#c62828", ls="--",
              label="adaptive split")], fontsize=7, loc="upper left")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return _fig_to_b64(fig)


def render_overview(params, zres_by, aoi_sc, transform, res_m, capture_info, keys):
    n = len(keys)
    ncol = 3
    nrow = int(math.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 4.4 * nrow),
                             facecolor="#ffffff")
    fig.suptitle(f"FIELD PARAMETER OVERVIEW\n{capture_info}",
                 fontsize=14, fontweight="bold", color="#212121")
    axes = np.atleast_1d(axes).ravel()

    ref = zres_by[keys[0]]["zone_grid"]
    any_valid = np.isfinite(params[keys[0]]) & (ref > 0)
    r0, r1, c0, c1 = _crop_bounds(any_valid)

    for ax, key in zip(axes, keys):
        meta = PARAMETERS[key]
        arr = params[key][r0:r1, c0:c1]
        zg = zres_by[key]["zone_grid"][r0:r1, c0:c1]
        vmask = np.isfinite(arr) & (zg > 0)
        if not np.any(vmask):
            ax.set_axis_off()
            continue
        vmin = float(np.nanpercentile(arr[vmask], 2))
        vmax = float(np.nanpercentile(arr[vmask], 98))
        if vmax <= vmin:
            vmax = vmin + 1e-6
        disp, dmask = _fill_for_display(arr, vmask, 3)
        im = ax.imshow(np.ma.masked_where(~dmask, disp), cmap=meta["cmap"],
                       vmin=vmin, vmax=vmax, interpolation="bilinear")
        cp = _aoi_pixel_path(aoi_sc, transform, r0, c0)
        im.set_clip_path(cp, transform=ax.transData)
        ax.add_patch(PathPatch(cp, transform=ax.transData, facecolor="none",
                               edgecolor="white", linewidth=2.4, zorder=6))
        n_z = int(zg.max())
        if n_z >= 2:
            cs = ax.contour(_contour_field(zg), levels=[i + 0.5 for i in range(1, n_z)],
                            colors="white", linewidths=1.5, zorder=5)
            _clip_contour(cs, cp, ax)
        cb = plt.colorbar(im, ax=ax, shrink=0.72, pad=0.02)
        cb.ax.tick_params(labelsize=7)
        ax.set_xlim(-0.5, arr.shape[1] - 0.5)
        ax.set_ylim(arr.shape[0] - 0.5, -0.5)
        ax.set_title(f"{meta['label']} ({meta['unit']})", fontsize=9.5, pad=4)
        ax.set_axis_off()

    for ax in axes[n:]:
        ax.set_axis_off()
    return _fig_to_b64(fig)


# ──────────────────────────────────────────────────────────────────────────
# REPORT
# ──────────────────────────────────────────────────────────────────────────
def build_text_report(res):
    m, crop = res["metadata"], res["crop"]
    hr = "=" * 78
    L = [hr, "  CROPGEN SOIL + VRA REPORT  (time-series, multi-index)", hr,
         f"  Crop            : {crop}",
         f"  Window          : {m['start_date']} -> {m['end_date']}",
         f"  Scenes analysed : {m['scenes_used']} of {m['scenes_found']} found",
         f"  Bare-soil scenes: {m['bare_scenes']}   Vegetation scenes: {m['veg_scenes']}",
         f"  Split threshold : NDVI {m['split_threshold']}  (adaptive to this field)",
         f"  Grid            : {m['grid_H']} x {m['grid_W']} px @ {m['res_m']:.1f} m/px",
         f"  AOI             : {m['aoi_pixels']} px, {m['field_area_ha']:.2f} ha",
         f"  Observations/px : min {m['obs_min']}, mean {m['obs_mean']:.1f}, max {m['obs_max']}",
         f"  Zones           : {m['n_zones']}  (multivariate PCA + k-means)",
         hr, ""]

    z = res.get("zone_info", {})
    if z and "explained_variance_ratio" in z:
        L += ["  -- ZONING BASIS --",
              f"  Features   : {', '.join(z['features'])}",
              f"  Components : {z['n_components']} "
              f"(captures {z['cumulative_variance']*100:.1f}% of variance)",
              f"  Ordered by : {z['ordered_by']} (zone 1 = least productive)", ""]

    L += ["  -- PARAMETERS --",
          f"  {'Parameter':<24}{'Mean':>10}{'Min':>10}{'Max':>10}  {'Status':<14}{'Confidence'}"]
    L.append("  " + "-" * 74)
    for k, st in res["param_stats"].items():
        meta = PARAMETERS[k]
        cal = res["calibration"].get(k, {}).get("status", "UNCALIBRATED")
        L.append(f"  {meta['label'][:23]:<24}{st['mean']:>10.2f}{st['min']:>10.2f}"
                 f"{st['max']:>10.2f}  {cal:<14}{meta['confidence']}")
    L.append("")

    st_count = {}
    for v in res["calibration"].values():
        st_count[v.get("status", CALIB_UNCAL)] = st_count.get(v.get("status", CALIB_UNCAL), 0) + 1
    reg = res.get("region", {})
    if reg:
        L += ["  -- REGION --",
              f"  Centroid    : {reg.get('lat')}, {reg.get('lon')}",
              f"  Basis       : {reg.get('basis')}",
              f"  Anchor      : {reg.get('resolved_region') or 'none'} "
              f"({reg.get('n_lab_records', 0)} lab records)", ""]
    sc = res.get("soc_composite", {})
    if sc:
        L += ["  -- SOC METHOD --", f"  {sc.get('method')}"]
        if "coverage" in sc:
            L.append(f"  Exposed-soil coverage of the field: {sc['coverage']*100:.1f}%")
        L.append("")

    L += ["  -- CALIBRATION STATUS --",
          "  " + "   ".join(f"{k}: {v}" for k, v in sorted(st_count.items()))]
    if m.get("prior_region"):
        L.append(f"  Regional prior : {m['prior_region']} ({m['prior_records']} lab records)")
        L.append("  RANGE-ANCHORED means the absolute scale matches soils measured in")
        L.append("  this region. It does NOT mean the within-field pattern is verified.")
    else:
        L.append("  No regional prior applied — pass district= / state= to anchor the")
        L.append("  absolute scale to measured lab distributions.")
    L.append("")

    cal_any = any(v.get("status") == CALIB_FULL for v in res["calibration"].values())
    if cal_any:
        L += ["  -- CALIBRATION --"]
        for k, c in res["calibration"].items():
            if c.get("status") == CALIB_FULL and "r2" in c:
                L.append(f"  {PARAMETERS[k]['label']:<26} n={c['n_samples']:<3} "
                         f"R2={c['r2']:.3f}  RMSE={c['rmse']:.3f}  "
                         f"LOO-RMSE={c['loo_rmse']:.3f}")
        L += ["  LOO-RMSE is the honest error estimate; R2 on few points flatters.", ""]
    else:
        L += ["  No geolocated ground samples supplied, so nothing reached CALIBRATED.",
              "  To get there, 8-12 samples with lat/lon inside this field are needed;",
              "  village-level lab records cannot be matched to pixels.", ""]

    L += [hr, "  -- VRA PRESCRIPTION --", ""]
    for nut in VRA_NUTRIENTS:
        if nut not in res["vra_rates"]:
            continue
        p = FERTILISER_PRODUCTS[nut]
        L.append(f"  > {nut} — {p['name']} ({p['nutrient_pct']:.0f}% nutrient)")
        L.append(f"    {'Zone':<13}{'Patch':>6}{'Area ha':>10}{'Area ac':>10}"
                 f"{'kg/ha':>9}{'Total kg':>11}")
        L.append("    " + "-" * 59)
        tot = 0.0
        for lbl, d in res["vra_rates"][nut].items():
            if d["pixel_count"] == 0:
                continue
            tot += d["total_product_kg"]
            L.append(f"    {lbl:<13}{d['patch_count']:>6}{d['area_ha']:>10.3f}"
                     f"{d['area_acres']:>10.3f}{d['product_dose_kg_ha']:>9.1f}"
                     f"{d['total_product_kg']:>11.1f}")
        flat = (FERTILISER_MAX_DOSE[nut] / (p["nutrient_pct"] / 100.0)
                * res["metadata"]["field_area_ha"])
        if flat > 0:
            L.append(f"    VRA {tot:.0f} kg vs flat-rate {flat:.0f} kg "
                     f"-> saving {(1-tot/flat)*100:.1f}%")
        L.append("")

    pg_ = res.get("prescription_grid")
    if pg_:
        s = pg_["summary"]
        L += [hr, "  -- PRESCRIPTION GRID --",
              f"  Cell size   : {s['cell_size_m']:.0f} m "
              f"({s['pixels_per_cell']} pixels per cell)",
              f"  Cells       : {s['n_cells']}   Area: {s['total_area_ha']:.2f} ha",
              f"  Uniform     : {s['uniform_cells']} ({s['uniform_pct']:.0f}%)   "
              f"Mixed: {s['mixed_cells']}",
              "  Set the cell size to the spreader working width. A 'mixed' cell",
              "  holds real variation the machine cannot resolve at that width.", ""]

    L += [hr, "  -- HOW TO READ THIS --",
          "  Zone 1 is the least productive part of the field and receives the",
          "  full dose; the highest zone is the most productive and receives the",
          "  smallest dose. Zones are relative to THIS field, so they always",
          "  split it into workable blocks even when the whole field is uniformly",
          "  good or uniformly poor. Check the parameter table above before",
          "  acting on any single number.", hr]
    return "\n".join(L)


# ══════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════
def run_analysis(aoi_geojson, start_date, end_date, crop="wheat",
                  n_zones=5, min_patch_ha=0.05, label_mode="percent",
                  ground_samples=None, max_scenes=MAX_SCENES,
                  zone_features=None, district=None, state=None,
                  auto_region=True, use_soilgrids=False,
                  grid_cell_m=DEFAULT_CELL_M, soc_method="published",
                  include_images=True, image_keys=None):
    """
    Full time-series soil + VRA analysis.

    ground_samples: optional list of lab results, e.g.
        [{"lat":20.1372,"lon":77.1561,"SOC":0.82,"N":210,"P":24,"K":180}, ...]
      Supply 8-12 well-spread points to switch parameters to CALIBRATED.
    image_keys: optional subset of image names to render (e.g. {"SOC"});
      None renders every map. Rendering is a large share of the runtime.
    """
    _validate_inputs(aoi_geojson, start_date, end_date, crop, n_zones)
    print(f"\n{'='*70}\n  CROPGEN SOIL + VRA ENGINE v6 (time-series)"
          f"\n  crop={crop}  window={start_date} -> {end_date}  zones={n_zones}\n{'='*70}")

    # 1 ── scenes
    print("\n[1/9] Searching all Sentinel-2 scenes in window ...")
    scenes = find_all_scenes(aoi_geojson, start_date, end_date, max_scenes)
    if not scenes:
        raise RuntimeError("No Sentinel-2 scenes found for this AOI / date range.")
    print(f"      {len(scenes)} candidate scenes "
          f"({scenes[0]['date']} .. {scenes[-1]['date']})")

    # 2 ── grid
    print("[2/9] Building native 10 m grid ...")
    crs = _item_crs(scenes[0]["item"])
    if crs is None:
        red_url = _pick_url(scenes[0]["item"].assets, "red", "B04")
        if not red_url:
            raise RuntimeError("No red-band asset — cannot derive CRS.")
        with rasterio.open(red_url) as ref:
            crs = ref.crs
    aoi_sc, dst_tf, H, W, res_m = build_grid(crs, aoi_geojson, NATIVE_RES_M)
    aoi_mask = geometry_mask([mapping(aoi_sc)], out_shape=(H, W),
                             transform=dst_tf, invert=True)
    aoi_px = int(aoi_mask.sum())
    if aoi_px == 0:
        raise RuntimeError("AOI covers zero pixels at 10 m — polygon may be too small.")
    field_ha = aoi_px * res_m * res_m / 10000.0
    print(f"      {H} x {W} px @ {res_m:.2f} m — {aoi_px} px inside AOI ({field_ha:.2f} ha)")

    # 3 ── time series
    print(f"[3/9] Building index time series ({len(INDEX_NAMES)} indices per scene) ...")
    stack, used, obs, raw_scene_bands = build_index_timeseries(
        scenes, aoi_geojson, dst_tf, H, W, aoi_mask)
    if stack is None:
        raise RuntimeError(
            "No scene met the clear-sky threshold. Widen the date window or "
            f"lower MIN_CLEAR_FRACTION (currently {MIN_CLEAR_FRACTION}).")
    print(f"      {len(used)} scenes accepted into the stack")

    enough = obs >= MIN_OBS_PER_PIXEL
    dropped = int((aoi_mask & ~enough).sum())
    if dropped:
        print(f"      dropping {dropped} px seen < {MIN_OBS_PER_PIXEL} times")
    aoi_mask = aoi_mask & enough
    if aoi_mask.sum() == 0:
        raise RuntimeError("Every pixel was cloudy too often. Widen the date window.")

    # 4 ── bare / vegetation split
    print("[4/9] Splitting scenes into bare-soil and vegetation groups ...")
    bare_idx, veg_idx, thr, scene_ndvi = split_bare_vegetation(stack, aoi_mask)
    print(f"      threshold NDVI={thr:.3f} -> {len(bare_idx)} bare, {len(veg_idx)} vegetation")

    # 5 ── composites
    print("[5/9] Temporal median composites ...")
    bare_comp = temporal_composite(stack, bare_idx)
    veg_comp  = temporal_composite(stack, veg_idx)

    # 6 ── correlation
    # Published exposed-soil composite: strict NDVI + NBR2 + spectral-shape
    # masking, multi-year median, normalised spectra. This is what the SOC
    # literature validates; the index blend below is kept as "legacy".
    soc_src = None
    if soc_method == "published" and raw_scene_bands:
        print("      building exposed-soil composite (NDVI + NBR2 + shape) ...")
        es_masks, diags = [], []
        for bd in raw_scene_bands:
            mk, dg = exposed_soil_mask(bd)
            es_masks.append(mk & aoi_mask)
            diags.append(dg)
        src_comp, src_nobs = build_soil_reflectance_composite(
            raw_scene_bands, es_masks, percentile=50, min_obs=2)
        if src_comp is not None:
            cov = float(np.isfinite(src_comp["B04"])[aoi_mask].mean())
            kept = int(np.isfinite(src_comp["B04"])[aoi_mask].sum())
            print(f"      exposed-soil coverage: {cov*100:.1f}% of the field "
                  f"({kept} px, {int(src_nobs[aoi_mask].max())} looks max)")
            if cov < 0.15:
                print("      coverage too low for the published method — the field "
                      "was rarely bare in this window. Widen the date range or "
                      "add earlier years; falling back to the index blend.")
                soc_src = None
            else:
                norm, bright = normalise_spectra(src_comp)
                soc_src = {"composite": src_comp, "normalised": norm,
                            "brightness": bright, "n_obs": src_nobs,
                            "coverage": round(cov, 4)}

    print("[6/9] Correlation analysis ...")
    corr_feats = {k: veg_comp[k] for k in ["NDVI", "EVI", "SAVI", "MSAVI", "NDRE", "RECI"]}
    corr_feats.update({f"{k}_bare": bare_comp[k]
                       for k in ["BSI", "SOCI", "CLAY", "BI", "SI", "NDMI"]})
    corr_names, corr_C = correlation_matrix(corr_feats, aoi_mask)
    finite = corr_C[np.isfinite(corr_C)]
    strong = int(((np.abs(corr_C) > 0.8) & np.isfinite(corr_C)).sum() - len(corr_names))
    print(f"      {len(corr_names)} features, {max(strong,0)//2} strongly redundant pairs")

    # 7 ── parameters + calibration
    print("[7/9] Retrieving parameters ...")
    clat, clon = aoi_centroid_wgs84(aoi_geojson)
    region_info = {"lat": clat, "lon": clon, "basis": "explicit district/state",
                   "district": district, "state": state, "anchored": bool(district or state)}
    if auto_region and not (district and state):
        det = detect_region(clat, clon)
        district = district or det["district"]
        state = state or det["state"]
        region_info = det
        print(f"      auto region: {det['basis']}")

    override = None
    if district or state:
        region, n_rec = region_support(district, state)
        region_info["resolved_region"] = region
        region_info["n_lab_records"] = n_rec
        print(f"      anchor: {region} ({n_rec} lab records)")
    elif use_soilgrids:
        print("      outside regions with local lab data — querying SoilGrids ...")
        override = soilgrids_prior(clat, clon)
        if override:
            region_info.update({"basis": "ISRIC SoilGrids 250 m",
                                "resolved_region": "SOILGRIDS", "anchored": True})
            print(f"      SoilGrids anchor for {len(override)} parameters")
    else:
        print("      no regional anchor — literature ranges will be used and the "
              "output stamped UNCALIBRATED")

    params, relative, base_status = retrieve_parameters(
        bare_comp, veg_comp, aoi_mask, district=district, state=state,
        override=override)
    if soc_src is not None:
        # Recompute SOC from the exposed-soil composite instead of the
        # index blend. Darker, less bright normalised soil spectra indicate
        # higher organic carbon.
        nb = soc_src["normalised"]
        dark = _norm01(-soc_src["brightness"], aoi_mask)
        vis  = _norm01(-(nb["B02"] + nb["B03"] + nb["B04"]) / 3.0, aoi_mask)
        soc_rel2 = np.clip(0.6 * dark + 0.4 * vis, 0, 1)
        soc_new, soc_stat = _anchor(soc_rel2, "SOC", district, state, override)
        keep = np.isfinite(soc_new) & aoi_mask
        params["SOC"] = np.where(keep, soc_new, params["SOC"]).astype("float32")
        base_status["SOC"] = soc_stat
        print(f"      SOC from exposed-soil composite "
              f"({soc_src['coverage']*100:.0f}% coverage)")

    for k, v in params.items():
        leak = int(np.count_nonzero(np.isfinite(v) & (~aoi_mask)))
        if leak:
            raise RuntimeError(f"AOI clip failed for {k}: {leak} px outside polygon.")
    print("      AOI clip verified: 0 px outside polygon")

    to_scene = Transformer.from_crs("EPSG:4326", crs.to_string(), always_xy=True)
    params, calib = calibrate_parameters(params, ground_samples, aoi_sc, dst_tf,
                                          aoi_mask, to_scene, base_status=base_status)
    from collections import Counter
    tally = Counter(c.get("status", CALIB_UNCAL) for c in calib.values())
    print("      status: " + "  ".join(f"{k}={v}" for k, v in tally.items()))

    # 8 ── zoning
    print(f"[8/9] Zoning ({n_zones} zones) ...")
    mv_grid, zone_info = multivariate_zones(params, aoi_mask, n_zones,
                                             feature_keys=zone_features)
    mv_zres = finalize_zones(mv_grid, dst_tf, res_m, aoi_mask, n_zones, min_patch_ha)
    print(f"      management zones: {len(mv_zres['patches'])} patches "
          f"from {zone_info.get('n_components','?')} PCA components")

    zres_by, vra_rates = {}, {}
    for key in params:
        g = classify_zones_relative(params[key], n_zones, "quantile")
        zres_by[key] = finalize_zones(g, dst_tf, res_m, aoi_mask, n_zones, min_patch_ha)
        if key in VRA_NUTRIENTS:
            vra_rates[key] = calc_vra_rates(zres_by[key], key, crop,
                                             params[key], aoi_mask)
        print(f"      {PARAMETERS[key]['label']}: {len(zres_by[key]['patches'])} patches")

    # 9 ── render (optional — skip for JSON-only API calls)
    dates = [u["date"] for u in used]
    capture_info = (f"Sentinel-2 L2A  |  {len(used)} scenes  "
                    f"{min(dates)} .. {max(dates)}  |  {res_m:.0f} m grid")

    images = {}
    if include_images:
        print("[9/9] Rendering ...")
        def want(name):
            return image_keys is None or name in image_keys

        for key in params:
            if not want(key):
                continue
            status = calib.get(key, {}).get("status", CALIB_UNCAL)
            images[key] = render_parameter_map(
                params[key], zres_by[key]["zone_grid"], aoi_sc, dst_tf, res_m,
                key, capture_info, vra_rates=vra_rates.get(key),
                zone_labels=zres_by[key]["zone_labels"], label_mode=label_mode,
                calib_status=status)
            print(f"      - {PARAMETERS[key]['label']}")

        if want("MANAGEMENT_ZONES"):
            images["MANAGEMENT_ZONES"] = render_parameter_map(
                params.get("VIGOUR", params["SOC"]), mv_zres["zone_grid"], aoi_sc, dst_tf,
                res_m, "VIGOUR", capture_info + "  |  multivariate zones",
                zone_labels=mv_zres["zone_labels"], label_mode="percent",
                calib_status=calib.get("VIGOUR", {}).get("status", CALIB_UNCAL))
        if want("CORRELATION"):
            images["CORRELATION"] = render_correlation_matrix(corr_names, corr_C)
        if want("TIMESERIES"):
            images["TIMESERIES"] = render_timeseries_chart(used, scene_ndvi, bare_idx, veg_idx, thr)
        if want("OVERVIEW"):
            images["OVERVIEW"] = render_overview(
                params, zres_by, aoi_sc, dst_tf, res_m, capture_info,
                [k for k in ["SOC", "N", "P", "K", "MOISTURE", "CLAY"] if k in params])
        print("      - management zones, correlation, time series, overview")
    else:
        print("[9/9] Skipping map render (include_images=False)")

    to_wgs = Transformer.from_crs(crs.to_string(), "EPSG:4326", always_xy=True)
    geojson = {k: patches_to_geojson(zres_by[k]["patches"], transformer=to_wgs)
               for k in params}
    geojson["MANAGEMENT_ZONES"] = patches_to_geojson(mv_zres["patches"], transformer=to_wgs)

    # machine-executable prescription grid
    print(f"      prescription grid at {grid_cell_m:.0f} m cells ...")
    conf_stamp = {k: calib.get(k, {}).get("status", CALIB_UNCAL)
                  for k in ("SOC", "N", "P", "K")}
    presc = build_prescription_grid(
        aoi_mask, dst_tf, res_m, params["SOC"],
        zone_grid=mv_zres["zone_grid"], zone_labels=mv_zres["zone_labels"],
        vra_rates=vra_rates,
        nutrient_maps={k: params[k] for k in VRA_NUTRIENTS if k in params},
        cell_m=grid_cell_m, to_wgs84=to_wgs, confidence=conf_stamp)
    print(f"      {presc['summary']['n_cells']} cells, "
          f"{presc['summary']['uniform_pct']:.0f}% uniform")

    param_stats = {}
    for k, v in params.items():
        vv = v[aoi_mask & np.isfinite(v)]
        param_stats[k] = {
            "mean": round(float(vv.mean()), 3) if vv.size else float("nan"),
            "min": round(float(vv.min()), 3) if vv.size else float("nan"),
            "max": round(float(vv.max()), 3) if vv.size else float("nan"),
            "std": round(float(vv.std()), 3) if vv.size else float("nan"),
            "unit": PARAMETERS[k]["unit"], "source_composite": PARAMETERS[k]["source"],
            "confidence": PARAMETERS[k]["confidence"],
        }

    result = {
        "images_b64": images,
        "param_stats": param_stats,
        "relative_index_note": "0-100 relative index also computed per parameter.",
        "vra_rates": vra_rates,
        "zone_geojson": geojson,
        "prescription_grid": presc,
        "region": region_info,
        "soc_composite": ({"coverage": soc_src["coverage"],
                            "method": "exposed-soil composite (NDVI+NBR2+shape, "
                                      "multi-year median, normalised spectra)"}
                           if soc_src else {"method": "index blend (legacy)"}),
        "zone_info": zone_info,
        "calibration": calib,
        "correlation": {"features": corr_names,
                         "matrix": [[None if not np.isfinite(v) else round(float(v), 4)
                                     for v in row] for row in corr_C]},
        "scenes_used": used,
        "crop": crop,
        "metadata": {
            "start_date": start_date, "end_date": end_date,
            "scenes_found": len(scenes), "scenes_used": len(used),
            "bare_scenes": len(bare_idx), "veg_scenes": len(veg_idx),
            "split_threshold": round(float(thr), 4) if np.isfinite(thr) else None,
            "grid_H": H, "grid_W": W, "res_m": res_m,
            "aoi_pixels": int(aoi_mask.sum()), "field_area_ha": round(field_ha, 3),
            "obs_min": int(obs[aoi_mask].min()), "obs_max": int(obs[aoi_mask].max()),
            "obs_mean": float(obs[aoi_mask].mean()),
            "n_zones": n_zones, "indices": INDEX_NAMES,
            "collection": "Sentinel-2 L2A",
            "district": district, "state": state,
            "prior_region": region_support(district, state)[0] if (district or state) else None,
            "prior_records": region_support(district, state)[1] if (district or state) else 0,
        },
    }
    result["text_report"] = build_text_report(result)
    print("\n" + result["text_report"])
    return result


def _json_safe(obj):
    """Convert numpy / nan values so FastAPI can emit JSON."""
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _json_safe(obj.tolist())
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        val = float(obj)
        if math.isnan(val) or math.isinf(val):
            return None
        return val
    if obj is None or isinstance(obj, (str, int, bool)):
        return obj
    return str(obj)


def to_api_result(result, include_images=True, include_prescription_geojson=True):
    """JSON-safe payload for the /v4/api/vra HTTP API."""
    images = result.get("images_b64") or {}
    payload = {
        "crop": result.get("crop"),
        "param_stats": result.get("param_stats"),
        "relative_index_note": result.get("relative_index_note"),
        "vra_rates": result.get("vra_rates"),
        "zone_geojson": result.get("zone_geojson"),
        "prescription_grid": result.get("prescription_grid"),
        "region": result.get("region"),
        "soc_composite": result.get("soc_composite"),
        "zone_info": result.get("zone_info"),
        "calibration": result.get("calibration"),
        "correlation": result.get("correlation"),
        "scenes_used": result.get("scenes_used"),
        "text_report": result.get("text_report"),
        "metadata": result.get("metadata"),
        "images": images if include_images else None,
    }
    if include_prescription_geojson and result.get("prescription_grid"):
        payload["prescription_geojson"] = grid_to_geojson(result["prescription_grid"])
    else:
        payload["prescription_geojson"] = None
    return _json_safe(payload)


def save_outputs(result, out_dir="./vra_output"):
    os.makedirs(out_dir, exist_ok=True)
    order = ["MANAGEMENT_ZONES", "SOC", "N", "P", "K", "MOISTURE", "CLAY",
             "EC", "PH", "VIGOUR", "OVERVIEW", "CORRELATION", "TIMESERIES"]
    seq = 1
    for key in order:
        b64 = result["images_b64"].get(key)
        if not b64:
            continue
        path = os.path.join(out_dir, f"{seq:02d}_{key}.png")
        with open(path, "wb") as f:
            f.write(base64.b64decode(b64))
        print(f"  Saved -> {path}")
        seq += 1
    for k, gj in result["zone_geojson"].items():
        path = os.path.join(out_dir, f"zones_{k}.geojson")
        with open(path, "w") as f:
            json.dump(gj, f, indent=2)
    print(f"  Saved -> {len(result['zone_geojson'])} GeoJSON zone files")
    pgd = result.get("prescription_grid")
    if pgd:
        p = os.path.join(out_dir, "prescription_grid.geojson")
        with open(p, "w") as f:
            json.dump(grid_to_geojson(pgd), f)
        print(f"  Saved -> {p}")
        p = os.path.join(out_dir, "prescription_grid.csv")
        n = grid_to_csv(pgd, p)
        print(f"  Saved -> {p} ({n} cells)")
        p = os.path.join(out_dir, "prescription_grid.txt")
        with open(p, "w") as f:
            f.write(grid_report(pgd))
        print(f"  Saved -> {p}")

    data = {k: v for k, v in result.items() if k != "images_b64"}
    path = os.path.join(out_dir, "analysis_data.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"  Saved -> {path}")
    path = os.path.join(out_dir, "report.txt")
    with open(path, "w") as f:
        f.write(result["text_report"])
    print(f"  Saved -> {path}")


# ──────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Any field, anywhere. Region is detected from the coordinates.
    aoi_geojson = {
        "type": "Polygon",
        "coordinates": [[
            [77.15618780277907, 20.137282565715527],
            [77.15840867184340, 20.136496870757790],
            [77.15745380543410, 20.135580221647306],
            [77.15575864933669, 20.136134240983303],
            [77.15618780277907, 20.137282565715527],
        ]]
    }

    # A long window matters twice over: the time series needs scenes, and
    # strict exposed-soil masking only reaches usable coverage across
    # several bare periods. 6-12 months minimum; multiple years is better.
    START_DATE  = "2024-06-01"
    END_DATE    = "2025-05-31"
    CROP        = "onion"
    N_ZONES     = 5
    GRID_CELL_M = 20.0      # set to the spreader working width

    # Geolocated lab samples inside this field lift parameters to
    # CALIBRATED. Village-level records cannot be used here — they have no
    # coordinates to match against a pixel.
    #   [{"lat":20.1372,"lon":77.1561,"SOC":0.82,"N":210,"P":24,"K":180}, ...]
    GROUND_SAMPLES = None

    result = run_analysis(
        aoi_geojson, START_DATE, END_DATE, crop=CROP, n_zones=N_ZONES,
        ground_samples=GROUND_SAMPLES,
        auto_region=True,        # resolve the region from coordinates
        use_soilgrids=False,     # set True outside India (needs network)
        grid_cell_m=GRID_CELL_M,
        soc_method="published",  # exposed-soil composite cascade
    )
    save_outputs(result, out_dir="./cropgen_output")
    print("\nDone. Check ./cropgen_output/")
