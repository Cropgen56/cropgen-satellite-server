"""
Layer rendering for the API.

Every layer comes out the same way:
    * a PNG in EPSG:4326 whose bounds are exactly the AOI bounds, so a web map
      can drop it straight onto the basemap with those four numbers
    * transparent everywhere outside the AOI, with an anti-aliased edge
    * a structured legend the frontend can draw itself

Why EPSG:4326 and not the working UTM grid: a UTM grid is rotated against
lat/lon by the meridian convergence (about 0.7 degrees for Washim in zone 43N).
An image rendered in UTM and placed with lat/lon corners is skewed by that much.

Fills are coloured in NumPy. Matplotlib is used only for lines (contours,
drainage, streamlines), and only through the object API — pyplot keeps global
state and is not safe across request threads.
"""
from __future__ import annotations

import base64
import io
import math
from dataclasses import dataclass

import numpy as np
from affine import Affine
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import (BoundaryNorm, LightSource, LinearSegmentedColormap,
                               ListedColormap, Normalize, to_hex, to_rgba)
from matplotlib.figure import Figure
from PIL import Image
from rasterio.enums import Resampling
from rasterio.features import geometry_mask
from rasterio.transform import from_bounds as tf_from_bounds
from rasterio.warp import reproject
from scipy.ndimage import gaussian_filter, zoom
from shapely.geometry import mapping

from . import engine as E

RENDER_PX = 900          # long side of the working image before reprojection
MAX_UPSAMPLE = 16
CROP_MARGIN_PX = 4       # grid pixels kept around the AOI so edges reproject cleanly
EDGE_SUPERSAMPLE = 4     # anti-aliasing for the AOI boundary

LAYER_NAMES = {
    "elevation": "Elevation",
    "slope": "Slope",
    "aspect": "Aspect",
    "water_flow": "Water Flow",
    "water_accumulation": "Water Accumulation",
    "wetness": "Wetness",
    "erosion_risk": "Erosion Risk",
}

# ── palettes ─────────────────────────────────────────────────────────────────
CMAP_ELEV = LinearSegmentedColormap.from_list(
    "elev", ["#1a6b3c", "#4caf50", "#c5e1a5", "#fff59d", "#ffb74d", "#e65100", "#6d2f0c"])
CMAP_ACC = LinearSegmentedColormap.from_list(
    "acc", ["#f7fbff", "#c6dbef", "#6baed6", "#2171b5", "#08306b"])
CMAP_TWI = LinearSegmentedColormap.from_list(
    "twi", ["#a1662f", "#dfc27d", "#f5f5f5", "#80cdc1", "#018571", "#003c30"])

SLOPE_CLASSES = [  # (lo, hi, label, fill colour, flow-line colour)
    (0, 1, "Flat", "#1a9850", "#1b7837"),
    (1, 3, "Very gentle", "#91cf60", "#4d9221"),
    (3, 5, "Gentle", "#d9ef8b", "#8c8c00"),
    (5, 8, "Moderate", "#fee08b", "#c77c02"),
    (8, 15, "Strong", "#fc8d59", "#d6604d"),
    (15, None, "Steep", "#d73027", "#a50026"),
]
ASPECT_NAMES = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
ASPECT_COLORS = ["#4575b4", "#74add1", "#abd9e9", "#fee090",
                 "#fdae61", "#f46d43", "#d73027", "#9e6ebd"]
FLAT_COLOR = "#d9d9d9"
EROSION_CLASSES = [("Low", "#2e7d32"), ("Moderate", "#f9a825"), ("High", "#c62828")]

COL_DRAIN = "#00e5ff"
COL_POND = "#ffb300"
COL_CONTOUR = "#3e2723"


# ──────────────────────────────────────────────────────────────────────────────
# Frame: the AOI crop of the working grid, upsampled for drawing
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class Frame:
    r0: int
    r1: int
    c0: int
    c1: int
    k: int

    @property
    def Hc(self): return self.r1 - self.r0 + 1

    @property
    def Wc(self): return self.c1 - self.c0 + 1


def _frame(T: E.Terrain) -> Frame:
    rr, cc = np.where(T.field_mask)
    H, W = T.field_mask.shape
    r0 = max(0, int(rr.min()) - CROP_MARGIN_PX)
    r1 = min(H - 1, int(rr.max()) + CROP_MARGIN_PX)
    c0 = max(0, int(cc.min()) - CROP_MARGIN_PX)
    c1 = min(W - 1, int(cc.max()) + CROP_MARGIN_PX)
    k = int(np.clip(math.ceil(RENDER_PX / max(r1 - r0 + 1, c1 - c0 + 1)), 1, MAX_UPSAMPLE))
    return Frame(r0, r1, c0, c1, k)


def _crop(F: Frame, a: np.ndarray) -> np.ndarray:
    return a[F.r0:F.r1 + 1, F.c0:F.c1 + 1]


def _up(F: Frame, a: np.ndarray) -> np.ndarray:
    """Bilinear upsample of the crop. grid_mode keeps pixel areas aligned with the transform."""
    sub = _crop(F, a).astype("float32")
    if not np.isfinite(sub).all():
        fill = float(np.nanmean(sub)) if np.isfinite(sub).any() else 0.0
        sub = np.where(np.isfinite(sub), sub, fill)
    if F.k == 1:
        return sub
    return zoom(sub, F.k, order=1, grid_mode=True, mode="nearest").astype("float32")


def _field_values(T, a):
    v = a[T.field_mask & np.isfinite(a)]
    if v.size == 0:
        raise E.DataSourceError("Layer has no valid values inside the AOI")
    return v


def _hillshade(F: Frame, T: E.Terrain) -> np.ndarray:
    dem = _up(F, T.dem)
    res = T.grid["res_m"] / F.k
    exag = float(np.clip(30.0 / max(T.relief_m, 0.5), 2.0, 15.0))
    return LightSource(azdeg=315, altdeg=45).hillshade(dem, vert_exag=exag, dx=res, dy=res)


def _shade(rgba: np.ndarray, sh: np.ndarray, strength: float = 0.35) -> np.ndarray:
    rgba[..., :3] *= (1.0 - strength) + strength * sh[..., None]
    return rgba


def _classify(v: np.ndarray, edges: list[float]) -> np.ndarray:
    return np.digitize(v, edges[1:-1], right=False)


def _class_rgba(idx: np.ndarray, colors: list[str]) -> np.ndarray:
    lut = np.array([to_rgba(c) for c in colors], dtype="float32")
    return lut[np.clip(idx, 0, len(colors) - 1)]


def _gradient_stops(cmap, lo, hi, n=5, fmt=lambda x: round(float(x), 2)):
    return [{"value": fmt(lo + (hi - lo) * i / (n - 1)), "color": to_hex(cmap(i / (n - 1)))}
            for i in range(n)]


# ──────────────────────────────────────────────────────────────────────────────
# Line overlays (the only matplotlib use)
# ──────────────────────────────────────────────────────────────────────────────
def _compose(F: Frame, rgba: np.ndarray, draw) -> np.ndarray:
    """Draw line work on top of an RGBA fill. Output pixels map 1:1 onto the fill."""
    h, w = rgba.shape[:2]
    fig = Figure(figsize=(w / 100.0, h / 100.0), dpi=100)
    canvas = FigureCanvasAgg(fig)
    fig.patch.set_alpha(0.0)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.patch.set_alpha(0.0)
    # extent in crop-grid pixel units, so line work can use grid coordinates
    ext = (-0.5, F.Wc - 0.5, F.Hc - 0.5, -0.5)
    ax.imshow(np.clip(rgba, 0, 1), extent=ext, interpolation="nearest", zorder=0)
    ax.set_xlim(ext[0], ext[1])
    ax.set_ylim(ext[2], ext[3])
    draw(ax)
    canvas.draw()
    out = np.asarray(canvas.buffer_rgba())[:h, :w].copy()
    if out.shape[0] != h or out.shape[1] != w:            # guard float rounding of figsize
        pad = np.zeros((h, w, 4), dtype=np.uint8)
        pad[:out.shape[0], :out.shape[1]] = out
        out = pad
    return out


def _upcoords(F: Frame):
    """Crop-grid coordinates of the upsampled pixel centres."""
    xs = (np.arange(F.Wc * F.k) + 0.5) / F.k - 0.5
    ys = (np.arange(F.Hc * F.k) + 0.5) / F.k - 0.5
    return np.meshgrid(xs, ys)


def _draw_drainage(ax, F, T):
    area = _up(F, np.log10(np.clip(T.upslope_m2, 1.0, None)))
    # smooth at roughly one grid cell, or the channel edge zig-zags along the
    # pixel staircase of the working grid
    area = gaussian_filter(area, sigma=0.7 * F.k)
    # Inside a filled hollow the epsilon gradient routes flow in dead-straight
    # lines. Water there pools rather than runs, so no channel is drawn in it.
    area[_up(F, T.ponding_m) >= E.PONDING_MIN_M] = -1.0
    thr = math.log10(E.DRAIN_AREA_M2)
    if float(area.max()) < thr:
        return False
    X, Y = _upcoords(F)
    # Fill the channel as a ribbon. Outlining the threshold with a contour line
    # drew both banks, so a single channel read as a thin closed loop.
    ax.contourf(X, Y, area, levels=[thr, 1e9], colors=[COL_DRAIN], alpha=0.95, zorder=6)
    cs = ax.contour(X, Y, area, levels=[thr], colors=["#0b2b40"], linewidths=0.8, zorder=6)
    del cs
    return True


def _draw_ponding(ax, F, T):
    p = _up(F, T.ponding_m)
    if float(p.max()) < E.PONDING_MIN_M:
        return False
    X, Y = _upcoords(F)
    ax.contourf(X, Y, p, levels=[E.PONDING_MIN_M, 1e6], colors=[COL_POND], alpha=0.8, zorder=5)
    ax.contour(X, Y, p, levels=[E.PONDING_MIN_M], colors=["#6d4c00"], linewidths=1.0, zorder=5)
    return True


def _nice_step(relief: float) -> float:
    for s in (0.1, 0.2, 0.25, 0.5, 1, 2, 2.5, 5, 10, 20, 25, 50):
        if relief / s <= 8:
            return float(s)
    return 100.0


# ──────────────────────────────────────────────────────────────────────────────
# Reprojection to EPSG:4326 + AOI clip
# ──────────────────────────────────────────────────────────────────────────────
def _to_png(F: Frame, T: E.Terrain, rgba_u8: np.ndarray, out_px: int):
    geom = T.grid["field_ll"]
    west, south, east, north = geom.bounds
    lat0 = math.radians((south + north) / 2.0)
    w_m = (east - west) * 111320.0 * math.cos(lat0)
    h_m = (north - south) * 110574.0
    px = max(w_m, h_m) / out_px
    W = max(1, int(round(w_m / px)))
    H = max(1, int(round(h_m / px)))

    src_tf = T.grid["transform"] * Affine.translation(F.c0, F.r0) * Affine.scale(1.0 / F.k)
    dst_tf = tf_from_bounds(west, south, east, north, W, H)

    out = np.zeros((4, H, W), dtype="float32")
    for b in range(4):
        reproject(rgba_u8[..., b].astype("float32"), out[b],
                  src_transform=src_tf, src_crs=T.grid["utm"],
                  dst_transform=dst_tf, dst_crs="EPSG:4326",
                  resampling=Resampling.bilinear)

    s = EDGE_SUPERSAMPLE
    inside = geometry_mask([mapping(geom)], out_shape=(H * s, W * s),
                           transform=tf_from_bounds(west, south, east, north, W * s, H * s),
                           invert=True)
    edge_alpha = inside.reshape(H, s, W, s).mean(axis=(1, 3))
    out[3] *= edge_alpha

    img = np.clip(np.moveaxis(out, 0, -1), 0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(img, "RGBA").save(buf, format="PNG", optimize=True)
    return (base64.b64encode(buf.getvalue()).decode("ascii"),
            {"west": west, "south": south, "east": east, "north": north})


def _u8(rgba_float: np.ndarray) -> np.ndarray:
    return (np.clip(rgba_float, 0, 1) * 255 + 0.5).astype(np.uint8)


# ──────────────────────────────────────────────────────────────────────────────
# Layers — each returns (rgba_u8, legend)
# ──────────────────────────────────────────────────────────────────────────────
def _elevation(F, T, C):
    v = _field_values(T, T.dem)
    lo, hi = float(np.percentile(v, 1)), float(np.percentile(v, 99))
    hi = hi if hi > lo else lo + 0.01
    dem_up = _up(F, T.dem)
    rgba = CMAP_ELEV(Normalize(lo, hi, clip=True)(dem_up)).astype("float32")
    _shade(rgba, _hillshade(F, T))

    step = _nice_step(float(v.max() - v.min()))
    # Levels span the FIELD's range only. Spanning the whole buffer's range put
    # ~280 levels on a flat field and cost 1.7 s per render.
    levels = np.arange(math.floor(v.min() / step) * step,
                       math.ceil(v.max() / step) * step + step / 2, step)

    def draw(ax):
        if len(levels) >= 1:
            X, Y = _upcoords(F)
            ax.contour(X, Y, nan_to(dem_up), levels=levels, colors=[COL_CONTOUR],
                       linewidths=0.9, alpha=0.7, zorder=4)

    legend = {"type": "gradient", "unit": "m", "min": round(lo, 2), "max": round(hi, 2),
              "stops": _gradient_stops(CMAP_ELEV, lo, hi),
              "symbols": [{"label": f"Contour every {step:g} m", "color": COL_CONTOUR,
                           "shape": "line"}]}
    return _compose(F, rgba, draw), legend


def nan_to(a):
    return np.nan_to_num(a, nan=float(np.nanmean(a)) if np.isfinite(a).any() else 0.0)


def _slope_edges():
    return [c[0] for c in SLOPE_CLASSES] + [1e9]


def _slope_items(color_idx=3):
    items = []
    for lo, hi, label, fill, line in SLOPE_CLASSES:
        items.append({"label": label,
                      "range": f"{lo}–{hi} %" if hi is not None else f"> {lo} %",
                      "color": fill if color_idx == 3 else line})
    return items


def _slope(F, T, C):
    s = _up(F, T.slope_pct)
    rgba = _class_rgba(_classify(s, _slope_edges()), [c[3] for c in SLOPE_CLASSES])
    _shade(rgba, _hillshade(F, T), 0.25)
    legend = {"type": "classes", "unit": "%", "items": _slope_items(3)}
    return _u8(rgba), legend


def _aspect(F, T, C):
    a = np.radians(np.nan_to_num(T.aspect, nan=0.0))
    s_up, c_up = _up(F, np.sin(a)), _up(F, np.cos(a))    # interpolate the angle safely
    ang = (np.degrees(np.arctan2(s_up, c_up)) + 360.0) % 360.0
    idx = (np.floor((ang + 22.5) / 45.0).astype(int)) % 8
    rgba = _class_rgba(idx, ASPECT_COLORS)
    flat = _up(F, T.slope_pct) < E.FLAT_SLOPE_PCT
    rgba[flat] = to_rgba(FLAT_COLOR)
    _shade(rgba, _hillshade(F, T), 0.25)
    items = [{"label": f"Faces {n}", "color": c} for n, c in zip(ASPECT_NAMES, ASPECT_COLORS)]
    items.append({"label": f"Flat (< {E.FLAT_SLOPE_PCT:g} % slope)", "color": FLAT_COLOR})
    return _u8(rgba), {"type": "classes", "items": items}


def _water_flow(F, T, C):
    sh = _hillshade(F, T)
    base = np.empty(sh.shape + (4,), dtype="float32")
    g = 0.72 + 0.26 * sh
    base[..., 0] = g
    base[..., 1] = g
    base[..., 2] = g * 1.02
    base[..., 3] = 1.0

    u, v = _crop(F, T.flow_u), _crop(F, T.flow_v)
    slope = np.nan_to_num(_crop(F, T.slope_pct), nan=0.0)
    acc = np.log10(np.clip(_crop(F, T.upslope_m2), 1.0, None))
    a_lo = float(np.nanpercentile(acc, 5))
    a_hi = max(float(np.nanpercentile(acc, 99)), a_lo + 0.5)
    lw = 0.7 + 2.8 * np.clip((acc - a_lo) / (a_hi - a_lo), 0, 1)
    edges = _slope_edges()
    cmap = ListedColormap([c[4] for c in SLOPE_CLASSES])
    norm = BoundaryNorm(edges, cmap.N)
    density = float(np.clip(max(F.Hc, F.Wc) / 30.0, 1.2, 2.6))

    def draw(ax):
        ax.streamplot(np.arange(F.Wc), np.arange(F.Hc), np.nan_to_num(u), np.nan_to_num(v),
                      density=density, color=slope, cmap=cmap, norm=norm,
                      linewidth=lw, arrowsize=1.3, arrowstyle="-|>",
                      minlength=0.08, zorder=5)
        _draw_drainage(ax, F, T)

    legend = {"type": "classes", "unit": "%",
              "title": "Flow line colour = slope, thickness = volume of water",
              "items": _slope_items(4),
              "symbols": [{"label": "Water flow path (arrow = direction)",
                           "color": "#1b1b1b", "shape": "arrow"},
                          {"label": "Drainage channel", "color": COL_DRAIN, "shape": "fill"}]}
    return _compose(F, base, draw), legend


def _water_accumulation(F, T, C):
    la = np.log10(np.clip(T.upslope_m2, 1.0, None))
    v = _field_values(T, la)
    lo = math.log10(max(T.grid["res_m"] ** 2, 1.0))
    hi = max(float(np.percentile(v, 99)), math.log10(E.DRAIN_AREA_M2) + 0.3)
    rgba = CMAP_ACC(Normalize(lo, hi, clip=True)(_up(F, la))).astype("float32")
    _shade(rgba, _hillshade(F, T), 0.22)
    marks = {}

    def draw(ax):
        marks["drain"] = _draw_drainage(ax, F, T)
        marks["pond"] = _draw_ponding(ax, F, T)

    img = _compose(F, rgba, draw)
    symbols = [{"label": f"Drainage channel (> {E.DRAIN_AREA_M2:g} m² draining in)",
                "color": COL_DRAIN, "shape": "fill"}]
    if marks.get("pond"):
        symbols.append({"label": "Standing water hollow", "color": COL_POND, "shape": "fill"})
    legend = {"type": "gradient", "unit": "m² upslope area", "scale": "log10",
              "min": round(10 ** lo), "max": round(10 ** hi),
              "stops": _gradient_stops(CMAP_ACC, lo, hi, fmt=lambda x: round(10 ** float(x))),
              "labels": {"min": "Little water passes", "max": "Most water passes"},
              "symbols": symbols}
    return img, legend


def _wetness(F, T, C):
    v = _field_values(T, T.twi)
    lo, hi = float(np.percentile(v, 2)), float(np.percentile(v, 98))
    hi = hi if hi > lo else lo + 0.1
    rgba = CMAP_TWI(Normalize(lo, hi, clip=True)(_up(F, T.twi))).astype("float32")
    _shade(rgba, _hillshade(F, T), 0.22)
    marks = {}

    def draw(ax):
        marks["pond"] = _draw_ponding(ax, F, T)

    img = _compose(F, rgba, draw)
    legend = {"type": "gradient", "unit": "TWI", "min": round(lo, 2), "max": round(hi, 2),
              "stops": _gradient_stops(CMAP_TWI, lo, hi),
              "labels": {"min": "Drier — sheds water", "max": "Wetter — collects water"}}
    if marks.get("pond"):
        legend["symbols"] = [{"label": "Standing water hollow", "color": COL_POND,
                              "shape": "fill"}]
    return img, legend


def _erosion(F, T, C):
    risk = E.erosion_risk(T, C)
    b1, b2 = E.EROSION_BREAKS
    idx = _classify(_up(F, risk), [0.0, b1, b2, 1e9])
    rgba = _class_rgba(idx, [c[1] for c in EROSION_CLASSES])
    _shade(rgba, _hillshade(F, T), 0.25)

    def draw(ax):
        _draw_drainage(ax, F, T)

    items = [{"label": "Low", "range": f"< {b1:g}", "color": EROSION_CLASSES[0][1]},
             {"label": "Moderate", "range": f"{b1:g}–{b2:g}", "color": EROSION_CLASSES[1][1]},
             {"label": "High", "range": f"> {b2:g}", "color": EROSION_CLASSES[2][1]}]
    legend = {"type": "classes", "unit": "RUSLE LS × C", "items": items,
              "symbols": [{"label": "Drainage channel", "color": COL_DRAIN, "shape": "fill"}],
              "source": C.source if C else "terrain only"}
    if C and C.scene_date:
        legend["scene_date"] = C.scene_date
    return _compose(F, rgba, draw), legend


_RENDERERS = {
    "elevation": _elevation,
    "slope": _slope,
    "aspect": _aspect,
    "water_flow": _water_flow,
    "water_accumulation": _water_accumulation,
    "wetness": _wetness,
    "erosion_risk": _erosion,
}


def render_layer(layer: str, T: E.Terrain, C: E.Cover | None, out_px: int) -> dict:
    F = _frame(T)
    rgba_u8, legend = _RENDERERS[layer](F, T, C)
    b64, bounds = _to_png(F, T, rgba_u8, out_px)
    warnings = list(T.warnings)
    if layer == "erosion_risk" and (C is None or C.cover is None):
        warnings.append("No usable satellite pass in the date range; erosion assumes "
                        "bare soil and reflects terrain alone.")
    return {"layer": layer, "name": LAYER_NAMES[layer], "image_base64": b64,
            "bounds": bounds, "legend": legend, "warnings": warnings or None}
