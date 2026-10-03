"""
CropGen terrain engine.

Computes terrain, hydrology and crop cover for one AOI. Plain functions only —
no web framework and no pyplot, so it is safe to call from request threads.

Data
----
    Elevation : Copernicus DEM GLO-30 (AWS Open Data) -> Planetary Computer fallbacks
    Cover     : Sentinel-2 L2A (AWS Element84 / Planetary Computer)
                -> Sentinel-1 RTC/GRD (Planetary Computer) when cloud covers the field

Only the erosion layer needs crop cover. Every other layer is terrain alone, so
those never touch the satellite imagery search — which is where almost all of
the old notebook's runtime went.
"""
from __future__ import annotations

import hashlib
import heapq
import logging
import math
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date, datetime

import numpy as np
import rasterio  # noqa: F401  (opened indirectly through rasterio.open)
import shapely
from affine import Affine
from pyproj import CRS, Transformer
from pystac_client import Client
from rasterio.enums import Resampling
from rasterio.features import geometry_mask
from rasterio.warp import reproject
from rasterio.windows import Window, from_bounds
from rasterio.windows import transform as win_transform
from scipy.ndimage import gaussian_filter
from shapely.geometry import mapping, shape
from shapely.ops import transform as shp_transform
from shapely.validation import explain_validity

try:
    import planetary_computer as pc
    HAS_PC = True
except ImportError:  # the service still runs on AWS sources alone
    pc = None
    HAS_PC = False

log = logging.getLogger("terrain.engine")

# Public COG access — no credentials involved
for _k, _v in {
    "CPL_VSIL_CURL_USE_HEAD": "FALSE",
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif,.tiff,.TIF,.TIFF",
    "AWS_NO_SIGN_REQUEST": "YES",
    "GDAL_HTTP_MULTIRANGE": "YES",
    "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",
    "GDAL_CACHEMAX": "512",
    "CPL_VSIL_CURL_CHUNK_SIZE": "65536",
    "GDAL_HTTP_TIMEOUT": "30",
    "GDAL_HTTP_MAX_RETRY": "3",
    "GDAL_HTTP_RETRY_DELAY": "1",
}.items():
    os.environ.setdefault(_k, _v)

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────
EARTH_SEARCH_URL = "https://earth-search.aws.element84.com/v1"
PLANETARY_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
COP_DEM_AWS_BASE = "https://copernicus-dem-30m.s3.amazonaws.com"

TARGET_GRID_PX = 240          # working grid across the buffered AOI
RES_MIN_M, RES_MAX_M = 2.0, 10.0
BUFFER_MIN_M, BUFFER_MAX_M, BUFFER_FACTOR = 150.0, 400.0, 2.0
DEM_SMOOTH_SIGMA = 1.2
FLOW_SMOOTH_SIGMA = 1.5
MFD_EXPONENT = 1.1

FLAT_RELIEF_M = 2.0           # below this a 30 m DSM cannot resolve slope
MIN_NATIVE_PX = 25            # below this many native 30 m pixels, inside detail is interpolation

# Scene search
S2_MAX_CANDIDATES = 8
S1_MAX_CANDIDATES = 4
MIN_FIELD_COVERAGE = 0.60
IO_WORKERS = 4
S2_SMOOTH_SIGMA = 1.5
S1_SPECKLE_SIGMA = 2.0
# SCL: 2 dark area, 4 vegetation, 5 bare soil, 6 water, 7 unclassified are usable.
# 0 no-data and 1 saturated were previously counted as clear because NaN was
# converted to 0 before the check; listing the GOOD classes avoids that trap.
SCL_GOOD = (2, 4, 5, 6, 7)

# Cover index endpoints
NDVI_BARE, NDVI_FULL = 0.15, 0.80
RVI_BARE, RVI_FULL = 0.20, 0.75

# Absolute thresholds. The notebook used within-field percentiles, which put
# "High" erosion on 18 % of EVERY field and drainage channels on 4 % of every
# field — including a flat plateau top with no runoff at all. These are fixed
# physical thresholds instead. They are sensible starting points, not
# calibrated values; tune them against fields your agronomists know.
EROSION_BREAKS = (0.5, 1.5)   # RUSLE LS x C:  <0.5 Low, 0.5-1.5 Moderate, >1.5 High
DRAIN_AREA_M2 = 3000.0        # upslope area at which runoff concentrates into a channel
PONDING_MIN_M = 0.15          # closed hollow deeper than this holds standing water
FLAT_SLOPE_PCT = 1.0          # below this, aspect is meaningless

_D8_OFF = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
_MFD_W = [0.354, 0.5, 0.354, 0.5, 0.5, 0.354, 0.5, 0.354]   # Quinn et al. 1991


class AOIError(ValueError):
    """The AOI supplied by the caller cannot be processed."""


class DataSourceError(RuntimeError):
    """An upstream data provider could not supply what was needed."""


# ──────────────────────────────────────────────────────────────────────────────
# AOI validation
# ──────────────────────────────────────────────────────────────────────────────
def utm_crs_for(lon: float, lat: float) -> CRS:
    zone = int((lon + 180) // 6) + 1
    return CRS.from_epsg((32600 if lat >= 0 else 32700) + zone)


def to_crs(geom, src_crs, dst_crs):
    t = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    return shp_transform(lambda x, y, z=None: t.transform(x, y), geom)


def _n_vertices(geom) -> int:
    polys = [geom] if geom.geom_type == "Polygon" else list(geom.geoms)
    return sum(len(p.exterior.coords) + sum(len(i.coords) for i in p.interiors)
               for p in polys)


def parse_aoi(aoi: dict, *, min_area_ha: float, max_area_ha: float,
              max_extent_m: float, max_vertices: int = 5000):
    """Validate a GeoJSON AOI. Returns (geometry in EPSG:4326, area in ha)."""
    if not isinstance(aoi, dict):
        raise AOIError("aoi must be a GeoJSON object")
    g = aoi.get("geometry") if aoi.get("type") == "Feature" else aoi
    if not isinstance(g, dict) or g.get("type") not in ("Polygon", "MultiPolygon"):
        raise AOIError("aoi must be a GeoJSON Polygon or MultiPolygon, "
                       "or a Feature wrapping one")
    try:
        geom = shape(g)
    except Exception as exc:
        raise AOIError("aoi coordinates are malformed") from exc
    if geom.is_empty:
        raise AOIError("aoi is empty")
    if _n_vertices(geom) > max_vertices:
        raise AOIError(f"aoi has more than {max_vertices} vertices")
    if not geom.is_valid:
        raise AOIError(f"aoi geometry is invalid: {explain_validity(geom)}")

    minx, miny, maxx, maxy = geom.bounds
    if not (-180 <= minx <= 180 and -180 <= maxx <= 180):
        raise AOIError("longitude out of range; coordinates must be [lon, lat] in EPSG:4326")
    if not (-84 <= miny <= 84 and -84 <= maxy <= 84):
        raise AOIError("latitude out of range; coordinates must be [lon, lat] "
                       "in EPSG:4326, between -84 and 84")

    c = geom.centroid
    g_utm = to_crs(geom, "EPSG:4326", utm_crs_for(c.x, c.y))
    area_ha = g_utm.area / 10000.0
    if area_ha < min_area_ha:
        raise AOIError(f"aoi is {area_ha:.3f} ha; minimum is {min_area_ha} ha")
    if area_ha > max_area_ha:
        raise AOIError(f"aoi is {area_ha:.1f} ha; maximum is {max_area_ha} ha")
    bx0, by0, bx1, by1 = g_utm.bounds
    if max(bx1 - bx0, by1 - by0) > max_extent_m:
        raise AOIError(f"aoi spans more than {max_extent_m:.0f} m; "
                       "split distant parcels into separate requests")
    return geom, area_ha


def aoi_key(geom) -> str:
    """Stable cache key: the same field sent twice hashes the same."""
    g = shapely.normalize(shapely.set_precision(geom, 1e-7))
    return hashlib.sha256(g.wkb).hexdigest()


# ──────────────────────────────────────────────────────────────────────────────
# Grid
# ──────────────────────────────────────────────────────────────────────────────
def build_grid(geom_ll) -> dict:
    """
    UTM grid over the AOI plus a buffer. The buffer scales with the field so a
    small plot still has real upslope terrain for water to run in from.
    """
    c = geom_ll.centroid
    utm = utm_crs_for(c.x, c.y)
    field_utm = to_crs(geom_ll, "EPSG:4326", utm)

    fx0, fy0, fx1, fy1 = field_utm.bounds
    span = max(fx1 - fx0, fy1 - fy0)
    buffer_m = float(np.clip(BUFFER_FACTOR * span, BUFFER_MIN_M, BUFFER_MAX_M))
    buf_utm = field_utm.buffer(buffer_m)

    res = round(float(np.clip((span + 2 * buffer_m) / TARGET_GRID_PX,
                              RES_MIN_M, RES_MAX_M)), 1)
    x0, y0, x1, y1 = buf_utm.bounds
    x0 = math.floor(x0 / res) * res
    y0 = math.floor(y0 / res) * res
    x1 = math.ceil(x1 / res) * res
    y1 = math.ceil(y1 / res) * res
    W = int(round((x1 - x0) / res))
    H = int(round((y1 - y0) / res))

    return {"utm": utm, "field_utm": field_utm, "buf_utm": buf_utm,
            "field_ll": geom_ll, "buf_ll": to_crs(buf_utm, utm, "EPSG:4326"),
            "transform": Affine.translation(x0, y1) * Affine.scale(res, -res),
            "H": H, "W": W, "res_m": res, "buffer_m": buffer_m}


# ──────────────────────────────────────────────────────────────────────────────
# COG reading
# ──────────────────────────────────────────────────────────────────────────────
def _s3_to_https(href: str) -> str:
    if href.startswith("s3://"):
        bucket, key = href[5:].split("/", 1)
        return f"https://{bucket}.s3.amazonaws.com/{key}"
    return href


def _prefer_https(asset) -> str | None:
    if asset is None:
        return None
    href = getattr(asset, "href", "") or ""
    alt = (getattr(asset, "extra_fields", {}) or {}).get("alternate", {}) or {}
    for k in ("https", "http"):
        v = alt.get(k)
        if isinstance(v, dict) and str(v.get("href", "")).startswith("http"):
            return v["href"]
    return href if href.startswith("http") else (_s3_to_https(href) if href else None)


def _read_cog(url: str, grid: dict, *, resamp=Resampling.bilinear,
              nodata: float | None = None, pad_px: int = 2) -> np.ndarray | None:
    """Windowed read of one COG band, reprojected onto the working grid."""
    with rasterio.open(url) as src:
        g = to_crs(grid["buf_utm"], grid["utm"], src.crs)
        pad = pad_px * max(abs(src.transform.a), abs(src.transform.e))
        b = g.bounds
        win = from_bounds(b[0] - pad, b[1] - pad, b[2] + pad, b[3] + pad, src.transform)
        win = win.round_offsets().round_lengths().intersection(
            Window(0, 0, src.width, src.height))
        if win.width <= 0 or win.height <= 0:
            return None
        fill = np.nan if nodata is None else nodata
        # cast BEFORE filling: integer bands (SCL, DN) cannot hold NaN
        arr = src.read(1, window=win, masked=True).astype("float32").filled(fill)
        dst = np.full((grid["H"], grid["W"]), np.nan, "float32")
        reproject(arr, dst, src_transform=win_transform(win, src.transform),
                  src_crs=src.crs, dst_transform=grid["transform"], dst_crs=grid["utm"],
                  src_nodata=nodata, dst_nodata=np.nan, resampling=resamp)
        return dst


def nan_smooth(arr: np.ndarray, sigma: float) -> np.ndarray:
    valid = np.isfinite(arr)
    if sigma <= 0 or not valid.any():
        return arr
    num = gaussian_filter(np.where(valid, arr, 0.0).astype("float32"), sigma)
    wgt = gaussian_filter(valid.astype("float32"), sigma)
    out = np.where(wgt > 1e-6, num / np.maximum(wgt, 1e-6), np.nan).astype("float32")
    return np.where(valid, out, np.nan)


def _safe_div(a, b):
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(np.abs(b) > 1e-9, a / b, np.nan).astype("float32")


# ──────────────────────────────────────────────────────────────────────────────
# DEM
# ──────────────────────────────────────────────────────────────────────────────
def _cop_tile_urls(bounds_ll) -> list[str]:
    x0, y0, x1, y1 = bounds_ll
    urls = []
    for lat in range(math.floor(y0), math.floor(y1) + 1):
        for lon in range(math.floor(x0), math.floor(x1) + 1):
            ns = "N" if lat >= 0 else "S"
            ew = "E" if lon >= 0 else "W"
            n = f"Copernicus_DSM_COG_10_{ns}{abs(lat):02d}_00_{ew}{abs(lon):03d}_00_DEM"
            urls.append(f"{COP_DEM_AWS_BASE}/{n}/{n}.tif")
    return urls


def _clean_dem(a):
    if a is None:
        return None
    a = a.copy()
    a[(a < -1000) | (a > 9000)] = np.nan
    return a


def _mosaic(parts):
    out = None
    for a in parts:
        if a is None:
            continue
        out = a if out is None else np.where(np.isfinite(out), out, a)
    return out


def fetch_dem(grid: dict) -> tuple[np.ndarray, str, float]:
    tiles = _cop_tile_urls(grid["buf_ll"].bounds)
    parts = []
    for url in tiles:
        try:
            parts.append(_clean_dem(_read_cog(url, grid)))
        except Exception as exc:
            log.warning("AWS DEM tile %s failed: %s", url.rsplit("/", 1)[-1], exc)
    m = _mosaic(parts)
    if m is not None and np.isfinite(m).mean() > 0.5:
        return m, "Copernicus DEM GLO-30", 30.0

    if HAS_PC:
        cat = Client.open(PLANETARY_URL, modifier=pc.sign_inplace)
        aoi = mapping(grid["buf_ll"])
        for coll, key, native, label in [
            ("cop-dem-glo-30", "data", 30.0, "Copernicus DEM GLO-30"),
            ("nasadem", "elevation", 30.0, "NASADEM"),
            ("alos-dem", "data", 30.0, "ALOS World 3D-30m"),
        ]:
            try:
                items = list(cat.search(collections=[coll], intersects=aoi, limit=8).items())
                parts = [_clean_dem(_read_cog(it.assets[key].href, grid))
                         for it in items if key in it.assets]
            except Exception as exc:
                log.warning("PC DEM %s failed: %s", coll, exc)
                continue
            m = _mosaic(parts)
            if m is not None and np.isfinite(m).mean() > 0.5:
                return m, label, native
    raise DataSourceError("No elevation model covers this AOI")


# ──────────────────────────────────────────────────────────────────────────────
# Terrain derivatives
# ──────────────────────────────────────────────────────────────────────────────
def _neighbours(z):
    p = np.pad(z, 1, mode="edge")
    H, W = z.shape
    return [p[r:r + H, c:c + W] for r in range(3) for c in range(3)]


def terrain_derivatives(dem: np.ndarray, cell: float) -> dict:
    Z1, Z2, Z3, Z4, _, Z6, Z7, Z8, Z9 = _neighbours(dem)
    dzdx = ((Z3 + 2 * Z6 + Z9) - (Z1 + 2 * Z4 + Z7)) / (8.0 * cell)   # Horn 1981
    dzdy = ((Z7 + 2 * Z8 + Z9) - (Z1 + 2 * Z2 + Z3)) / (8.0 * cell)
    slope_rad = np.arctan(np.hypot(dzdx, dzdy))
    # downhill vector (east, north) = (-dzdx, +dzdy); row index grows southward
    aspect = np.degrees(np.arctan2(-dzdx, dzdy)).astype("float32")
    aspect = np.where(aspect < 0, aspect + 360.0, aspect)
    return {"slope_deg": np.degrees(slope_rad).astype("float32"),
            "slope_pct": (np.tan(slope_rad) * 100.0).astype("float32"),
            "aspect": aspect.astype("float32")}


# ──────────────────────────────────────────────────────────────────────────────
# Hydrology
#
# Both loops run over flat Python lists. Indexing a NumPy array one element at a
# time from Python costs roughly ten times a list lookup, and these loops touch
# every cell eight times.
# ──────────────────────────────────────────────────────────────────────────────
def fill_depressions(dem: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    """Priority-flood with epsilon (Barnes et al. 2014): every cell drains somewhere."""
    H, W = dem.shape
    valid = np.isfinite(dem)
    # seeds = valid cells on the grid edge or next to no-data (vectorised)
    pad = np.pad(~valid, 1, constant_values=True)
    touches_edge = np.zeros_like(valid)
    for dr, dc in _D8_OFF:
        touches_edge |= pad[1 + dr:1 + dr + H, 1 + dc:1 + dc + W]
    seeds = np.flatnonzero(valid & touches_edge)

    z = np.where(valid, dem, np.inf).astype("float64").ravel().tolist()
    closed = (~valid).ravel().tolist()
    heap = [(z[i], int(i)) for i in seeds]
    heapq.heapify(heap)
    for i in seeds:
        closed[int(i)] = True

    while heap:
        zi, i = heapq.heappop(heap)
        r, c = divmod(i, W)
        for dr, dc in _D8_OFF:
            rr, cc = r + dr, c + dc
            if 0 <= rr < H and 0 <= cc < W:
                j = rr * W + cc
                if not closed[j]:
                    closed[j] = True
                    zj = z[j]
                    if zj <= zi:
                        zj = zi + eps
                        z[j] = zj
                    heapq.heappush(heap, (zj, j))

    out = np.array(z, dtype="float64").reshape(H, W)
    out[~valid] = np.nan
    return out.astype("float32")


def mfd_accumulation(filled: np.ndarray, cell: float, p: float = MFD_EXPONENT) -> np.ndarray:
    """
    Multiple-flow-direction accumulation (Quinn et al. 1991), in cells.

    D8 sends each cell's water to a single neighbour, which on gentle ground
    draws parallel stripes that look like channels but are grid artefacts. MFD
    shares flow among all downslope neighbours in proportion to slope.
    """
    H, W = filled.shape
    valid = np.isfinite(filled)
    n = int(valid.sum())
    z = np.where(valid, filled, np.inf).astype("float64").ravel().tolist()
    ok = valid.ravel().tolist()
    acc = [1.0] * (H * W)
    nb = [(dr, dc, dr * W + dc, 1.0 / (math.hypot(dr, dc) * cell), w)
          for (dr, dc), w in zip(_D8_OFF, _MFD_W)]
    order = np.argsort(np.where(valid, -filled, np.inf), axis=None, kind="stable")[:n]

    for i in order.tolist():
        r, c = divmod(i, W)
        zi = z[i]
        tgt = []
        tot = 0.0
        for dr, dc, di, invd, cw in nb:
            rr, cc = r + dr, c + dc
            if 0 <= rr < H and 0 <= cc < W:
                j = i + di
                if ok[j]:
                    s = (zi - z[j]) * invd
                    if s > 0.0:
                        w = (s ** p) * cw
                        tgt.append((j, w))
                        tot += w
        if tot > 0.0:
            a = acc[i] / tot
            for j, w in tgt:
                acc[j] += a * w

    out = np.array(acc, dtype="float32").reshape(H, W)
    out[~valid] = np.nan
    return out


def flow_vectors(filled: np.ndarray, cell: float) -> tuple[np.ndarray, np.ndarray]:
    """Continuous downslope field for streamlines: +u east (column), +v down (row)."""
    z = nan_smooth(filled, FLOW_SMOOTH_SIGMA)
    z = np.nan_to_num(z, nan=float(np.nanmean(filled)))
    dz_dr, dz_dc = np.gradient(z, cell, cell)
    return (-dz_dc).astype("float32"), (-dz_dr).astype("float32")


def wetness_index(acc_cells, slope_deg, cell):
    sca = np.clip(acc_cells, 1.0, None) * cell            # specific catchment area, m
    tanb = np.tan(np.radians(np.clip(slope_deg, 0.1, None)))
    return np.log(sca / tanb).astype("float32")


def ls_factor(acc_cells, slope_deg, cell, m=0.4, n=1.3):
    """RUSLE topographic factor (Moore & Burch 1986)."""
    sca = np.clip(acc_cells, 1.0, None) * cell
    beta = np.radians(np.clip(slope_deg, 0.05, None))
    return ((sca / 22.13) ** m * (np.sin(beta) / 0.0896) ** n).astype("float32")


def cover_factor(cover01, alpha=2.0, beta=1.0):
    """RUSLE C-factor (Van der Knijff 2000) from a 0..1 cover fraction."""
    nd = np.clip(NDVI_BARE + np.clip(cover01, 0, 1) * (NDVI_FULL - NDVI_BARE), -0.95, 0.95)
    return np.clip(np.exp(-alpha * nd / (beta - nd)), 0.001, 1.0).astype("float32")


# ──────────────────────────────────────────────────────────────────────────────
# Terrain bundle
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class Terrain:
    grid: dict
    field_mask: np.ndarray
    dem: np.ndarray
    slope_pct: np.ndarray
    slope_deg: np.ndarray
    aspect: np.ndarray
    upslope_m2: np.ndarray
    twi: np.ndarray
    ls: np.ndarray
    ponding_m: np.ndarray
    flow_u: np.ndarray
    flow_v: np.ndarray
    relief_m: float
    native_px: float
    dem_source: str

    @property
    def warnings(self) -> list[str]:
        w = []
        if self.native_px < MIN_NATIVE_PX:
            w.append(f"Field covers about {self.native_px:.0f} native 30 m elevation pixels; "
                     "patterns inside the boundary are interpolated. Overall flow "
                     "direction and which edge is high are reliable; small patches are not.")
        if self.relief_m < FLAT_RELIEF_M:
            w.append(f"Field relief is {self.relief_m:.2f} m, within the vertical error of "
                     "a 30 m elevation model. Treat slope and flow as indicative only.")
        return w


def compute_terrain(geom_ll) -> Terrain:
    grid = build_grid(geom_ll)
    dem_raw, source, _native = fetch_dem(grid)
    dem = nan_smooth(dem_raw, DEM_SMOOTH_SIGMA)

    mask = geometry_mask([mapping(grid["field_utm"])], out_shape=(grid["H"], grid["W"]),
                         transform=grid["transform"], invert=True)
    mask &= np.isfinite(dem)
    if not mask.any():
        raise DataSourceError("The elevation model has no valid pixels inside this AOI")

    res = grid["res_m"]
    d = terrain_derivatives(dem, res)
    filled = fill_depressions(dem)
    acc = mfd_accumulation(filled, res)
    u, v = flow_vectors(filled, res)
    ef = dem[mask]
    return Terrain(
        grid=grid, field_mask=mask, dem=dem,
        slope_pct=d["slope_pct"], slope_deg=d["slope_deg"], aspect=d["aspect"],
        upslope_m2=(acc * res * res).astype("float32"),
        twi=wetness_index(acc, d["slope_deg"], res),
        ls=ls_factor(acc, d["slope_deg"], res),
        ponding_m=np.where(np.isfinite(dem), np.clip(filled - dem, 0, None),
                           np.nan).astype("float32"),
        flow_u=u, flow_v=v,
        relief_m=float(ef.max() - ef.min()),
        native_px=float(grid["field_utm"].area / 900.0),
        dem_source=source,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Crop cover (erosion layer only)
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class Cover:
    cover: np.ndarray | None      # 0 bare .. 1 closed canopy
    source: str
    scene_date: str | None


def _search(catalog, collection, aoi, start, end, limit):
    try:
        kw = {"modifier": pc.sign_inplace} if (catalog == PLANETARY_URL and HAS_PC) else {}
        return list(Client.open(catalog, **kw).search(
            collections=[collection], intersects=aoi,
            datetime=f"{start}/{end}", limit=limit).items())
    except Exception as exc:
        log.warning("STAC %s/%s failed: %s", catalog.split("/")[2], collection, exc)
        return []


def _item_date(it) -> str:
    return (it.properties.get("datetime") or it.properties.get("start_datetime") or "")[:10]


def _scale_offset(item, asset) -> tuple[float, float]:
    """
    DN -> reflectance. Since processing baseline 04.00 (25 Jan 2022) ESA adds a
    +1000 offset to L2A digital numbers. Dividing by 10000 without removing it
    inflates both red and NIR, which drags NDVI down and makes every field look
    barer than it is — and the erosion map redder than it should be.
    """
    rb = (getattr(asset, "extra_fields", {}) or {}).get("raster:bands") or []
    if rb and isinstance(rb[0], dict) and rb[0].get("scale") is not None:
        return float(rb[0]["scale"]), float(rb[0].get("offset") or 0.0)
    pb = item.properties.get("s2:processing_baseline")
    try:
        if pb is not None and float(pb) >= 4.0:
            return 1e-4, -0.1
    except (TypeError, ValueError):
        pass
    return 1e-4, 0.0


def _scl_coverage(item, grid, mask) -> tuple[float, np.ndarray | None]:
    a = item.assets.get("scl") or item.assets.get("SCL")
    url = _prefer_https(a)
    if not url:
        return -1.0, None
    try:
        scl = _read_cog(url, grid, resamp=Resampling.nearest)
    except Exception as exc:
        log.warning("SCL read %s failed: %s", _item_date(item), exc)
        return -1.0, None
    if scl is None:
        return -1.0, None
    good = np.isin(np.nan_to_num(scl, nan=0).astype("int16"), SCL_GOOD)
    return float(good[mask].mean()), good


def _read_ndvi(item, grid, good: np.ndarray | None) -> np.ndarray | None:
    bands = {}
    for band, keys in (("red", ("red", "B04")), ("nir", ("nir", "B08"))):
        asset = next((item.assets[k] for k in keys if k in item.assets), None)
        url = _prefer_https(asset)
        if not url:
            return None
        dn = _read_cog(url, grid, nodata=0.0)
        if dn is None:
            return None
        scale, offset = _scale_offset(item, asset)
        bands[band] = dn * scale + offset
    ndvi = _safe_div(bands["nir"] - bands["red"], bands["nir"] + bands["red"])
    if good is not None:
        ndvi[~good] = np.nan
    return nan_smooth(ndvi, S2_SMOOTH_SIGMA)


def _s2_cover(aoi, start, end, grid, mask):
    cands, seen = [], set()
    for url in (EARTH_SEARCH_URL, PLANETARY_URL):
        if url == PLANETARY_URL and not HAS_PC:
            continue
        for it in _search(url, "sentinel-2-l2a", aoi, start, end, 20):
            key = _item_date(it)            # both catalogues list the same pass
            if key and key not in seen:
                seen.add(key)
                cands.append(it)
    # least cloudy first, most recent breaks ties
    cands.sort(key=lambda it: (it.properties.get("eo:cloud_cover", 100),
                               -datetime.strptime(_item_date(it), "%Y-%m-%d").toordinal()))
    cands = cands[:S2_MAX_CANDIDATES]
    if not cands:
        return None, None, -1.0

    # The tile cloud % describes 110 km, not this field. Check the field itself
    # with the small SCL band — in parallel — before paying for red and NIR.
    with ThreadPoolExecutor(max_workers=IO_WORKERS) as ex:
        results = list(ex.map(lambda it: _scl_coverage(it, grid, mask), cands))

    rows = [(it, cov, good) for it, (cov, good) in zip(cands, results)]
    chosen = next((r for r in rows if r[1] >= MIN_FIELD_COVERAGE),
                  max(rows, key=lambda r: r[1]))
    it, cov, good = chosen
    if cov <= 0:
        return None, None, cov
    try:
        return _read_ndvi(it, grid, good), _item_date(it), cov
    except Exception as exc:
        log.warning("S2 band read %s failed: %s", _item_date(it), exc)
        return None, None, -1.0


def _s1_rvi(item, grid, calibrated) -> np.ndarray | None:
    vv_url = _prefer_https(item.assets.get("vv"))
    vh_url = _prefer_https(item.assets.get("vh"))
    if not vv_url or not vh_url:
        return None
    vv = _read_cog(vv_url, grid, nodata=0.0)
    vh = _read_cog(vh_url, grid, nodata=0.0)
    if vv is None or vh is None:
        return None
    if not calibrated:              # GRD is DN amplitude; power is its square
        vv, vh = vv ** 2, vh ** 2
    vv = nan_smooth(np.where(vv > 0, vv, np.nan), S1_SPECKLE_SIGMA)
    vh = nan_smooth(np.where(vh > 0, vh, np.nan), S1_SPECKLE_SIGMA)
    return np.clip(_safe_div(4.0 * vh, vv + vh), 0, 1).astype("float32")


def _s1_cover(aoi, start, end, grid, mask):
    """
    Radar sees through cloud. RTC is terrain-flattened and calibrated. GRD is
    uncalibrated DN, which still gives a usable within-scene RVI ratio.
    Element84's Sentinel-1 GRD sits in a requester-pays bucket, so it is not
    tried here — it would need AWS credentials and billing.
    """
    if not HAS_PC:
        return None, None, None
    for coll, calib, label in (("sentinel-1-rtc", True, "Sentinel-1 radar (RTC)"),
                               ("sentinel-1-grd", False, "Sentinel-1 radar (GRD)")):
        items = _search(PLANETARY_URL, coll, aoi, start, end, 12)
        items.sort(key=lambda it: _item_date(it), reverse=True)
        for it in items[:S1_MAX_CANDIDATES]:
            try:
                rvi = _s1_rvi(it, grid, calib)
            except Exception as exc:
                log.warning("S1 %s read failed: %s", coll, exc)
                continue
            if rvi is not None and np.isfinite(rvi[mask]).mean() >= MIN_FIELD_COVERAGE:
                return rvi, _item_date(it), label
    return None, None, None


def _rescale(a, lo, hi):
    return np.clip((a - lo) / (hi - lo), 0.0, 1.0).astype("float32")


def compute_cover(T: Terrain, start: date, end: date) -> Cover:
    aoi = mapping(T.grid["buf_ll"])
    s, e = start.isoformat(), end.isoformat()
    mask = T.field_mask

    ndvi, s2_date, s2_cov = _s2_cover(aoi, s, e, T.grid, mask)
    fc_opt = None
    if ndvi is not None:
        fc_opt = np.where(np.isfinite(ndvi), _rescale(ndvi, NDVI_BARE, NDVI_FULL), np.nan)

    if fc_opt is not None and np.isfinite(fc_opt[mask]).mean() >= MIN_FIELD_COVERAGE:
        return Cover(fc_opt, "Sentinel-2 optical", s2_date)

    rvi, s1_date, s1_label = _s1_cover(aoi, s, e, T.grid, mask)
    if rvi is not None:
        fc_rad = _rescale(rvi, RVI_BARE, RVI_FULL)
        if fc_opt is not None and np.isfinite(fc_opt[mask]).any():
            merged = np.where(np.isfinite(fc_opt), fc_opt, fc_rad).astype("float32")
            return Cover(merged, f"Sentinel-2 optical + {s1_label}",
                         f"{s2_date} / {s1_date}")
        return Cover(fc_rad, s1_label, s1_date)

    if fc_opt is not None and np.isfinite(fc_opt[mask]).any():
        return Cover(fc_opt, "Sentinel-2 optical (partly clouded)", s2_date)
    return Cover(None, "terrain only — no usable satellite pass in the date range", None)


def erosion_risk(T: Terrain, C: Cover | None) -> np.ndarray:
    """Risk index = RUSLE LS x C. Terrain-only when cover is unavailable."""
    if C is None or C.cover is None or not np.isfinite(C.cover[T.field_mask]).any():
        risk = T.ls * 0.70     # C for bare soil at NDVI_BARE: the conservative assumption
    else:
        c = np.where(np.isfinite(C.cover), C.cover,
                     float(np.nanmedian(C.cover[T.field_mask])))
        risk = T.ls * cover_factor(c)
    return nan_smooth(risk.astype("float32"), 1.5)
