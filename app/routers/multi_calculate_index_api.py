"""Calculate a satellite index for many fields at once (GeoJSON FeatureCollection).

Every field goes through the same steps as the single-field /calculate/index
(the same tile reader, cloud-gap fill, SCL masking and renderer), so each
result has the same image, bounds and legend as calling that endpoint for the
field on its own. What changes is where the pixels come from: fields that sit
close together share their network work.

1. Once per request: pick the best scene over the bounding box of all fields
   and list that pass's tiles. Cloud-gap fill scenes are also searched once,
   only if a field needs them.
2. Fields are grouped into ~10 km cells, and cells run in parallel. Per cell:
   keep the tiles that touch it and download each tile band's native pixels
   covering the cell into memory. A cell no tile of that pass touches (fields
   on another orbit) picks a scene for itself.
3. Per field: run the unchanged reader against those in-memory copies.

Downloaded windows are kept in a bounded process-wide cache, so requests for
the same fields and date (the dashboard asks for NDVI and NDMI together, both
of which need B08 and SCL) download each window once.

The one intended difference: all fields use the scene ranked best for their
cell, instead of each field ranking scenes on its own (which could give
neighbouring fields different tiles of the same pass). The request-wide pick
uses STAC metadata (footprint coverage, cloud cover) when the requested day has
scenes; each cell then puts the tile with the most cloud-free pixels over it
first, so overlapping tiles are resolved as utils.pick_best_item would.

TRUE_COLOR and Sentinel-1 requests keep the plain per-field pipeline. A field
that fails (no valid pixels, ...) is reported in `errors` without failing the
rest.
"""
import hashlib
import json
import logging
import math
import os
import threading
import time
from collections import OrderedDict, defaultdict
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, Hashable, List, Optional, Tuple

import numpy as np
import rasterio
import shapely
from fastapi import APIRouter, HTTPException
from rasterio.features import geometry_mask
from rasterio.io import MemoryFile
from rasterio.windows import Window, from_bounds
from shapely.geometry import mapping, shape
from shapely.geometry.base import BaseGeometry

from app.models import CalculateRequest, MultiCalculateRequest, MultiCalculateResponse
from app.routers.calculate_index_api import calculate_index, render_index_result, resolve_index
from app.services import utils
from app.services.feature_collection import ParsedFeature, parse_feature_collection
from app.services.response_cache import TTLCache

router = APIRouter()
log = logging.getLogger("multi.calculate")

# Per-field fallback path: each field reads its own tiles, so keep this modest
# to avoid overloading the STAC/COG providers.
FIELD_WORKERS = 6
# Cells processed at once. Each downloads its windows with up to utils.THREADS
# connections, so this also bounds the load on the COG providers.
GROUP_WORKERS = int(os.getenv("MULTI_GROUP_WORKERS", "4"))
# Per-field work is CPU-bound (in-memory reads + rendering). One pool for all
# cells and requests keeps a single dense village on every core while capping
# CPU threads when several requests run at once.
_RENDER_POOL = ThreadPoolExecutor(max_workers=os.cpu_count() or 4, thread_name_prefix="multi-render")
# Nearby fields (by centroid) share one scene download; ~10 km cells keep each
# in-memory copy to a few MB per band even for batches spread across districts.
GROUP_CELL_DEG = 0.1
# Extra native pixels kept around the cell so resampling at field edges sees
# the same neighbours as a read from the full file.
PREFETCH_PAD_PX = 4
# Same thresholds as the single-field pipeline.
MIN_VALID_FRAC = 0.7
FILL_WINDOW_DAYS = 14
SCL_CLOUD_CLASSES = [8, 9, 10, 11]
# Downloaded windows, per worker process. Scene pixels never change, so the TTL
# only keeps memory from holding windows nobody asks for any more.
PIXEL_CACHE_BYTES = int(os.getenv("MULTI_PIXEL_CACHE_MB", "256")) * 1024 * 1024
PIXEL_CACHE_TTL_SECONDS = 30 * 60
# Whole responses, for repeated clicks on the same date and index. Large
# responses are not kept: they would crowd out everything else.
_RESPONSE_CACHE = TTLCache(ttl_seconds=10 * 60, max_entries=32)
RESPONSE_CACHE_MAX_BYTES = 8 * 1024 * 1024


def _run_one(feature: ParsedFeature, req: MultiCalculateRequest) -> Dict[str, Any]:
    single = CalculateRequest(
        geometry=feature.geometry,
        date=req.date,
        index_name=req.index_name,
        provider=req.provider,
        satellite=req.satellite,
        width=req.width,
        height=req.height,
        supersample=req.supersample,
        smooth=req.smooth,
        gaussian_sigma=req.gaussian_sigma,
    )
    return _outcome(feature, lambda: calculate_index(single))


def _outcome(feature: ParsedFeature, compute: Callable[[], Dict[str, Any]]) -> Dict[str, Any]:
    try:
        out = compute()
        return {"ok": True, "result": {"id": feature.id, "properties": feature.properties, **out}}
    except HTTPException as e:
        return {"ok": False, "status_code": e.status_code, "detail": str(e.detail)}
    except Exception:
        log.exception("multi calculate: field %s failed", feature.id)
        return {"ok": False, "status_code": 500, "detail": "Internal error while processing this field"}


def _bbox(shapes: List[BaseGeometry]) -> BaseGeometry:
    """The envelope of the shapes' union, without building the union."""
    return shapely.box(*shapely.total_bounds(shapes))


# ---------------------------------------------------------------------------
# In-memory copies of the COG windows a group of fields needs
# ---------------------------------------------------------------------------

def _url_key(url: str) -> str:
    # Planetary Computer SAS tokens can be re-issued between calls; the blob path identifies the file.
    return url.split("?", 1)[0]


def _band_urls(item, bands: List[str], want_scl: bool) -> List[str]:
    """The exact URLs utils._read_tile_into_stack / temporal_fill_median will open."""
    assets = item.assets or {}
    urls = []
    for band in bands:
        asset = utils.resolve_asset(assets, band)
        href = utils.prefer_http_from_asset(asset) if asset else None
        if href:
            urls.append(utils.sign_href_if_pc(href))
    scl = assets.get("scl") or assets.get("SCL")
    if want_scl and scl:
        href = utils.prefer_http_from_asset(scl)
        if href:
            urls.append(utils.sign_href_if_pc(href))
    return urls


# (pixels, rasterio profile) of one window of one file
_Window = Tuple[np.ndarray, Dict[str, Any]]


class _PixelCache:
    """Byte-bounded LRU of downloaded windows, shared by every request in the process.

    Concurrent requests for the same window wait for one download instead of
    each fetching it. Failed downloads are not cached.
    """

    def __init__(self, max_bytes: int, ttl_seconds: float):
        self.max_bytes = max_bytes
        self.ttl_seconds = ttl_seconds
        self._data: "OrderedDict[Hashable, Tuple[float, _Window]]" = OrderedDict()
        self._inflight: Dict[Hashable, Future] = {}
        self._bytes = 0
        self._lock = threading.Lock()

    def get_or_fetch(self, key: Hashable, fetch: Callable[[], Optional[_Window]]) -> Optional[_Window]:
        with self._lock:
            hit = self._data.get(key)
            if hit is not None and time.monotonic() - hit[0] <= self.ttl_seconds:
                self._data.move_to_end(key)
                return hit[1]
            pending = self._inflight.get(key)
            if pending is None:
                pending = self._inflight[key] = Future()
                owner = True
            else:
                owner = False
        if not owner:
            return pending.result()
        value = None
        try:
            value = fetch()
        finally:
            with self._lock:
                del self._inflight[key]
                if value is not None:
                    self._store(key, value)
            pending.set_result(value)
        return value

    def _store(self, key: Hashable, value: _Window) -> None:
        old = self._data.pop(key, None)
        if old is not None:
            self._bytes -= old[1][0].nbytes
        size = value[0].nbytes
        if size > self.max_bytes:
            return
        self._data[key] = (time.monotonic(), value)
        self._bytes += size
        while self._bytes > self.max_bytes:
            _, (_, evicted) = self._data.popitem(last=False)
            self._bytes -= evicted[0].nbytes


_PIXELS = _PixelCache(PIXEL_CACHE_BYTES, PIXEL_CACHE_TTL_SECONDS)


def _download_window(url: str, area_ll: Dict[str, Any]) -> Optional[_Window]:
    """Native pixels of `url` covering `area_ll` (plus a small pad), unresampled."""
    try:
        with rasterio.Env(), rasterio.open(url) as ds:
            aoi_sc = utils.aoi_to_scene(area_ll, ds.crs.to_string())
            w = from_bounds(*aoi_sc.bounds, ds.transform)
            c0 = max(0, int(math.floor(w.col_off)) - PREFETCH_PAD_PX)
            r0 = max(0, int(math.floor(w.row_off)) - PREFETCH_PAD_PX)
            c1 = min(ds.width, int(math.ceil(w.col_off + w.width)) + PREFETCH_PAD_PX)
            r1 = min(ds.height, int(math.ceil(w.row_off + w.height)) + PREFETCH_PAD_PX)
            if c1 <= c0 or r1 <= r0:
                return None
            win = Window(c0, r0, c1 - c0, r1 - r0)
            data = ds.read(1, window=win)
            profile = {
                "driver": "GTiff", "count": 1, "dtype": data.dtype,
                "width": data.shape[1], "height": data.shape[0],
                "crs": ds.crs, "transform": ds.window_transform(win), "nodata": ds.nodata,
            }
        return data, profile
    except Exception as e:
        log.warning("multi calculate: window download failed for %s: %s", _url_key(url), e)
        return None  # the reader falls back to the remote file


class _WindowCache:
    """Serves rasterio datasets for whole-file URLs from small in-memory windows.

    Each window is an exact, unresampled pixel subset of the source file (same
    CRS, resolution and pixel alignment, same nodata), so reading a field from
    it gives the same pixels as reading the field from the file. URLs that were
    not prefetched fall through to rasterio.open.
    """

    def __init__(self, area_ll: Dict[str, Any], area_shape: BaseGeometry):
        self.area_ll = area_ll
        self._area_key = tuple(round(b, 7) for b in area_shape.bounds)
        self._files: Dict[str, MemoryFile] = {}
        self._lock = threading.Lock()

    def prefetch(self, urls: List[str]) -> None:
        todo = [u for u in dict.fromkeys(urls) if _url_key(u) not in self._files]
        if not todo:
            return
        with ThreadPoolExecutor(max_workers=min(utils.THREADS, len(todo))) as ex:
            for url, window in zip(todo, ex.map(self._fetch, todo)):
                if window is None:
                    continue
                data, profile = window
                memfile = MemoryFile()
                with memfile.open(**profile) as dst:
                    dst.write(data, 1)
                with self._lock:
                    self._files[_url_key(url)] = memfile

    def _fetch(self, url: str) -> Optional[_Window]:
        return _PIXELS.get_or_fetch(
            (_url_key(url), self._area_key), lambda: _download_window(url, self.area_ll)
        )

    def open(self, url: str):
        memfile = self._files.get(_url_key(url))
        return memfile.open() if memfile is not None else rasterio.open(url)

    def read_cached(self, url: str) -> Optional[np.ndarray]:
        """The prefetched window's pixels, or None if it was not downloaded."""
        memfile = self._files.get(_url_key(url))
        if memfile is None:
            return None
        with memfile.open() as ds:
            return ds.read(1)

    def close(self) -> None:
        for memfile in self._files.values():
            memfile.close()
        self._files.clear()


class _Scene:
    """Scene choice for an area: the best item, the tiles of its pass, and fill scenes."""

    def __init__(self, area_ll: Dict[str, Any], req: MultiCalculateRequest):
        self.req = req
        self.area_ll = area_ll
        self.satellite = (req.satellite or "s2").lower()
        self.search_order = utils.get_provider_search_order(req.provider, prefer_pc_default=True)
        self.prefer_pc = self.search_order[0] == "planetary"
        self._fill_items: Optional[List[Any]] = None
        self._fill_lock = threading.Lock()

        item, has_scl, coll = self._pick_same_day(area_ll)
        if not item:
            item, has_scl, coll = utils.pick_best_item(
                area_ll, req.date, req.date, prefer_pc=self.prefer_pc, satellite=self.satellite
            )
        if not item:
            provider_diag = utils.get_provider_error_summary()
            if provider_diag:
                raise HTTPException(
                    status_code=503,
                    detail=(
                        "No suitable item found for date/AOI. "
                        f"Provider diagnostics: {provider_diag}. "
                        "This may indicate upstream timeout/unavailability."
                    ),
                )
            raise HTTPException(status_code=404, detail="No suitable item found for date/AOI")
        self.item, self.has_scl = item, has_scl
        self.collection = coll or utils.get_collections_for_satellite(self.satellite)[0]
        tiles = utils.items_for_date(
            area_ll, item.properties.get("datetime", ""), self.collection, prefer_pc=self.prefer_pc, limit=6
        ) or [item]
        # Where tiles of the same pass overlap, the first tile read wins. The
        # single-field endpoint takes whichever download finishes first; here
        # the best-ranked scene always goes first, so results are repeatable.
        best = getattr(item, "id", None)
        self.tiles = sorted(tiles, key=lambda t: getattr(t, "id", None) != best)

    def _pick_same_day(self, area_ll: Dict[str, Any]) -> Tuple[Any, bool, Optional[str]]:
        """Best tile among the requested day's scenes, ranked from STAC metadata.

        Every tile of that pass is read anyway (see items_for_date below), so the
        ranking only decides which tile wins where tiles overlap. That does not
        need utils.pick_best_item's per-tile pixel probe (several COG reads per
        candidate). Returns (None, False, None) when the day has no scene, so the
        caller can fall back to pick_best_item's wider date search.
        """
        order = ["planetary", "aws"] if self.prefer_pc else ["aws", "planetary"]
        items = utils.search_stac_items(
            utils.get_collections_for_satellite(self.satellite), area_ll,
            f"{self.req.date}/{self.req.date}", limit=12, search_order=order,
        )
        if not items:
            return None, False, None
        best = _rank_tiles(items, shape(area_ll))[0]
        assets = getattr(best, "assets", {}) or {}
        return best, bool(assets.get("scl") or assets.get("SCL")), utils._item_collection_id(best)

    def tiles_for(self, area_shape: BaseGeometry) -> List[Any]:
        """The tiles a search over `area_shape` alone would have returned."""
        return [it for it in self.tiles if _footprint_intersects(it, area_shape)]

    def fill_items(self) -> List[Any]:
        """Nearby-date scenes over the whole area for cloud-gap fill: searched once, on first need."""
        with self._fill_lock:
            if self._fill_items is None:
                fmt = "%Y-%m-%d"
                start = (datetime.strptime(self.req.date, fmt) - timedelta(days=FILL_WINDOW_DAYS)).strftime(fmt)
                end = (datetime.strptime(self.req.date, fmt) + timedelta(days=FILL_WINDOW_DAYS)).strftime(fmt)
                items = []
                for provider_name in self.search_order:
                    items = utils.search_stac_items(
                        [self.collection], self.area_ll, f"{start}/{end}", limit=12, search_order=[provider_name]
                    )
                    if items:
                        break
                self._fill_items = items
            return self._fill_items


def _rank_tiles(items: List[Any], area_shape: BaseGeometry) -> List[Any]:
    """Most of `area_shape` covered first, then least cloudy (stable for ties)."""
    def rank(it) -> Tuple[float, float]:
        try:
            covered = shape(it.geometry).intersection(area_shape).area if it.geometry else 0.0
        except Exception:
            covered = 0.0
        cloud = (it.properties or {}).get("eo:cloud_cover")
        return -covered, float(cloud) if cloud is not None else 100.0
    return sorted(items, key=rank)


def _footprint_intersects(item, area_shape: BaseGeometry) -> bool:
    geom = getattr(item, "geometry", None)
    if not geom:
        return True  # no footprint: let the reader decide
    try:
        return shape(geom).intersects(area_shape)
    except Exception:
        return True


class _Group:
    """Tiles + in-memory pixels shared by the fields of one cell."""

    def __init__(self, fields: List[ParsedFeature], req: MultiCalculateRequest,
                 required_bands: List[str], scene: _Scene):
        self.req = req
        self.required_bands = required_bands
        area_shape = _bbox([f.shape for f in fields])
        self.area_shape = area_shape
        self.area_ll = mapping(area_shape)
        self.cache = _WindowCache(self.area_ll, area_shape)
        self._fill_items: Optional[List[Any]] = None
        self._fill_lock = threading.Lock()

        tiles = scene.tiles_for(area_shape)
        if not tiles:
            # The request's pass does not reach this cell (fields on another orbit): pick for the cell alone.
            scene = _Scene(self.area_ll, req)
            tiles = scene.tiles
        self.scene, self.has_scl = scene, scene.has_scl
        self.cache.prefetch([u for t in tiles for u in _band_urls(t, required_bands, self.has_scl)])
        self.tiles = self._clearest_first(tiles) if len(tiles) > 1 else tiles
        self.crs = utils.get_item_crs(self.tiles[0])
        if self.crs is None:
            raise HTTPException(status_code=500, detail="Could not determine reference band CRS")

    def _clearest_first(self, tiles: List[Any]) -> List[Any]:
        """Put the tile with the most cloud-free pixels over the cell first.

        Same criterion as utils.pick_best_item's probe, but measured on the
        windows already in memory instead of extra COG reads. Ties keep the
        search order, so results are repeatable.
        """
        def clear_fraction(tile) -> float:
            assets = tile.assets or {}
            scl = assets.get("scl") or assets.get("SCL")
            href = utils.prefer_http_from_asset(scl) if (self.has_scl and scl) else None
            if not href:
                urls = _band_urls(tile, self.required_bands[:1], False)
                href = urls[0] if urls else None
            if not href:
                return 0.0
            a = self.cache.read_cached(utils.sign_href_if_pc(href))
            if a is None:
                return 0.0  # nothing downloaded: the tile has no pixels here
            valid = a != 0
            if scl is not None and self.has_scl:
                valid &= ~np.isin(a, SCL_CLOUD_CLASSES)
            return float(np.count_nonzero(valid)) / max(a.size, 1)

        scores = [clear_fraction(t) for t in tiles]
        best = max(range(len(tiles)), key=lambda i: scores[i])  # first wins ties
        return [tiles[best]] + tiles[:best] + tiles[best + 1:]

    def fill_items(self) -> List[Any]:
        """The scene's fill scenes that touch this cell, downloaded once, on first need."""
        with self._fill_lock:
            if self._fill_items is None:
                # Like the single-field pipeline: every nearby scene except the one being filled.
                primary = getattr(self.tiles[0], "id", None)
                items = [
                    it for it in self.scene.fill_items()
                    if getattr(it, "id", None) != primary and _footprint_intersects(it, self.area_shape)
                ]
                # temporal_fill_median reads at most the first 6
                self.cache.prefetch([u for it in items[:6] for u in _band_urls(it, self.required_bands, False)])
                self._fill_items = items
            return self._fill_items

    def compute_field(self, feature: ParsedFeature, index_name: str) -> Dict[str, Any]:
        """calculate_index's per-field steps, reading pixels from the group's cache."""
        req, required_bands, geom = self.req, self.required_bands, feature.geometry
        aoi_sc, dst_transform, H, W, res_m = utils.build_adaptive_grid(self.crs, geom, native_res_m=10.0)
        aoi_mask = geometry_mask([mapping(aoi_sc)], out_shape=(H, W), transform=dst_transform, invert=True)

        band_dict = {b: np.full((H, W), np.nan, dtype="float32") for b in required_bands}
        have_data = np.zeros((H, W), dtype=bool)
        S_stack = np.zeros((H, W), dtype="int16") if self.has_scl else None
        for tile in self.tiles:
            res = utils._read_tile_into_stack(
                tile, geom, dst_transform, H, W, S_stack is not None, required_bands,
                open_dataset=self.cache.open,
            )
            if not res.get("used"):
                continue
            arrays = [res["bands"].get(b) for b in required_bands]
            if any(a is None for a in arrays):
                continue
            take = np.logical_and.reduce([np.isfinite(a) for a in arrays]) & ~have_data
            if np.any(take):
                for b, a in zip(required_bands, arrays):
                    band_dict[b][take] = a[take]
                if S_stack is not None and res.get("S") is not None:
                    S_stack[take] = res["S"][take]
                have_data |= take
            if have_data.all():
                break  # later tiles only fill pixels that are still empty

        try:
            index_arr = utils.compute_index_array_by_name(index_name, band_dict)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Index compute error: {e}")

        aoi_count = float(max(np.count_nonzero(aoi_mask), 1))
        if np.count_nonzero(np.isfinite(index_arr) & aoi_mask) / aoi_count < MIN_VALID_FRAC:
            try:
                items = self.fill_items()
                medians = {
                    b: utils.temporal_fill_median(
                        b, items, geom, dst_transform, H, W, False, 6, open_dataset=self.cache.open
                    )
                    for b in required_bands
                }
                if all(medians[b] is not None for b in required_bands):
                    fill_mask = (~np.isfinite(index_arr)) & aoi_mask
                    valid_fill = np.logical_and.reduce([np.isfinite(medians[b]) for b in required_bands])
                    fill_here = fill_mask & valid_fill
                    if np.any(fill_here):
                        for b in required_bands:
                            if np.isfinite(medians[b]).any() and np.nanmax(medians[b]) > 1.5:
                                medians[b] = medians[b] * (1 / 10000.0)
                            band_dict[b][fill_here] = medians[b][fill_here]
                        index_arr = utils.compute_index_array_by_name(index_name, band_dict)
            except Exception as e:
                log.warning("multi calculate: cloud-gap fill failed for field %s: %s", feature.id, e)

        if S_stack is not None:
            index_arr[np.isin(S_stack, SCL_CLOUD_CLASSES)] = np.nan
        index_arr[~aoi_mask] = np.nan

        return render_index_result(
            index_name, index_arr, aoi_mask, res_m, geom, dst_transform, self.crs,
            W, H, int(req.width or 800), int(req.height or 800),
            int(req.supersample or 4), bool(req.smooth if req.smooth is not None else True),
            float(req.gaussian_sigma or 3.0), req.date, png_optimize=False,
        )


def _group_features(features: List[ParsedFeature]) -> List[List[ParsedFeature]]:
    groups: Dict[tuple, List[ParsedFeature]] = defaultdict(list)
    for f in features:
        c = f.shape.centroid
        groups[(math.floor(c.x / GROUP_CELL_DEG), math.floor(c.y / GROUP_CELL_DEG))].append(f)
    return list(groups.values())


def _failed(fields: List[ParsedFeature], status_code: int, detail: str) -> List[Dict[str, Any]]:
    return [{"ok": False, "status_code": status_code, "detail": detail} for _ in fields]


def _run_group(fields: List[ParsedFeature], req: MultiCalculateRequest,
               index_name: str, required_bands: List[str], scene: _Scene) -> List[Dict[str, Any]]:
    try:
        group = _Group(fields, req, required_bands, scene)
    except HTTPException as e:
        return _failed(fields, e.status_code, str(e.detail))
    except Exception:
        log.exception("multi calculate: preparing a cell of %d fields failed", len(fields))
        return _failed(fields, 500, "Internal error while reading the scene for these fields")
    try:
        return list(_RENDER_POOL.map(
            lambda f: _outcome(f, lambda: group.compute_field(f, index_name)),
            fields,
        ))
    finally:
        group.cache.close()


def _run_shared_scene(features: List[ParsedFeature], req: MultiCalculateRequest,
                      index_name: str, required_bands: List[str]) -> Tuple[List[Dict[str, Any]], int]:
    try:
        scene = _Scene(mapping(_bbox([f.shape for f in features])), req)
    except HTTPException as e:
        return _failed(features, e.status_code, str(e.detail)), 0
    except Exception:
        log.exception("multi calculate: scene search failed for %d fields", len(features))
        return _failed(features, 500, "Internal error while searching for a scene"), 0

    groups = _group_features(features)
    with ThreadPoolExecutor(max_workers=max(1, min(GROUP_WORKERS, len(groups)))) as ex:
        group_outcomes = list(ex.map(
            lambda fields: _run_group(fields, req, index_name, required_bands, scene), groups
        ))
    by_id: Dict[str, Dict[str, Any]] = {}
    for fields, outcomes in zip(groups, group_outcomes):
        for f, outcome in zip(fields, outcomes):
            by_id[f.id] = outcome
    return [by_id[f.id] for f in features], len(groups)


def _request_cache_key(req: MultiCalculateRequest) -> str:
    raw = json.dumps(req.model_dump(), sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()


def _cacheable(response: Dict[str, Any]) -> bool:
    # 5xx errors may be transient provider trouble: retrying should try again.
    if any(e["status_code"] >= 500 for e in response["errors"]):
        return False
    size = sum(len(r.get("image_base64") or "") for r in response["results"])
    return size <= RESPONSE_CACHE_MAX_BYTES


@router.post("/index", response_model=MultiCalculateResponse)
def multi_calculate_index(req: MultiCalculateRequest):
    try:
        datetime.strptime(req.date, "%Y-%m-%d")
    except Exception:
        raise HTTPException(status_code=400, detail="date must be YYYY-MM-DD")

    satellite = (req.satellite or "s2").lower()
    # Request-level problems (bad index name, index/satellite mismatch) fail fast with 400.
    index_name, required_bands = resolve_index(req.index_name, satellite)

    # A cached answer was validated when it was computed, so skip parsing on a hit.
    cache_key = _request_cache_key(req)
    cached = _RESPONSE_CACHE.get(cache_key)
    if cached is not None:
        return cached

    t0 = time.monotonic()
    features = parse_feature_collection(req.geojson)
    if index_name == "TRUE_COLOR" or satellite.startswith("s1"):
        with ThreadPoolExecutor(max_workers=min(FIELD_WORKERS, len(features))) as ex:
            outcomes = list(ex.map(lambda f: _run_one(f, req), features))
        n_groups = len(features)
    else:
        outcomes, n_groups = _run_shared_scene(features, req, index_name, required_bands)

    results, errors = [], []
    for feature, outcome in zip(features, outcomes):
        if outcome["ok"]:
            results.append(outcome["result"])
        elif outcome["status_code"] == 400:
            raise HTTPException(status_code=400, detail=outcome["detail"])
        else:
            errors.append({
                "id": feature.id,
                "status_code": outcome["status_code"],
                "detail": outcome["detail"],
            })

    response = {
        "date": req.date,
        "index_name": index_name,
        "total_features": len(features),
        "succeeded": len(results),
        "failed": len(errors),
        "results": results,
        "errors": errors,
    }
    log.info("multi calculate %s %s: %d fields in %d cells, %d ok, %d failed in %.2fs",
             index_name, req.date, len(features), n_groups, len(results), len(errors), time.monotonic() - t0)
    if _cacheable(response):
        _RESPONSE_CACHE.set(cache_key, response)
    return response
