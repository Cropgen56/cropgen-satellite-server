"""Calculate a satellite index for many fields at once (GeoJSON FeatureCollection).

Every field goes through the same steps as the single-field /calculate/index
(the same tile reader, cloud-gap fill, SCL masking and renderer), so each
result has the same image, bounds and legend as calling that endpoint for the
field on its own. What changes is where the pixels come from: fields that sit
close together share their network work.

1. Fields are grouped into ~10 km cells.
2. Per cell, once: pick the best scene, list its tiles, and download each
   tile band's native pixels covering the cell into memory.
3. Per field: run the unchanged reader against those in-memory copies.
   Cloud-gap fill scenes are searched and downloaded once per cell, only if a
   field needs them.

The one intended difference: all fields in a cell use the scene ranked best for
the cell, instead of each field ranking scenes on its own (which could give
neighbouring fields different tiles of the same pass).

TRUE_COLOR and Sentinel-1 requests keep the plain per-field pipeline. A field
that fails (no valid pixels, ...) is reported in `errors` without failing the
rest.
"""
import math
import os
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import rasterio
from fastapi import APIRouter, HTTPException
from rasterio.features import geometry_mask
from rasterio.io import MemoryFile
from rasterio.windows import Window, from_bounds
from shapely.geometry import mapping
from shapely.ops import unary_union

from app.models import CalculateRequest, MultiCalculateRequest, MultiCalculateResponse
from app.routers.calculate_index_api import calculate_index, render_index_result, resolve_index
from app.services import utils
from app.services.feature_collection import ParsedFeature, parse_feature_collection

router = APIRouter()

# Per-field fallback path: each field reads its own tiles, so keep this modest
# to avoid overloading the STAC/COG providers.
FIELD_WORKERS = 6
# Shared-scene path: per-field work is CPU-bound (in-memory reads + rendering).
RENDER_WORKERS = max(4, os.cpu_count() or 4)
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
    except Exception as e:
        return {"ok": False, "status_code": 500, "detail": str(e)}


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


class _WindowCache:
    """Serves rasterio datasets for whole-file URLs from small in-memory windows.

    Each window is an exact, unresampled pixel subset of the source file (same
    CRS, resolution and pixel alignment, same nodata), so reading a field from
    it gives the same pixels as reading the field from the file. URLs that were
    not prefetched fall through to rasterio.open.
    """

    def __init__(self, area_ll: Dict[str, Any]):
        self.area_ll = area_ll
        self._files: Dict[str, MemoryFile] = {}
        self._lock = threading.Lock()

    def prefetch(self, urls: List[str]) -> None:
        todo = [u for u in dict.fromkeys(urls) if _url_key(u) not in self._files]
        if not todo:
            return
        with ThreadPoolExecutor(max_workers=min(utils.THREADS, len(todo))) as ex:
            for url, memfile in zip(todo, ex.map(self._download, todo)):
                if memfile is not None:
                    with self._lock:
                        self._files[_url_key(url)] = memfile

    def _download(self, url: str) -> Optional[MemoryFile]:
        try:
            with rasterio.Env(), rasterio.open(url) as ds:
                aoi_sc = utils.aoi_to_scene(self.area_ll, ds.crs.to_string())
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
            memfile = MemoryFile()
            with memfile.open(**profile) as dst:
                dst.write(data, 1)
            return memfile
        except Exception:
            return None  # the reader falls back to the remote file

    def open(self, url: str):
        memfile = self._files.get(_url_key(url))
        return memfile.open() if memfile is not None else rasterio.open(url)

    def close(self) -> None:
        for memfile in self._files.values():
            memfile.close()
        self._files.clear()


class _Group:
    """Scene choice + in-memory pixels shared by the fields of one cell."""

    def __init__(self, fields: List[ParsedFeature], req: MultiCalculateRequest, required_bands: List[str]):
        self.req = req
        self.required_bands = required_bands
        self.satellite = (req.satellite or "s2").lower()
        self.search_order = utils.get_provider_search_order(req.provider, prefer_pc_default=True)
        self.prefer_pc = self.search_order[0] == "planetary"
        self.area_ll = mapping(unary_union([f.shape for f in fields]).envelope)
        self.cache = _WindowCache(self.area_ll)
        self._fill_items: Optional[List[Any]] = None
        self._fill_lock = threading.Lock()

        item, has_scl, coll = utils.pick_best_item(
            self.area_ll, req.date, req.date, prefer_pc=self.prefer_pc, satellite=self.satellite
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
        self.crs = utils.get_item_crs(item)
        if self.crs is None:
            raise HTTPException(status_code=500, detail="Could not determine reference band CRS")
        tiles = utils.items_for_date(
            self.area_ll, item.properties.get("datetime", ""), self.collection, prefer_pc=self.prefer_pc, limit=6
        ) or [item]
        # Where tiles of the same pass overlap, the first tile read wins. The
        # single-field endpoint takes whichever download finishes first; here
        # the best-ranked scene always goes first, so results are repeatable.
        best = getattr(item, "id", None)
        self.tiles = sorted(tiles, key=lambda t: getattr(t, "id", None) != best)
        self.cache.prefetch([u for t in self.tiles for u in _band_urls(t, required_bands, has_scl)])

    def fill_items(self) -> List[Any]:
        """Nearby-date scenes for cloud-gap fill: searched and downloaded once, on first need."""
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
                items = [it for it in items if getattr(it, "id", None) != getattr(self.item, "id", None)]
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
            except Exception:
                pass

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


def _run_group(fields: List[ParsedFeature], req: MultiCalculateRequest,
               index_name: str, required_bands: List[str]) -> List[Dict[str, Any]]:
    try:
        group = _Group(fields, req, required_bands)
    except HTTPException as e:
        return [{"ok": False, "status_code": e.status_code, "detail": str(e.detail)} for _ in fields]
    except Exception as e:
        return [{"ok": False, "status_code": 500, "detail": str(e)} for _ in fields]
    try:
        with ThreadPoolExecutor(max_workers=min(RENDER_WORKERS, len(fields))) as ex:
            return list(ex.map(
                lambda f: _outcome(f, lambda: group.compute_field(f, index_name)),
                fields,
            ))
    finally:
        group.cache.close()


@router.post("/index", response_model=MultiCalculateResponse)
def multi_calculate_index(req: MultiCalculateRequest):
    try:
        datetime.strptime(req.date, "%Y-%m-%d")
    except Exception:
        raise HTTPException(status_code=400, detail="date must be YYYY-MM-DD")

    features = parse_feature_collection(req.geojson)
    satellite = (req.satellite or "s2").lower()
    # Request-level problems (bad index name, index/satellite mismatch) fail fast with 400.
    index_name, required_bands = resolve_index(req.index_name, satellite)

    if index_name == "TRUE_COLOR" or satellite.startswith("s1"):
        with ThreadPoolExecutor(max_workers=min(FIELD_WORKERS, len(features))) as ex:
            outcomes = list(ex.map(lambda f: _run_one(f, req), features))
    else:
        by_id: Dict[str, Dict[str, Any]] = {}
        for fields in _group_features(features):
            for f, outcome in zip(fields, _run_group(fields, req, index_name, required_bands)):
                by_id[f.id] = outcome
        outcomes = [by_id[f.id] for f in features]

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

    return {
        "date": req.date,
        "index_name": index_name,
        "total_features": len(features),
        "succeeded": len(results),
        "failed": len(errors),
        "results": results,
        "errors": errors,
    }
