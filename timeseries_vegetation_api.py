# timeseries_vegetation_api.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Any, Dict, List, Optional
import numpy as np
import traceback
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# helpers from utils.py
from utils import (
    prefer_http_from_asset,
    sign_href_if_pc,
    aoi_to_scene,
    read_band_window,
    compute_index_array_by_name,
    search_planetary,
    search_aws,
    THREADS,
    get_collections_for_satellite,
    get_provider_search_order,
)

import rasterio
from rasterio.enums import Resampling

router = APIRouter()

# ✅ Only keep the required vegetation indices
SUPPORTED = ["NDVI", "EVI", "SAVI", "SUCROSE"]
MAX_THREADS = min(4, THREADS or 4)
SAMPLE_SIZE = 12
DEFAULT_MAX_POINTS = 24
MAX_RETURN_POINTS = 36
MAX_SEARCH_ITEMS = 96
RESPONSE_CACHE_TTL_SECONDS = 10 * 60
_RESPONSE_CACHE: Dict[str, tuple[float, Dict[str, Any]]] = {}

class TSRequest(BaseModel):
    geometry: Dict[str, Any]
    start_date: str
    end_date: str
    index: str
    provider: Optional[str] = "both"
    satellite: Optional[str] = "s2"
    max_items: Optional[int] = 8

class TimePoint(BaseModel):
    date: str
    value: float
    status: str

class Summary(BaseModel):
    min: Optional[float]
    mean: Optional[float]
    max: Optional[float]

class TSResponse(BaseModel):
    index: str
    summary: Summary
    timeseries: List[TimePoint]


def _normalize_max_points(value: Optional[int]) -> int:
    try:
        parsed = int(value or DEFAULT_MAX_POINTS)
    except (TypeError, ValueError):
        parsed = DEFAULT_MAX_POINTS
    return max(8, min(parsed, MAX_RETURN_POINTS))


def _request_cache_key(req: TSRequest, idx: str) -> str:
    return json.dumps(
        {
            "geometry": req.geometry,
            "start_date": req.start_date,
            "end_date": req.end_date,
            "index": idx,
            "provider": (req.provider or "both").lower(),
            "satellite": (req.satellite or "s2").lower(),
            "max_items": _normalize_max_points(req.max_items),
        },
        sort_keys=True,
    )


def _get_cached_response(cache_key: str) -> Optional[Dict[str, Any]]:
    cached = _RESPONSE_CACHE.get(cache_key)
    if not cached:
        return None
    timestamp, value = cached
    if time.time() - timestamp > RESPONSE_CACHE_TTL_SECONDS:
        _RESPONSE_CACHE.pop(cache_key, None)
        return None
    return value


def _set_cached_response(cache_key: str, value: Dict[str, Any]) -> Dict[str, Any]:
    _RESPONSE_CACHE[cache_key] = (time.time(), value)
    return value


def _item_date(item) -> str:
    return str(item.properties.get("datetime") or item.properties.get("acquired") or "")[:10]


def _item_cloud(item) -> float:
    cloud = item.properties.get("eo:cloud_cover") or item.properties.get("cloud_cover")
    try:
        return float(cloud)
    except Exception:
        return 999.0


def _pick_best_items(items: List[Any], max_points: int) -> List[Any]:
    by_date: Dict[str, Any] = {}
    for item in items:
        date_key = _item_date(item)
        if not date_key:
            continue
        existing = by_date.get(date_key)
        if existing is None or _item_cloud(item) < _item_cloud(existing):
            by_date[date_key] = item

    deduped = sorted(by_date.values(), key=lambda item: _item_date(item))
    target = min(len(deduped), max(24, max_points * 3))
    if len(deduped) <= target:
        return deduped

    idxs = [
        int(round(i * (len(deduped) - 1) / float(max(target - 1, 1))))
        for i in range(target)
    ]
    return [deduped[i] for i in idxs]

def classify_vegetation(index: str, v: Optional[float]) -> str:
    if v is None:
        return "No Data"
    if index in ("NDVI", "EVI", "SAVI"):
        if v < 0.2: return "Very Poor"
        if v < 0.4: return "Moderate"
        if v < 0.6: return "Good"
        return "Very Good"
    if index == "SUCROSE":
        if v < 0.2: return "Immature"
        if v < 0.4: return "Early Maturity"
        if v < 0.6: return "Optimal"
        return "Overmature"
    return "Unknown"

def _signed_asset_map(item):
    assets = getattr(item, "assets", {}) or {}
    signed = {}
    for k, a in assets.items():
        try:
            url = prefer_http_from_asset(a)
            signed[k.lower()] = sign_href_if_pc(url) if url else None
        except Exception:
            signed[k.lower()] = None
    return signed

def _item_has_required_assets(item, required_bands):
    assets_keys = set(k.lower() for k in (item.assets or {}).keys())
    return required_bands.issubset(assets_keys)

def _read_bands_from_signed(signed_assets, needed, geom, out_h=16, out_w=16):
    band_arrays = {}
    for b in needed:
        url = signed_assets.get(b.lower()) or signed_assets.get(b)
        if not url:
            band_arrays[b] = None
            continue
        try:
            with rasterio.Env():
                with rasterio.open(url) as ds:
                    arr = read_band_window(
                        ds,
                        aoi_to_scene(geom, ds.crs.to_string()),
                        out_h,
                        out_w,
                        ds.transform,
                        Resampling.bilinear
                    )
                    if np.nanmax(arr) > 1.5:
                        arr = arr * (1/10000.0)
                    band_arrays[b] = arr
        except Exception:
            band_arrays[b] = None
    return band_arrays

def _compute_index_for_item(item, geom, idx, out_h=SAMPLE_SIZE, out_w=SAMPLE_SIZE):
    try:
        cloud = item.properties.get("eo:cloud_cover") or item.properties.get("cloud_cover")
        if cloud is not None:
            try:
                if float(cloud) > 60.0:
                    return None
            except Exception:
                pass

        # ✅ Only needed bands for the 4 indices
        if idx == "NDVI":
            needed = ["B08","B04"]
        elif idx == "EVI":
            needed = ["B08","B04","B02"]
        elif idx == "SAVI":
            needed = ["B08","B04"]
        elif idx == "SUCROSE":
            needed = ["B11","B04"]
        else:
            return None

        reqset = set(b.lower() for b in needed)
        if not _item_has_required_assets(item, reqset):
            return None

        signed = _signed_asset_map(item)
        bands = _read_bands_from_signed(signed, needed, geom, out_h=out_h, out_w=out_w)
        if any(bands.get(b) is None for b in needed):
            return None

        band_dict = {k: bands.get(k) for k in bands}

        try:
            arr = compute_index_array_by_name(idx, band_dict)
        except Exception:
            eps = 1e-6
            if idx == "NDVI":
                arr = (band_dict["B08"] - band_dict["B04"]) / (band_dict["B08"] + band_dict["B04"] + eps)
            elif idx == "EVI":
                NIR, RED, BLUE = band_dict["B08"], band_dict["B04"], band_dict["B02"]
                arr = 2.5 * (NIR - RED) / (NIR + 6*RED - 7.5*BLUE + 1.0 + eps)
            elif idx == "SAVI":
                NIR, RED = band_dict["B08"], band_dict["B04"]
                L = 0.5
                arr = ((NIR - RED) / (NIR + RED + L + eps)) * (1.0 + L)
            elif idx == "SUCROSE":
                arr = (band_dict["B11"] - band_dict["B04"]) / (band_dict["B11"] + band_dict["B04"] + eps)
            else:
                return None

        mask = np.isfinite(arr)
        if not np.any(mask):
            return None
        mean_val = round(float(np.nanmean(arr[mask])), 3)
        date = str(item.properties.get("datetime") or item.properties.get("acquired") or "")[:10]
        return (date, mean_val)
    except Exception:
        return None

@router.post("/vegetation", response_model=TSResponse)
def vegetation_timeseries(req: TSRequest):
    idx = req.index.upper()
    if idx not in SUPPORTED:
        raise HTTPException(status_code=400, detail=f"Unsupported index. Supported: {SUPPORTED}")

    satellite = (req.satellite or "s2").lower()
    if satellite.startswith("s1"):
        raise HTTPException(
            status_code=400,
            detail=f"Index {idx} requires Sentinel-2 (optical). Sentinel-1 is radar-only."
        )

    try:
        max_points = _normalize_max_points(req.max_items)
        cache_key = _request_cache_key(req, idx)
        cached = _get_cached_response(cache_key)
        if cached is not None:
            return cached

        collections = get_collections_for_satellite(req.satellite or "s2")
        search_order = get_provider_search_order(req.provider, prefer_pc_default=True)
        dt = f"{req.start_date}/{req.end_date}"
        search_limit = min(max(24, max_points * 3), MAX_SEARCH_ITEMS)

        items = []
        for provider_name in search_order:
            if provider_name == "planetary":
                items = search_planetary(collections, req.geometry, dt, limit=search_limit)
            else:
                items = search_aws(collections, req.geometry, dt, limit=search_limit)
            if items:
                break

        if not items:
            return _set_cached_response(
                cache_key,
                {"index": idx, "summary": {"min": None, "mean": None, "max": None}, "timeseries": []},
            )

        items = _pick_best_items(items, max_points)

        results = []
        with ThreadPoolExecutor(max_workers=min(MAX_THREADS, max(1, len(items)))) as ex:
            futures = {
                ex.submit(_compute_index_for_item, it, req.geometry, idx, SAMPLE_SIZE, SAMPLE_SIZE): it
                for it in items
            }
            for fut in as_completed(futures):
                try:
                    res = fut.result()
                except Exception:
                    res = None
                if res:
                    results.append(res)

        if not results:
            return _set_cached_response(
                cache_key,
                {"index": idx, "summary": {"min": None, "mean": None, "max": None}, "timeseries": []},
            )

        date_map: Dict[str, List[float]] = {}
        for date_str, val in results:
            if not date_str:
                continue
            date_map.setdefault(date_str, []).append(val)

        aggregated = [(d, round(float(sum(vals) / len(vals)), 3)) for d, vals in date_map.items()]
        aggregated_sorted = sorted(aggregated, key=lambda x: x[0])

        max_pts = max_points or len(aggregated_sorted)
        if len(aggregated_sorted) > max_pts:
            n = max_pts
            idxs = [int(round(i * (len(aggregated_sorted) - 1) / float(max(n - 1, 1)))) for i in range(n)]
            aggregated_sorted = [aggregated_sorted[i] for i in idxs]

        times = []
        vals = []
        for d, v in aggregated_sorted:
            times.append({"date": d, "value": v, "status": classify_vegetation(idx, v)})
            vals.append(v)

        summary = {
            "min": round(min(vals), 3),
            "mean": round(float(sum(vals) / len(vals)), 3),
            "max": round(max(vals), 3)
        }
        return _set_cached_response(
            cache_key,
            {"index": idx, "summary": summary, "timeseries": times},
        )
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
