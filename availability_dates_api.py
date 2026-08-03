from fastapi import APIRouter, HTTPException
from models import AvailabilityRequest, AvailabilityResponse, AvailabilityItem
import utils
import json
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

router = APIRouter()

AVAILABILITY_SEARCH_LIMIT = 500
RESPONSE_CACHE_TTL_SECONDS = 10 * 60
_RESPONSE_CACHE: Dict[str, tuple[float, Dict[str, Any]]] = {}


def _request_cache_key(req: AvailabilityRequest) -> str:
    return json.dumps(
        {
            "geometry": req.geometry,
            "start_date": req.start_date,
            "end_date": req.end_date,
            "provider": (req.provider or "both").lower(),
            "satellite": (req.satellite or "s2").lower(),
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


def _aggregate_availability_items(all_items: List[Any]) -> List[AvailabilityItem]:
    date_map: Dict[str, List[float]] = {}
    for it in all_items:
        item_dt = it.properties.get("datetime") or it.properties.get("acquired") or ""
        if not item_dt:
            continue
        date_key = str(item_dt)[:10]
        cloud = it.properties.get("eo:cloud_cover") or it.properties.get("cloud_cover") or None
        try:
            cloud = float(cloud) if cloud is not None else None
        except Exception:
            cloud = None
        date_map.setdefault(date_key, []).append(cloud if cloud is not None else 999.0)

    out_items: List[AvailabilityItem] = []
    for d, clouds in sorted(date_map.items()):
        clouds_valid = [c for c in clouds if c is not None and c < 999.0]
        best = float(min(clouds_valid)) if clouds_valid else None
        out_items.append(AvailabilityItem(date=d, cloud_cover=best))
    return out_items


@router.post("/", response_model=AvailabilityResponse)
def availability(req: AvailabilityRequest):
    geom = req.geometry
    try:
        datetime.strptime(req.start_date, "%Y-%m-%d")
        datetime.strptime(req.end_date, "%Y-%m-%d")
    except Exception:
        raise HTTPException(status_code=400, detail="start_date and end_date must be YYYY-MM-DD")

    cache_key = _request_cache_key(req)
    cached = _get_cached_response(cache_key)
    if cached is not None:
        return cached

    collections = utils.get_collections_for_satellite(req.satellite or "s2")
    search_order = utils.get_provider_search_order(req.provider, prefer_pc_default=True)

    try:
        dt = f"{req.start_date}/{req.end_date}"
        all_items = utils.search_stac_items(
            collections,
            geom,
            dt,
            limit=AVAILABILITY_SEARCH_LIMIT,
            search_order=search_order,
            metadata_only=True,
        )

        if not all_items:
            # Do not cache empty results: transient STAC/provider failures would
            # otherwise pin {"items": []} for RESPONSE_CACHE_TTL_SECONDS.
            return {"items": []}

        out_items = _aggregate_availability_items(all_items)
        return _set_cached_response(cache_key, {"items": out_items})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
