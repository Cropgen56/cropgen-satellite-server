"""Availability dates for many fields at once (GeoJSON FeatureCollection).

One STAC search covers every field (over the convex hull of all polygons); each
scene is then matched back to the fields it actually intersects, so a date is
listed with exactly the fields it covers.
"""
import json
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from shapely import STRtree
from shapely.geometry import mapping, shape
from shapely.ops import unary_union

from app.models import MultiAvailabilityItem, MultiAvailabilityRequest, MultiAvailabilityResponse
from app.services import utils
from app.services.response_cache import TTLCache
from app.services.feature_collection import parse_feature_collection

router = APIRouter()

# STAC page size (all pages are still fetched). Planetary rejects pages above 1000.
AVAILABILITY_SEARCH_LIMIT = 500
RESPONSE_CACHE_TTL_SECONDS = 10 * 60
_RESPONSE_CACHE = TTLCache(ttl_seconds=RESPONSE_CACHE_TTL_SECONDS, max_entries=256)


def _request_cache_key(req: MultiAvailabilityRequest) -> str:
    return json.dumps(
        {
            "geojson": req.geojson,
            "start_date": req.start_date,
            "end_date": req.end_date,
            "provider": (req.provider or "both").lower(),
            "satellite": (req.satellite or "s2").lower(),
        },
        sort_keys=True,
    )


def _item_cloud(it: Any) -> Optional[float]:
    cloud = it.properties.get("eo:cloud_cover") or it.properties.get("cloud_cover") or None
    try:
        return float(cloud) if cloud is not None else None
    except Exception:
        return None


@router.post("/", response_model=MultiAvailabilityResponse)
def multi_availability(req: MultiAvailabilityRequest):
    try:
        datetime.strptime(req.start_date, "%Y-%m-%d")
        datetime.strptime(req.end_date, "%Y-%m-%d")
    except Exception:
        raise HTTPException(status_code=400, detail="start_date and end_date must be YYYY-MM-DD")

    features = parse_feature_collection(req.geojson)

    cache_key = _request_cache_key(req)
    cached = _RESPONSE_CACHE.get(cache_key)
    if cached is not None:
        return cached

    collections = utils.get_collections_for_satellite(req.satellite or "s2")
    search_order = utils.get_provider_search_order(req.provider, prefer_pc_default=True)
    # A hull keeps the STAC request small even with hundreds of detailed polygons.
    search_geom = mapping(unary_union([f.shape for f in features]).convex_hull)

    try:
        all_items = utils.search_stac_items(
            collections,
            search_geom,
            f"{req.start_date}/{req.end_date}",
            limit=AVAILABILITY_SEARCH_LIMIT,
            search_order=search_order,
            metadata_only=True,
        )
        if not all_items:
            # Not cached: an empty result may be a transient provider failure.
            return {"total_features": len(features), "items": []}

        tree = STRtree([f.shape for f in features])
        # date -> (best cloud cover, indices of fields covered that day)
        by_date: Dict[str, tuple[Optional[float], set]] = {}
        for it in all_items:
            item_dt = it.properties.get("datetime") or it.properties.get("acquired") or ""
            if not item_dt:
                continue
            if it.geometry:
                hit = set(tree.query(shape(it.geometry), predicate="intersects").tolist())
            else:
                hit = set(range(len(features)))
            if not hit:
                continue
            date_key = str(item_dt)[:10]
            cloud = _item_cloud(it)
            best, covered = by_date.get(date_key, (None, set()))
            if cloud is not None and (best is None or cloud < best):
                best = cloud
            by_date[date_key] = (best, covered | hit)

        items: List[MultiAvailabilityItem] = []
        for d, (best, covered) in sorted(by_date.items()):
            ids = [features[i].id for i in sorted(covered)]
            items.append(MultiAvailabilityItem(
                date=d, cloud_cover=best, feature_count=len(ids), feature_ids=ids,
            ))
        return _RESPONSE_CACHE.set(cache_key, {"total_features": len(features), "items": items})
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
