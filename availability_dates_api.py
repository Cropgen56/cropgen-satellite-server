from fastapi import APIRouter, HTTPException
from models import AvailabilityRequest, AvailabilityResponse, AvailabilityItem
import utils
from datetime import datetime

router = APIRouter()

@router.post("/", response_model=AvailabilityResponse)
def availability(req: AvailabilityRequest):
    geom = req.geometry
    try:
        datetime.strptime(req.start_date, "%Y-%m-%d")
        datetime.strptime(req.end_date, "%Y-%m-%d")
    except Exception:
        raise HTTPException(status_code=400, detail="start_date and end_date must be YYYY-MM-DD")

    collections = utils.get_collections_for_satellite(req.satellite or "s2")
    search_order = utils.get_provider_search_order(req.provider, prefer_pc_default=True)

    try:
        dt = f"{req.start_date}/{req.end_date}"
        all_items = []
        for provider_name in search_order:
            if provider_name == "planetary":
                items = utils.search_planetary(collections, geom, dt, limit=500)
            else:
                items = utils.search_aws(collections, geom, dt, limit=500)
            if items:
                all_items.extend(items)
                # For default "both", use the first provider with results to avoid double-fetching.
                break

        if not all_items:
            return {"items": []}

        date_map = {}
        for it in all_items:
            dt = it.properties.get("datetime") or it.properties.get("acquired") or ""
            if not dt:
                continue
            date_key = str(dt)[:10]
            cloud = it.properties.get("eo:cloud_cover") or it.properties.get("cloud_cover") or None
            try:
                cloud = float(cloud) if cloud is not None else None
            except Exception:
                cloud = None
            date_map.setdefault(date_key, []).append(cloud if cloud is not None else 999.0)
        out_items = []
        for d, clouds in sorted(date_map.items()):
            clouds_valid = [c for c in clouds if c is not None and c < 999.0]
            best = float(min(clouds_valid)) if clouds_valid else None
            out_items.append(AvailabilityItem(date=d, cloud_cover=best))
        return {"items": out_items}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
