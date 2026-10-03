"""Calculate a satellite index for many fields at once (GeoJSON FeatureCollection).

Each polygon is run through the single-field calculate_index pipeline, so every
result (image, bounds, legend) is identical to calling /calculate/index for that
field on its own. A field that fails (no scene, clouds, ...) is reported in
`errors` without failing the rest of the batch.
"""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any, Dict

from fastapi import APIRouter, HTTPException

from app.models import CalculateRequest, MultiCalculateRequest, MultiCalculateResponse
from app.routers.calculate_index_api import calculate_index
from app.services import utils
from app.services.feature_collection import ParsedFeature, parse_feature_collection

router = APIRouter()

# Fields processed in parallel. Each one already reads its tiles with its own
# thread pool, so keep this modest to avoid overloading the STAC/COG providers.
FIELD_WORKERS = 6


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
    try:
        out = calculate_index(single)
        return {"ok": True, "result": {"id": feature.id, "properties": feature.properties, **out}}
    except HTTPException as e:
        return {"ok": False, "status_code": e.status_code, "detail": str(e.detail)}
    except Exception as e:
        return {"ok": False, "status_code": 500, "detail": str(e)}


@router.post("/index", response_model=MultiCalculateResponse)
def multi_calculate_index(req: MultiCalculateRequest):
    try:
        datetime.strptime(req.date, "%Y-%m-%d")
    except Exception:
        raise HTTPException(status_code=400, detail="date must be YYYY-MM-DD")

    features = parse_feature_collection(req.geojson)

    with ThreadPoolExecutor(max_workers=min(FIELD_WORKERS, len(features))) as ex:
        outcomes = list(ex.map(lambda f: _run_one(f, req), features))

    results, errors = [], []
    for feature, outcome in zip(features, outcomes):
        if outcome["ok"]:
            results.append(outcome["result"])
        elif outcome["status_code"] == 400:
            # 400s from calculate_index are request-level (bad index name,
            # index/satellite mismatch), so they would fail every field the same way.
            raise HTTPException(status_code=400, detail=outcome["detail"])
        else:
            errors.append({
                "id": feature.id,
                "status_code": outcome["status_code"],
                "detail": outcome["detail"],
            })

    index_name = req.index_name.upper()
    if utils.is_s1_index(index_name):
        index_name = utils.normalize_s1_index(index_name)

    return {
        "date": req.date,
        "index_name": index_name,
        "total_features": len(features),
        "succeeded": len(results),
        "failed": len(errors),
        "results": results,
        "errors": errors,
    }
