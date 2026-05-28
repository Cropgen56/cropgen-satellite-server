from datetime import datetime
from fastapi import APIRouter, HTTPException

from calculate_index_api import calculate_index
from models import (
    CalculateRequest,
    NpkAvailabilityRequest,
    NpkAvailabilityResponse,
    NpkNutrientAvailability,
)

router = APIRouter()

_LABEL_WEIGHTS = {
    "cloud": 50,
    "clouds": 50,
    "very poor": 12,
    "poor": 28,
    "fair": 42,
    "moderate": 55,
    "good": 72,
    "very good": 82,
    "excellent": 90,
    "dense": 92,
    "very dense": 95,
    "extreme": 98,
}


def _legend_health_score(legend):
    if not isinstance(legend, list) or not legend:
        return None
    weighted = 0.0
    total_pct = 0.0
    for row in legend:
        pct = float(row.get("percent", 0) or 0)
        if pct <= 0:
            continue
        label = str(row.get("label", "")).strip().lower()
        weight = _LABEL_WEIGHTS.get(label, 55)
        weighted += (weight * pct) / 100.0
        total_pct += pct
    if total_pct <= 0:
        return None
    return round(max(0.0, min(100.0, weighted)), 2)


def _score_to_factor(score):
    if score is None:
        return 0.7
    if score >= 80:
        return 1.0
    if score >= 65:
        return 0.85
    if score >= 50:
        return 0.7
    if score >= 35:
        return 0.55
    return 0.4


def _index_health(geometry, date, index_name, provider, satellite):
    res = calculate_index(
        CalculateRequest(
            geometry=geometry,
            date=date,
            index_name=index_name,
            provider=provider,
            satellite=satellite,
            width=320,
            height=320,
            supersample=1,
            smooth=False,
            gaussian_sigma=1.0,
        )
    )
    score = _legend_health_score(res.get("legend"))
    return score


@router.post("/availability", response_model=NpkAvailabilityResponse)
def npk_availability(req: NpkAvailabilityRequest):
    try:
        datetime.strptime(req.date, "%Y-%m-%d")
    except Exception:
        raise HTTPException(status_code=400, detail="date must be YYYY-MM-DD")

    try:
        n_score = _index_health(
            req.geometry, req.date, "NITROGEN", req.provider, req.satellite
        )
        p_score = _index_health(
            req.geometry, req.date, "CCC", req.provider, req.satellite
        )
        k_score = _index_health(
            req.geometry, req.date, "NDMI", req.provider, req.satellite
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"NPK availability failed: {exc}")

    nutrients = {
        "nitrogen": NpkNutrientAvailability(
            health_score=n_score,
            factor=_score_to_factor(n_score),
            source_index="NITROGEN",
        ),
        "phosphorous": NpkNutrientAvailability(
            health_score=p_score,
            factor=_score_to_factor(p_score),
            source_index="CCC",
        ),
        "potassium": NpkNutrientAvailability(
            health_score=k_score,
            factor=_score_to_factor(k_score),
            source_index="NDMI",
        ),
    }

    return NpkAvailabilityResponse(
        date=req.date,
        provider=req.provider,
        satellite=req.satellite,
        stage_context={
            "bbch_stage": req.bbch_stage,
            "stage_name": req.stage_name,
        },
        nutrients=nutrients,
        debug={
            "mapping": "factor from health_score buckets [80,65,50,35]",
        },
    )
