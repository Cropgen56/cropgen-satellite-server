from fastapi import APIRouter, HTTPException

from app.services import cropgen_soil_vra
from app.models import VraAnalysisRequest, VraAnalysisResponse
from app.services.vra_service import run_vra

router = APIRouter()


@router.get("/crops")
def list_crops():
    crops = sorted(k for k in cropgen_soil_vra.CROP_DEMAND if k != "default")
    return {"crops": crops, "default": cropgen_soil_vra.CROP_DEMAND["default"]}


@router.get("/regions")
def list_regions():
    return cropgen_soil_vra.available_regions()


@router.post("/analysis", response_model=VraAnalysisResponse)
def vra_analysis(req: VraAnalysisRequest):
    try:
        return VraAnalysisResponse(**run_vra(req))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        detail = str(exc)
        status = 503 if "Provider diagnostics" in detail else 404
        raise HTTPException(status_code=status, detail=detail) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"VRA analysis failed: {exc}") from exc
