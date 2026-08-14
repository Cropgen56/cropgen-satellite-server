from fastapi import APIRouter, HTTPException

from models import VraAnalysisRequest, VraAnalysisResponse
import vra_core

router = APIRouter()


@router.post("/analysis", response_model=VraAnalysisResponse)
def vra_analysis(req: VraAnalysisRequest):
    try:
        result = vra_core.run_vra_analysis(
            geometry=req.geometry,
            start_date=req.start_date,
            end_date=req.end_date,
            crop=req.crop,
            provider=req.provider,
            satellite=req.satellite,
            include_images=req.include_images,
            n_zones=req.n_zones,
            zone_method=req.zone_method,
            min_patch_ha=req.min_patch_ha,
        )
        return VraAnalysisResponse(**result)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        detail = str(exc)
        status = 503 if "Provider diagnostics" in detail else 404
        raise HTTPException(status_code=status, detail=detail) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"VRA analysis failed: {exc}") from exc
