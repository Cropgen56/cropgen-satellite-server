from fastapi import APIRouter, HTTPException

from models import SocAnalysisRequest, SocAnalysisResponse
import vra_core

router = APIRouter()


@router.post("/analysis", response_model=SocAnalysisResponse)
def soc_analysis(req: SocAnalysisRequest):
    try:
        result = vra_core.run_soc_analysis(
            geometry=req.geometry,
            start_date=req.start_date,
            end_date=req.end_date,
            provider=req.provider,
            satellite=req.satellite,
        )
        return SocAnalysisResponse(**result)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        detail = str(exc)
        status = 503 if "Provider diagnostics" in detail else 404
        raise HTTPException(status_code=status, detail=detail) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"SOC analysis failed: {exc}") from exc
