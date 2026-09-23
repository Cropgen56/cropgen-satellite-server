from fastapi import APIRouter, HTTPException

from models import SocAnalysisRequest, SocAnalysisResponse, VraAnalysisRequest
from vra_service import run_vra

router = APIRouter()


@router.post("/analysis", response_model=SocAnalysisResponse)
def soc_analysis(req: SocAnalysisRequest):
    """SOC extract from the same cropgen_soil_vra engine used by /vra/analysis."""
    try:
        vra_req = VraAnalysisRequest(
            geometry=req.geometry,
            start_date=req.start_date,
            end_date=req.end_date,
            include_images=True,
            include_prescription_geojson=False,
        )
        payload = run_vra(vra_req, image_keys={"SOC"})
        images = payload.get("images") or {}
        image = images.get("SOC")
        soc_stats = payload.get("soc_stats")
        if not image or not soc_stats:
            raise RuntimeError("SOC map was not produced for this field / date range.")
        return SocAnalysisResponse(
            date=payload.get("date") or req.end_date,
            cloud_cover=payload.get("cloud_cover"),
            image_base64=image,
            soc_stats=soc_stats,
            metadata=payload.get("metadata") or {},
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        detail = str(exc)
        status = 503 if "Provider diagnostics" in detail else 404
        raise HTTPException(status_code=status, detail=detail) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"SOC analysis failed: {exc}") from exc
