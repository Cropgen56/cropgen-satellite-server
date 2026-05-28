from datetime import datetime
from typing import Dict, Optional

import numpy as np
import rasterio
from fastapi import APIRouter, HTTPException
from rasterio.features import geometry_mask
from shapely.geometry import mapping

import utils
from models import CropHealthRequest, CropHealthResponse

router = APIRouter()

CLOUD_SCL_CLASSES = [3, 8, 9, 10, 11]

STAGE_EXPECTED = {
    "germination": {"ndvi": (0.15, 0.35), "ndre": (0.05, 0.15)},
    "vegetative": {"ndvi": (0.45, 0.80), "ndre": (0.20, 0.45)},
    "flowering": {"ndvi": (0.40, 0.70), "ndre": (0.18, 0.38)},
    "maturity": {"ndvi": (0.25, 0.55), "ndre": (0.10, 0.28)},
}


def _infer_stage(sowing_date: Optional[str], target_date: str) -> str:
    if not sowing_date:
        return "vegetative"
    try:
        sow = datetime.strptime(sowing_date, "%Y-%m-%d")
        cur = datetime.strptime(target_date, "%Y-%m-%d")
        days = (cur - sow).days
    except Exception:
        return "vegetative"

    if days <= 20:
        return "germination"
    if days <= 60:
        return "vegetative"
    if days <= 100:
        return "flowering"
    return "maturity"


def _stage_normalized_score(value: float, lo: float, hi: float) -> float:
    if not np.isfinite(value):
        return 0.0
    baseline = ((value + 1.0) / 2.0) * 100.0
    span = max(hi - lo, 1e-6)
    if value < lo:
        penalty = min(40.0, ((lo - value) / span) * 35.0)
        baseline -= penalty
    elif value > hi:
        penalty = min(25.0, ((value - hi) / span) * 20.0)
        baseline -= penalty
    else:
        baseline += 10.0
    return float(np.clip(baseline, 0.0, 100.0))


def _status_from_health(health: int) -> str:
    if health >= 75:
        return "Healthy"
    if health >= 50:
        return "Moderate"
    return "Poor"


def _stress_from_health(health: int) -> str:
    if health >= 75:
        return "Low"
    if health >= 50:
        return "Medium"
    return "High"


@router.post("/score", response_model=CropHealthResponse)
def crop_health_score(req: CropHealthRequest):
    date_str = req.date
    try:
        datetime.strptime(date_str, "%Y-%m-%d")
    except Exception:
        raise HTTPException(status_code=400, detail="date must be YYYY-MM-DD")

    satellite = (req.satellite or "s2").lower()
    if satellite.startswith("s1"):
        raise HTTPException(status_code=400, detail="crop health scoring requires Sentinel-2 imagery")

    search_order = utils.get_provider_search_order(req.provider, prefer_pc_default=True)
    prefer_pc = search_order[0] == "planetary"

    item, has_scl, _ = utils.pick_best_item(
        req.geometry,
        date_str,
        date_str,
        prefer_pc=prefer_pc,
        satellite=satellite,
    )
    if not item:
        raise HTTPException(status_code=404, detail="No suitable item found for crop health scoring")

    first_assets = item.assets or {}
    first_candidate = utils.prefer_http_from_asset(first_assets.get("red") or first_assets.get("B04"))
    if not first_candidate:
        for a in first_assets.values():
            h = utils.prefer_http_from_asset(a)
            if h:
                first_candidate = h
                break
    first_red = utils.sign_href_if_pc(first_candidate) if first_candidate else None
    if not first_red:
        raise HTTPException(status_code=500, detail="Could not determine reference band URL")

    with rasterio.open(first_red) as fr:
        target_crs = fr.crs

    _, dst_transform, H, W, _ = utils.build_adaptive_grid(target_crs, req.geometry, native_res_m=10.0)

    # memory safety for large polygons
    max_pixels = 600 * 600
    pixels = H * W
    if pixels > max_pixels:
        scale = np.sqrt(float(max_pixels) / float(max(pixels, 1)))
        H = max(32, int(H * scale))
        W = max(32, int(W * scale))

    read = utils._read_tile_into_stack(
        item,
        req.geometry,
        dst_transform,
        H,
        W,
        True,
        ["B08", "B04", "B8A", "B05"],
    )
    bands = read.get("bands", {})
    scl = read.get("S")
    b08, b04, b8a, b05 = bands.get("B08"), bands.get("B04"), bands.get("B8A"), bands.get("B05")

    if any(b is None for b in (b08, b04, b8a, b05)):
        raise HTTPException(status_code=424, detail="Required Sentinel-2 bands unavailable for NDVI/NDRE")

    aoi_sc = utils.aoi_to_scene(req.geometry, target_crs.to_string())
    aoi_mask = geometry_mask([mapping(aoi_sc)], out_shape=(H, W), transform=dst_transform, invert=True)

    valid = (
        np.isfinite(b08)
        & np.isfinite(b04)
        & np.isfinite(b8a)
        & np.isfinite(b05)
        & aoi_mask
    )
    if scl is not None:
        valid = valid & (~np.isin(scl, CLOUD_SCL_CLASSES))
        cloud_coverage = float(
            np.count_nonzero(np.isin(scl, CLOUD_SCL_CLASSES) & aoi_mask)
            / float(max(np.count_nonzero(aoi_mask), 1))
            * 100.0
        )
    else:
        cloud_coverage = None

    if not np.any(valid):
        raise HTTPException(status_code=424, detail="No valid cloud-free pixels inside farm polygon")

    eps = 1e-6
    ndvi = (b08 - b04) / (b08 + b04 + eps)
    ndre = (b8a - b05) / (b8a + b05 + eps)

    ndvi_mean = float(np.nanmean(ndvi[valid]))
    ndre_mean = float(np.nanmean(ndre[valid]))

    stage = _infer_stage(req.sowing_date, date_str)
    expected = STAGE_EXPECTED.get(stage, STAGE_EXPECTED["vegetative"])
    ndvi_score = _stage_normalized_score(ndvi_mean, expected["ndvi"][0], expected["ndvi"][1])
    ndre_score = _stage_normalized_score(ndre_mean, expected["ndre"][0], expected["ndre"][1])

    health = int(round(np.clip((ndvi_score * 0.6) + (ndre_score * 0.4), 0, 100)))
    status = _status_from_health(health)
    stress = _stress_from_health(health)

    return {
        "health": health,
        "status": status,
        "ndvi": round(ndvi_mean, 3),
        "ndre": round(ndre_mean, 3),
        "stress": stress,
        "stage": stage,
        "cloud_coverage": round(cloud_coverage, 2) if cloud_coverage is not None else None,
    }
