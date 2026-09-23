"""HTTP adapter for cropgen_soil_vra.run_analysis.

The engine in cropgen_soil_vra.py is the source of truth. This module only
turns a request into kwargs, JSON-safes the result, and adds a few fields
the Zoning UI already consumes (date, cloud_cover, soc_stats).
"""

from typing import Any, Dict, Optional

import cropgen_soil_vra

HA_ACRE = 2.47105


def _dump(model) -> dict:
    if hasattr(model, "model_dump"):
        return model.model_dump(exclude_none=True)
    return model.dict(exclude_none=True)


def _scene_date(payload: Dict[str, Any]) -> Optional[str]:
    dates = [
        scene.get("date")
        for scene in (payload.get("scenes_used") or [])
        if scene.get("date")
    ]
    if dates:
        return dates[-1]
    meta = payload.get("metadata") or {}
    return meta.get("end_date")


def _cloud_cover(payload: Dict[str, Any]) -> Optional[float]:
    vals = [
        scene.get("scene_cloud_pct")
        for scene in (payload.get("scenes_used") or [])
        if scene.get("scene_cloud_pct") is not None
    ]
    if not vals:
        return None
    return round(sum(vals) / len(vals), 2)


def _soc_stats(payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    soc = (payload.get("param_stats") or {}).get("SOC") or {}
    if not soc:
        return None

    field_ha = (payload.get("metadata") or {}).get("field_area_ha")
    classes: Dict[str, Dict[str, Any]] = {}
    features = ((payload.get("zone_geojson") or {}).get("SOC") or {}).get(
        "features"
    ) or []
    for feat in features:
        props = feat.get("properties") or {}
        label = props.get("zone_class") or "Zone"
        rec = classes.setdefault(
            label, {"pixels": 0, "ha": 0.0, "acres": 0.0, "pct_area": 0.0}
        )
        rec["pixels"] += int(props.get("pixel_count") or 0)
        rec["ha"] += float(props.get("area_ha") or 0)

    total_ha = sum(c["ha"] for c in classes.values()) or (field_ha or 0)
    for rec in classes.values():
        rec["ha"] = round(rec["ha"], 4)
        rec["acres"] = round(rec["ha"] * HA_ACRE, 4)
        rec["pct_area"] = round(100.0 * rec["ha"] / total_ha, 2) if total_ha else 0.0

    total_area_ha = field_ha if field_ha is not None else (total_ha or None)
    return {
        "mean_pct": soc.get("mean"),
        "min_pct": soc.get("min"),
        "max_pct": soc.get("max"),
        "std_pct": soc.get("std"),
        "total_area_ha": total_area_ha,
        "total_area_acres": (
            round(float(total_area_ha) * HA_ACRE, 4)
            if total_area_ha is not None
            else None
        ),
        "classes": classes,
        "unit": soc.get("unit"),
        "confidence": soc.get("confidence"),
    }


def run_vra(req, image_keys=None) -> Dict[str, Any]:
    samples = None
    if req.ground_samples:
        samples = [_dump(s) for s in req.ground_samples]

    kwargs = dict(
        aoi_geojson=req.geometry,
        start_date=req.start_date,
        end_date=req.end_date,
        crop=req.crop,
        n_zones=req.n_zones,
        min_patch_ha=req.min_patch_ha,
        label_mode=req.label_mode,
        ground_samples=samples,
        district=req.district,
        state=req.state,
        auto_region=req.auto_region,
        use_soilgrids=req.use_soilgrids,
        grid_cell_m=req.grid_cell_m,
        soc_method=req.soc_method,
        include_images=req.include_images,
    )
    if req.max_scenes is not None:
        kwargs["max_scenes"] = req.max_scenes
    if image_keys is not None:
        kwargs["image_keys"] = image_keys
    if getattr(req, "zone_features", None):
        kwargs["zone_features"] = req.zone_features

    result = cropgen_soil_vra.run_analysis(**kwargs)
    payload = cropgen_soil_vra.to_api_result(
        result,
        include_images=req.include_images,
        include_prescription_geojson=req.include_prescription_geojson,
    )
    payload["date"] = _scene_date(payload)
    payload["cloud_cover"] = _cloud_cover(payload)
    payload["soc_stats"] = _soc_stats(payload)
    return payload
