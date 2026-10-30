"""Parse a GeoJSON FeatureCollection of field polygons for the multi-polygon APIs."""
import os
from dataclasses import dataclass
from typing import Any, Dict, List

from fastapi import HTTPException
from shapely.geometry import MultiPolygon, mapping, shape
from shapely.geometry.base import BaseGeometry
from shapely.validation import explain_validity, make_valid

POLYGON_TYPES = {"Polygon", "MultiPolygon"}
# No cap by default: any number of fields is accepted. Set MULTI_MAX_FEATURES to a positive
# number to cap one request (callers such as the dashboard already split large collections).
MAX_FEATURES = int(os.getenv("MULTI_MAX_FEATURES", "0"))


@dataclass
class ParsedFeature:
    id: str
    geometry: Dict[str, Any]
    shape: BaseGeometry
    properties: Dict[str, Any]


def _repair(geom: BaseGeometry) -> BaseGeometry:
    """Fix a self-intersecting or otherwise invalid outline, keeping only its polygon parts."""
    fixed = make_valid(geom)
    if fixed.geom_type in POLYGON_TYPES:
        return fixed
    polygons = []
    for part in getattr(fixed, "geoms", []):
        if part.geom_type == "Polygon":
            polygons.append(part)
        elif part.geom_type == "MultiPolygon":
            polygons.extend(part.geoms)
    return MultiPolygon(polygons)


def _feature_id(feature: Dict[str, Any], index: int) -> str:
    props = feature.get("properties") or {}
    for value in (feature.get("id"), props.get("id"), props.get("name")):
        if value is not None and str(value).strip():
            return str(value)
    return f"feature-{index + 1}"


def parse_feature_collection(fc: Dict[str, Any]) -> List[ParsedFeature]:
    """Validate the collection and return one ParsedFeature per polygon.

    Any bad feature rejects the whole request with 400, naming the feature, so the
    caller can fix the file instead of getting a partial answer.
    """
    if not isinstance(fc, dict) or fc.get("type") != "FeatureCollection":
        raise HTTPException(status_code=400, detail="geojson must be a GeoJSON FeatureCollection")
    features = fc.get("features")
    if not isinstance(features, list) or not features:
        raise HTTPException(status_code=400, detail="FeatureCollection has no features")
    if MAX_FEATURES > 0 and len(features) > MAX_FEATURES:
        raise HTTPException(
            status_code=413,
            detail=f"FeatureCollection has {len(features)} features; send at most {MAX_FEATURES} per request",
        )

    parsed: List[ParsedFeature] = []
    seen_ids: Dict[str, int] = {}
    for i, feature in enumerate(features):
        if not isinstance(feature, dict):
            raise HTTPException(status_code=400, detail=f"Feature #{i + 1} is not an object")
        fid = _feature_id(feature, i)
        if fid in seen_ids:
            # Keep ids unique so results can be matched back to fields.
            seen_ids[fid] += 1
            fid = f"{fid}-{seen_ids[fid]}"
        else:
            seen_ids[fid] = 1

        geometry = feature.get("geometry")
        if not isinstance(geometry, dict) or geometry.get("type") not in POLYGON_TYPES:
            raise HTTPException(
                status_code=400,
                detail=f"Feature '{fid}' must have a Polygon or MultiPolygon geometry",
            )
        try:
            geom_shape = shape(geometry)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Feature '{fid}' has malformed geometry: {e}")
        if not geom_shape.is_empty and not geom_shape.is_valid:
            # Hand-drawn and uploaded outlines often cross themselves; repair them rather than
            # failing every field in the request.
            repaired = _repair(geom_shape)
            if not repaired.is_empty and repaired.is_valid:
                geom_shape, geometry = repaired, mapping(repaired)
        if geom_shape.is_empty or not geom_shape.is_valid:
            raise HTTPException(
                status_code=400,
                detail=f"Feature '{fid}' has invalid geometry: {explain_validity(geom_shape)}",
            )

        parsed.append(ParsedFeature(
            id=fid,
            geometry=geometry,
            shape=geom_shape,
            properties=feature.get("properties") or {},
        ))
    return parsed
