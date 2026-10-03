"""Parse a GeoJSON FeatureCollection of field polygons for the multi-polygon APIs."""
from dataclasses import dataclass
from typing import Any, Dict, List

from fastapi import HTTPException
from shapely.geometry import shape
from shapely.geometry.base import BaseGeometry
from shapely.validation import explain_validity

POLYGON_TYPES = {"Polygon", "MultiPolygon"}


@dataclass
class ParsedFeature:
    id: str
    geometry: Dict[str, Any]
    shape: BaseGeometry
    properties: Dict[str, Any]


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
