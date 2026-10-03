"""
Terrain layers: elevation, slope, aspect, water flow, water accumulation,
wetness and erosion risk, as AOI-clipped PNG overlays with legends.

    POST /v4/api/terrain/{layer}   one layer
    POST /v4/api/terrain           every layer in one response

Every call takes the same body:
    {"aoi": <GeoJSON Polygon/MultiPolygon or Feature>,
     "start_date": "YYYY-MM-DD", "end_date": "YYYY-MM-DD"}

Only erosion_risk uses the dates — they select the satellite pass that supplies
crop cover. Terrain is computed once per AOI and cached, so after the first call
the other layers for the same field come straight from memory.
"""
from __future__ import annotations

import logging
from datetime import date, datetime, timezone
from enum import Enum
from typing import Any

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

from app.terrain import engine as E
from app.terrain import render as R
from app.terrain.cache import SingleFlightCache
from app.terrain.settings import load_settings

log = logging.getLogger("terrain.api")

SETTINGS = load_settings()

router = APIRouter()

_TERRAIN: SingleFlightCache[E.Terrain] = SingleFlightCache(
    SETTINGS.cache_max_items, SETTINGS.cache_ttl_s)
_COVER: SingleFlightCache[E.Cover] = SingleFlightCache(
    SETTINGS.cache_max_items, SETTINGS.cache_ttl_s)

S2_ARCHIVE_START = date(2017, 1, 1)


# ──────────────────────────────────────────────────────────────────────────────
# Schemas
# ──────────────────────────────────────────────────────────────────────────────
class Layer(str, Enum):
    elevation = "elevation"
    slope = "slope"
    aspect = "aspect"
    water_flow = "water_flow"
    water_accumulation = "water_accumulation"
    wetness = "wetness"
    erosion_risk = "erosion_risk"


class TerrainRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_extra={"example": {
        "aoi": {"type": "Polygon", "coordinates": [[
            [77.15664790353888, 20.135857699751913], [77.15749548158759, 20.13559579948349],
            [77.15725944719428, 20.134800022897924], [77.15636895380133, 20.13477987660266],
            [77.15664790353888, 20.135857699751913]]]},
        "start_date": "2026-08-01", "end_date": "2026-08-31"}})

    aoi: dict[str, Any] = Field(..., description="GeoJSON Polygon/MultiPolygon or Feature, EPSG:4326")
    start_date: date
    end_date: date
    _geom: Any = PrivateAttr(default=None)

    @model_validator(mode="after")
    def _validate(self) -> "TerrainRequest":
        today = datetime.now(timezone.utc).date()
        if self.start_date > self.end_date:
            raise ValueError("start_date must be on or before end_date")
        if self.end_date > today:
            raise ValueError(f"end_date cannot be in the future (today is {today.isoformat()})")
        if self.start_date < S2_ARCHIVE_START:
            raise ValueError(f"start_date must be on or after {S2_ARCHIVE_START.isoformat()}")
        if (self.end_date - self.start_date).days > SETTINGS.max_window_days:
            raise ValueError(f"date range cannot exceed {SETTINGS.max_window_days} days")
        try:
            geom, _area = E.parse_aoi(self.aoi, min_area_ha=SETTINGS.min_area_ha,
                                      max_area_ha=SETTINGS.max_area_ha,
                                      max_extent_m=SETTINGS.max_extent_m)
        except E.AOIError as exc:
            raise ValueError(str(exc)) from exc
        self._geom = geom
        return self

    @property
    def geom(self):
        return self._geom


class Bounds(BaseModel):
    west: float
    south: float
    east: float
    north: float


class LayerResponse(BaseModel):
    layer: Layer
    name: str
    image_base64: str = Field(..., description="PNG, EPSG:4326, transparent outside the AOI")
    bounds: Bounds = Field(..., description="Image corners; place the PNG on these exactly")
    legend: dict[str, Any]
    warnings: list[str] | None = None


class AllLayersResponse(BaseModel):
    bounds: Bounds
    layers: list[LayerResponse]
    warnings: list[str] | None = None


# ──────────────────────────────────────────────────────────────────────────────
# Core
# ──────────────────────────────────────────────────────────────────────────────
def _terrain_for(req: TerrainRequest) -> tuple[str, E.Terrain]:
    key = E.aoi_key(req.geom)
    return key, _TERRAIN.get_or_compute(key, lambda: E.compute_terrain(req.geom))


def _cover_for(key: str, T: E.Terrain, req: TerrainRequest) -> E.Cover:
    ckey = (key, req.start_date.isoformat(), req.end_date.isoformat())
    return _COVER.get_or_compute(ckey, lambda: E.compute_cover(T, req.start_date, req.end_date))


def _render(layer: str, req: TerrainRequest) -> dict:
    key, T = _terrain_for(req)
    C = _cover_for(key, T, req) if layer == Layer.erosion_risk.value else None
    return R.render_layer(layer, T, C, SETTINGS.output_px)


def _guard(fn, aoi_hash_src):
    """Map engine failures onto HTTP errors without leaking internals."""
    try:
        return fn()
    except E.DataSourceError as exc:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        # log the AOI hash, not its coordinates
        log.exception("render failed for aoi %s", E.aoi_key(aoi_hash_src)[:12])
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR,
                            "Terrain processing failed") from exc


# ──────────────────────────────────────────────────────────────────────────────
# Routes
# Endpoints are plain `def`, so FastAPI runs them in its thread pool and the
# CPU-bound work never blocks the event loop.
# ──────────────────────────────────────────────────────────────────────────────
@router.post("/{layer}", response_model=LayerResponse, response_model_exclude_none=True)
def terrain_layer(layer: Layer, req: TerrainRequest) -> dict:
    return _guard(lambda: _render(layer.value, req), req.geom)


@router.post("", response_model=AllLayersResponse, response_model_exclude_none=True)
def terrain_all(req: TerrainRequest) -> dict:
    def _all():
        layers = [_render(lay.value, req) for lay in Layer]
        # terrain warnings are identical on every layer; surface them once
        shared = layers[0].get("warnings") or []
        shared = [w for w in shared if all(w in (l.get("warnings") or []) for l in layers)]
        for l in layers:
            rest = [w for w in (l.get("warnings") or []) if w not in shared]
            l["warnings"] = rest or None
        return {"bounds": layers[0]["bounds"], "layers": layers, "warnings": shared or None}
    return _guard(_all, req.geom)
