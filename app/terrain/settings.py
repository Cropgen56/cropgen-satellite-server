"""
Terrain runtime limits, read from the environment once at import.

Auth is handled by the server's shared CROPGEN_API_KEY (auth.py), so there is
no terrain-specific key here.
"""
from __future__ import annotations

import os
from dataclasses import dataclass


def _float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be a number, got {raw!r}") from exc


def _int(name: str, default: int) -> int:
    return int(_float(name, default))


@dataclass(frozen=True)
class Settings:
    min_area_ha: float
    max_area_ha: float
    max_extent_m: float
    max_window_days: int
    cache_ttl_s: int
    cache_max_items: int
    output_px: int


def load_settings() -> Settings:
    return Settings(
        min_area_ha=_float("TERRAIN_MIN_AREA_HA", 0.05),
        max_area_ha=_float("TERRAIN_MAX_AREA_HA", 500.0),
        max_extent_m=_float("TERRAIN_MAX_EXTENT_M", 5000.0),
        max_window_days=_int("TERRAIN_MAX_WINDOW_DAYS", 366),
        cache_ttl_s=_int("TERRAIN_CACHE_TTL_S", 24 * 3600),
        cache_max_items=_int("TERRAIN_CACHE_MAX_ITEMS", 128),
        output_px=_int("TERRAIN_OUTPUT_PX", 768),
    )
