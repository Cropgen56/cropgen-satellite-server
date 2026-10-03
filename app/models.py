from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field

class AvailabilityRequest(BaseModel):
    geometry: Dict[str, Any]
    start_date: str
    end_date: str
    provider: Optional[str] = "both"
    satellite: Optional[str] = "s2"

class AvailabilityItem(BaseModel):
    date: str
    cloud_cover: Optional[float] = None

class AvailabilityResponse(BaseModel):
    items: List[AvailabilityItem]

class CalculateRequest(BaseModel):
    geometry: Dict[str, Any]
    date: str
    index_name: str
    provider: Optional[str] = "both"
    satellite: Optional[str] = "s2"
    width: Optional[int] = 800
    height: Optional[int] = 800
    supersample: Optional[int] = 1
    smooth: Optional[bool] = False
    gaussian_sigma: Optional[float] = 1.0

class AreaStat(BaseModel):
    label: str
    hectares: float
    percent: float

class CalculateResponse(BaseModel):
    date: str
    index_name: str
    image_base64: str
    bounds: Optional[List[float]] = None
    legend: Optional[List[Dict[str, Any]]] = None
    area_stats: Optional[List[AreaStat]] = None


class NpkAvailabilityRequest(BaseModel):
    geometry: Dict[str, Any]
    date: str
    provider: Optional[str] = "both"
    satellite: Optional[str] = "s2"
    bbch_stage: Optional[float] = None
    stage_name: Optional[str] = None


class NpkNutrientAvailability(BaseModel):
    health_score: Optional[float] = None
    factor: float
    source_index: str


class NpkAvailabilityResponse(BaseModel):
    date: str
    provider: str
    satellite: str
    stage_context: Optional[Dict[str, Any]] = None
    nutrients: Dict[str, NpkNutrientAvailability]
    debug: Optional[Dict[str, Any]] = None


class CropHealthRequest(BaseModel):
    geometry: Dict[str, Any]
    date: str
    sowing_date: Optional[str] = None
    provider: Optional[str] = "both"
    satellite: Optional[str] = "s2"


class CropHealthResponse(BaseModel):
    health: int
    status: str
    ndvi: float
    ndre: float
    stress: str
    stage: str
    cloud_coverage: Optional[float] = None


class SocAnalysisRequest(BaseModel):
    geometry: Dict[str, Any]
    start_date: str
    end_date: str
    provider: Optional[str] = "both"
    satellite: Optional[str] = "s2"


class SocClassStat(BaseModel):
    pixels: int
    ha: float
    acres: float
    pct_area: float


class SocStats(BaseModel):
    mean_pct: Optional[float] = None
    min_pct: Optional[float] = None
    max_pct: Optional[float] = None
    std_pct: Optional[float] = None
    total_area_ha: Optional[float] = None
    total_area_acres: Optional[float] = None
    classes: Dict[str, SocClassStat] = {}
    unit: Optional[str] = None
    confidence: Optional[str] = None


class SocAnalysisResponse(BaseModel):
    date: str
    cloud_cover: Optional[float] = None
    image_base64: str
    soc_stats: SocStats
    metadata: Dict[str, Any]


class VraGroundSample(BaseModel):
    lat: float
    lon: float
    SOC: Optional[float] = None
    N: Optional[float] = None
    P: Optional[float] = None
    K: Optional[float] = None
    PH: Optional[float] = None
    EC: Optional[float] = None
    MOISTURE: Optional[float] = None
    CLAY: Optional[float] = None

    class Config:
        extra = "allow"


class VraAnalysisRequest(BaseModel):
    """Request body for cropgen_soil_vra.run_analysis."""

    geometry: Dict[str, Any]
    start_date: str
    end_date: str
    crop: str = "wheat"
    n_zones: int = Field(default=5, ge=2, le=7)
    min_patch_ha: float = 0.05
    include_images: bool = True
    include_prescription_geojson: bool = False
    label_mode: Literal["percent", "dose"] = "percent"
    ground_samples: Optional[List[VraGroundSample]] = None
    max_scenes: Optional[int] = Field(default=None, ge=2, le=40)
    district: Optional[str] = None
    state: Optional[str] = None
    auto_region: bool = True
    use_soilgrids: bool = False
    grid_cell_m: float = Field(default=20.0, ge=10.0, le=100.0)
    soc_method: Literal["published", "legacy"] = "published"
    zone_features: Optional[List[str]] = None
    # Ignored by the engine; kept so older clients do not 422.
    provider: Optional[str] = None
    satellite: Optional[str] = None
    zone_method: Optional[str] = None


class VraParamStat(BaseModel):
    mean: Optional[float] = None
    min: Optional[float] = None
    max: Optional[float] = None
    std: Optional[float] = None
    unit: Optional[str] = None
    source_composite: Optional[str] = None
    confidence: Optional[str] = None


class VraAnalysisResponse(BaseModel):
    crop: str
    date: Optional[str] = None
    cloud_cover: Optional[float] = None
    param_stats: Dict[str, VraParamStat]
    relative_index_note: Optional[str] = None
    vra_rates: Dict[str, Any]
    soc_stats: Optional[SocStats] = None
    zone_geojson: Optional[Dict[str, Any]] = None
    prescription_grid: Optional[Dict[str, Any]] = None
    prescription_geojson: Optional[Dict[str, Any]] = None
    region: Optional[Dict[str, Any]] = None
    soc_composite: Optional[Dict[str, Any]] = None
    zone_info: Optional[Dict[str, Any]] = None
    calibration: Optional[Dict[str, Any]] = None
    correlation: Optional[Dict[str, Any]] = None
    scenes_used: Optional[List[Dict[str, Any]]] = None
    images: Optional[Dict[str, str]] = None
    text_report: Optional[str] = None
    metadata: Dict[str, Any]
