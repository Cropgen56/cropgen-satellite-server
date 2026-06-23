from typing import Any, Dict, List, Optional
from pydantic import BaseModel

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
    mean_pct: float
    min_pct: float
    max_pct: float
    std_pct: float
    total_area_ha: float
    total_area_acres: float
    classes: Dict[str, SocClassStat]


class SocAnalysisResponse(BaseModel):
    date: str
    cloud_cover: Optional[float] = None
    image_base64: str
    soc_stats: SocStats
    metadata: Dict[str, Any]


class VraZoneRate(BaseModel):
    pixel_count: int
    area_ha: float
    area_acres: float
    nutrient_dose_kg_ha: float
    product: str
    product_dose_kg_ha: float
    total_product_kg: float


class VraAnalysisRequest(BaseModel):
    geometry: Dict[str, Any]
    start_date: str
    end_date: str
    crop: str = "wheat"
    provider: Optional[str] = "both"
    satellite: Optional[str] = "s2"
    include_images: bool = True


class VraAnalysisResponse(BaseModel):
    date: str
    crop: str
    cloud_cover: Optional[float] = None
    vra_rates: Dict[str, Dict[str, VraZoneRate]]
    soc_stats: SocStats
    images: Optional[Dict[str, str]] = None
    text_report: Optional[str] = None
    metadata: Dict[str, Any]
