import os

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv(override=True)  # override=True ensures .env always wins over shell/conda env vars

from app.routers.availability_dates_api import router as availability_router
from app.routers.calculate_index_api import router as calculate_router
from app.routers.npk_availability_api import router as npk_router
from app.routers.timeseries_vegetation_api import router as veg_router
from app.routers.timeseries_water_api import router as water_router
from app.routers.crop_health_api import router as crop_health_router
from app.routers.soc_api import router as soc_router
from app.routers.vra_api import router as vra_router
from app.routers.terrain_api import router as terrain_router
from app.routers.multi_availability_api import router as multi_availability_router
from app.routers.multi_calculate_index_api import router as multi_calculate_router
from app.auth import get_expected_api_key, validate_api_key

# Docs + OpenAPI must live under /v4/ so the browser requests /v4/openapi.json (same prefix as
# /v4/docs). Default /openapi.json hits the site root and is often routed to the wrong upstream
# (502 Bad Gateway). Nginx should forward full paths starting with /v4 to this app, e.g.:
#   location /v4/ { proxy_pass http://127.0.0.1:8001; ... }   # no trailing slash after port
app = FastAPI(
    title="CropGen Satellite API",
    docs_url="/v4/docs",
    openapi_url="/v4/openapi.json",
    redoc_url="/v4/redoc",
)
get_expected_api_key()

default_origins = [
    "http://localhost:3000",
    "http://localhost:3001",
    "http://localhost:5173",
    "http://localhost:5174",
    "http://localhost:5175",
    "http://localhost:5176",
    "http://localhost:5177",
    "http://localhost:5178",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:5174",
    "http://127.0.0.1:5175",
    "http://127.0.0.1:5176",
    "http://127.0.0.1:5177",
    "http://127.0.0.1:5178",
    "https://cropydeals.cropgenapp.com",
    "https://app.cropgenapp.com",
    "https://admin.cropgenapp.com",
    "https://biodrops.cropgenapp.com",
    "https://satagro.ai",
    "https://app.satagro.ai",
]

env_origins = [
    origin.strip()
    for origin in os.getenv("CORS_ALLOWED_ORIGINS", "").split(",")
    if origin.strip()
]

origins = list(dict.fromkeys(default_origins + env_origins))

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🔐 Apply auth to the routers
app.include_router(
    availability_router,
    prefix="/v4/api/availability",
    tags=["Availability"],
    dependencies=[Depends(validate_api_key)],
)

app.include_router(
    calculate_router,
    prefix="/v4/api/calculate",
    tags=["Calculate Index"],
    dependencies=[Depends(validate_api_key)],
)

app.include_router(
    multi_availability_router,
    prefix="/v4/api/multi/availability",
    tags=["Multi-Polygon Availability"],
    dependencies=[Depends(validate_api_key)],
)

app.include_router(
    multi_calculate_router,
    prefix="/v4/api/multi/calculate",
    tags=["Multi-Polygon Calculate Index"],
    dependencies=[Depends(validate_api_key)],
)

app.include_router(
    veg_router,
    prefix="/v4/api/timeseries/vegetation",
    tags=["Vegetation Timeseries"],
    dependencies=[Depends(validate_api_key)],
)

app.include_router(
    water_router,
    prefix="/v4/api/timeseries/water",
    tags=["Water Timeseries"],
    dependencies=[Depends(validate_api_key)],
)

app.include_router(
    npk_router,
    prefix="/v4/api/npk",
    tags=["NPK Availability"],
    dependencies=[Depends(validate_api_key)],
)

app.include_router(
    crop_health_router,
    prefix="/v4/api/crop-health",
    tags=["Crop Health"],
    dependencies=[Depends(validate_api_key)],
)

app.include_router(
    soc_router,
    prefix="/v4/api/soc",
    tags=["SOC"],
    dependencies=[Depends(validate_api_key)],
)

app.include_router(
    vra_router,
    prefix="/v4/api/vra",
    tags=["VRA"],
    dependencies=[Depends(validate_api_key)],
)

# Same engine and routes as /vra (cropgen_soil_vra). Kept so older clients
# that already call /soil-vra keep working.
app.include_router(
    vra_router,
    prefix="/v4/api/soil-vra",
    tags=["VRA"],
    dependencies=[Depends(validate_api_key)],
)

app.include_router(
    terrain_router,
    prefix="/v4/api/terrain",
    tags=["Terrain"],
    dependencies=[Depends(validate_api_key)],
)

@app.get("/")
def root():
    return {"message": "CropGen API v4 is running"}
