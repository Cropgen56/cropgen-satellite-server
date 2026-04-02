import os

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from availability_dates_api import router as availability_router
from calculate_index_api import router as calculate_router
from timeseries_vegetation_api import router as veg_router
from timeseries_water_api import router as water_router
from auth import get_expected_api_key, validate_api_key

load_dotenv(override=True)  # override=True ensures .env always wins over shell/conda env vars

# Routes are mounted at /v4/api/... matching production nginx prefix.
app = FastAPI()
get_expected_api_key()

default_origins = [
    "http://localhost:3000",
    "http://localhost:3001",
    "http://localhost:5173",
    "http://localhost:5176",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:5176",
    "https://cropydeals.cropgenapp.com",
    "https://app.cropgenapp.com",
    "https://soilsense.com",
    "https://admin.cropgenapp.com",
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

@app.get("/")
def root():
    return {"message": "CropGen API v4 is running"}
