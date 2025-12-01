from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from availability_dates_api import router as availability_router
from calculate_index_api import router as calculate_router
from timeseries_vegetation_api import router as veg_router
from timeseries_water_api import router as water_router
from auth import validate_api_key 

app = FastAPI(root_path="/v4")

origins = [
    "http://localhost:3000",
    "http://localhost:5173",
    "https://cropydeals.cropgenapp.com",
    "https://app.cropgenapp.com",
    "https://soilsense.com",
]

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
    prefix="/api/availability",
    tags=["Availability"],
    dependencies=[Depends(validate_api_key)],
)

app.include_router(
    calculate_router,
    prefix="/api/calculate",
    tags=["Calculate Index"],
    dependencies=[Depends(validate_api_key)],
)

app.include_router(
    veg_router,
    prefix="/api/timeseries/vegetation",
    tags=["Vegetation Timeseries"],
    dependencies=[Depends(validate_api_key)],
)

app.include_router(
    water_router,
    prefix="/api/timeseries/water",
    tags=["Water Timeseries"],
    dependencies=[Depends(validate_api_key)],
)

@app.get("/")
def root():
    return {"message": "CropGen API v4 is running"}
