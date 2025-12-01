
from fastapi import Security, HTTPException, status
from fastapi.security import APIKeyHeader
import os

# Get API key from env var or config
API_KEY = os.getenv("CROPGEN_API_KEY", "CROPGEN_230498adklfjadsljf")  
API_KEY_NAME = "x-api-key"

api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

async def validate_api_key(api_key: str = Security(api_key_header)):
    if not api_key or api_key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied. This CropGen API is protected. Please provide a valid API key to continue",
        )
    return api_key
