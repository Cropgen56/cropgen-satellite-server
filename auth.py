
from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader
import os
import secrets

API_KEY_NAME = "x-api-key"

api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)


def get_expected_api_key() -> str:
    api_key = os.getenv("CROPGEN_API_KEY")
    if not api_key:
        raise RuntimeError("CROPGEN_API_KEY environment variable is required")
    return api_key


async def validate_api_key(api_key: str = Security(api_key_header)):
    expected_api_key = get_expected_api_key()
    if not api_key or not secrets.compare_digest(api_key, expected_api_key):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied. This CropGen API is protected. Please provide a valid API key to continue",
        )
    return api_key
