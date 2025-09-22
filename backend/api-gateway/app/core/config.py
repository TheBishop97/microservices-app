# backend/api-gateway/app/core/config.py
from pydantic_settings import BaseSettings
from typing import List
import os


class Settings(BaseSettings):
    # API Configuration
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "API Gateway (BFF)"
    PROJECT_VERSION: str = "1.0.0"
    
    # Service URLs
    AUTH_SERVICE_URL: str = os.getenv(
        "AUTH_SERVICE_URL", 
        "http://auth-service:8001"
    )
    PRODUCT_SERVICE_URL: str = os.getenv(
        "PRODUCT_SERVICE_URL", 
        "http://product-service:8002"
    )
    ORDER_SERVICE_URL: str = os.getenv(
        "ORDER_SERVICE_URL", 
        "http://order-service:8003"
    )
    
    # CORS Configuration
    BACKEND_CORS_ORIGINS: List[str] = [
        "http://localhost:3000",  # Next.js frontend
        "https://localhost:3000",
        "http://127.0.0.1:3000",
    ]
    
    # Rate Limiting
    RATE_LIMIT_PER_MINUTE: int = int(os.getenv("RATE_LIMIT_PER_MINUTE", "100"))
    
    # Timeouts (seconds)
    SERVICE_TIMEOUT: float = float(os.getenv("SERVICE_TIMEOUT", "10.0"))
    SERVICE_CONNECT_TIMEOUT: float = float(os.getenv("SERVICE_CONNECT_TIMEOUT", "5.0"))
    
    # Environment
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"
    
    class Config:
        case_sensitive = True
        env_file = ".env"


# Create global settings instance
settings = Settings()
