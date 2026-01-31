"""
IMSKOS Backend - Configuration Module
Handles environment variables with graceful fallback to MOCK MODE.
"""
import os
from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # Application
    app_name: str = "IMSKOS API"
    app_version: str = "1.1.0"
    debug: bool = False
    environment: str = "development"
    
    # Supabase
    supabase_url: Optional[str] = None
    supabase_anon_key: Optional[str] = None
    supabase_service_role_key: Optional[str] = None
    
    # DataStax Astra DB
    astra_db_application_token: Optional[str] = None
    astra_db_id: Optional[str] = None
    astra_db_region: Optional[str] = None
    
    # LLM Providers
    groq_api_key: Optional[str] = None
    huggingface_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    
    # Storage
    s3_endpoint: Optional[str] = None
    s3_access_key: Optional[str] = None
    s3_secret_key: Optional[str] = None
    s3_bucket: Optional[str] = None
    
    # Redis
    upstash_redis_rest_url: Optional[str] = None
    upstash_redis_rest_token: Optional[str] = None
    
    # Security
    jwt_secret: Optional[str] = None
    
    # Monitoring
    sentry_dsn: Optional[str] = None
    
    # Notifications
    sendgrid_api_key: Optional[str] = None
    slack_webhook_url: Optional[str] = None
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
    
    def get_mock_status(self) -> dict[str, bool]:
        """Return which services are in mock mode due to missing env vars."""
        return {
            "supabase": not all([self.supabase_url, self.supabase_anon_key]),
            "astra_db": not all([self.astra_db_application_token, self.astra_db_id]),
            "groq": not self.groq_api_key,
            "huggingface": not self.huggingface_api_key,
            "redis": not all([self.upstash_redis_rest_url, self.upstash_redis_rest_token]),
            "s3": not all([self.s3_endpoint, self.s3_access_key, self.s3_secret_key]),
        }
    
    def log_mock_mode_warnings(self) -> list[str]:
        """Generate log messages for missing environment variables."""
        warnings = []
        mock_status = self.get_mock_status()
        
        env_var_mapping = {
            "supabase": ["SUPABASE_URL", "SUPABASE_ANON_KEY"],
            "astra_db": ["ASTRA_DB_APPLICATION_TOKEN", "ASTRA_DB_ID"],
            "groq": ["GROQ_API_KEY"],
            "huggingface": ["HUGGINGFACE_API_KEY"],
            "redis": ["UPSTASH_REDIS_REST_URL", "UPSTASH_REDIS_REST_TOKEN"],
            "s3": ["S3_ENDPOINT", "S3_ACCESS_KEY", "S3_SECRET_KEY"],
        }
        
        for service, is_mock in mock_status.items():
            if is_mock:
                for var in env_var_mapping.get(service, []):
                    if not os.getenv(var):
                        warnings.append(f"MOCK MODE: missing {var}")
        
        return warnings


@lru_cache()
def get_settings() -> Settings:
    """Cached settings instance."""
    return Settings()
