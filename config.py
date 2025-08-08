"""
Configuration settings for the Candidate Recommendation Engine.
"""
import os
from typing import List
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # API Configuration
    openai_api_key: str = Field("", env="OPENAI_API_KEY")  # Optional
    cohere_api_key: str = Field("", env="COHERE_API_KEY")  # Optional
    hf_api_token: str = Field("", env="HF_API_TOKEN")  # Optional
    
    # Application Settings
    app_name: str = Field("Candidate Recommendation Engine", env="APP_NAME")
    app_version: str = Field("1.0.0", env="APP_VERSION")
    debug: bool = Field(False, env="DEBUG")
    
    # API Settings
    api_host: str = Field("0.0.0.0", env="API_HOST")
    api_port: int = Field(8000, env="API_PORT")
    
    # Streamlit Settings
    streamlit_host: str = Field("0.0.0.0", env="STREAMLIT_HOST")
    streamlit_port: int = Field(8501, env="STREAMLIT_PORT")
    
    # Vector Database Settings
    vector_dimension: int = Field(1536, env="VECTOR_DIMENSION")  # OpenAI ada-002
    max_candidates: int = Field(50, env="MAX_CANDIDATES")
    
    # File Upload Settings
    max_file_size: int = Field(10485760, env="MAX_FILE_SIZE")  # 10MB
    allowed_extensions: List[str] = Field(
        default=[".pdf", ".docx", ".txt"]
    )
    
    # Rate Limiting
    rate_limit_requests: int = Field(100, env="RATE_LIMIT_REQUESTS")
    rate_limit_window: int = Field(3600, env="RATE_LIMIT_WINDOW")  # 1 hour
    
    # AI Settings
    embedding_model: str = Field("text-embedding-ada-002", env="EMBEDDING_MODEL")
    chat_model: str = Field("gpt-3.5-turbo", env="CHAT_MODEL")
    max_summary_length: int = Field(200, env="MAX_SUMMARY_LENGTH")
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()
