from pydantic_settings import BaseSettings
from typing import List

class Settings(BaseSettings):
    API_V1_STR: str = "/v1"
    PROJECT_NAME: str = "CrediCouncil AI API"
    PROJECT_DESCRIPTION: str = "Credit Scoring & Creditworthiness Assessment for Underbanked & Micro SMEs"
    VERSION: str = "0.1.0"
    
    # CORS setup
    BACKEND_CORS_ORIGINS: List[str] = ["*"]
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()
