import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    app_name: str = "DAIZEN API"
    
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL", 
        "postgresql://daizen_user:daizen_password@localhost:5432/daizen_db"
    )
    
    REDIS_URL: str = os.getenv(
        "REDIS_URL", 
        "redis://localhost:6379/0"
    )
    
    SECRET_KEY: str = os.getenv(
        "SECRET_KEY",
        "",
    )
    
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    # 🌟 신규 추가: 프론트엔드 접속을 허용할 주소 목록
    # 로컬 개발용 포트(React, Vue 등) 및 추후 상용화될 도메인을 입력함
    BACKEND_CORS_ORIGINS: list[str] = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
    ]

settings = Settings()