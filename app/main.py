from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware  # 신규 추가: CORS 미들웨어 임포트

from app.core.config import settings

from app.api.activity_logs import router as activity_log_router
from app.api.ai_tasks import router as ai_task_router
from app.api.auth import router as auth_router
from app.api.bearings import router as bearing_router
from app.api.data_files import router as data_file_router
from app.api.datasets import router as dataset_router
from app.api.project_invites import router as project_invite_router
from app.api.project_members import router as project_member_router
from app.api.projects import router as project_router
from app.api.users import router as user_router
from app.db.database import Base, engine
from app.routers.root import router

app = FastAPI(title=settings.app_name)

# 신규 추가: CORS 미들웨어 등록 (가장 먼저 요청을 검사하는 문지기 역할)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.BACKEND_CORS_ORIGINS, # 허용할 프론트엔드 주소
    allow_credentials=True,                      # 쿠키, 인증 헤더 허용
    allow_methods=["*"],                         # 모든 HTTP 메서드(GET, POST 등) 허용
    allow_headers=["*"],                         # 모든 헤더 허용
)

Base.metadata.create_all(bind=engine)

app.include_router(
    router,
    prefix="/api/v1",
    tags=["Root"],
)

app.include_router(
    user_router,
    prefix="/api/v1/users",
    tags=["Users"],
)

app.include_router(
    auth_router,
    prefix="/api/v1/auth",
    tags=["Authentication"],
)

app.include_router(
    project_router,
    prefix="/api/v1/projects",
    tags=["Projects"],
)

app.include_router(
    project_member_router,
    prefix="/api/v1/project-members",
    tags=["Project Members"],
)

app.include_router(
    project_invite_router,
    prefix="/api/v1/project-invites",
    tags=["Project Invites"],
)

app.include_router(
    activity_log_router,
    prefix="/api/v1/activity-logs",
    tags=["Activity Logs"],
)

app.include_router(
    dataset_router,
    prefix="/api/v1/datasets",
    tags=["Datasets"],
)

app.include_router(
    bearing_router,
    prefix="/api/v1/bearings",
    tags=["Bearings"],
)

app.include_router(
    data_file_router,
    prefix="/api/v1/data-files",
    tags=["Data Files"],
)

app.include_router(
    ai_task_router,
    prefix="/api/v1/ai-tasks",
    tags=["AI Tasks"],
)