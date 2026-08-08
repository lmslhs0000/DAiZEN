from fastapi import APIRouter, Query

from app.schemas.response import ApiResponse
from app.schemas.message import MessageRequest

router = APIRouter()


@router.get("/", response_model=ApiResponse)
def root():
    return ApiResponse(
        success=True,
        message="DAIZEN API is running",
        data=None,
    )


@router.post("/echo", response_model=ApiResponse)
def echo(request: MessageRequest):
    return ApiResponse(
        success=True,
        message="메시지를 성공적으로 받았습니다.",
        data={
            "message": request.message
        }
    )


@router.get("/users/{user_id}", response_model=ApiResponse)
def get_user(user_id: int):
    return ApiResponse(
        success=True,
        message="사용자 조회 성공",
        data={
            "user_id": user_id
        }
    )    


@router.get("/users", response_model=ApiResponse)
def get_users(page: int = 1, size: int = 10):
    return ApiResponse(
        success=True,
        message="사용자 목록 조회 성공",
        data={
            "page": page,
            "size": size
        }
    )


@router.get("/projects/{project_id}/tasks", response_model=ApiResponse)
def get_project_tasks(
    project_id: int,
    page: int = Query(1, ge=1),
    size: int = Query(10, ge=1, le=100),
):
    return ApiResponse(
        success=True,
        message="프로젝트 작업 목록 조회 성공",
        data={
            "project_id": project_id,
            "page": page,
            "size": size,
        },
    )