from fastapi import APIRouter, Depends, Query, status
from sqlalchemy.orm import Session

from app.core.dependencies import get_current_user
from app.db.database import get_db
from app.models.user import User
from app.schemas.ai_task import AITaskCreate, AITaskResponse, AITaskUpdate
from app.services.ai_task_service import AITaskService

router = APIRouter()


@router.post(
    "/bearing/{bearing_id}",
    response_model=AITaskResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_ai_task(
    bearing_id: int,
    task_in: AITaskCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """특정 베어링에 대한 새로운 AI 분석 작업을 요청(생성)합니다."""
    return AITaskService.create(db, bearing_id, task_in, current_user)


@router.get(
    "/bearing/{bearing_id}",
    response_model=list[AITaskResponse],
)
def get_bearing_ai_tasks(
    bearing_id: int,
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=500),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """특정 베어링에 등록된 모든 AI 분석 작업 목록을 조회합니다."""
    return AITaskService.get_all(db, bearing_id, current_user, skip, limit)


@router.get(
    "/{task_id}",
    response_model=AITaskResponse,
)
def get_ai_task(
    task_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """단일 AI 분석 작업의 상세 정보(진행 상태, 결과 데이터 등)를 조회합니다."""
    return AITaskService.get(db, task_id, current_user)


@router.patch(
    "/{task_id}",
    response_model=AITaskResponse,
)
def update_ai_task(
    task_id: int,
    task_in: AITaskUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """AI 분석 작업의 상태(status)나 결과 데이터(result_data)를 갱신합니다."""
    return AITaskService.update(db, task_id, task_in, current_user)


@router.delete(
    "/{task_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
def delete_ai_task(
    task_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """AI 분석 기록을 완전히 삭제합니다."""
    AITaskService.delete(db, task_id, current_user)