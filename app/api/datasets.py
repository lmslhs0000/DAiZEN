from fastapi import APIRouter, Depends, Query, status
from sqlalchemy.orm import Session

from app.core.dependencies import get_current_user
from app.db.database import get_db
from app.models.user import User
from app.schemas.dataset import DatasetCreate, DatasetResponse, DatasetUpdate
from app.services.dataset_service import DatasetService

router = APIRouter()


@router.post(
    "/project/{project_id}",
    response_model=DatasetResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_dataset(
    project_id: int,
    dataset_in: DatasetCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """프로젝트 내에 새로운 데이터셋을 생성합니다."""
    return DatasetService.create(db, project_id, dataset_in, current_user)


@router.get(
    "/project/{project_id}",
    response_model=list[DatasetResponse],
)
def get_project_datasets(
    project_id: int,
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=500),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """특정 프로젝트의 모든 데이터셋 목록을 조회합니다."""
    return DatasetService.get_all(db, project_id, current_user, skip, limit)


@router.get(
    "/{dataset_id}",
    response_model=DatasetResponse,
)
def get_dataset(
    dataset_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """단일 데이터셋의 상세 정보를 조회합니다."""
    return DatasetService.get(db, dataset_id, current_user)


@router.patch(
    "/{dataset_id}",
    response_model=DatasetResponse,
)
def update_dataset(
    dataset_id: int,
    dataset_in: DatasetUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """데이터셋 정보를 수정합니다."""
    return DatasetService.update(db, dataset_id, dataset_in, current_user)


@router.delete(
    "/{dataset_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
def delete_dataset(
    dataset_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """데이터셋을 삭제합니다."""
    DatasetService.delete(db, dataset_id, current_user)