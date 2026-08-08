from fastapi import APIRouter, Depends, Query, status
from sqlalchemy.orm import Session

from app.core.dependencies import get_current_user
from app.db.database import get_db
from app.models.user import User
from app.schemas.bearing import BearingCreate, BearingResponse, BearingUpdate
from app.services.bearing_service import BearingService

router = APIRouter()


@router.post(
    "/dataset/{dataset_id}",
    response_model=BearingResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_bearing(
    dataset_id: int,
    bearing_in: BearingCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """데이터셋 내에 새로운 베어링(기기/센서) 메타데이터를 추가합니다."""
    return BearingService.create(db, dataset_id, bearing_in, current_user)


@router.get(
    "/dataset/{dataset_id}",
    response_model=list[BearingResponse],
)
def get_dataset_bearings(
    dataset_id: int,
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=500),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """특정 데이터셋에 속한 베어링 목록을 조회합니다."""
    return BearingService.get_all(db, dataset_id, current_user, skip, limit)


@router.get(
    "/{bearing_id}",
    response_model=BearingResponse,
)
def get_bearing(
    bearing_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """단일 베어링 상세 정보를 조회합니다."""
    return BearingService.get(db, bearing_id, current_user)


@router.patch(
    "/{bearing_id}",
    response_model=BearingResponse,
)
def update_bearing(
    bearing_id: int,
    bearing_in: BearingUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """베어링 메타데이터(이름, 상태 등)를 수정합니다."""
    return BearingService.update(db, bearing_id, bearing_in, current_user)


@router.delete(
    "/{bearing_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
def delete_bearing(
    bearing_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """베어링 메타데이터를 삭제합니다."""
    BearingService.delete(db, bearing_id, current_user)