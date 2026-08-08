from fastapi import APIRouter, Depends, File, Query, UploadFile, status
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.core.dependencies import get_current_user
from app.db.database import get_db
from app.models.user import User
from app.schemas.data_file import DataFileResponse
from app.services.data_file_service import DataFileService

router = APIRouter()


@router.post(
    "/bearing/{bearing_id}",
    response_model=DataFileResponse,
    status_code=status.HTTP_201_CREATED,
)
def upload_data_file(
    bearing_id: int,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """베어링 기기에 새로운 측정 데이터 파일(CSV 등)을 업로드합니다."""
    return DataFileService.upload_file(db, bearing_id, file, current_user)


@router.get(
    "/bearing/{bearing_id}",
    response_model=list[DataFileResponse],
)
def get_bearing_files(
    bearing_id: int,
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=500),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """특정 베어링에 업로드된 모든 측정 파일 목록(메타데이터)을 조회합니다."""
    return DataFileService.get_all(db, bearing_id, current_user, skip, limit)


@router.get(
    "/{file_id}",
    response_model=DataFileResponse,
)
def get_data_file_info(
    file_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """단일 측정 파일의 메타데이터(이름, 용량, 업로드 시간 등)를 조회합니다."""
    return DataFileService.get(db, file_id, current_user)


@router.get(
    "/{file_id}/download",
)
def download_data_file(
    file_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """서버에 저장된 실제 측정 데이터 파일을 클라이언트로 다운로드합니다."""
    data_file = DataFileService.get(db, file_id, current_user)
    return FileResponse(
        path=data_file.saved_filepath,
        filename=data_file.original_filename,
        media_type=data_file.content_type or "application/octet-stream",
    )


@router.delete(
    "/{file_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
def delete_data_file(
    file_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """측정 파일을 서버에서 완전히 삭제합니다."""
    DataFileService.delete(db, file_id, current_user)