import os
import shutil
import uuid
from fastapi import HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from app.crud.data_file import DataFileCRUD
from app.models.data_file import DataFile
from app.models.project_member import ProjectRole
from app.models.user import User
from app.schemas.data_file import DataFileCreate
from app.services.activity_log_service import ActivityLogService
from app.services.bearing_service import BearingService
from app.services.dataset_service import DatasetService
from app.services.project_member_service import ProjectMemberService

# 파일이 저장될 기본 디렉토리 설정
UPLOAD_DIR = "uploads/bearings"


class DataFileService:
    @staticmethod
    def upload_file(
        db: Session,
        bearing_id: int,
        file: UploadFile,
        current_user: User,
    ) -> DataFile:
        # 1. 권한 검증 (MEMBER 이상만 업로드 가능)
        bearing = BearingService.get(db, bearing_id, current_user)
        dataset = DatasetService.get(db, bearing.dataset_id, current_user)
        ProjectMemberService.ensure_role(
            db, dataset.project_id, current_user.id, ProjectRole.MEMBER
        )

        # 2. 저장 디렉토리 생성 (없으면 자동 생성)
        os.makedirs(UPLOAD_DIR, exist_ok=True)

        # 3. 파일명 충돌 방지를 위한 고유 식별자(UUID) 적용 및 디스크 저장
        ext = os.path.splitext(file.filename)[1]
        unique_filename = f"{uuid.uuid4().hex}{ext}"
        saved_filepath = os.path.join(UPLOAD_DIR, unique_filename)

        with open(saved_filepath, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        file_size = os.path.getsize(saved_filepath)

        # 4. DB 기록
        obj_in = DataFileCreate(
            original_filename=file.filename,
            saved_filepath=saved_filepath,
            file_size=file_size,
            content_type=file.content_type,
        )
        data_file = DataFileCRUD.create(db, bearing_id, obj_in)

        # 5. 활동 로그 기록
        ActivityLogService.log_activity(
            db=db,
            project_id=dataset.project_id,
            user_id=current_user.id,
            action="FILE_UPLOADED",
            details=f"File '{file.filename}' uploaded to bearing '{bearing.name}'.",
        )

        return data_file

    @staticmethod
    def get(
        db: Session,
        file_id: int,
        current_user: User,
    ) -> DataFile:
        data_file = DataFileCRUD.get_by_id(db, file_id)
        if not data_file:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="File not found.",
            )
        
        # 권한 검증 (소속 프로젝트 멤버인지 간접 확인)
        BearingService.get(db, data_file.bearing_id, current_user)
        return data_file

    @staticmethod
    def get_all(
        db: Session,
        bearing_id: int,
        current_user: User,
        skip: int = 0,
        limit: int = 100,
    ) -> list[DataFile]:
        # 권한 검증
        BearingService.get(db, bearing_id, current_user)
        return DataFileCRUD.get_bearing_files(db, bearing_id, skip, limit)

    @staticmethod
    def delete(
        db: Session,
        file_id: int,
        current_user: User,
    ) -> None:
        data_file = DataFileService.get(db, file_id, current_user)
        bearing = BearingService.get(db, data_file.bearing_id, current_user)
        dataset = DatasetService.get(db, bearing.dataset_id, current_user)

        # 권한 검증 (ADMIN 이상만 파일 삭제 가능)
        ProjectMemberService.ensure_role(
            db, dataset.project_id, current_user.id, ProjectRole.ADMIN
        )

        # 1. 서버 디스크에서 실제 파일 삭제
        if os.path.exists(data_file.saved_filepath):
            os.remove(data_file.saved_filepath)

        # 2. DB에서 메타데이터 삭제
        DataFileCRUD.delete(db, data_file)

        # 3. 활동 로그 기록
        ActivityLogService.log_activity(
            db=db,
            project_id=dataset.project_id,
            user_id=current_user.id,
            action="FILE_DELETED",
            details=f"File '{data_file.original_filename}' deleted from bearing '{bearing.name}'.",
        )