from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from app.crud.dataset import DatasetCRUD
from app.models.dataset import Dataset
from app.models.project_member import ProjectRole
from app.models.user import User
from app.schemas.dataset import DatasetCreate, DatasetUpdate
from app.services.activity_log_service import ActivityLogService
from app.services.project_member_service import ProjectMemberService
from app.services.project_service import ProjectService


class DatasetService:
    @staticmethod
    def create(
        db: Session,
        project_id: int,
        dataset_in: DatasetCreate,
        current_user: User,
    ) -> Dataset:
        # 1. 프로젝트 존재 여부 확인
        ProjectService.get(db, project_id)
        
        # 2. 권한 확인 (최소 MEMBER 이상만 데이터셋 생성 가능)
        ProjectMemberService.ensure_role(
            db, project_id, current_user.id, ProjectRole.MEMBER
        )

        dataset = DatasetCRUD.create(db, project_id, dataset_in)

        # 3. 활동 로그 기록
        ActivityLogService.log_activity(
            db=db,
            project_id=project_id,
            user_id=current_user.id,
            action="DATASET_CREATED",
            details=f"Dataset '{dataset.name}' was created.",
        )
        return dataset

    @staticmethod
    def get(
        db: Session,
        dataset_id: int,
        current_user: User,
    ) -> Dataset:
        dataset = DatasetCRUD.get_by_id(db, dataset_id)
        if not dataset:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Dataset not found.",
            )

        # 권한 확인 (프로젝트 소속 멤버인지 검증)
        ProjectMemberService.ensure_member(db, dataset.project_id, current_user.id)
        return dataset

    @staticmethod
    def get_all(
        db: Session,
        project_id: int,
        current_user: User,
        skip: int = 0,
        limit: int = 100,
    ) -> list[Dataset]:
        ProjectService.get(db, project_id)
        ProjectMemberService.ensure_member(db, project_id, current_user.id)
        
        return DatasetCRUD.get_project_datasets(db, project_id, skip, limit)

    @staticmethod
    def update(
        db: Session,
        dataset_id: int,
        dataset_in: DatasetUpdate,
        current_user: User,
    ) -> Dataset:
        dataset = DatasetService.get(db, dataset_id, current_user)
        
        ProjectMemberService.ensure_role(
            db, dataset.project_id, current_user.id, ProjectRole.MEMBER
        )

        updated = DatasetCRUD.update(db, dataset, dataset_in)
        
        ActivityLogService.log_activity(
            db=db,
            project_id=dataset.project_id,
            user_id=current_user.id,
            action="DATASET_UPDATED",
            details=f"Dataset '{dataset.name}' was updated.",
        )
        return updated

    @staticmethod
    def delete(
        db: Session,
        dataset_id: int,
        current_user: User,
    ) -> None:
        dataset = DatasetService.get(db, dataset_id, current_user)
        
        # 삭제는 최소 ADMIN 권한 요구
        ProjectMemberService.ensure_role(
            db, dataset.project_id, current_user.id, ProjectRole.ADMIN
        )

        ActivityLogService.log_activity(
            db=db,
            project_id=dataset.project_id,
            user_id=current_user.id,
            action="DATASET_DELETED",
            details=f"Dataset '{dataset.name}' was deleted.",
        )
        
        DatasetCRUD.delete(db, dataset)