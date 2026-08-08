from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from app.crud.bearing import BearingCRUD
from app.models.bearing import Bearing
from app.models.project_member import ProjectRole
from app.models.user import User
from app.schemas.bearing import BearingCreate, BearingUpdate
from app.services.activity_log_service import ActivityLogService
from app.services.dataset_service import DatasetService
from app.services.project_member_service import ProjectMemberService


class BearingService:
    @staticmethod
    def create(
        db: Session,
        dataset_id: int,
        bearing_in: BearingCreate,
        current_user: User,
    ) -> Bearing:
        dataset = DatasetService.get(db, dataset_id, current_user)
        
        ProjectMemberService.ensure_role(
            db, dataset.project_id, current_user.id, ProjectRole.MEMBER
        )

        bearing = BearingCRUD.create(db, dataset_id, bearing_in)

        ActivityLogService.log_activity(
            db=db,
            project_id=dataset.project_id,
            user_id=current_user.id,
            action="BEARING_CREATED",
            details=f"Bearing '{bearing.name}' added to dataset '{dataset.name}'.",
        )
        return bearing

    @staticmethod
    def get(
        db: Session,
        bearing_id: int,
        current_user: User,
    ) -> Bearing:
        bearing = BearingCRUD.get_by_id(db, bearing_id)
        if not bearing:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Bearing not found.",
            )

        # 데이터셋 조회를 통해 간접적으로 프로젝트 권한 검증 수행
        DatasetService.get(db, bearing.dataset_id, current_user)
        return bearing

    @staticmethod
    def get_all(
        db: Session,
        dataset_id: int,
        current_user: User,
        skip: int = 0,
        limit: int = 100,
    ) -> list[Bearing]:
        # 권한 검증
        DatasetService.get(db, dataset_id, current_user)
        
        return BearingCRUD.get_dataset_bearings(db, dataset_id, skip, limit)

    @staticmethod
    def update(
        db: Session,
        bearing_id: int,
        bearing_in: BearingUpdate,
        current_user: User,
    ) -> Bearing:
        bearing = BearingService.get(db, bearing_id, current_user)
        dataset = DatasetService.get(db, bearing.dataset_id, current_user)
        
        ProjectMemberService.ensure_role(
            db, dataset.project_id, current_user.id, ProjectRole.MEMBER
        )

        updated = BearingCRUD.update(db, bearing, bearing_in)
        
        ActivityLogService.log_activity(
            db=db,
            project_id=dataset.project_id,
            user_id=current_user.id,
            action="BEARING_UPDATED",
            details=f"Bearing '{bearing.name}' status/info updated.",
        )
        return updated

    @staticmethod
    def delete(
        db: Session,
        bearing_id: int,
        current_user: User,
    ) -> None:
        bearing = BearingService.get(db, bearing_id, current_user)
        dataset = DatasetService.get(db, bearing.dataset_id, current_user)
        
        ProjectMemberService.ensure_role(
            db, dataset.project_id, current_user.id, ProjectRole.ADMIN
        )

        ActivityLogService.log_activity(
            db=db,
            project_id=dataset.project_id,
            user_id=current_user.id,
            action="BEARING_DELETED",
            details=f"Bearing '{bearing.name}' deleted.",
        )
        
        BearingCRUD.delete(db, bearing)