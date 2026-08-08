from typing import Optional
from sqlalchemy.orm import Session

from app.crud.activity_log import ActivityLogCRUD
from app.models.activity_log import ActivityLog
from app.schemas.activity_log import ActivityLogCreate


class ActivityLogService:
    @staticmethod
    def log_activity(
        db: Session,
        project_id: int,
        user_id: Optional[int],
        action: str,
        details: Optional[str] = None,
    ) -> ActivityLog:
        log_data = ActivityLogCreate(
            project_id=project_id,
            user_id=user_id,
            action=action,
            details=details,
        )
        return ActivityLogCRUD.create(db, log_data)

    @staticmethod
    def get_logs(
        db: Session,
        project_id: int,
        limit: int = 100,
        offset: int = 0,
    ) -> list[ActivityLog]:
        return ActivityLogCRUD.get_project_logs(
            db=db,
            project_id=project_id,
            limit=limit,
            offset=offset,
        )