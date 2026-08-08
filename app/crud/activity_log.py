from sqlalchemy.orm import Session

from app.models.activity_log import ActivityLog
from app.schemas.activity_log import ActivityLogCreate


class ActivityLogCRUD:
    @staticmethod
    def create(
        db: Session,
        log_data: ActivityLogCreate,
    ) -> ActivityLog:
        activity_log = ActivityLog(
            project_id=log_data.project_id,
            user_id=log_data.user_id,
            action=log_data.action,
            details=log_data.details,
        )
        db.add(activity_log)
        db.commit()
        db.refresh(activity_log)
        return activity_log

    @staticmethod
    def get_project_logs(
        db: Session,
        project_id: int,
        limit: int = 100,
        offset: int = 0,
    ) -> list[ActivityLog]:
        return (
            db.query(ActivityLog)
            .filter(ActivityLog.project_id == project_id)
            .order_order_by(ActivityLog.created_at.desc())
            .offset(offset)
            .limit(limit)
            .all()
        )