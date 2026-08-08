from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.core.dependencies import get_current_user
from app.db.database import get_db
from app.models.project_member import ProjectRole
from app.models.user import User
from app.schemas.activity_log import ActivityLogResponse
from app.services.activity_log_service import ActivityLogService
from app.services.project_member_service import ProjectMemberService
from app.services.project_service import ProjectService

router = APIRouter()


@router.get(
    "/project/{project_id}",
    response_model=list[ActivityLogResponse],
)
def get_activity_logs(
    project_id: int,
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    project = ProjectService.get(db, project_id)

    # 최소 MEMBER 이상만 프로젝트 감사 로그 조회 가능
    ProjectMemberService.ensure_role(
        db=db,
        project_id=project.id,
        user_id=current_user.id,
        required_role=ProjectRole.MEMBER,
    )

    return ActivityLogService.get_logs(
        db=db,
        project_id=project_id,
        limit=limit,
        offset=offset,
    )