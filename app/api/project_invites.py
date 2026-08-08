from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session

from app.core.dependencies import get_current_user
from app.db.database import get_db
from app.models.project_member import ProjectRole
from app.models.user import User
from app.schemas.project_invite import (
    ProjectInviteAcceptRequest,
    ProjectInviteCreate,
    ProjectInviteResponse,
)
from app.services.project_invite_service import ProjectInviteService
from app.services.project_member_service import ProjectMemberService
from app.services.project_service import ProjectService

router = APIRouter()


@router.post(
    "/project/{project_id}",
    response_model=ProjectInviteResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_invite(
    project_id: int,
    request: ProjectInviteCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    project = ProjectService.get(db, project_id)

    # 최소 ADMIN 이상만 초대 생성 가능
    ProjectMemberService.ensure_role(
        db=db,
        project_id=project.id,
        user_id=current_user.id,
        required_role=ProjectRole.ADMIN,
    )

    return ProjectInviteService.create_invite(
        db=db,
        project_id=project_id,
        inviter_id=current_user.id,
        email=request.email,
        role=request.role.value,
    )


@router.get(
    "/project/{project_id}",
    response_model=list[ProjectInviteResponse],
)
def get_invites(
    project_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    project = ProjectService.get(db, project_id)

    # 최소 ADMIN 이상만 초대 목록 조회 가능
    ProjectMemberService.ensure_role(
        db=db,
        project_id=project.id,
        user_id=current_user.id,
        required_role=ProjectRole.ADMIN,
    )

    return ProjectInviteService.get_invites(db, project_id)


@router.post(
    "/accept",
    response_model=ProjectInviteResponse,
)
def accept_invite(
    request: ProjectInviteAcceptRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return ProjectInviteService.accept_invite(
        db=db,
        token=request.token,
        current_user=current_user,
    )


@router.post(
    "/reject",
    response_model=ProjectInviteResponse,
)
def reject_invite(
    request: ProjectInviteAcceptRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return ProjectInviteService.reject_invite(
        db=db,
        token=request.token,
        current_user=current_user,
    )