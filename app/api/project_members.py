from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session

from app.core.dependencies import get_current_user
from app.db.database import get_db
from app.models.project_member import ProjectRole
from app.models.user import User
from app.schemas.message import MessageResponse
from app.schemas.project_member import (
    AddProjectMemberRequest,
    ProjectMemberResponse,
    UpdateMemberRoleRequest,
)
from app.services.project_member_service import ProjectMemberService
from app.services.project_service import ProjectService

router = APIRouter(
    prefix="/project-members",
    tags=["Project Members"],
)


@router.get(
    "/project/{project_id}",
    response_model=list[ProjectMemberResponse],
)
def get_project_members(
    project_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    project = ProjectService.get(
        db,
        project_id,
    )

    # 프로젝트 멤버 전체 목록 조회는 최소 VIEWER 이상 권한 필요
    ProjectMemberService.ensure_role(
        db=db,
        project_id=project.id,
        user_id=current_user.id,
        required_role=ProjectRole.VIEWER,
    )

    return ProjectMemberService.get_members(
        db,
        project_id,
    )


@router.post(
    "/project/{project_id}",
    response_model=ProjectMemberResponse,
    status_code=status.HTTP_201_CREATED,
)
def add_member(
    project_id: int,
    request: AddProjectMemberRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    project = ProjectService.get(
        db,
        project_id,
    )

    # 멤버 초대 및 추가는 최소 ADMIN 이상 권한 필요
    ProjectMemberService.ensure_role(
        db=db,
        project_id=project.id,
        user_id=current_user.id,
        required_role=ProjectRole.ADMIN,
    )

    return ProjectMemberService.add_member(
        db=db,
        project_id=project_id,
        user_id=request.user_id,
        role=request.role,
    )


@router.patch(
    "/{member_id}/role",
    response_model=ProjectMemberResponse,
)
def update_role(
    member_id: int,
    request: UpdateMemberRoleRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    member = ProjectMemberService.get_member(
        db,
        member_id,
    )

    # 멤버 역할 변경은 최소 ADMIN 이상 권한 필요
    ProjectMemberService.ensure_role(
        db=db,
        project_id=member.project_id,
        user_id=current_user.id,
        required_role=ProjectRole.ADMIN,
    )

    return ProjectMemberService.update_role(
        db,
        member,
        request.role,
    )


@router.delete(
    "/{member_id}",
    response_model=MessageResponse,
)
def remove_member(
    member_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    member = ProjectMemberService.get_member(
        db,
        member_id,
    )

    # 멤버 추출/삭제는 최소 ADMIN 이상 권한 필요
    ProjectMemberService.ensure_role(
        db=db,
        project_id=member.project_id,
        user_id=current_user.id,
        required_role=ProjectRole.ADMIN,
    )

    ProjectMemberService.remove_member(
        db,
        member,
    )

    return MessageResponse(
        message="Project member deleted successfully.",
    )