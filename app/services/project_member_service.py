from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from app.crud.project_member import ProjectMemberCRUD
from app.models.project_member import ProjectMember, ProjectRole
from app.services.activity_log_service import ActivityLogService


ROLE_HIERARCHY = {
    ProjectRole.OWNER: 40,
    ProjectRole.ADMIN: 30,
    ProjectRole.MEMBER: 20,
    ProjectRole.VIEWER: 10,
}


class ProjectMemberService:
    @staticmethod
    def add_member(
        db: Session,
        project_id: int,
        user_id: int,
        role: str = ProjectRole.MEMBER.value,
        operator_id: int = None,
    ) -> ProjectMember:
        existing_member = ProjectMemberCRUD.get_by_project_and_user(
            db,
            project_id,
            user_id,
        )

        if existing_member:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="User is already a member of this project.",
            )

        member = ProjectMemberCRUD.create(
            db=db,
            project_id=project_id,
            user_id=user_id,
            role=role,
        )

        ActivityLogService.log_activity(
            db=db,
            project_id=project_id,
            user_id=operator_id or user_id,
            action="MEMBER_ADDED",
            details=f"User ID {user_id} was added with role '{role}'.",
        )

        return member

    @staticmethod
    def get_members(
        db: Session,
        project_id: int,
    ) -> list[ProjectMember]:
        return ProjectMemberCRUD.get_project_members(
            db,
            project_id,
        )

    @staticmethod
    def get_member(
        db: Session,
        member_id: int,
    ) -> ProjectMember:
        member = ProjectMemberCRUD.get_by_id(
            db,
            member_id,
        )

        if member is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Project member not found.",
            )

        return member

    @staticmethod
    def is_member(
        db: Session,
        project_id: int,
        user_id: int,
    ) -> bool:
        member = ProjectMemberCRUD.get_by_project_and_user(
            db,
            project_id,
            user_id,
        )

        return member is not None

    @staticmethod
    def ensure_member(
        db: Session,
        project_id: int,
        user_id: int,
    ) -> None:
        if not ProjectMemberService.is_member(
            db,
            project_id,
            user_id,
        ):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You are not a member of this project.",
            )

    @staticmethod
    def has_role(
        db: Session,
        project_id: int,
        user_id: int,
        required_role: ProjectRole,
    ) -> bool:
        member = ProjectMemberCRUD.get_by_project_and_user(
            db,
            project_id,
            user_id,
        )

        if not member:
            return False

        user_level = ROLE_HIERARCHY.get(ProjectRole(member.role), 0)
        required_level = ROLE_HIERARCHY.get(required_role, 0)

        return user_level >= required_level

    @staticmethod
    def ensure_role(
        db: Session,
        project_id: int,
        user_id: int,
        required_role: ProjectRole,
    ) -> None:
        if not ProjectMemberService.has_role(
            db,
            project_id,
            user_id,
            required_role,
        ):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permission denied. Required minimum role: {required_role.value}",
            )

    @staticmethod
    def update_role(
        db: Session,
        member: ProjectMember,
        role: str,
        operator_id: int = None,
    ) -> ProjectMember:
        updated = ProjectMemberCRUD.update_role(
            db,
            member,
            role,
        )

        ActivityLogService.log_activity(
            db=db,
            project_id=member.project_id,
            user_id=operator_id or member.user_id,
            action="MEMBER_ROLE_UPDATED",
            details=f"Member ID {member.id} role changed to '{role}'.",
        )

        return updated

    @staticmethod
    def remove_member(
        db: Session,
        member: ProjectMember,
        operator_id: int = None,
    ) -> None:
        ActivityLogService.log_activity(
            db=db,
            project_id=member.project_id,
            user_id=operator_id or member.user_id,
            action="MEMBER_REMOVED",
            details=f"User ID {member.user_id} was removed from project.",
        )

        ProjectMemberCRUD.delete(
            db,
            member,
        )