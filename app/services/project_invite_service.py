import secrets
from datetime import datetime, timedelta, timezone
from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from app.crud.project_invite import ProjectInviteCRUD
from app.crud.project_member import ProjectMemberCRUD
from app.models.project_invite import InviteStatus, ProjectInvite
from app.models.user import User
from app.services.activity_log_service import ActivityLogService


class ProjectInviteService:
    @staticmethod
    def create_invite(
        db: Session,
        project_id: int,
        inviter_id: int,
        email: str,
        role: str,
    ) -> ProjectInvite:
        existing_invite = ProjectInviteCRUD.get_pending_invite_by_email(
            db,
            project_id,
            email,
        )
        if existing_invite:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Pending invite already exists for this email.",
            )

        token = secrets.token_urlsafe(32)
        expires_at = datetime.now(timezone.utc) + timedelta(hours=24)

        invite = ProjectInviteCRUD.create(
            db=db,
            project_id=project_id,
            inviter_id=inviter_id,
            email=email,
            role=role,
            token=token,
            expires_at=expires_at,
        )

        ActivityLogService.log_activity(
            db=db,
            project_id=project_id,
            user_id=inviter_id,
            action="INVITE_SENT",
            details=f"Invite sent to '{email}' with role '{role}'.",
        )

        return invite

    @staticmethod
    def get_invites(
        db: Session,
        project_id: int,
    ) -> list[ProjectInvite]:
        return ProjectInviteCRUD.get_project_invites(db, project_id)

    @staticmethod
    def accept_invite(
        db: Session,
        token: str,
        current_user: User,
    ) -> ProjectInvite:
        invite = ProjectInviteCRUD.get_by_token(db, token)
        if not invite:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Invite token not found.",
            )

        if invite.status != InviteStatus.PENDING.value:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invite is no longer pending. Current status: {invite.status}",
            )

        now = datetime.now(timezone.utc)
        if invite.expires_at < now:
            ProjectInviteCRUD.update_status(db, invite, InviteStatus.EXPIRED)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invite token has expired.",
            )

        if invite.email != current_user.email:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="This invite was sent to a different email address.",
            )

        existing_member = ProjectMemberCRUD.get_by_project_and_user(
            db,
            invite.project_id,
            current_user.id,
        )
        if existing_member:
            ProjectInviteCRUD.update_status(db, invite, InviteStatus.ACCEPTED)
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="User is already a member of this project.",
            )

        ProjectMemberCRUD.create(
            db=db,
            project_id=invite.project_id,
            user_id=current_user.id,
            role=invite.role,
        )

        updated_invite = ProjectInviteCRUD.update_status(db, invite, InviteStatus.ACCEPTED)

        ActivityLogService.log_activity(
            db=db,
            project_id=invite.project_id,
            user_id=current_user.id,
            action="INVITE_ACCEPTED",
            details=f"User '{current_user.email}' accepted project invite.",
        )

        return updated_invite

    @staticmethod
    def reject_invite(
        db: Session,
        token: str,
        current_user: User,
    ) -> ProjectInvite:
        invite = ProjectInviteCRUD.get_by_token(db, token)
        if not invite:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Invite token not found.",
            )

        if invite.status != InviteStatus.PENDING.value:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invite is no longer pending. Current status: {invite.status}",
            )

        if invite.email != current_user.email:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="This invite was sent to a different email address.",
            )

        rejected = ProjectInviteCRUD.update_status(db, invite, InviteStatus.REJECTED)

        ActivityLogService.log_activity(
            db=db,
            project_id=invite.project_id,
            user_id=current_user.id,
            action="INVITE_REJECTED",
            details=f"User '{current_user.email}' rejected project invite.",
        )

        return rejected