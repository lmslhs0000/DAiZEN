from datetime import datetime
from sqlalchemy.orm import Session

from app.models.project_invite import InviteStatus, ProjectInvite


class ProjectInviteCRUD:
    @staticmethod
    def create(
        db: Session,
        project_id: int,
        inviter_id: int,
        email: str,
        role: str,
        token: str,
        expires_at: datetime,
    ) -> ProjectInvite:
        invite = ProjectInvite(
            project_id=project_id,
            inviter_id=inviter_id,
            email=email,
            role=role,
            token=token,
            expires_at=expires_at,
            status=InviteStatus.PENDING.value,
        )
        db.add(invite)
        db.commit()
        db.refresh(invite)
        return invite

    @staticmethod
    def get_by_id(
        db: Session,
        invite_id: int,
    ) -> ProjectInvite | None:
        return (
            db.query(ProjectInvite)
            .filter(ProjectInvite.id == invite_id)
            .first()
        )

    @staticmethod
    def get_by_token(
        db: Session,
        token: str,
    ) -> ProjectInvite | None:
        return (
            db.query(ProjectInvite)
            .filter(ProjectInvite.token == token)
            .first()
        )

    @staticmethod
    def get_pending_invite_by_email(
        db: Session,
        project_id: int,
        email: str,
    ) -> ProjectInvite | None:
        return (
            db.query(ProjectInvite)
            .filter(
                ProjectInvite.project_id == project_id,
                ProjectInvite.email == email,
                ProjectInvite.status == InviteStatus.PENDING.value,
            )
            .first()
        )

    @staticmethod
    def get_project_invites(
        db: Session,
        project_id: int,
    ) -> list[ProjectInvite]:
        return (
            db.query(ProjectInvite)
            .filter(ProjectInvite.project_id == project_id)
            .all()
        )

    @staticmethod
    def update_status(
        db: Session,
        invite: ProjectInvite,
        status: InviteStatus,
    ) -> ProjectInvite:
        invite.status = status.value
        db.commit()
        db.refresh(invite)
        return invite