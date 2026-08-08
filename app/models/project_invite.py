import enum
from sqlalchemy import (
    Column,
    DateTime,
    Enum,
    ForeignKey,
    Integer,
    String,
    func,
)
from sqlalchemy.orm import relationship

from app.db.database import Base
from app.models.project_member import ProjectRole


class InviteStatus(str, enum.Enum):
    PENDING = "PENDING"
    ACCEPTED = "ACCEPTED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


class ProjectInvite(Base):
    __tablename__ = "project_invites"

    id = Column(Integer, primary_key=True, index=True)

    project_id = Column(
        Integer,
        ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False,
    )

    inviter_id = Column(
        Integer,
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )

    email = Column(
        String(255),
        nullable=False,
        index=True,
    )

    role = Column(
        String(20),
        nullable=False,
        default=ProjectRole.MEMBER.value,
    )

    token = Column(
        String(255),
        unique=True,
        nullable=False,
        index=True,
    )

    status = Column(
        String(20),
        nullable=False,
        default=InviteStatus.PENDING.value,
    )

    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
    )

    expires_at = Column(
        DateTime(timezone=True),
        nullable=False,
    )

    project = relationship(
        "Project",
        back_populates="invites",
    )

    inviter = relationship(
        "User",
        back_populates="sent_invites",
    )