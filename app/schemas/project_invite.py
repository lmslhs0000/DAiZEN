from datetime import datetime
from pydantic import BaseModel, ConfigDict, EmailStr

from app.models.project_invite import InviteStatus
from app.models.project_member import ProjectRole


class ProjectInviteCreate(BaseModel):
    email: EmailStr
    role: ProjectRole = ProjectRole.MEMBER


class ProjectInviteResponse(BaseModel):
    id: int
    project_id: int
    inviter_id: int
    email: EmailStr
    role: str
    token: str
    status: InviteStatus
    created_at: datetime
    expires_at: datetime

    model_config = ConfigDict(from_attributes=True)


class ProjectInviteAcceptRequest(BaseModel):
    token: str