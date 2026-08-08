from datetime import datetime
from typing import Optional
from pydantic import BaseModel, ConfigDict


class ActivityLogCreate(BaseModel):
    project_id: int
    user_id: Optional[int] = None
    action: str
    details: Optional[str] = None


class ActivityLogResponse(BaseModel):
    id: int
    project_id: int
    user_id: Optional[int] = None
    action: str
    details: Optional[str] = None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)