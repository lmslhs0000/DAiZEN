from datetime import datetime
from typing import Optional
from pydantic import BaseModel, ConfigDict


class BearingCreate(BaseModel):
    name: str
    status: str = "NORMAL"


class BearingUpdate(BaseModel):
    name: Optional[str] = None
    status: Optional[str] = None


class BearingResponse(BaseModel):
    id: int
    dataset_id: int
    name: str
    status: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)