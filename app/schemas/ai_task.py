from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict


class AITaskBase(BaseModel):
    task_name: str


class AITaskCreate(AITaskBase):
    data_file_id: int


class AITaskUpdate(BaseModel):
    status: Optional[str] = None
    result_data: Optional[str] = None
    error_message: Optional[str] = None
    completed_at: Optional[datetime] = None


class AITaskResponse(AITaskBase):
    id: int
    bearing_id: int
    data_file_id: int
    file_name: Optional[str] = None
    status: str
    result_data: Optional[str] = None
    error_message: Optional[str] = None
    created_at: datetime
    completed_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)