from datetime import datetime

from pydantic import BaseModel, ConfigDict


class DataFileBase(BaseModel):
    original_filename: str
    file_size: int
    content_type: str | None = None


class DataFileCreate(DataFileBase):
    saved_filepath: str


class DataFileResponse(DataFileBase):
    id: int
    bearing_id: int
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)