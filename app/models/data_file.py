from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, func
from sqlalchemy.orm import relationship

from app.db.database import Base


class DataFile(Base):
    __tablename__ = "data_files"

    id = Column(Integer, primary_key=True, index=True)

    bearing_id = Column(
        Integer,
        ForeignKey("bearings.id", ondelete="CASCADE"),
        nullable=False,
    )

    original_filename = Column(String(255), nullable=False)

    saved_filepath = Column(String(500), nullable=False)

    file_size = Column(Integer, nullable=False)

    content_type = Column(String(100), nullable=True)

    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
    )

    bearing = relationship(
        "Bearing",
        back_populates="data_files",
    )

    ai_tasks = relationship(
    "AITask",
    back_populates="data_file",
    cascade="all, delete-orphan",
    )