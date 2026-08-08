from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import relationship

from app.db.database import Base


class AITask(Base):
    __tablename__ = "ai_tasks"

    id = Column(Integer, primary_key=True, index=True)

    bearing_id = Column(
        Integer,
        ForeignKey("bearings.id", ondelete="CASCADE"),
        nullable=False,
    )

    data_file_id = Column(
        Integer,
        ForeignKey("data_files.id", ondelete="CASCADE"),
        nullable=False,
    )

    task_name = Column(String(100), nullable=False)

    status = Column(
        String(50),
        nullable=False,
        default="PENDING",  # PENDING, RUNNING, COMPLETED, FAILED
    )

    # AI 모델의 추론 결과(예: 잔여 수명, 결함 확률 등)를 JSON 문자열 형태로 저장
    result_data = Column(Text, nullable=True)

    error_message = Column(Text, nullable=True)

    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
    )

    completed_at = Column(
        DateTime(timezone=True),
        nullable=True,
    )

    bearing = relationship(
        "Bearing",
        back_populates="ai_tasks",
    )

    data_file = relationship(
    "DataFile",
    back_populates="ai_tasks",
    )  