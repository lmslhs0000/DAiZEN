from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, func
from sqlalchemy.orm import relationship

from app.db.database import Base


class Bearing(Base):
    __tablename__ = "bearings"

    id = Column(Integer, primary_key=True, index=True)

    dataset_id = Column(
        Integer,
        ForeignKey("datasets.id", ondelete="CASCADE"),
        nullable=False,
    )

    name = Column(String(100), nullable=False)

    status = Column(
        String(50),
        nullable=False,
        default="NORMAL",
    )

    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
    )

    dataset = relationship(
        "Dataset",
        back_populates="bearings",
    )

    data_files = relationship(
        "DataFile",
        back_populates="bearing",
        cascade="all, delete-orphan",
    )

    # 신규 추가된 관계: 1개의 베어링은 여러 번의 AI 분석 작업을 가질 수 있음
    ai_tasks = relationship(
        "AITask",
        back_populates="bearing",
        cascade="all, delete-orphan",
    )