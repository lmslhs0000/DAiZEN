from sqlalchemy.orm import Session

from app.models.ai_task import AITask
from app.schemas.ai_task import AITaskCreate, AITaskUpdate


class AITaskCRUD:
    @staticmethod
    def create(
        db: Session,
        bearing_id: int,
        obj_in: AITaskCreate,
    ) -> AITask:
        db_obj = AITask(
            bearing_id=bearing_id,
            data_file_id=obj_in.data_file_id,
            task_name=obj_in.task_name,
            status="PENDING",
        )
        db.add(db_obj)
        db.commit()
        db.refresh(db_obj)
        return db_obj

    @staticmethod
    def get_by_id(
        db: Session,
        task_id: int,
    ) -> AITask | None:
        return db.query(AITask).filter(AITask.id == task_id).first()

    @staticmethod
    def get_by_bearing(
        db: Session,
        bearing_id: int,
        skip: int = 0,
        limit: int = 100,
    ) -> list[AITask]:
        return (
            db.query(AITask)
            .filter(AITask.bearing_id == bearing_id)
            .offset(skip)
            .limit(limit)
            .all()
        )

    @staticmethod
    def update(
        db: Session,
        db_obj: AITask,
        obj_in: AITaskUpdate,
    ) -> AITask:
        update_data = obj_in.model_dump(exclude_unset=True)
        for field, value in update_data.items():
            setattr(db_obj, field, value)
        
        db.add(db_obj)
        db.commit()
        db.refresh(db_obj)
        return db_obj

    @staticmethod
    def delete(
        db: Session,
        db_obj: AITask,
    ) -> None:
        db.delete(db_obj)
        db.commit()