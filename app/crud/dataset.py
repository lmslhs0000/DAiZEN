from sqlalchemy.orm import Session

from app.models.dataset import Dataset
from app.schemas.dataset import DatasetCreate, DatasetUpdate


class DatasetCRUD:
    @staticmethod
    def create(
        db: Session,
        project_id: int,
        obj_in: DatasetCreate,
    ) -> Dataset:
        db_obj = Dataset(
            project_id=project_id,
            name=obj_in.name,
            description=obj_in.description,
        )
        db.add(db_obj)
        db.commit()
        db.refresh(db_obj)
        return db_obj

    @staticmethod
    def get_by_id(
        db: Session,
        dataset_id: int,
    ) -> Dataset | None:
        return db.query(Dataset).filter(Dataset.id == dataset_id).first()

    @staticmethod
    def get_project_datasets(
        db: Session,
        project_id: int,
        skip: int = 0,
        limit: int = 100,
    ) -> list[Dataset]:
        return (
            db.query(Dataset)
            .filter(Dataset.project_id == project_id)
            .offset(skip)
            .limit(limit)
            .all()
        )

    @staticmethod
    def update(
        db: Session,
        db_obj: Dataset,
        obj_in: DatasetUpdate,
    ) -> Dataset:
        update_data = obj_in.model_dump(exclude_unset=True)
        for field, value in update_data.items():
            setattr(db_obj, field, value)
        db.commit()
        db.refresh(db_obj)
        return db_obj

    @staticmethod
    def delete(
        db: Session,
        db_obj: Dataset,
    ) -> None:
        db.delete(db_obj)
        db.commit()