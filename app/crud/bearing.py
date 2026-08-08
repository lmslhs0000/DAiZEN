from sqlalchemy.orm import Session

from app.models.bearing import Bearing
from app.schemas.bearing import BearingCreate, BearingUpdate


class BearingCRUD:
    @staticmethod
    def create(
        db: Session,
        dataset_id: int,
        obj_in: BearingCreate,
    ) -> Bearing:
        db_obj = Bearing(
            dataset_id=dataset_id,
            name=obj_in.name,
            status=obj_in.status,
        )
        db.add(db_obj)
        db.commit()
        db.refresh(db_obj)
        return db_obj

    @staticmethod
    def get_by_id(
        db: Session,
        bearing_id: int,
    ) -> Bearing | None:
        return db.query(Bearing).filter(Bearing.id == bearing_id).first()

    @staticmethod
    def get_dataset_bearings(
        db: Session,
        dataset_id: int,
        skip: int = 0,
        limit: int = 100,
    ) -> list[Bearing]:
        return (
            db.query(Bearing)
            .filter(Bearing.dataset_id == dataset_id)
            .offset(skip)
            .limit(limit)
            .all()
        )

    @staticmethod
    def update(
        db: Session,
        db_obj: Bearing,
        obj_in: BearingUpdate,
    ) -> Bearing:
        update_data = obj_in.model_dump(exclude_unset=True)
        for field, value in update_data.items():
            setattr(db_obj, field, value)
        db.commit()
        db.refresh(db_obj)
        return db_obj

    @staticmethod
    def delete(
        db: Session,
        db_obj: Bearing,
    ) -> None:
        db.delete(db_obj)
        db.commit()