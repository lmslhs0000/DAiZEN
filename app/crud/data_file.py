from sqlalchemy.orm import Session

from app.models.data_file import DataFile
from app.schemas.data_file import DataFileCreate


class DataFileCRUD:
    @staticmethod
    def create(
        db: Session,
        bearing_id: int,
        obj_in: DataFileCreate,
    ) -> DataFile:
        db_obj = DataFile(
            bearing_id=bearing_id,
            original_filename=obj_in.original_filename,
            saved_filepath=obj_in.saved_filepath,
            file_size=obj_in.file_size,
            content_type=obj_in.content_type,
        )
        db.add(db_obj)
        db.commit()
        db.refresh(db_obj)
        return db_obj

    @staticmethod
    def get_by_id(
        db: Session,
        file_id: int,
    ) -> DataFile | None:
        return db.query(DataFile).filter(DataFile.id == file_id).first()

    @staticmethod
    def get_bearing_files(
        db: Session,
        bearing_id: int,
        skip: int = 0,
        limit: int = 100,
    ) -> list[DataFile]:
        return (
            db.query(DataFile)
            .filter(DataFile.bearing_id == bearing_id)
            .offset(skip)
            .limit(limit)
            .all()
        )

    @staticmethod
    def delete(
        db: Session,
        db_obj: DataFile,
    ) -> None:
        db.delete(db_obj)
        db.commit()