from sqlalchemy.orm import Session

from app.models.project import Project
from app.schemas.project import ProjectCreate, ProjectUpdate


def create_project(
    db: Session,
    project: ProjectCreate,
    owner_id: int,
) -> Project:
    db_project = Project(
        name=project.name,
        description=project.description,
        owner_id=owner_id,
    )

    db.add(db_project)
    db.commit()
    db.refresh(db_project)

    return db_project


def get_projects(db: Session):
    return db.query(Project).all()


def get_project_by_id(
    db: Session,
    project_id: int,
):
    return (
        db.query(Project)
        .filter(Project.id == project_id)
        .first()
    )


def update_project(
    db: Session,
    project: Project,
    update_data: ProjectUpdate,
) -> Project:
    if update_data.name is not None:
        project.name = update_data.name

    if update_data.description is not None:
        project.description = update_data.description

    db.commit()
    db.refresh(project)

    return project


def delete_project(
    db: Session,
    project: Project,
):
    db.delete(project)
    db.commit()