from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from app.crud.project import (
    create_project,
    delete_project,
    get_project_by_id,
    get_projects,
    update_project,
)
from app.models.project import Project
from app.models.user import User
from app.schemas.project import (
    ProjectCreate,
    ProjectUpdate,
)
from app.services.activity_log_service import ActivityLogService

from app.models.project_member import ProjectMember

class ProjectService:
    @staticmethod
    def create(
        db: Session,
        project: ProjectCreate,
        owner_id: int,
    ) -> Project:
        # 1. 프로젝트 껍데기 생성
        created = create_project(
            db=db,
            project=project,
            owner_id=owner_id,
        )

        # 🌟 2. 누락되었던 로직 추가: 만든 사람을 프로젝트의 멤버(최고 권한)로 등록!
        # (만약 백엔드에서 역할 이름이 "OWNER"가 아니라 "MEMBER"나 "ADMIN"이라면 맞춰서 변경해 줘)
        new_member = ProjectMember(
            project_id=created.id,
            user_id=owner_id,
            role="OWNER" 
        )
        db.add(new_member)
        db.commit()

        # 3. 활동 로그 기록
        ActivityLogService.log_activity(
            db=db,
            project_id=created.id,
            user_id=owner_id,
            action="PROJECT_CREATED",
            details=f"Project '{created.name}' was created.",
        )

        return created

    @staticmethod
    def get_all(
        db: Session,
    ) -> list[Project]:
        return get_projects(db)

    @staticmethod
    def get(
        db: Session,
        project_id: int,
    ) -> Project:
        project = get_project_by_id(
            db=db,
            project_id=project_id,
        )

        if project is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="프로젝트를 찾을 수 없습니다.",
            )

        return project

    @staticmethod
    def ensure_owner(
        project: Project,
        current_user: User,
    ) -> None:
        if project.owner_id != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="프로젝트에 대한 권한이 없습니다.",
            )

    @staticmethod
    def update(
        db: Session,
        project: Project,
        update_data: ProjectUpdate,
        current_user_id: int,
    ) -> Project:
        updated = update_project(
            db=db,
            project=project,
            update_data=update_data,
        )

        ActivityLogService.log_activity(
            db=db,
            project_id=updated.id,
            user_id=current_user_id,
            action="PROJECT_UPDATED",
            details=f"Project info was updated.",
        )

        return updated

    @staticmethod
    def delete(
        db: Session,
        project: Project,
        current_user_id: int,
    ) -> None:
        ActivityLogService.log_activity(
            db=db,
            project_id=project.id,
            user_id=current_user_id,
            action="PROJECT_DELETED",
            details=f"Project '{project.name}' was deleted.",
        )

        delete_project(
            db=db,
            project=project,
        )