from sqlalchemy.orm import Session

from app.models.project_member import ProjectMember


class ProjectMemberCRUD:
    @staticmethod
    def create(
        db: Session,
        project_id: int,
        user_id: int,
        role: str = "MEMBER",
    ) -> ProjectMember:
        member = ProjectMember(
            project_id=project_id,
            user_id=user_id,
            role=role,
        )

        db.add(member)
        db.commit()
        db.refresh(member)

        return member

    @staticmethod
    def get_by_id(
        db: Session,
        member_id: int,
    ) -> ProjectMember | None:
        return (
            db.query(ProjectMember)
            .filter(ProjectMember.id == member_id)
            .first()
        )

    @staticmethod
    def get_project_members(
        db: Session,
        project_id: int,
    ) -> list[ProjectMember]:
        return (
            db.query(ProjectMember)
            .filter(ProjectMember.project_id == project_id)
            .all()
        )

    @staticmethod
    def get_by_project_and_user(
        db: Session,
        project_id: int,
        user_id: int,
    ) -> ProjectMember | None:
        return (
            db.query(ProjectMember)
            .filter(
                ProjectMember.project_id == project_id,
                ProjectMember.user_id == user_id,
            )
            .first()
        )

    @staticmethod
    def update_role(
        db: Session,
        member: ProjectMember,
        role: str,
    ) -> ProjectMember:
        member.role = role

        db.commit()
        db.refresh(member)

        return member

    @staticmethod
    def delete(
        db: Session,
        member: ProjectMember,
    ) -> None:
        db.delete(member)
        db.commit()