from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from app.crud.ai_task import AITaskCRUD
from app.models.ai_task import AITask
from app.models.project_member import ProjectRole
from app.models.user import User
from app.schemas.ai_task import AITaskCreate, AITaskUpdate
from app.services.activity_log_service import ActivityLogService
from app.services.bearing_service import BearingService
from app.services.dataset_service import DatasetService
from app.services.project_member_service import ProjectMemberService
from app.services.data_file_service import DataFileService

# 신규 추가: Celery Task 불러오기
from app.worker.tasks import analyze_sensor_data


class AITaskService:
    @staticmethod
    def create(
        db: Session,
        bearing_id: int,
        obj_in: AITaskCreate,
        current_user: User,
    ) -> AITask:
        # 1. 권한 검증 (MEMBER 이상)
        bearing = BearingService.get(db, bearing_id, current_user)
        dataset = DatasetService.get(db, bearing.dataset_id, current_user)
        
        ProjectMemberService.ensure_role(
            db, dataset.project_id, current_user.id, ProjectRole.MEMBER
        )
        
        data_file = DataFileService.get(
        db,
        obj_in.data_file_id,
        current_user,
        )

        if not data_file:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Data file not found.",
            )


        # 2. 작업 생성
        task = AITaskCRUD.create(db, bearing_id, obj_in)

        # 3. 활동 로그 기록
        ActivityLogService.log_activity(
            db=db,
            project_id=dataset.project_id,
            user_id=current_user.id,
            action="AI_TASK_CREATED",
            details=f"AI Task '{task.task_name}' requested for bearing '{bearing.name}'.",
        )

        print("TASK APP:", analyze_sensor_data.app)
        print("BROKER :", analyze_sensor_data.app.conf.broker_url)
        print("BACKEND:", analyze_sensor_data.app.conf.result_backend)
        
        # 🌟 4. Celery 워커로 비동기 작업 전달 (대기열에 주문 넣기)
        analyze_sensor_data.delay(
            task_id=task.id,
            file_path=data_file.saved_filepath,
        )

        return task

    @staticmethod
    def get(
        db: Session,
        task_id: int,
        current_user: User,
    ) -> AITask:
        task = AITaskCRUD.get_by_id(db, task_id)
        if not task:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="AI Task not found.",
            )
        
        # 권한 검증
        BearingService.get(db, task.bearing_id, current_user)
        from app.schemas.ai_task import AITaskResponse

        return AITaskResponse(
            id=task.id,
            task_name=task.task_name,
            bearing_id=task.bearing_id,
            data_file_id=task.data_file_id,
            file_name=task.data_file.original_filename,
            status=task.status,
            result_data=task.result_data,
            error_message=task.error_message,
            created_at=task.created_at,
            completed_at=task.completed_at,
        )       

    @staticmethod
    def get_all(
        db: Session,
        bearing_id: int,
        current_user: User,
        skip: int = 0,
        limit: int = 100,
    ) -> list[AITask]:
        # 권한 검증
        BearingService.get(db, bearing_id, current_user)
        tasks = AITaskCRUD.get_by_bearing(db, bearing_id, skip, limit)

        from app.schemas.ai_task import AITaskResponse

        return [
            AITaskResponse(
                id=task.id,
                task_name=task.task_name,
                bearing_id=task.bearing_id,
                data_file_id=task.data_file_id,
                file_name=task.data_file.original_filename,
                status=task.status,
                result_data=task.result_data,
                error_message=task.error_message,
                created_at=task.created_at,
                completed_at=task.completed_at,
            )
            for task in tasks
        ]

    @staticmethod
    def update(
        db: Session,
        task_id: int,
        obj_in: AITaskUpdate,
        current_user: User,
    ) -> AITask:
        task = AITaskService.get(db, task_id, current_user)
        bearing = BearingService.get(db, task.bearing_id, current_user)
        dataset = DatasetService.get(db, bearing.dataset_id, current_user)

        # 권한 검증 (MEMBER 이상)
        ProjectMemberService.ensure_role(
            db, dataset.project_id, current_user.id, ProjectRole.MEMBER
        )

        updated_task = AITaskCRUD.update(db, task, obj_in)

        # 상태가 변경된 경우에만 로그 기록
        if obj_in.status:
            ActivityLogService.log_activity(
                db=db,
                project_id=dataset.project_id,
                user_id=current_user.id,
                action="AI_TASK_UPDATED",
                details=f"AI Task '{task.task_name}' status changed to '{obj_in.status}'.",
            )
        return updated_task

    @staticmethod
    def delete(
        db: Session,
        task_id: int,
        current_user: User,
    ) -> None:
        task = AITaskService.get(db, task_id, current_user)
        bearing = BearingService.get(db, task.bearing_id, current_user)
        dataset = DatasetService.get(db, bearing.dataset_id, current_user)

        # 권한 검증 (ADMIN 이상만 삭제 가능)
        ProjectMemberService.ensure_role(
            db, dataset.project_id, current_user.id, ProjectRole.ADMIN
        )

        AITaskCRUD.delete(db, task)

        ActivityLogService.log_activity(
            db=db,
            project_id=dataset.project_id,
            user_id=current_user.id,
            action="AI_TASK_DELETED",
            details=f"AI Task '{task.task_name}' deleted.",
        ) 