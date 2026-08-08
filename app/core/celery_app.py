from celery import Celery

from app.core.config import settings

# Celery 인스턴스 생성
celery_app = Celery(
    "daizen_worker",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,
    include=["app.worker.tasks"],  # 신규 추가: 워커가 실행할 작업 파일 경로 등록
)

# Celery 기본 설정
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="Asia/Seoul",
    enable_utc=False,
)