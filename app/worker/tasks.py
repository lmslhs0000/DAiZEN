import json
import os
import pandas as pd
#import requests
from datetime import datetime, timezone

from app.core.celery_app import celery_app
from app.crud.ai_task import AITaskCRUD
from app.db.database import SessionLocal
from app.schemas.ai_task import AITaskUpdate

from app.models.user import User
from app.models.project import Project
from app.models.project_member import ProjectMember
from app.models.project_invite import ProjectInvite
from app.models.activity_log import ActivityLog
from app.models.dataset import Dataset
from app.models.data_file import DataFile
from app.models.bearing import Bearing
from app.models.ai_task import AITask

from openai import OpenAI

# HF_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
    # base_url="https://router.huggingface.co/v1",
    # api_key=HF_TOKEN,
    timeout=60,
)

@celery_app.task(name="analyze_sensor_data", bind=True, max_retries=3)
def analyze_sensor_data(self, task_id: int, file_path: str):
    """
    업로드된 측정 데이터를 기반으로 예지보전 분석을 수행하고 DB를 업데이트하는 백그라운드 작업.
    """
    db = SessionLocal()
    task = None
    
    try:
        # 1. 상태를 RUNNING(분석 중)으로 업데이트
        task = AITaskCRUD.get_by_id(db, task_id)
        if task:
            AITaskCRUD.update(db, task, AITaskUpdate(status="RUNNING"))
            
        # 2. 허깅페이스 API 호출
        # CSV 읽기
        df = pd.read_csv(file_path)

        # 처음 30행만 AI에게 전달 (토큰 절약)
        csv_data = df.head(30).to_csv(index=False)

        prompt = f"""
        You are a predictive maintenance AI.

        Analyze the following bearing sensor data.

        {csv_data}

        Please answer with:

        - Is the bearing normal or abnormal?
        - Estimated fault probability (%)
        - Possible cause
        - Recommended maintenance action
"""

        completion = client.chat.completions.create(
            model="nvidia/nemotron-3-nano-30b-a3b:free",
            messages=[
                {
                "role": "user",
                "content": prompt,
                }
            ],
            max_completion_tokens=2000, 
        )


        print(completion)

        result_data = {
            "huggingface_result": completion.model_dump(),
            "message": "Kimi-K3 분석 완료",
        }

        # 3. 상태를 COMPLETED(완료)로 업데이트하고 결과물(JSON) 저장
        if task:
            AITaskCRUD.update(
                db, 
                task, 
                AITaskUpdate(
                    status="COMPLETED",
                    result_data=json.dumps(result_data),
                    completed_at=datetime.now(timezone.utc)
                )
            )
            
        return {"task_id": task_id, "status": "COMPLETED"}

    except Exception as exc:
        if task:
            AITaskCRUD.update(
                db, 
                task, 
                AITaskUpdate(status="FAILED", error_message=str(exc))
            )
        raise self.retry(exc=exc, countdown=5)
        
    finally:
        db.close()