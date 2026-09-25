# DAIZEN 자동차 판매량 예측

2026-09-25 갱신 데이터와 저장 모델을 반영한 전달본이다. 기본 모델은 TFT A이며 중국·유럽·한국은 재학습 모델, 나머지 4개 시장은 재사용 모델이다.

압축을 푼 최상위 폴더에서 실행한다.

```powershell
uv run --frozen predict.py
```

전체 결과를 Python의 `result`로 보려면 다음 명령을 사용한다.

```powershell
uv run --frozen python -i predict.py
```

예측이 끝난 `>>>`에서 `result`를 입력한다. 결과는 `predictions`와 `excluded`가 들어 있는 딕셔너리다.

`outputs/forecast/forecast.json`과 `outputs/forecast/forecasft_data.csv`에 최신 결과를 덮어쓴다. 실행마다 새 날짜 폴더를 만들지 않는다. CSV는 6열이며 제외 사유는 `Predicted Sales`에 넣는다. `forecast_dict.py`는 생성하지 않는다.

입력 `raw/data.csv`는 187,570행·2,515개 국가별 차종이다. 2026-08~2028-07의 24개월 결과는 예측 53,448행(2,227개 차종)과 제외 316개다. 제외에는 기존 288개와 원강 요청 차종의 상태 명부 28개가 포함된다. `raw/vehicle_status.csv`의 28개 상태 기록도 매 결과의 `excluded`와 6열 CSV에 사유로 포함한다. 이 기록을 수치 판매량으로 학습하거나 예측하지 않는다. 수치 입력에 같은 Country·Brand·Model이 있으면 해당 입력의 결과를 우선하고 명부 행을 중복 추가하지 않는다.

API 서버는 `uv run --frozen python -m service.api`로 실행한다. `POST /predict`에 `Country,Month,Brand,Model,Sales` CSV를 multipart `file`로 전달한다.

- [실행·결과 안내](service/README.md)
- [백엔드 요청·응답 계약](service/PREDICT_API_CONTRACT.md)
- [전체 입력 CSV](raw/data.csv)
- [전체 JSON 예시](service/examples/response_tft_a.json)
- [6열 CSV 예시](service/examples/forecasft_data.csv)

XGBoost A/B, TFT A/B 네 설정의 미래용 모델과 전처리를 함께 제공한다. 실행 중 재학습하거나 모델을 자동 선택하지 않는다. 베어링 수요 변환은 백엔드에서 처리한다. 의존성 버전은 기존 잠금 파일을 유지했다.
