# DAIZEN 자동차 판매량 예측

압축을 푼 최상위 폴더에서 실행한다.

```powershell
uv run predict.py
```

`raw/data.csv`를 읽고 차량별 다음 24개월을 예측한다. 항상 `outputs/forecast/`에 `forecast.json`과 `forecasft_data.csv` 두 파일을 저장하고 종료한다. 성공한 예측의 최신 결과로 두 파일을 덮어쓰며 새 날짜 폴더는 만들지 않는다. 예측이나 결과 변환에 실패하면 기존 파일을 유지한다. 이전 날짜별 폴더는 그대로 남는다.

CSV 열은 `Country,Brand,Model,Forecast Month,Horizon,Predicted Sales` 여섯 개다. 성공 차량은 월별 정수 판매량을 기록한다. 제외 차량은 차량당 한 행에 예측월·Horizon을 공란으로 두고 `Predicted Sales`에 제외 이유를 표시한다. 제외 이유는 판매량 0이 아니다.

Python에서 직접 실행하고 결과를 받는다. 같은 실행에서 JSON과 CSV도 저장한다.

```python
from predict import predict

result = predict()
```

실행 후 Python에서 `result`를 확인하려면 다음 명령을 사용한다.

```powershell
uv run python -i predict.py
```

예측과 파일 저장이 끝나면 `>>>`에서 `result`를 사용할 수 있다. `forecast_dict.py`는 생성하지 않는다.

- [실행·결과 안내](service/README.md)
- [백엔드 요청·응답 계약 v4](service/PREDICT_API_CONTRACT.md)
- [전체 입력 CSV](raw/data.csv)
- [전체 JSON 예시](service/examples/response_tft_a.json)
- [통합 CSV 예시](service/examples/forecasft_data.csv)

저장된 원본 TFT A가 기본 모델이다. 원본 데이터와 모델을 유지하며 실행 중 새로 학습하지 않는다.
