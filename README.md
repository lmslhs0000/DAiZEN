# DAIZEN 자동차 판매량 예측

이 폴더에서 터미널을 열고 실행한다.

```powershell
uv run predict.py
```

`raw/data.csv` 전체를 읽어 차량별 24개월 판매량을 예측하고 종료한다. 실행할 때마다 `outputs/forecast_날짜_시간_마이크로초/`에 다음 세 파일을 새로 저장한다.

- `forecast.json`: `predictions`와 `excluded`만 포함하는 JSON.
- `forecasft_data.csv`: 모든 성공 차량의 월별 예측값. 파일명은 요청한 표기 그대로다.
- `excluded_vehicles.csv`: 제외 차량과 사유.

CSV는 UTF-8 BOM으로 저장한다. JSON과 예측 CSV의 항목은 `Country, Brand, Model, Forecast Month, Horizon, Predicted Sales`다. `Predicted Sales`는 정수 반올림하며 0.5는 올린다. 반올림은 저장과 응답에만 적용하고 재귀 예측 계산에는 원래 소수를 사용한다. 이전 결과는 덮어쓰지 않는다.

다른 입력은 `uv run predict.py --csv "다른파일.csv"`로 지정한다. `--output "새폴더/결과.json"`을 쓰면 같은 폴더에 두 CSV도 저장된다. 기존 파일이 있으면 거절한다. 첫 실행에는 uv가 잠금 파일에 맞는 패키지를 설치할 수 있다.

## 자료

- [전체 입력 CSV](raw/data.csv): 7개 시장, 187,382행, 2,511개 차량 시계열.
- [실행된 전체 JSON](service/examples/response_tft_a.json)
- [실행된 전체 예측 CSV](service/examples/forecasft_data.csv): 2,223개 차량 × 24개월 = 53,352행.
- [제외 차량과 사유](service/examples/excluded_vehicles.csv): 288개.
- [백엔드 요청·응답 계약](service/PREDICT_API_CONTRACT.md)
- [연동 안내](service/README.md)

기본 모델은 저장된 원본 TFT A이며, 현재 입력으로 2026-08~2028-07을 예측한다. 모델은 `models/tft/A/canada/`처럼 알고리즘·A/B·시장 이름으로 구분한다. 전체 CSV와 원본 모델 가중치는 보존했다. 백엔드 연결 시 `POST /predict`에 multipart 필드 `file`로 입력 CSV를 보내며 응답은 같은 JSON 구조다.
