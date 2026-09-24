# DAIZEN 판매량 예측

기업 CSV의 `Country,Month,Brand,Model,Sales` 다섯 열을 읽어 차량별 다음 24개월 판매량을 예측한다. 기본 모델은 저장된 원본 TFT A다. 결과의 각 행은 `Country,Brand,Model,Forecast Month,Horizon,Predicted Sales` 여섯 열이다.

## 실행

프로젝트 또는 압축을 푼 `DAIZEN_판매량예측` 최상위 폴더에서 실행한다.

```powershell
uv run predict.py
```

기본 입력은 `raw/data.csv`다. 실행마다 새 결과 폴더를 만들고 완료 시 저장 경로를 표시한다.

```text
outputs/forecast_날짜_시간_마이크로초/
├─ forecast.json
├─ forecasft_data.csv
└─ excluded_vehicles.csv
```

- `forecast.json`: 예측 목록 `predictions`와 제외 목록 `excluded`.
- `forecasft_data.csv`: 성공 차량의 월별 예측. 파일명은 요청한 철자 그대로 사용한다.
- `excluded_vehicles.csv`: 제외 차량의 `Country,Brand,Model,status,reason`.

CSV는 엑셀에서도 한글을 읽을 수 있는 UTF-8 BOM으로 저장한다. 결과가 없어도 열 제목은 남긴다. 별도 서버를 실행할 필요가 없다.

다른 CSV를 사용할 때만 경로를 추가한다.

```powershell
uv run predict.py --csv "다른파일.csv"
```

저장 위치를 직접 지정할 수도 있다.

```powershell
uv run predict.py --output "outputs/이번예측/forecast.json"
```

두 CSV는 지정한 JSON과 같은 폴더에 저장한다. 세 결과 파일 중 하나라도 이미 있으면 덮어쓰지 않으므로 새 폴더를 지정한다.

## 결과 형식

공개 JSON은 아래 두 배열만 반환한다. 실행 ID·모델 정보·요약 등 기술 정보는 넣지 않는다. 아래는 전체 결과 중 한 행을 보여 주는 형식 예시다.

```json
{
  "predictions": [
    {
      "Country": "Canada",
      "Brand": "Acura",
      "Model": "ADX",
      "Forecast Month": "2026-08",
      "Horizon": 1,
      "Predicted Sales": 385
    }
  ],
  "excluded": []
}
```

`Forecast Month`는 예측월, `Horizon`은 입력 마지막 월 다음 달부터의 거리다. 현재 CSV는 2026-07까지 있으므로 성공 차량마다 **2026-08~2028-07의 24행**이 나온다. 누적 판매량이 아닌 각 월의 판매량이다.

JSON과 CSV의 `Predicted Sales`는 정수로 반올림한다. 소수 부분이 0.5 이상이면 올린다. 반올림은 최종 출력에만 적용하며, 다음 달 예측에 사용하는 내부 계산값은 소수를 유지한다.

## 전체 입력과 검산 자료

- [전체 입력 CSV](../raw/data.csv): 7개 시장, 187,382행, 2,511개 `(Country, Brand, Model)` 시계열. 전달 ZIP에도 전체를 포함한다.
- [전체 실제 응답](examples/response_tft_a.json): 전체 입력의 원본 TFT A 24개월 결과.
- [전체 예측 CSV](examples/forecasft_data.csv): 같은 53,352개 월별 예측 행과 정수 판매량.
- [제외 차량 전체 목록](examples/excluded_vehicles.csv): 제외된 288개 시계열의 이름·상태·이유.
- [입력 요약](examples/data_summary.json): 시장별 건수와 원본 파일 해시.
- [요청·응답 계약 v4](PREDICT_API_CONTRACT.md): 백엔드 요청 형식·응답·제외·오류 규칙.

현재 전체 입력의 기본 TFT A 결과는 **2,223개 성공 × 24개월 = 53,352행**, **288개 제외**다. 제외 사유는 이력 부족 260개와 해당 모델 미학습 28개다. 다른 입력의 결과 개수를 이 값에 맞추지 않는다.

실제 `Country` 값은 `Canada`, `China`, `Europe`, `Germany`, `Japan`, `South Korea`, `United States`다. Europe과 Germany는 별개로 유지한다.

입력은 UTF-8(BOM 허용)이며 `market` 열을 추가할 필요가 없다. 최근 연속 12개월의 유효한 이력과 해당 저장 모델이 학습한 정확한 시계열 키가 필요하다. 이름을 임의로 맞추거나 누락을 0으로 채우지 않는다. 판매량 0과 소수는 유효하다.

## 백엔드 연결

백엔드는 `POST /predict`에 `multipart/form-data`로 CSV를 전달한다. 필수 파일 필드명은 `file`이며 파일만 보내면 24개월을 예측한다. `<ML_BASE_URL>`은 백엔드 담당자가 연결할 서비스 주소다.

```powershell
curl.exe -X POST "<ML_BASE_URL>/predict" -F "file=@raw/data.csv;type=text/csv" -o response.json
```

선택 필드는 `origin`과 `horizon_months`다. API 응답은 위 JSON 형식이며, CSV 저장은 로컬 실행 기능이다. CSV 다운로드용 별도 API는 추가하지 않는다.

Python에서 직접 호출해도 같은 공개 JSON 구조를 받는다. 프로세스당 Predictor를 재사용한다.

```python
from service.inference import Predictor

predictor = Predictor()
result = predictor.predict_csv(csv_bytes)
```

`predict_csv_detailed()`는 원래 소수 예측값과 실행 정보를 확인하는 내부 검산용이다. 백엔드 전달에는 `predict_csv()`를 사용한다.

## 전달 구성과 모델

전달 폴더 이름은 `DAIZEN_판매량예측`이다. 모델은 `models/tft/A/canada/`처럼 알고리즘·A/B·시장별로 정리했다. 원본 실험·결과·모델 출처와 해시는 보존한다.

```text
DAIZEN_판매량예측/
├─ predict.py
├─ raw/data.csv
├─ outputs/
├─ models/tft/
├─ models/xgboost/
├─ model_info/
└─ service/
   ├─ configs/
   └─ examples/
```

TFT A/B와 XGBoost A/B 각각 7개 시장, 총 28개 미래용 모델 번들을 포함한다. 설정은 `tft_a.json`(기본), `tft_b.json`, `xgboost_a.json`, `xgboost_b.json`이다. 설정 선택은 백엔드 배포 담당자가 수행하며, 요청에서 모델 경로·알고리즘을 받지 않는다.

저장 모델과 전처리는 2026-07까지 학습했다. 요청 중 새 학습·튜닝·모델 선택을 실행하지 않는다. 모델은 CPU에서 실행하며 필요한 시장의 모델을 로드해 재사용한다. 입력 origin은 학습 종료월보다 이를 수 없다. 새 차종 지원에는 별도 학습이 필요하다.

ZIP 전체를 풀어 폴더 관계를 유지한다. 실행 의존성은 `pyproject.toml`과 `uv.lock`에 고정되어 있다. 차량 분류·베어링 수량·프론트 UI는 백엔드와 프론트 담당 범위다. 24개월 출력 기능과 24개월 정확도 검증은 별개다.
