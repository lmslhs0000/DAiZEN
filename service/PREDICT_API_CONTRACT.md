# POST /predict 계약 v4.0.0

기본 모델은 7개 시장의 저장된 원본 TFT A다. 입력 CSV는 `Country,Month,Brand,Model,Sales` 다섯 열 그대로 받는다. 예측 행은 `Country,Brand,Model,Forecast Month,Horizon,Predicted Sales` 여섯 열이다. 요청 중 새 학습을 실행하지 않는다.

## 요청

```text
POST <ML_BASE_URL>/predict
Content-Type: multipart/form-data; boundary=<클라이언트가 생성>

file: CSV 파일 (필수)
origin: YYYY-MM (선택, 기본 CSV 전체의 마지막 유효 Month)
horizon_months: 1~24 정수 (선택, 기본 24)
```

파일만 보내면 된다. 아래 경로는 프로젝트 또는 전달 ZIP의 최상위 폴더 기준이다. `<ML_BASE_URL>`은 백엔드에서 연결할 서비스 주소다.

```powershell
curl.exe -X POST "<ML_BASE_URL>/predict" -F "file=@raw/data.csv;type=text/csv" -o response.json
```

선택 값을 명시하려면 `-F "origin=2026-07" -F "horizon_months=24"`를 추가한다. JSON 본문이나 쿼리가 아닌 multipart 필드다. `market`·알고리즘·A/B·모델 경로는 보내지 않는다.

필수 CSV 열의 순서는 달라도 된다. 시계열 키는 `(Country, Brand, Model)`, 행 고유키는 여기에 `Month`를 더한 값이다.

- 인코딩은 UTF-8(BOM 허용), CSV 최대 크기는 25 MiB다. 전체 multipart 한도는 여기에 64 KiB를 더한 값이다.
- CSV의 Month는 `YYYY-MM` 또는 `YYYY-M`을 읽고 월 단위로 정규화한다. 요청 origin은 `YYYY-MM`이다.
- Sales의 0·소수는 유효하다. 빈 Sales는 누락으로 남기며 문자열·음수·무한대는 오류다.
- 중복 키를 합산하지 않는다. 중복/빈 CSV 헤더와 중복/추가 multipart 필드는 거절한다. 추가 CSV 열은 예측 입력으로 쓰지 않는다.
- origin은 CSV 마지막 월보다 뒤일 수 없고, 적용 모델의 학습 종료월(2026-07)보다 이를 수 없다. origin 이후 행은 검사 후 추론 이력에서 제외한다. 관련 건수는 내부 검산 결과에만 남긴다.

## 200 응답

상위 키는 아래 두 개뿐이다. `predictions`가 먼저 나오고 `excluded`가 뒤에 나온다.

| 키 | 내용 |
|---|---|
| predictions | 성공 차량의 월별 예측 행 배열 |
| excluded | 예측 불가 차량의 식별키·상태·제외 이유 |

아래는 전체 결과 중 한 행을 보여 주는 형식 예시다. 예측값은 실제 원본 TFT A 결과를 정수로 반올림했다.

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

각 예측 행은 위 여섯 키만 갖는다. `request_id`, `origin`, `horizon_months`, `models`, `summary`, `warnings` 등 기술 정보는 공개 응답에 포함하지 않는다. 요청의 origin·horizon 설정은 입력 계약으로 유지한다.

`Forecast Month`는 예측월, `Horizon`은 origin 다음 달부터의 거리인 정수다. 기본 응답은 성공 차량마다 1~24를 모두 포함하며, 길이를 지정한 요청에서는 1~horizon_months를 포함한다. origin=2026-07이면 2026-08~2028-07의 모든 월을 반환한다. 오늘 날짜를 기준으로 바꾸지 않는다.

`Predicted Sales`는 원 판매 대수 단위의 음수가 아닌 정수다. 최종 출력에서 소수 부분이 0.5 이상이면 올리는 방식으로 반올림한다. **내부 재귀 예측에는 반올림 전 소수 값을 사용한다.** 누적값·미래 actual·오차·WAPE는 반환하지 않는다.

성공 차량은 요청한 전체 기간을 반환하며 부분 성공은 허용하지 않는다. 성공과 제외에 동일 시계열이 중복되지 않는다. 모두 제외되면 `predictions`는 빈 배열이고 제외 사유는 남는다.

현재 [전체 입력 CSV](../raw/data.csv)는 7개 시장·187,382행·2,511개 시계열이다. 기본 TFT A 결과는 **2,223개 성공 × 24개월 = 53,352행**, **288개 제외**다. 개수는 설명용이며 다른 입력에 고정하지 않는다. [전체 실제 응답](examples/response_tft_a.json), [전체 예측 CSV](examples/forecasft_data.csv), [시장별 입력 건수·원본 해시](examples/data_summary.json)를 제공한다.

## 차량별 제외

최근 연속 12개월의 유효한 이력과 해당 모델이 학습한 정확한 `(Country, Brand, Model)`이 필요하다. 이름을 임의로 바꾸거나 다른 차량으로 대체하지 않는다. 누락을 0으로 채우지 않으며 판매량 0 자체는 제외 사유가 아니다.

`excluded`의 각 행은 정확히 `Country,Brand,Model,status,reason` 다섯 키를 갖는다. `status`는 기계 판독용 코드, `reason`은 개별 제외 이유다.

| status | 의미 |
|---|---|
| INSUFFICIENT_HISTORY | origin까지 12개월의 관측 기간이 없거나 origin 이하 행 자체가 없음 |
| INCOMPLETE_HISTORY | 최근 12개월의 월 또는 Sales 누락 |
| UNSEEN_SERIES | 해당 저장 모델이 학습한 정확한 Country·Brand·Model이 아님 |
| UNSUPPORTED_COUNTRY | 배포 설정에 해당 Country 모델이 없음 |
| PREDICTION_ERROR | 전체 기간 예측 실패, 부분/비유한/잘못된 월 출력 등. 상세 원인은 서비스 로그에 기록 |

현재 전체 CSV는 `INSUFFICIENT_HISTORY` 260개, `UNSEEN_SERIES` 28개다. [제외 차량 전체 목록](examples/excluded_vehicles.csv)에 288개 차량의 이름과 이유를 제공한다. 정상 파일의 다른 차량은 계속 처리하며 HTTP 200의 `excluded`에 제외 차량을 남긴다. origin 이후에만 나타나는 차량도 제외 목록에 남는다.

## 로컬 JSON·CSV 저장

프로젝트 또는 전달 폴더 최상위에서 실행한다.

```powershell
uv run predict.py
```

`raw/data.csv`를 읽고 새 `outputs/forecast_날짜_시간_마이크로초/` 폴더에 세 파일을 저장한다.

| 파일 | 내용 |
|---|---|
| forecast.json | 위 공개 응답과 같은 구조 |
| forecasft_data.csv | predictions와 같은 여섯 열, 같은 정수 판매량 |
| excluded_vehicles.csv | excluded와 같은 다섯 열 |

CSV 파일명 `forecasft_data.csv`는 사용자가 지정한 철자 그대로다. 두 CSV는 UTF-8 BOM이며 결과가 없어도 헤더를 남긴다. `--csv "다른파일.csv"`로 입력을 지정할 수 있다. `--output "새폴더/forecast.json"`을 지정하면 두 CSV도 같은 폴더에 생성한다. 세 대상 파일 중 하나라도 이미 있으면 덮어쓰지 않는다.

API는 JSON을 반환한다. CSV 저장은 로컬 실행 기능이며 새 다운로드 엔드포인트는 없다. Python의 `Predictor.predict_csv()`도 공개 응답 구조를 반환한다. `predict_csv_detailed()`는 소수 예측값·모델·실행 정보를 확인하는 내부 검산용이다.

## 요청·서비스 오류

```json
{
  "error": {
    "code": "ORIGIN_BEFORE_MODEL_TRAINING",
    "message": "origin은 선택된 모델의 학습 종료월(2026-07) 이후 또는 같은 월이어야 합니다."
  }
}
```

| HTTP | 주요 코드 | 처리 |
|---|---|---|
| 422 | INVALID_REQUEST, EMPTY_FILE, INVALID_ENCODING, INVALID_CSV | 요청 또는 CSV 수정 |
| 422 | INVALID_ORIGIN, INVALID_HORIZON | origin·horizon 형식/범위 수정 |
| 422 | ORIGIN_BEFORE_MODEL_TRAINING | 모델 학습 종료월 이후 또는 같은 월을 origin으로 사용 |
| 422 | ORIGIN_AFTER_DATA | CSV 마지막 월 이하를 origin으로 사용 |
| 413 | REQUEST_TOO_LARGE, FILE_TOO_LARGE | 파일/요청 크기 한도 확인 |
| 415 | UNSUPPORTED_MEDIA_TYPE | multipart와 CSV 파일 형식으로 전송 |
| 503 | MODEL_UNAVAILABLE | 배포 설정·manifest·모델/전처리 파일 점검 |
| 503 | INVALID_MODEL_OUTPUT, INFERENCE_FAILED, INFERENCE_UNAVAILABLE | 서비스 로그와 예측 상태 점검 |

모델 파일의 누락·손상은 503으로 알리며 차량별 제외로 숨기지 않는다.

## 설정과 v4 전환

설정 `tft_a.json`(기본), `tft_b.json`, `xgboost_a.json`, `xgboost_b.json`은 유지한다. 설정 선택은 백엔드 배포 담당자가 수행한다. 모델은 필요할 때 로드해 재사용하므로 최초 호출에 로딩 시간이 포함된다. [Python 클라이언트](client.py)의 기본 timeout은 600초다.

v4는 v3의 여섯 열 예측 구조와 다섯 열 입력을 유지하면서 공개 응답을 `predictions`·`excluded`로 줄이고, 출력 판매량을 정수로 반올림한다. 상세 정보와 원래 소수 예측은 내부 검산 경로에 보존한다. 기계 판독용 명세는 [openapi.json](examples/openapi.json)이다.
