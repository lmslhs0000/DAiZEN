# POST /predict 계약 v4.0.0

기본 방식은 7개 시장의 TFT A를 유지하며, 최신 데이터 갱신 실행 `run_20260924T190014_42dc7a14`의 미래용 모델을 사용한다. 입력 CSV는 `Country,Month,Brand,Model,Sales` 다섯 열 그대로 받는다. 예측 행은 `Country,Brand,Model,Forecast Month,Horizon,Predicted Sales` 여섯 열이다. 함께 배포한 `raw/vehicle_status.csv`의 판매·자료 상태도 각 응답의 제외 목록에 포함한다. 요청 중 새 학습을 실행하지 않는다.

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
| excluded | 입력에서 예측 불가인 차량과 배포 상태 명부 차량의 식별키·상태·제외 이유 |

아래는 전체 결과에서 예측과 상태 명부의 제외 행을 하나씩 발췌한 형식 예시다. 예측값은 이번 갱신 실행의 TFT A 결과를 정수로 반올림했다.

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
  "excluded": [
    {
      "Country": "Germany",
      "Brand": "Tesla",
      "Model": "Cybertruck",
      "status": "NOT_SOLD",
      "reason": "현재 판매 안함"
    }
  ]
}
```

각 예측 행은 위 여섯 키만 갖는다. `request_id`, `origin`, `horizon_months`, `models`, `summary`, `warnings` 등 기술 정보는 공개 응답에 포함하지 않는다. 요청의 origin·horizon 설정은 입력 계약으로 유지한다.

`Forecast Month`는 예측월, `Horizon`은 origin 다음 달부터의 거리인 정수다. 기본 응답은 성공 차량마다 1~24를 모두 포함하며, 길이를 지정한 요청에서는 1~horizon_months를 포함한다. origin=2026-07이면 2026-08~2028-07의 모든 월을 반환한다. 오늘 날짜를 기준으로 바꾸지 않는다.

`Predicted Sales`는 원 판매 대수 단위의 음수가 아닌 정수다. 최종 출력에서 소수 부분이 0.5 이상이면 올리는 방식으로 반올림한다. **내부 재귀 예측에는 반올림 전 소수 값을 사용한다.** 누적값·미래 actual·오차·WAPE는 반환하지 않는다.

성공 차량은 요청한 전체 기간을 반환하며 부분 성공은 허용하지 않는다. 성공과 제외에 동일 시계열이 중복되지 않는다. 모두 제외되면 `predictions`는 빈 배열이고 제외 사유는 남는다.

현재 [전체 입력 CSV](../raw/data.csv)는 7개 시장·187,570행·2,515개 수치 시계열이며 관측 기간은 2016-01~2026-07이다. 이번 갱신 실행의 수치 입력 TFT A 결과는 **2,227개 성공 × 24개월 = 53,448행**, **288개 제외**다. 별도 상태 명부 28개를 제외 목록에 더해 최종 응답은 **총 2,543개 차량: 성공 2,227개·제외 316개**를 표시한다. 수치 입력과 예측 행 수는 그대로다. 개수는 설명용이며 다른 입력의 예측·제외 개수에 고정하지 않는다. [전체 실제 응답](examples/response_tft_a.json), [전체 통합 CSV](examples/forecasft_data.csv), [시장별 입력 건수·원본 해시](examples/data_summary.json)를 제공한다.

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
| NOT_SOLD | 배포 상태 명부에서 현재 판매하지 않는 것으로 기록된 차량 |
| NO_PUBLIC_DATA | 배포 상태 명부에서 정확한 공개 판매 데이터가 부족한 것으로 기록된 차량 |
| AGGREGATED_SERIES | 배포 상태 명부에서 다른 시리즈로 통합 집계되는 것으로 기록된 차량 |

현재 전체 수치 CSV의 제외는 `INSUFFICIENT_HISTORY` 260개, `UNSEEN_SERIES` 28개다. 상태 명부는 `NOT_SOLD` 16개, `NO_PUBLIC_DATA` 10개, `AGGREGATED_SERIES` 2개이며 수치 입력의 제외와 별도로 합친다. [전체 통합 CSV](examples/forecasft_data.csv)의 `Forecast Month`와 `Horizon`이 공란인 총 316행에서 제외 차량의 이름을, 해당 행의 `Predicted Sales`에서 제외 이유를 확인할 수 있다. 상태 코드는 JSON과 반환 딕셔너리의 `excluded`에 보존한다. 정상 파일의 다른 차량은 계속 처리하며 HTTP 200의 `excluded`에 제외 차량을 남긴다. origin 이후에만 나타나는 입력 차량도 제외 목록에 남는다.

별도 [판매·자료 상태 명부](../raw/vehicle_status.csv)의 28행은 원강산업 요청 차종 중 원본에서 `Month`가 `-`였던 차량이다. 수치 학습 대상 2,515개에 포함하거나 모델에 전달하지 않으며, 판매량 0이나 가짜 월 관측으로 만들지 않는다. `Country`, `Brand`, `Model`, `status`, `reason`의 철자와 문구를 그대로 결과에 보존한다.

기본 `Predictor`는 이 배포 명부를 모든 정상 응답의 `excluded`에 결합한다. 다른 CSV나 소규모 CSV를 업로드해도 28개 상태 목록을 함께 반환한다. 다만 수치 입력에 정확히 같은 `(Country, Brand, Model)`이 있으면 해당 입력에서 나온 예측 또는 제외 결과를 우선하고 명부 행은 중복 추가하지 않는다. 브랜드·모델 대소문자나 이름을 임의로 바꾸어 같은 차량으로 취급하지 않는다. 명부는 업로드 파일이나 추가 multipart 필드로 받지 않는다.

명부 파일이 없는 과거 프로젝트는 기존처럼 수치 입력의 예측·제외 결과만 반환한다. 파일이 존재하지만 형식이나 내용이 잘못된 경우에는 `ServiceError`의 `INVALID_STATUS_CATALOG` 코드로 HTTP 503을 반환한다. 명부를 조용히 생략하지 않는다. 명부는 `Predictor` 생성 시 읽으므로 상시 API 프로세스에서 변경을 반영하려면 재시작한다.

## 로컬 JSON·통합 CSV 저장과 Python 딕셔너리 반환

프로젝트 또는 전달 폴더 최상위에서 실행한다.

```powershell
uv run predict.py
```

`raw/data.csv`를 읽고 고정된 `outputs/forecast/` 폴더에 두 파일을 저장한 뒤 종료한다. 실행이 성공하면 같은 이름의 기존 파일을 최신 결과로 덮어쓴다.

| 파일 | 내용 |
|---|---|
| forecast.json | 위 공개 응답과 같은 구조 |
| forecasft_data.csv | 여섯 열에 성공 예측과 제외 이유를 함께 저장 |

통합 CSV의 열은 정확히 `Country,Brand,Model,Forecast Month,Horizon,Predicted Sales` 여섯 개다.

- 성공 차량은 월별 한 행씩 저장한다. 예측 필드 여섯 개와 정수 판매량은 JSON의 `predictions`와 같다.
- 제외 차량은 차량당 한 행을 저장한다. `Country,Brand,Model`은 JSON의 `excluded`와 같고, `Forecast Month,Horizon`은 공란이다. `Predicted Sales`에는 해당 제외 행의 `reason` 문구를 그대로 적는다. 이 문구는 판매량 0이 아니다.
- CSV의 `Predicted Sales` 열에는 성공 예측의 정수와 제외 이유의 문자열이 함께 들어간다. 별도 `status`, `reason` 열은 없다. JSON과 반환 딕셔너리의 `excluded`에는 기존 `status`, `reason`을 유지한다.
- 현재 전체 입력의 통합 CSV는 예측 53,448행과 제외 316행(수치 입력 288개 + 상태 명부 28개)을 합친 **53,764행**이다. 헤더는 이 건수에 포함하지 않는다.

상태 명부 차량의 CSV 예시는 다음과 같다.

```csv
Country,Brand,Model,Forecast Month,Horizon,Predicted Sales
Germany,Tesla,Cybertruck,,,현재 판매 안함
```

CSV 파일명 `forecasft_data.csv`는 사용자가 지정한 철자 그대로다. UTF-8 BOM으로 저장하며 결과가 없어도 헤더를 남긴다. 제외 사유 CSV를 따로 생성하지 않는다. 과거 날짜별 결과 폴더는 보존한다.

프로젝트 최상위의 `predict.py`는 `predict(csv_path=None, output=None)` 함수를 제공한다. 함수는 JSON·CSV를 저장하고 공개 응답과 같은 Python 딕셔너리를 직접 반환한다.

```python
from predict import predict

result = predict()
predictions = result["predictions"]
excluded = result["excluded"]
```

반환값은 `{'predictions': [...], 'excluded': [...]}`이며 같은 실행의 JSON을 읽은 값과 같다. `csv_path`와 `output`으로 입력 CSV와 출력 JSON 경로를 지정할 수 있다. `from predict import predict`만 실행하면 모델을 실행하지 않는다.

예측 뒤 Python 세션을 유지하며 결과를 확인하려면 아래 명령을 사용한다.

```powershell
uv run python -i predict.py
```

두 파일을 저장한 뒤 `>>>` 프롬프트에서 `result`를 바로 사용할 수 있다. `main(argv=None)`도 같은 딕셔너리를 반환하며 직접 실행 시 `result = main()`으로 받는다. `forecast_dict.py`는 더 이상 생성하지 않으며 기존 파일은 과거 실행 결과로 보존한다.

`--csv "다른파일.csv"`로 입력을 지정할 수 있다. `--output "결과폴더/응답.json"`을 지정하면 JSON 파일명만 `응답.json`으로 바뀌고, 같은 폴더에 `forecasft_data.csv`를 저장한다. 직접 지정한 경로에서도 실행이 성공하면 두 결과 파일을 최신 결과로 덮어쓴다. 예측이나 결과 직렬화가 실패하면 기존 결과 파일을 유지한다. 과거에 생성한 날짜·시간별 결과 폴더는 그대로 보존한다.

API는 JSON을 반환한다. JSON·CSV 파일 저장과 `predict()`의 딕셔너리 반환은 로컬 실행 기능이며 새 다운로드 엔드포인트는 없다. Python의 `Predictor.predict_csv()`도 공개 응답 구조를 반환한다. `predict_csv_detailed()`는 소수 예측값·모델·실행 정보를 확인하는 내부 검산용이다.

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
| 503 | INVALID_STATUS_CATALOG | 배포된 `raw/vehicle_status.csv`의 형식·식별키·상태·사유 점검 |
| 503 | INVALID_MODEL_OUTPUT, INFERENCE_FAILED, INFERENCE_UNAVAILABLE | 서비스 로그와 예측 상태 점검 |

모델 파일의 누락·손상은 503으로 알리며 차량별 제외로 숨기지 않는다.

## 설정과 v4 전환

설정 `tft_a.json`(기본), `tft_b.json`, `xgboost_a.json`, `xgboost_b.json`은 유지한다. 설정 선택은 백엔드 배포 담당자가 수행한다. 모델은 필요할 때 로드해 재사용하므로 최초 호출에 로딩 시간이 포함된다. [Python 클라이언트](client.py)의 기본 timeout은 600초다.

이번 모델 출처는 `run_20260924T190014_42dc7a14`다. China, Europe, South Korea는 기존 선택 설정으로 재학습했고 Canada, Germany, Japan, United States는 기존 모델과 전처리를 재사용했다. 네 설정 모두 미래용 모델의 학습 종료월은 2026-07이다. 기본 TFT A 방식과 후보 설정은 새로 선택하지 않았다.

v4는 v3의 여섯 열 예측 구조와 다섯 열 입력을 유지하면서 공개 응답을 `predictions`·`excluded`로 줄이고, 출력 판매량을 정수로 반올림한다. 상세 정보와 원래 소수 예측은 내부 검산 경로에 보존한다. 로컬 CSV는 여섯 열이며 Python 함수는 딕셔너리를 직접 반환한다. 이번 입력·모델 갱신에서도 HTTP JSON의 예측·제외 구조는 그대로이므로 API 계약은 v4.0.0을 유지한다. 기계 판독용 명세는 [openapi.json](examples/openapi.json)이다.
