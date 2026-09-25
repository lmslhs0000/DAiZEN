# DAIZEN 판매량 예측

기업 CSV의 `Country,Month,Brand,Model,Sales` 다섯 열을 읽어 차량별 다음 24개월 판매량을 예측한다. 기본 방식은 TFT A를 유지하며, 최신 데이터 갱신 실행 `run_20260924T190014_42dc7a14`의 미래용 모델을 사용한다. JSON과 Python 딕셔너리의 예측 행 및 CSV의 열은 `Country,Brand,Model,Forecast Month,Horizon,Predicted Sales` 여섯 필드다. CSV의 제외 차량은 `Predicted Sales`에 제외 이유를 표시한다. 함께 배포한 `raw/vehicle_status.csv`의 판매·자료 상태도 각 응답의 제외 목록에 포함한다.

## 실행

프로젝트 또는 압축을 푼 최상위 폴더에서 실행한다.

```powershell
uv run predict.py
```

기본 입력은 `raw/data.csv`다. 실행이 성공하면 고정된 `outputs/forecast/` 폴더에 최신 결과를 저장하고 완료 시 저장 경로를 표시한다. 같은 이름의 결과 파일이 있으면 덮어쓴다.

```text
outputs/forecast/
├─ forecast.json
└─ forecasft_data.csv
```

- `forecast.json`: 예측 목록 `predictions`와 제외 목록 `excluded`.
- `forecasft_data.csv`: 성공 차량의 월별 예측과 제외 차량의 사유를 합친 표. 파일명은 요청한 철자 그대로 사용한다.

CSV는 엑셀에서도 한글을 읽을 수 있는 UTF-8 BOM으로 저장한다. 결과가 없어도 열 제목은 남긴다. 제외 사유 CSV를 따로 생성하지 않으며, 과거 날짜별 결과 폴더는 보존한다. 별도 서버를 실행할 필요가 없다.

다른 CSV를 사용할 때만 경로를 추가한다.

```powershell
uv run predict.py --csv "다른파일.csv"
```

저장 위치를 직접 지정할 수도 있다.

```powershell
uv run predict.py --output "outputs/이번예측/forecast.json"
```

CSV는 지정한 JSON과 같은 폴더에 저장한다. `--output`으로 바꾸는 것은 JSON 경로와 파일명이며, CSV 파일명은 `forecasft_data.csv`로 유지한다. 직접 지정한 경로에서도 실행이 성공하면 두 결과 파일을 최신 결과로 덮어쓴다. 예측이나 결과 직렬화가 실패하면 기존 결과 파일을 유지한다. 과거에 생성한 날짜·시간별 결과 폴더는 그대로 보존한다.

Python 코드에서 바로 딕셔너리를 받으려면 프로젝트 최상위의 `predict` 함수를 호출한다. 같은 실행에서 JSON과 CSV도 저장한다.

```python
from predict import predict

result = predict()
predictions = result["predictions"]
excluded = result["excluded"]
```

함수 형식은 `predict(csv_path=None, output=None)`이다. 예를 들어 `predict(csv_path="다른파일.csv", output="outputs/이번예측/forecast.json")`으로 입력과 저장 위치를 지정한다. 모듈을 불러오는 것만으로는 예측을 실행하지 않는다.

실행을 마친 뒤 같은 Python 세션에서 `result`를 확인하려면 다음 명령을 사용한다.

```powershell
uv run python -i predict.py
```

예측과 파일 저장을 마치면 `>>>` 프롬프트가 나오며 `result`, `result["predictions"]`, `result["excluded"]`를 바로 사용할 수 있다. 일반 `uv run predict.py`는 두 파일을 저장한 뒤 종료한다. `main(argv=None)`도 같은 딕셔너리를 반환하며, 직접 실행할 때 `result = main()`으로 받는다. `forecast_dict.py`는 새로 생성하지 않고 과거 파일만 보존한다.

## 결과 형식

공개 JSON은 아래 두 배열만 반환한다. 실행 ID·모델 정보·요약 등 기술 정보는 넣지 않는다. 아래는 전체 결과에서 예측과 상태 명부의 제외 행을 하나씩 발췌한 형식 예시다.

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

`Forecast Month`는 예측월, `Horizon`은 입력 마지막 월 다음 달부터의 거리다. 현재 CSV는 2026-07까지 있으므로 성공 차량마다 **2026-08~2028-07의 24행**이 나온다. 누적 판매량이 아닌 각 월의 판매량이다.

성공 예측의 JSON·CSV·Python 딕셔너리 `Predicted Sales`는 같은 정수로 반올림한다. 소수 부분이 0.5 이상이면 올린다. 반올림은 최종 출력에만 적용하며, 다음 달 예측에 사용하는 내부 계산값은 소수를 유지한다.

통합 CSV의 열은 아래 여섯 개다.

```text
Country,Brand,Model,Forecast Month,Horizon,Predicted Sales
```

성공 차량은 월별로 한 행씩 저장하며 `Predicted Sales`는 정수다. 제외 차량은 차량당 한 행을 남겨 `Country`, `Brand`, `Model`을 표시하고 `Forecast Month`, `Horizon`은 공란으로 둔다. `Predicted Sales`에는 JSON의 `excluded`에 있는 `reason` 문구를 그대로 적는다. 이 문구는 판매량 0을 뜻하지 않는다. 따라서 CSV의 `Predicted Sales` 열에는 성공 예측의 정수와 제외 이유의 문자열이 함께 들어간다.

JSON과 반환 딕셔너리는 `predictions`와 `excluded` 구조를 유지한다. `excluded`의 `Country,Brand,Model,status,reason`도 그대로 보존한다. 반환된 `result`는 같은 실행의 `forecast.json`을 읽은 값과 같다.

## 전체 입력과 검산 자료

- [전체 입력 CSV](../raw/data.csv): 7개 시장, 187,570행, 2,515개 `(Country, Brand, Model)` 시계열. 관측 기간은 2016-01~2026-07이며 전달 ZIP에도 전체를 포함한다.
- [전체 실제 응답](examples/response_tft_a.json): 최신 갱신 모델의 TFT A 24개월 결과.
- [전체 통합 CSV](examples/forecasft_data.csv): 53,448개 월별 예측 행과 제외 차량 316행, 총 53,764행.
- [입력 요약](examples/data_summary.json): 시장별 건수와 원본 파일 해시.
- [요청·응답 계약 v4](PREDICT_API_CONTRACT.md): 백엔드 요청 형식·응답·제외·오류 규칙.

이번 갱신 실행의 수치 입력 TFT A 결과는 **2,227개 성공 × 24개월 = 53,448행**, **288개 제외**다. 수치 입력의 제외 사유는 이력 부족 260개와 해당 모델 미학습 28개다. 여기에 별도 상태 명부 28개를 더해 최종 응답은 **총 2,543개 차량: 성공 2,227개·제외 316개**를 표시한다. 예측 행 수는 53,448행 그대로이며, 다른 입력의 예측·제외 개수를 이 값에 맞추지 않는다.

입력 SHA-256은 `5347bcff6c995566bc9d850a05c1ed5a1bf7ffc155e9d7aef12cf5e651da3208`이다. 별도 [판매·자료 상태 명부](../raw/vehicle_status.csv)의 28행은 원강산업 요청 차종 중 원본에서 `Month`가 `-`였던 차량이다. 상태는 `NOT_SOLD` 16개, `NO_PUBLIC_DATA` 10개, `AGGREGATED_SERIES` 2개다. 이 명부를 수치 학습 대상 2,515개에 포함하거나 모델에 전달하지 않으며, 판매량 0이나 가짜 월 관측으로 바꾸지 않는다.

기본 `Predictor`는 배포 프로젝트의 상태 명부를 읽어 모든 정상 응답의 `excluded`에 결합한다. 다른 CSV나 소규모 CSV를 입력해도 이 28개 상태 목록을 함께 표시한다. 단, 수치 입력에 정확히 같은 `(Country, Brand, Model)`이 있으면 그 입력의 예측 또는 제외 결과를 우선하고 명부 행은 중복 추가하지 않는다. 명부의 브랜드·차종 철자와 `status`, `reason`은 그대로 보존한다. JSON·반환 딕셔너리·로컬 CSV 모두 같은 제외 목록을 사용하며, 명부 차량도 CSV의 예측월·Horizon은 공란이고 `Predicted Sales`에 이유를 적는다.

명부는 업로드 CSV에서 받는 값이 아니라 배포 파일이다. 명부 파일이 없는 과거 프로젝트는 기존처럼 수치 입력 결과만 반환한다. 파일이 있는데 형식이나 내용이 잘못된 경우에는 `INVALID_STATUS_CATALOG` 서비스 오류로 알리고 조용히 생략하지 않는다. 명부는 `Predictor` 생성 시 읽으므로 파일을 변경한 상시 API 프로세스는 재시작해야 반영된다.

실제 `Country` 값은 `Canada`, `China`, `Europe`, `Germany`, `Japan`, `South Korea`, `United States`다. Europe과 Germany는 별개로 유지한다.

입력은 UTF-8(BOM 허용)이며 `market` 열을 추가할 필요가 없다. 최근 연속 12개월의 유효한 이력과 해당 저장 모델이 학습한 정확한 시계열 키가 필요하다. 이름을 임의로 맞추거나 누락을 0으로 채우지 않는다. 판매량 0과 소수는 유효하다.

## 백엔드 연결

백엔드는 `POST /predict`에 `multipart/form-data`로 CSV를 전달한다. 필수 파일 필드명은 `file`이며 파일만 보내면 24개월을 예측한다. `<ML_BASE_URL>`은 백엔드 담당자가 연결할 서비스 주소다.

```powershell
curl.exe -X POST "<ML_BASE_URL>/predict" -F "file=@raw/data.csv;type=text/csv" -o response.json
```

선택 필드는 `origin`과 `horizon_months`다. API 응답은 위 JSON 형식이며, JSON·CSV 파일 저장과 `predict()`의 딕셔너리 반환은 로컬 실행 기능이다. 다운로드용 별도 API는 추가하지 않는다. 이번 입력·모델 갱신에서도 HTTP JSON 구조와 로컬 CSV의 여섯 열을 유지하므로 계약 버전은 v4다.

Python에서 직접 호출해도 같은 공개 JSON 구조를 받는다. 프로세스당 Predictor를 재사용한다.

```python
from service.inference import Predictor

predictor = Predictor()
result = predictor.predict_csv(csv_bytes)
```

`predict_csv_detailed()`는 원래 소수 예측값과 실행 정보를 확인하는 내부 검산용이다. 백엔드 전달에는 `predict_csv()`를 사용한다.

## 전달 구성과 모델

전달 폴더는 `service/delivery/DAIZEN_판매량예측_최신모델/`, ZIP은 같은 위치의 `DAIZEN_판매량예측_최신모델.zip`이다. 이전 전달본은 당시 결과로 보존한다.

모델은 `models/tft/A/canada/`처럼 알고리즘·A/B·시장별로 정리했다. 출처 실행은 `run_20260924T190014_42dc7a14`이며 원본 실험·결과·모델 출처와 해시는 보존한다.

```text
압축을 푼 최상위 폴더/
├─ predict.py
├─ raw/
│  ├─ data.csv
│  └─ vehicle_status.csv
├─ outputs/
├─ models/tft/
├─ models/xgboost/
├─ model_info/
└─ service/
   ├─ configs/
   └─ examples/
```

TFT A/B와 XGBoost A/B 각각 7개 시장, 총 28개 미래용 모델 번들을 포함한다. 설정은 `tft_a.json`(기본), `tft_b.json`, `xgboost_a.json`, `xgboost_b.json`이다. 설정 선택은 백엔드 배포 담당자가 수행하며, 요청에서 모델 경로·알고리즘을 받지 않는다.

China, Europe, South Korea는 기존 선택 설정으로 재학습했고 Canada, Germany, Japan, United States는 기존 모델과 전처리를 재사용했다. 후보 탐색이나 기본 방식 재선택은 하지 않았다. 추가된 China / Genesis / G90, Europe / Chevrolet / Silverado, Europe / Chevrolet / Tahoe, Europe / Genesis / G90은 미래 예측에 포함된다.

저장 모델과 전처리는 2026-07까지 학습했다. 요청 중 새 학습·튜닝·모델 선택을 실행하지 않는다. 모델은 CPU에서 실행하며 필요한 시장의 모델을 로드해 재사용한다. 입력 origin은 학습 종료월보다 이를 수 없다. 새 차종 지원에는 별도 학습이 필요하다.

전달본의 예제 JSON·CSV는 CPU에서 실제 실행한 결과다. 학습 보고서의 중국·유럽·한국 미래 결과는 CUDA에서 생성됐으므로 동일 체크포인트라도 일부 값이 다르다. 이번 TFT A 전체 대조에서 반올림 후 4,942행에 차이가 있었고 최대 차이는 34대였다. 기존 CPU 모델을 재사용한 나머지 4개 시장은 일치했다. 장치별 결과를 섞지 않고 전달본의 예제와 검산 기준을 CPU 실행으로 통일했다.

ZIP 전체를 풀어 폴더 관계를 유지한다. 실행 의존성은 `pyproject.toml`과 `uv.lock`에 고정되어 있다. 차량 분류·베어링 수량·프론트 UI는 백엔드와 프론트 담당 범위다. 24개월 출력 기능과 24개월 정확도 검증은 별개다.
