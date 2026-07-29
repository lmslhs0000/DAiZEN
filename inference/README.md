# inference — 원강산업 수요예측

원강산업(자동차용 정밀가공품 제조)의 주문·생산·재고 데이터를 기반으로 수요를 예측하고,
생산계획 및 재고관리에 활용할 수 있는 형태로 결과를 제공하는 파트

## 개발 환경

- Python 3.12
- 패키지 관리: uv

## 사용 패키지

| 패키지 | 용도 |
| --- | --- |
| pandas | 데이터 처리 및 시계열 집계 |
| numpy | 수치 계산 |
| openpyxl | 엑셀(.xlsx) 파일 읽기 |
| pyyaml | 설정 파일 읽기 |
| scikit-learn | 모델 및 평가 지표 |(추후 데이터에 따른 모델 변경)
| pytest (dev) | 테스트 |
| ruff (dev) | 코드 검사 및 포맷 |

## 실행 방법

의존성 설치:

    uv sync

실행:

    uv run main.py

테스트 및 코드 검사:

    uv run pytest
    uv run ruff check .

## 폴더 구조

    inference/
    ├── config/            # 예측 설정 (예측 기간, 시간 단위, 평가 지표)
    ├── data/
    │   ├── raw/           # 원본 데이터 (git 제외)
    │   └── processed/     # 정제된 데이터 (git 제외)
    ├── notebooks/         # 데이터 탐색(EDA)용 노트북
    ├── src/
    │   ├── config.py      # 설정 파일 로더
    │   ├── data/          # 데이터 읽기 및 정제
    │   ├── features/      # 파생 변수 생성 (과거값, 이동평균, 달력 등)
    │   ├── models/        # 예측 모델
    │   └── evaluation/    # 평가 지표 및 백테스트
    └── tests/             # 테스트

## 데이터 취급 원칙

회사 데이터는 저장소에 올리지 않습니다. `data/` 아래의 엑셀·CSV 파일은 `.gitignore`로 제외하고,
폴더 구조만 `.gitkeep`으로 유지합니다.

## 진행 계획

1. 데이터 확보 및 탐색(EDA)
2. 데이터 정제 및 시계열 정렬
3. 검증 방식·평가 지표 확정
4. 베이스라인 수립
5. 모델 후보 비교 및 오차 분석