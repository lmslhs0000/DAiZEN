## 환경 설정

이 프로젝트는 `uv`를 사용합니다.

프로젝트를 clone한 뒤:

```powershell
uv sync
```

설치가 정상적으로 되었는지 확인:

```powershell
uv run python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

GPU 환경에서는 CUDA 사용 가능 여부가 `True`로 출력되는지 확인합니다.

## 미래 예측

이미 학습된 `.ckpt` 파일이 있다면 재학습 없이 바로 예측할 수 있습니다.

```powershell
uv run python predict.py
```

예측 시 최근 3개월 동안 판매 기록이 있는 차량을 Active 차량 후보로 사용합니다.

모델 입력 조건을 충족하지 못하는 차량은 예측 대상에서 제외될 수 있습니다. 이는 단종 차량만을 의미하지 않으며, 신규 차량처럼 충분한 과거 이력이 없는 차량도 포함될 수 있습니다.

## 예측 결과

`predict.py` 실행 후 다음 CSV가 생성됩니다.

```text
output/future_forecast_6months.csv
output/future_forecast_1_3_6months.csv
```

1 / 3 / 6개월 결과 예:

```text
market,brand,model,date,horizon,predicted_sales
국산,기아,EV6,2026-08-01,1,760
국산,기아,EV6,2026-10-01,3,812
국산,기아,EV6,2027-01-01,6,735
```

이미 학습된 checkpoint가 포함되어 있는 경우:

uv sync
uv run python predict.py
