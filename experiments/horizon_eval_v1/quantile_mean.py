"""(ga) 분위수 예측 → 평균 예측. 분위수 함수 Q(p)를 [0, 1]에서 적분한다: 주어진 수준 사이는 사다리꼴,
가장 낮은 수준 아래와 가장 높은 수준 위는 끝값으로 연장(평균 = ∫Q(p)dp). numpy만 쓴다."""

import numpy as np


def mean_from_quantiles(levels, values):
    """levels: 오름차순 분위수 수준 K개, values: (..., K) 분위수 값 → (...) 평균."""
    levels = np.asarray(levels, dtype=float)
    values = np.asarray(values, dtype=float)
    middle = np.trapezoid(values, levels, axis=-1) if hasattr(np, "trapezoid") else np.trapz(values, levels, axis=-1)
    return middle + values[..., 0] * levels[0] + values[..., -1] * (1.0 - levels[-1])
