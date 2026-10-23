"""백테스트 챔피언(기본 ens11_npm, --recipe로 선택)을 최신 자료(2026-07까지)로 재현해 납품용 미래 24개월 예측(2026-08~2028-07)을 만든다.

재료
- tft_a: `forecast_tft.py --origins 2026-07 --no-check`로 미리 학습한 run
- basic_xgboost_v2: v2 run의 시장별 선택 설정으로 이 스크립트가 2026-07에 새로 학습
- rich49 또는 rich_cal(레시피가 정함): 별도 프로세스로 2026-07에 새로 학습(rich_core import 충돌 때문)
- ets: ETS 전용 환경(statsmodels)으로 2026-07에 새로 학습
- 작년 그대로

조립(백테스트와 같은 함수): combo_horizon_v2 = 1~12개월 XGBoost v2, 13~24개월 XGBoost v2와 작년 그대로 반반.
레시피(RECIPES): ens4 = 네 재료 평균, ens4_median = rich49 포함 중앙값, ens4_median_cal = rich_cal 포함 중앙값,
ens5_tftgp = 여기에 공동 TFT + 동력원(tft_global_pt, --extra-run)을 더한 다섯 재료 중앙값,
ens5_tftgp_a3 = 그 TFT A 자리를 seed 42·43·44 평균(tft_a3, --seed-runs)으로 바꾼 것,
ens6_tw = ens5_tftgp_a3에 rich_tw(rich_cal 입력 + Tweedie 손실, --extra2-run)를 여섯 번째로 더한 중앙값,
ens8_dn = ens6_tw에 nhits_global·deepar_global(--more-runs, 이 순서)을 더한 여덟 재료 중앙값,
ens9_vw = ens8_dn에 nhits_vw(판매 규모 가중 N-HiTS; --more-runs 세 번째)를 더한 아홉 재료 중앙값,
ens10_rvw = ens9_vw에 rich_cal_vw(판매 규모 가중 rich_cal; --more-runs 네 번째)를 더한 열 재료 중앙값,
ens10_xlvws = ens10_rvw에서 XGBoost 결합의 재료를 basic_xgboost_lvw(로그 정답·판매 규모 가중 시장별 XGBoost; --xgb-run)로
바꾼 것. 레시피의 xgb_model이 --xgb-run 모델 이름을 정한다(기본 basic_xgboost_v2).
ens11_npm = ens10_xlvws에 nhits_pm(시장별 N-HiTS; --more-runs 다섯 번째)을 더한 열한 재료 중앙값(현재 챔피언).
ens12_chrxl = ens11_npm에 chronos2_xl(--more-runs 여섯 번째)을 더한 대안(규칙 통과, 확인용 조금 나빠 비채택; 사용자 판단용).
ens12_chrxl_xcn = 중국은 ens11_npm, 나머지 6개 시장은 ens12_chrxl인 대안(같은 --more-runs 여섯; 사용자 판단용).
ens12_chrft = ens11_npm에 chronos2_ft(기준월 미세조정 Chronos-2; --more-runs 여섯 번째)를 더한 대안(사용자 판단용).
ens12_ent = ens11_npm에 rich_cal_vw_ent(시장 신차 비중; --more-runs 여섯 번째)를 더한 대안(사용자 판단용).
ens13_entft = ens11_npm에 rich_cal_vw_ent·chronos2_ft(--more-runs 여섯·일곱 번째)를 더한 대안(사용자 판단용).
ens12_brand = ens11_npm에 rich_cal_vw_brand(브랜드 흐름; --more-runs 여섯 번째)를 더한 대안(사용자 판단용).
ens12_chrft_xbr = ens12_chrft에서 XGBoost 결합 재료를 basic_xgboost_blvw로 바꾼 대안(백테스트 최저; 사용자 판단용).
ens11_tweb = ens11_npm에서 rich_tw를 rich_tw_eb(--extra2-run)로 바꾼 대안(사용자 판단용).
예측할 수 없는 차종은 24개월 모두 예측 공란 + 상태.
바탕화면 서비스는 바꾸지 않는다.

사용: PY experiments/horizon_eval_v1/deliver.py --tft-run <production/tft_a_...> --v2-run <reports/basic_xgboost_v2_...>
"""

import argparse
import subprocess
import sys

import numpy as np
import pandas as pd

from common import (CSV_PATH, HERE, HORIZON, KEY, PROJECT_ROOT, ROOT, file_sha256, guarded, new_run_dir, read_json,
                    require_unchanged, stack_outputs, write_csv, write_json)
import baseline
import forecast_xgboost
from challengers import RULES, combine, read_statuses
from ensembles import average_many, iqm_many, median_many, route_by_market
from naive import seasonal_naive
from score import read_run

ORIGIN = "2026-07"
ETS_PYTHON = ROOT / "experiments" / "ets_v1" / ".venv" / "Scripts" / "python.exe"
FUTURE_COLUMNS = (["run_id", "algorithm", "selection_method", "phase", "train_end", "origin"] + KEY
                  + ["Month", "horizon", "prediction", "status"])


def build_combo(xgb, naive):
    return combine(xgb, naive, "combo_horizon_v2", RULES["combo_horizon"][0])


def build_ens4(tft, xgb, naive, rich, ets):
    """ens4 규칙: (tft_a + combo_horizon_v2 + rich49 + ets) ÷ 4."""
    return average_many([tft, build_combo(xgb, naive), rich, ets], "ens4")


def build_ens4_median(tft, xgb, naive, rich, ets):
    """ens4_median 규칙(2026-10-01 챔피언): tft_a·combo_horizon_v2·rich49·ets의 점별 중앙값."""
    return median_many([tft, build_combo(xgb, naive), rich, ets], "ens4_median")


def build_ens4_median_cal(tft, xgb, naive, rich, ets):
    """ens4_median_cal 규칙(2026-10-01 챔피언): tft_a·combo_horizon_v2·rich_cal·ets의 점별 중앙값."""
    return median_many([tft, build_combo(xgb, naive), rich, ets], "ens4_median_cal")


def build_ens5_tftgp(tft, xgb, naive, rich, ets, extra):
    """ens5_tftgp 규칙(2026-10-01 챔피언): tft_a·tft_global_pt·combo_horizon_v2·rich_cal·ets의 점별 중앙값."""
    return median_many([tft, extra, build_combo(xgb, naive), rich, ets], "ens5_tftgp")


def build_ens5_tftgp_a3(tft, xgb, naive, rich, ets, extra, seeds):
    """ens5_tftgp_a3 규칙(2026-10-01 챔피언): tft_a3(TFT A seed 42·43·44 점별 평균)·tft_global_pt·combo_horizon_v2·
    rich_cal·ets의 점별 중앙값."""
    tft_a3 = average_many([tft, *seeds], "tft_a3")
    return median_many([tft_a3, extra, build_combo(xgb, naive), rich, ets], "ens5_tftgp_a3")


def build_ens6_tw(tft, xgb, naive, rich, ets, extra, seeds, extra2):
    """ens6_tw 규칙(2026-10-02 챔피언): tft_a3·tft_global_pt·combo_horizon_v2·rich_cal·rich_tw·ets 여섯 재료의
    점별 중앙값(가운데 두 값의 평균)."""
    tft_a3 = average_many([tft, *seeds], "tft_a3")
    return median_many([tft_a3, extra, build_combo(xgb, naive), rich, extra2, ets], "ens6_tw")


def build_ens8_dn(tft, xgb, naive, rich, ets, extra, seeds, extra2, more):
    """ens8_dn 규칙(2026-10-02 챔피언): ens6_tw 여섯 재료 + nhits_global + deepar_global의 점별 중앙값(가운데 두 값 평균)."""
    tft_a3 = average_many([tft, *seeds], "tft_a3")
    return median_many([tft_a3, extra, build_combo(xgb, naive), rich, extra2, ets, *more], "ens8_dn")


def build_ens9_vw(tft, xgb, naive, rich, ets, extra, seeds, extra2, more):
    """ens9_vw 규칙(2026-10-02 챔피언): ens8_dn 여덟 재료 + nhits_vw의 점별 중앙값."""
    tft_a3 = average_many([tft, *seeds], "tft_a3")
    return median_many([tft_a3, extra, build_combo(xgb, naive), rich, extra2, ets, *more], "ens9_vw")


def build_ens10_rvw(tft, xgb, naive, rich, ets, extra, seeds, extra2, more):
    """ens10_rvw 규칙(2026-10-03 챔피언): ens9_vw 아홉 재료 + rich_cal_vw의 점별 중앙값(가운데 두 값 평균)."""
    tft_a3 = average_many([tft, *seeds], "tft_a3")
    return median_many([tft_a3, extra, build_combo(xgb, naive), rich, extra2, ets, *more], "ens10_rvw")


def build_ens10_xlvws(tft, xgb, naive, rich, ets, extra, seeds, extra2, more):
    """ens10_xlvws 규칙(2026-10-03 챔피언): ens10_rvw와 같은 열 재료 중앙값이되 XGBoost 결합은 basic_xgboost_lvw로."""
    tft_a3 = average_many([tft, *seeds], "tft_a3")
    return median_many([tft_a3, extra, build_combo(xgb, naive), rich, extra2, ets, *more], "ens10_xlvws")


def build_ens11_npm(tft, xgb, naive, rich, ets, extra, seeds, extra2, more):
    """ens11_npm 규칙(2026-10-03 챔피언): ens10_xlvws 열 재료 + nhits_pm의 점별 중앙값."""
    tft_a3 = average_many([tft, *seeds], "tft_a3")
    return median_many([tft_a3, extra, build_combo(xgb, naive), rich, extra2, ets, *more], "ens11_npm")


def build_ens12_chrxl(tft, xgb, naive, rich, ets, extra, seeds, extra2, more):
    """ens12_chrxl(2026-10-03, 규칙 통과·확인용 조금 나빠 비채택, 사용자 판단용 대안): ens11_npm 재료 + chronos2_xl."""
    tft_a3 = average_many([tft, *seeds], "tft_a3")
    return median_many([tft_a3, extra, build_combo(xgb, naive), rich, extra2, ets, *more], "ens12_chrxl")


def build_ens12_chrft(tft, xgb, naive, rich, ets, extra, seeds, extra2, more):
    """ens12_chrft((au), 규칙 통과·확인용 1~3개월 +0.03%p로 비채택, 사용자 판단용 대안): ens11_npm 재료 + chronos2_ft."""
    tft_a3 = average_many([tft, *seeds], "tft_a3")
    return median_many([tft_a3, extra, build_combo(xgb, naive), rich, extra2, ets, *more], "ens12_chrft")


def build_ens12_ent(tft, xgb, naive, rich, ets, extra, seeds, extra2, more):
    """ens12_ent((ay), 규칙 통과·확인용 1~3개월 +0.04%p로 비채택, 사용자 판단용 대안): ens11_npm 재료 + rich_cal_vw_ent."""
    tft_a3 = average_many([tft, *seeds], "tft_a3")
    return median_many([tft_a3, extra, build_combo(xgb, naive), rich, extra2, ets, *more], "ens12_ent")


def build_ens12_brand(tft, xgb, naive, rich, ets, extra, seeds, extra2, more):
    """ens12_brand((bc), 규칙 통과·확인용 1~3개월 +0.03%p로 비채택, 사용자 판단용 대안): ens11_npm 재료 + rich_cal_vw_brand."""
    tft_a3 = average_many([tft, *seeds], "tft_a3")
    return median_many([tft_a3, extra, build_combo(xgb, naive), rich, extra2, ets, *more], "ens12_brand")


def build_ens13_entft(tft, xgb, naive, rich, ets, extra, seeds, extra2, more):
    """ens13_entft((az), 규칙 통과·확인용 1~3개월 +0.03%p로 비채택, 사용자 판단용 대안):
    ens11_npm 재료 + rich_cal_vw_ent + chronos2_ft(13개 중앙값)."""
    tft_a3 = average_many([tft, *seeds], "tft_a3")
    return median_many([tft_a3, extra, build_combo(xgb, naive), rich, extra2, ets, *more], "ens13_entft")


def build_ens12_chrft_xbr(tft, xgb, naive, rich, ets, extra, seeds, extra2, more):
    """ens12_chrft_xbr((bs), 백테스트 최저 26.274 — 확인용은 보지 않은 사용자 판단용 대안): ens12_chrft와 같은 재료에서
    XGBoost 결합 재료만 basic_xgboost_blvw(브랜드 흐름)로 만든 것(--xgb-run에 production basic_xgboost_blvw)."""
    tft_a3 = average_many([tft, *seeds], "tft_a3")
    return median_many([tft_a3, extra, build_combo(xgb, naive), rich, extra2, ets, *more], "ens12_chrft_xbr")


def build_ens11_tweb(tft, xgb, naive, rich, ets, extra, seeds, extra2, more):
    """ens11_tweb((bl), 백테스트 26.354 — 확인용은 보지 않은 사용자 판단용 대안): ens11_npm에서 rich_tw 자리를
    rich_tw_eb(신차 비중 + 브랜드 흐름)로 바꾼 열한 재료 중앙값(--extra2-run에 production rich_tw_eb)."""
    tft_a3 = average_many([tft, *seeds], "tft_a3")
    return median_many([tft_a3, extra, build_combo(xgb, naive), rich, extra2, ets, *more], "ens11_tweb")


def build_ens13_final(tft, xgb, naive, rich, ets, extra, seeds, extra2, more):
    """ens13_final((ce), 백테스트 26.167 — 확인용은 보지 않은 사용자 판단용 대안): ens11_npm에서 rich_tw → rich_tw_all
    (--extra2-run), XGBoost 결합 재료 → basic_xgboost_blvw(--xgb-run), + chronos2_ft·rich_cal_vw_all(more 끝 둘).
    중국 13~24개월 조정(cn12)과 중국 밖 4~24개월 작년 그대로 반 조정(nv)은 postprocess_delivery.py로 따로 한다."""
    tft_a3 = average_many([tft, *seeds], "tft_a3")
    return median_many([tft_a3, extra, build_combo(xgb, naive), rich, extra2, ets, *more], "ens13_final")


def build_ens14_tfm(tft, xgb, naive, rich, ets, extra, seeds, extra2, more):
    """ens14_tfm((cj), 백테스트 26.121, cn12·nv까지 26.025 — 확인용은 보지 않은 사용자 판단용 대안): ens13_final 재료 +
    timesfm(more 마지막, production timesfm). 중국·중국 밖 조정은 postprocess_delivery.py로 따로 한다."""
    tft_a3 = average_many([tft, *seeds], "tft_a3")
    return median_many([tft_a3, extra, build_combo(xgb, naive), rich, extra2, ets, *more], "ens14_tfm")


def build_ens14_iqm(tft, xgb, naive, rich, ets, extra, seeds, extra2, more):
    """ens14_iqm((co), 백테스트 26.074, cn12·nv까지 25.993 — 확인용은 보지 않은 사용자 판단용 대안): ens14_tfm과 같은
    열넷 재료를 사분위 절사평균(아래·위 3개씩 빼고 가운데 8개 평균)으로 결합. 조정은 postprocess_delivery.py로 따로."""
    tft_a3 = average_many([tft, *seeds], "tft_a3")
    return iqm_many([tft_a3, extra, build_combo(xgb, naive), rich, extra2, ets, *more], "ens14_iqm")


def build_ens15_cbq(tft, xgb, naive, rich, ets, extra, seeds, extra2, more):
    """ens15_cbq((cl), 조정까지 백테스트 25.961 — 확인용은 보지 않은 사용자 판단용 대안): ens14_iqm 열넷 + rich_cb_all
    (more 마지막, production rich_cb_all; CatBoost)을 사분위 절사평균으로 결합. 조정은 postprocess_delivery.py로 따로."""
    tft_a3 = average_many([tft, *seeds], "tft_a3")
    return iqm_many([tft_a3, extra, build_combo(xgb, naive), rich, extra2, ets, *more], "ens15_cbq")


def build_ens11_iqm(tft, xgb, naive, rich, ets, extra, seeds, extra2, more):
    """ens11_iqm((co2), 백테스트 26.327 — 챔피언 재료 그대로 결합만 바꾼 가장 작은 변경, 사용자 판단용 대안):
    ens11_npm 열한 재료를 사분위 절사평균(아래·위 2개씩 빼고 가운데 7개 평균)으로 결합."""
    tft_a3 = average_many([tft, *seeds], "tft_a3")
    return iqm_many([tft_a3, extra, build_combo(xgb, naive), rich, extra2, ets, *more], "ens11_iqm")


def build_ens16_cbpm(tft, xgb, naive, rich, ets, extra, seeds, extra2, more):
    """ens16_cbpm((cs), 조정까지 백테스트 25.902 — 확인용은 보지 않은 사용자 판단용 대안): ens15_cbq 열다섯 +
    rich_cb_pm_all(more 마지막, production 시장별 CatBoost)을 사분위 절사평균으로 결합. 조정은 postprocess_delivery.py로 따로."""
    tft_a3 = average_many([tft, *seeds], "tft_a3")
    return iqm_many([tft_a3, extra, build_combo(xgb, naive), rich, extra2, ets, *more], "ens16_cbpm")


def build_ens17_twpm(tft, xgb, naive, rich, ets, extra, seeds, extra2, more):
    """ens17_twpm((cu), 조정 없이 백테스트 25.840 — 확인용은 보지 않은 사용자 판단용 대안): ens16_cbpm 열여섯 +
    rich_tw_pm_all(more 마지막, production 시장별 XGBoost Tweedie)을 사분위 절사평균으로 결합."""
    tft_a3 = average_many([tft, *seeds], "tft_a3")
    return iqm_many([tft_a3, extra, build_combo(xgb, naive), rich, extra2, ets, *more], "ens17_twpm")


def build_ens18_yoy(tft, xgb, naive, rich, ets, extra, seeds, extra2, more):
    """ens18_yoy((dc), 2026-10-04 챔피언, 백테스트 25.798 — 새 확인 규칙 통과): ens17_twpm 열일곱 +
    rich_cb_yoy(more 마지막, production 작년 대비 변화 CatBoost)을 사분위 절사평균으로 결합."""
    tft_a3 = average_many([tft, *seeds], "tft_a3")
    return iqm_many([tft_a3, extra, build_combo(xgb, naive), rich, extra2, ets, *more], "ens18_yoy")


def build_ens28_front(tft, xgb, naive, rich, ets, extra, seeds, extra2, more):
    """ens28_front((ec)): ens22_tft + nhits_global·nhits_pm·tft_global_pt seed 43·44 재료(스물여덟), 사분위 절사평균.
    챔피언 ens_part_front2는 1~3개월만 이것(postprocess_delivery.py --route-from)."""
    tft_a3 = average_many([tft, *seeds], "tft_a3")
    return iqm_many([tft_a3, extra, build_combo(xgb, naive), rich, extra2, ets, *more], "ens28_front")


def build_ens22_tft(tft, xgb, naive, rich, ets, extra, seeds, extra2, more):
    """ens22_tft((ea)·(eb)): ens20_lrel 스물 + TFT A seed 45·46 재료, 사분위 절사평균. 챔피언 ens_part_front는 1~3개월만
    이것(+ 중국 밖 nv; nv는 4~24개월만 바꾸므로 1~3개월에는 영향 없음)."""
    tft_a3 = average_many([tft, *seeds], "tft_a3")
    return iqm_many([tft_a3, extra, build_combo(xgb, naive), rich, extra2, ets, *more], "ens22_tft")


def build_ens22_seed(tft, xgb, naive, rich, ets, extra, seeds, extra2, more):
    """ens22_seed((dy)·(dz)): ens20_lrel 스물 + rich_cb_yoy_lcg seed 43·44 재료, 사분위 절사평균. 챔피언 ens_part_loo는
    13~24개월만 이것(+ 중국 밖 nv)을 쓰고 1~12개월은 ens20_lrel_nv(postprocess_delivery.py --route-from)."""
    tft_a3 = average_many([tft, *seeds], "tft_a3")
    return iqm_many([tft_a3, extra, build_combo(xgb, naive), rich, extra2, ets, *more], "ens22_seed")


def build_ens20_lrel(tft, xgb, naive, rich, ets, extra, seeds, extra2, more):
    """ens20_lrel((dg)·(dh), 2026-10-04 챔피언 ens20_lrel_nv의 조정 전 판): ens19_ypm에서 rich_cb_yoy 자리를 rich_cb_yoy_lcg
    (전 시장 수명 특징)로 바꾸고 rich_cb_rel(정답 기준 = 직전 12개월 평균)을 더한 스물을 사분위 절사평균으로 결합.
    중국 밖 조정은 postprocess_delivery.py로 따로."""
    tft_a3 = average_many([tft, *seeds], "tft_a3")
    return iqm_many([tft_a3, extra, build_combo(xgb, naive), rich, extra2, ets, *more], "ens20_lrel")


def build_ens19_ypm(tft, xgb, naive, rich, ets, extra, seeds, extra2, more):
    """ens19_ypm((df), 2026-10-04 챔피언 ens19_ypm_nv의 조정 전 판): ens18_yoy 열여덟 + rich_cb_yoy_pm(more 마지막,
    production 시장별 작년 대비 CatBoost)을 사분위 절사평균으로 결합. 중국 밖 조정은 postprocess_delivery.py로 따로."""
    tft_a3 = average_many([tft, *seeds], "tft_a3")
    return iqm_many([tft_a3, extra, build_combo(xgb, naive), rich, extra2, ets, *more], "ens19_ypm")


def build_ens12_chrxl_xcn(tft, xgb, naive, rich, ets, extra, seeds, extra2, more):
    """ens12_chrxl_xcn((ax), 규칙 통과·확인용 1~3개월 +0.005%p로 비채택, 사용자 판단용 대안):
    중국은 ens11_npm(more 앞 다섯), 나머지 시장은 ens12_chrxl(more 여섯 = + chronos2_xl)."""
    base = build_ens11_npm(tft, xgb, naive, rich, ets, extra, seeds, extra2, more[:5])
    plus = build_ens12_chrxl(tft, xgb, naive, rich, ets, extra, seeds, extra2, more)
    return route_by_market(base, plus, ["China"], "ens12_chrxl_xcn")


RICH_MODELS = {"rich_log_l2": "rich49", "calendar_l2": "rich_cal"}
RECIPES = {"ens4": {"build": build_ens4, "rich_variant": "rich_log_l2",
                    "text": "(tft_a + combo_horizon_v2 + rich49 + ets) / 4"},
           "ens4_median": {"build": build_ens4_median, "rich_variant": "rich_log_l2",
                           "text": "median(tft_a, combo_horizon_v2, rich49, ets)"},
           "ens4_median_cal": {"build": build_ens4_median_cal, "rich_variant": "calendar_l2",
                               "text": "median(tft_a, combo_horizon_v2, rich_cal, ets); rich_cal = rich49 + "
                                       "target-month calendar days and weekday counts"},
           "ens5_tftgp": {"build": build_ens5_tftgp, "rich_variant": "calendar_l2", "extra_model": "tft_global_pt",
                          "text": "median(tft_a, tft_global_pt, combo_horizon_v2, rich_cal, ets); tft_global_pt = "
                                  "one TFT over all 7 markets + static powertrain code"},
           "ens5_tftgp_a3": {"build": build_ens5_tftgp_a3, "rich_variant": "calendar_l2", "extra_model": "tft_global_pt",
                             "seed_models": ["tft_a_s43", "tft_a_s44"],
                             "text": "median(tft_a3, tft_global_pt, combo_horizon_v2, rich_cal, ets); tft_a3 = mean of "
                                     "TFT A trained with seeds 42, 43, 44"},
           "ens6_tw": {"build": build_ens6_tw, "rich_variant": "calendar_l2", "extra_model": "tft_global_pt",
                       "seed_models": ["tft_a_s43", "tft_a_s44"], "extra2_model": "rich_tw",
                       "text": "median(tft_a3, tft_global_pt, combo_horizon_v2, rich_cal, rich_tw, ets); rich_tw = "
                               "rich_cal inputs with raw-sales target and Tweedie loss (power 1.5)"},
           "ens8_dn": {"build": build_ens8_dn, "rich_variant": "calendar_l2", "extra_model": "tft_global_pt",
                       "seed_models": ["tft_a_s43", "tft_a_s44"], "extra2_model": "rich_tw",
                       "more_models": ["nhits_global", "deepar_global"],
                       "text": "median(tft_a3, tft_global_pt, combo_horizon_v2, rich_cal, rich_tw, ets, nhits_global, "
                               "deepar_global); nhits_global = one N-HiTS over 7 markets, deepar_global = one DeepAR "
                               "(negative binomial) over 7 markets"},
           "ens9_vw": {"build": build_ens9_vw, "rich_variant": "calendar_l2", "extra_model": "tft_global_pt",
                       "seed_models": ["tft_a_s43", "tft_a_s44"], "extra2_model": "rich_tw",
                       "more_models": ["nhits_global", "deepar_global", "nhits_vw"],
                       "text": "median(tft_a3, tft_global_pt, combo_horizon_v2, rich_cal, rich_tw, ets, nhits_global, "
                               "deepar_global, nhits_vw); nhits_vw = N-HiTS trained with per-series weight = "
                               "normalizer scale / mean scale"},
           "ens10_rvw": {"build": build_ens10_rvw, "rich_variant": "calendar_l2", "extra_model": "tft_global_pt",
                         "seed_models": ["tft_a_s43", "tft_a_s44"], "extra2_model": "rich_tw",
                         "more_models": ["nhits_global", "deepar_global", "nhits_vw", "rich_cal_vw"],
                         "text": "median of 10: ens9_vw members + rich_cal_vw (rich_cal trained with per-series weight = "
                                 "series mean Sales / mean over series)"},
           "ens10_xlvws": {"build": build_ens10_xlvws, "rich_variant": "calendar_l2", "extra_model": "tft_global_pt",
                           "seed_models": ["tft_a_s43", "tft_a_s44"], "extra2_model": "rich_tw",
                           "more_models": ["nhits_global", "deepar_global", "nhits_vw", "rich_cal_vw"],
                           "xgb_model": "basic_xgboost_lvw",
                           "text": "ens10_rvw with combo_horizon built from basic_xgboost_lvw (per-market XGBoost, log1p "
                                   "target, sample weight = series mean Sales / market mean)"},
           "ens11_npm": {"build": build_ens11_npm, "rich_variant": "calendar_l2", "extra_model": "tft_global_pt",
                         "seed_models": ["tft_a_s43", "tft_a_s44"], "extra2_model": "rich_tw",
                         "more_models": ["nhits_global", "deepar_global", "nhits_vw", "rich_cal_vw", "nhits_pm"],
                         "xgb_model": "basic_xgboost_lvw",
                         "text": "median of 11: ens10_xlvws members + nhits_pm (N-HiTS trained per market)"},
           "ens12_chrxl": {"build": build_ens12_chrxl, "rich_variant": "calendar_l2", "extra_model": "tft_global_pt",
                           "seed_models": ["tft_a_s43", "tft_a_s44"], "extra2_model": "rich_tw",
                           "more_models": ["nhits_global", "deepar_global", "nhits_vw", "rich_cal_vw", "nhits_pm",
                                           "chronos2_xl"],
                           "xgb_model": "basic_xgboost_lvw",
                           "text": "median of 12: ens11_npm members + chronos2_xl (Chronos-2 zero-shot, market chunks); "
                                   "alternative, not champion (check origin slightly worse)"},
           "ens12_chrft": {"build": build_ens12_chrft, "rich_variant": "calendar_l2", "extra_model": "tft_global_pt",
                           "seed_models": ["tft_a_s43", "tft_a_s44"], "extra2_model": "rich_tw",
                           "more_models": ["nhits_global", "deepar_global", "nhits_vw", "rich_cal_vw", "nhits_pm",
                                           "chronos2_ft"],
                           "xgb_model": "basic_xgboost_lvw",
                           "text": "median of 12: ens11_npm members + chronos2_ft (Chronos-2 fine-tuned at the origin); "
                                   "alternative, not champion (check origin front +0.03pp)"},
           "ens12_ent": {"build": build_ens12_ent, "rich_variant": "calendar_l2", "extra_model": "tft_global_pt",
                         "seed_models": ["tft_a_s43", "tft_a_s44"], "extra2_model": "rich_tw",
                         "more_models": ["nhits_global", "deepar_global", "nhits_vw", "rich_cal_vw", "nhits_pm",
                                         "rich_cal_vw_ent"],
                         "xgb_model": "basic_xgboost_lvw",
                         "text": "median of 12: ens11_npm members + rich_cal_vw_ent (rich_cal_vw + market entrant shares); "
                                 "alternative, not champion (check origin front +0.04pp)"},
           "ens12_brand": {"build": build_ens12_brand, "rich_variant": "calendar_l2", "extra_model": "tft_global_pt",
                           "seed_models": ["tft_a_s43", "tft_a_s44"], "extra2_model": "rich_tw",
                           "more_models": ["nhits_global", "deepar_global", "nhits_vw", "rich_cal_vw", "nhits_pm",
                                           "rich_cal_vw_brand"],
                           "xgb_model": "basic_xgboost_lvw",
                           "text": "median of 12: ens11_npm members + rich_cal_vw_brand (rich_cal_vw + brand momentum); "
                                   "alternative, not champion (check origin front +0.03pp, middle -0.17pp)"},
           "ens13_entft": {"build": build_ens13_entft, "rich_variant": "calendar_l2", "extra_model": "tft_global_pt",
                           "seed_models": ["tft_a_s43", "tft_a_s44"], "extra2_model": "rich_tw",
                           "more_models": ["nhits_global", "deepar_global", "nhits_vw", "rich_cal_vw", "nhits_pm",
                                           "rich_cal_vw_ent", "chronos2_ft"],
                           "xgb_model": "basic_xgboost_lvw",
                           "text": "median of 13: ens11_npm members + rich_cal_vw_ent + chronos2_ft; "
                                   "alternative, not champion (check origin front +0.03pp, middle -0.20pp)"},
           "ens12_chrft_xbr": {"build": build_ens12_chrft_xbr, "rich_variant": "calendar_l2",
                               "extra_model": "tft_global_pt", "seed_models": ["tft_a_s43", "tft_a_s44"],
                               "extra2_model": "rich_tw",
                               "more_models": ["nhits_global", "deepar_global", "nhits_vw", "rich_cal_vw", "nhits_pm",
                                               "chronos2_ft"],
                               "xgb_model": "basic_xgboost_blvw",
                               "text": "median of 12: ens12_chrft members with the XGBoost combo built from "
                                       "basic_xgboost_blvw (brand momentum); backtest only, check origin not used"},
           "ens11_tweb": {"build": build_ens11_tweb, "rich_variant": "calendar_l2", "extra_model": "tft_global_pt",
                          "seed_models": ["tft_a_s43", "tft_a_s44"], "extra2_model": "rich_tw_eb",
                          "more_models": ["nhits_global", "deepar_global", "nhits_vw", "rich_cal_vw", "nhits_pm"],
                          "xgb_model": "basic_xgboost_lvw",
                          "text": "median of 11: ens11_npm members with rich_tw replaced by rich_tw_eb "
                                  "(entrant share + brand momentum); backtest only, check origin not used"},
           "ens13_final": {"build": build_ens13_final, "rich_variant": "calendar_l2", "extra_model": "tft_global_pt",
                           "seed_models": ["tft_a_s43", "tft_a_s44"], "extra2_model": "rich_tw_all",
                           "more_models": ["nhits_global", "deepar_global", "nhits_vw", "rich_cal_vw", "nhits_pm",
                                           "chronos2_ft", "rich_cal_vw_all"],
                           "xgb_model": "basic_xgboost_blvw",
                           "text": "median of 13: ens11_npm members with rich_tw -> rich_tw_all (12 China/external "
                                   "signals), XGBoost combo from basic_xgboost_blvw, plus chronos2_ft and rich_cal_vw_all; "
                                   "backtest only, check origin not used"},
           "ens14_tfm": {"build": build_ens14_tfm, "rich_variant": "calendar_l2", "extra_model": "tft_global_pt",
                         "seed_models": ["tft_a_s43", "tft_a_s44"], "extra2_model": "rich_tw_all",
                         "more_models": ["nhits_global", "deepar_global", "nhits_vw", "rich_cal_vw", "nhits_pm",
                                         "chronos2_ft", "rich_cal_vw_all", "timesfm"],
                         "xgb_model": "basic_xgboost_blvw",
                         "text": "median of 14: ens13_final members plus timesfm (TimesFM 2.5 zero-shot); "
                                 "backtest only, check origin not used"},
           "ens14_iqm": {"build": build_ens14_iqm, "rich_variant": "calendar_l2", "extra_model": "tft_global_pt",
                         "seed_models": ["tft_a_s43", "tft_a_s44"], "extra2_model": "rich_tw_all",
                         "more_models": ["nhits_global", "deepar_global", "nhits_vw", "rich_cal_vw", "nhits_pm",
                                         "chronos2_ft", "rich_cal_vw_all", "timesfm"],
                         "xgb_model": "basic_xgboost_blvw",
                         "text": "interquartile mean of 14 (drop 3 lowest and 3 highest, average the middle 8): "
                                 "ens14_tfm members; backtest only, check origin not used"},
           "ens15_cbq": {"build": build_ens15_cbq, "rich_variant": "calendar_l2", "extra_model": "tft_global_pt",
                         "seed_models": ["tft_a_s43", "tft_a_s44"], "extra2_model": "rich_tw_all",
                         "more_models": ["nhits_global", "deepar_global", "nhits_vw", "rich_cal_vw", "nhits_pm",
                                         "chronos2_ft", "rich_cal_vw_all", "timesfm", "rich_cb_all"],
                         "xgb_model": "basic_xgboost_blvw",
                         "text": "interquartile mean of 15: ens14_iqm members plus rich_cb_all (CatBoost on the "
                                 "all-signal rich features); backtest only, check origin not used"},
           "ens11_iqm": {"build": build_ens11_iqm, "rich_variant": "calendar_l2", "extra_model": "tft_global_pt",
                         "seed_models": ["tft_a_s43", "tft_a_s44"], "extra2_model": "rich_tw",
                         "more_models": ["nhits_global", "deepar_global", "nhits_vw", "rich_cal_vw", "nhits_pm"],
                         "xgb_model": "basic_xgboost_lvw",
                         "text": "interquartile mean of the 11 ens11_npm members (drop 2 lowest and 2 highest); "
                                 "backtest only, check origin not used"},
           "ens16_cbpm": {"build": build_ens16_cbpm, "rich_variant": "calendar_l2", "extra_model": "tft_global_pt",
                          "seed_models": ["tft_a_s43", "tft_a_s44"], "extra2_model": "rich_tw_all",
                          "more_models": ["nhits_global", "deepar_global", "nhits_vw", "rich_cal_vw", "nhits_pm",
                                          "chronos2_ft", "rich_cal_vw_all", "timesfm", "rich_cb_all", "rich_cb_pm_all"],
                          "xgb_model": "basic_xgboost_blvw",
                          "text": "interquartile mean of 16: ens15_cbq members plus rich_cb_pm_all (CatBoost trained "
                                  "per market); backtest only, check origin not used"},
           "ens17_twpm": {"build": build_ens17_twpm, "rich_variant": "calendar_l2", "extra_model": "tft_global_pt",
                          "seed_models": ["tft_a_s43", "tft_a_s44"], "extra2_model": "rich_tw_all",
                          "more_models": ["nhits_global", "deepar_global", "nhits_vw", "rich_cal_vw", "nhits_pm",
                                          "chronos2_ft", "rich_cal_vw_all", "timesfm", "rich_cb_all", "rich_cb_pm_all",
                                          "rich_tw_pm_all"],
                          "xgb_model": "basic_xgboost_blvw",
                          "text": "interquartile mean of 17: ens16_cbpm members plus rich_tw_pm_all (XGBoost Tweedie "
                                  "trained per market); backtest only, check origin not used"},
           "ens18_yoy": {"build": build_ens18_yoy, "rich_variant": "calendar_l2", "extra_model": "tft_global_pt",
                         "seed_models": ["tft_a_s43", "tft_a_s44"], "extra2_model": "rich_tw_all",
                         "more_models": ["nhits_global", "deepar_global", "nhits_vw", "rich_cal_vw", "nhits_pm",
                                         "chronos2_ft", "rich_cal_vw_all", "timesfm", "rich_cb_all", "rich_cb_pm_all",
                                         "rich_tw_pm_all", "rich_cb_yoy"],
                         "xgb_model": "basic_xgboost_blvw",
                         "text": "interquartile mean of 18: ens17_twpm members plus rich_cb_yoy (CatBoost on the "
                                 "year-over-year log change); champion 2026-10-04"},
           "ens28_front": {"build": build_ens28_front, "rich_variant": "calendar_l2", "extra_model": "tft_global_pt",
                           "seed_models": ["tft_a_s43", "tft_a_s44"], "extra2_model": "rich_tw_all",
                           "more_models": ["nhits_global", "deepar_global", "nhits_vw", "rich_cal_vw", "nhits_pm",
                                           "chronos2_ft", "rich_cal_vw_all", "timesfm", "rich_cb_all", "rich_cb_pm_all",
                                           "rich_tw_pm_all", "rich_cb_yoy_lcg", "rich_cb_yoy_pm", "rich_cb_rel",
                                           "tft_a_s45", "tft_a_s46", "nhits_global_s43", "nhits_global_s44",
                                           "nhits_pm_s43", "nhits_pm_s44", "tft_global_pt_s43", "tft_global_pt_s44"],
                           "xgb_model": "basic_xgboost_blvw",
                           "text": "interquartile mean of 28: ens22_tft members plus N-HiTS global/per-market and global "
                                   "TFT seeds 43 and 44; champion ens_part_front2 uses this for months 1-3 only"},
           "ens22_tft": {"build": build_ens22_tft, "rich_variant": "calendar_l2", "extra_model": "tft_global_pt",
                         "seed_models": ["tft_a_s43", "tft_a_s44"], "extra2_model": "rich_tw_all",
                         "more_models": ["nhits_global", "deepar_global", "nhits_vw", "rich_cal_vw", "nhits_pm",
                                         "chronos2_ft", "rich_cal_vw_all", "timesfm", "rich_cb_all", "rich_cb_pm_all",
                                         "rich_tw_pm_all", "rich_cb_yoy_lcg", "rich_cb_yoy_pm", "rich_cb_rel",
                                         "tft_a_s45", "tft_a_s46"],
                         "xgb_model": "basic_xgboost_blvw",
                         "text": "interquartile mean of 22: ens20_lrel members plus TFT A seeds 45 and 46; champion "
                                 "ens_part_front uses this for months 1-3 only"},
           "ens22_seed": {"build": build_ens22_seed, "rich_variant": "calendar_l2", "extra_model": "tft_global_pt",
                          "seed_models": ["tft_a_s43", "tft_a_s44"], "extra2_model": "rich_tw_all",
                          "more_models": ["nhits_global", "deepar_global", "nhits_vw", "rich_cal_vw", "nhits_pm",
                                          "chronos2_ft", "rich_cal_vw_all", "timesfm", "rich_cb_all", "rich_cb_pm_all",
                                          "rich_tw_pm_all", "rich_cb_yoy_lcg", "rich_cb_yoy_pm", "rich_cb_rel",
                                          "rich_cb_yoy_lcg_s43", "rich_cb_yoy_lcg_s44"],
                          "xgb_model": "basic_xgboost_blvw",
                          "text": "interquartile mean of 22: ens20_lrel members plus rich_cb_yoy_lcg seeds 43 and 44; "
                                  "champion ens_part_loo uses this (+ non-China nv) for months 13-24 only"},
           "ens20_lrel": {"build": build_ens20_lrel, "rich_variant": "calendar_l2", "extra_model": "tft_global_pt",
                          "seed_models": ["tft_a_s43", "tft_a_s44"], "extra2_model": "rich_tw_all",
                          "more_models": ["nhits_global", "deepar_global", "nhits_vw", "rich_cal_vw", "nhits_pm",
                                          "chronos2_ft", "rich_cal_vw_all", "timesfm", "rich_cb_all", "rich_cb_pm_all",
                                          "rich_tw_pm_all", "rich_cb_yoy_lcg", "rich_cb_yoy_pm", "rich_cb_rel"],
                          "xgb_model": "basic_xgboost_blvw",
                          "text": "interquartile mean of 20: ens19_ypm members with rich_cb_yoy replaced by rich_cb_yoy_lcg "
                                  "(all-market lifecycle features) plus rich_cb_rel (target relative to previous 12-month "
                                  "mean); champion ens20_lrel_nv = this + non-China nv adjustment"},
           "ens19_ypm": {"build": build_ens19_ypm, "rich_variant": "calendar_l2", "extra_model": "tft_global_pt",
                         "seed_models": ["tft_a_s43", "tft_a_s44"], "extra2_model": "rich_tw_all",
                         "more_models": ["nhits_global", "deepar_global", "nhits_vw", "rich_cal_vw", "nhits_pm",
                                         "chronos2_ft", "rich_cal_vw_all", "timesfm", "rich_cb_all", "rich_cb_pm_all",
                                         "rich_tw_pm_all", "rich_cb_yoy", "rich_cb_yoy_pm"],
                         "xgb_model": "basic_xgboost_blvw",
                         "text": "interquartile mean of 19: ens18_yoy members plus rich_cb_yoy_pm (year-over-year "
                                 "CatBoost trained per market); champion ens19_ypm_nv = this + non-China nv adjustment"},
           "ens12_chrxl_xcn": {"build": build_ens12_chrxl_xcn, "rich_variant": "calendar_l2",
                               "extra_model": "tft_global_pt", "seed_models": ["tft_a_s43", "tft_a_s44"],
                               "extra2_model": "rich_tw",
                               "more_models": ["nhits_global", "deepar_global", "nhits_vw", "rich_cal_vw", "nhits_pm",
                                               "chronos2_xl"],
                               "xgb_model": "basic_xgboost_lvw",
                               "text": "China: ens11_npm; other 6 markets: ens12_chrxl (adds chronos2_xl); "
                                       "alternative, not champion (check origin front +0.005pp)"}}


def chronos_cn_recipe(name):
    """(el) 챔피언 el_champ용: name 레시피에서 chronos2_ft 자리만 chronos2_ft_cnlc(중국은 수명 공변량 Chronos)로."""
    recipe = dict(RECIPES[name])
    recipe["more_models"] = ["chronos2_ft_cnlc" if m == "chronos2_ft" else m for m in recipe["more_models"]]
    recipe["text"] = (f"{name} with chronos2_ft replaced by chronos2_ft_cnlc (China: lifecycle covariates); "
                      "champion el_champ")
    return recipe


for _name in ("ens20_lrel", "ens22_seed", "ens28_front"):
    RECIPES[f"{_name}_cn"] = chronos_cn_recipe(_name)


def future_table(predictions, statuses, origin, horizon, run_id, algorithm, selection_method="goal_v2"):
    """AGENTS.md 11절 미래 결과 형식. 상태표의 모든 차종이 horizon개 월을 갖고, OK가 아니면 예측은 공란이다."""
    origin = pd.Period(origin, freq="M")
    months = pd.DataFrame({"horizon": range(1, horizon + 1)})
    months["Month"] = [str(origin + h) for h in months["horizon"]]
    table = statuses[KEY + ["prediction_status"]].merge(months, how="cross")
    table = table.merge(predictions[KEY + ["horizon", "prediction"]], on=KEY + ["horizon"], how="left")
    table.loc[table["prediction_status"].ne("OK"), "prediction"] = np.nan
    table = table.rename(columns={"prediction_status": "status"}).assign(
        run_id=run_id, algorithm=algorithm, selection_method=selection_method, phase="production_refit",
        train_end=str(origin), origin=str(origin))
    return table[FUTURE_COLUMNS].sort_values(KEY + ["horizon"]).reset_index(drop=True)


def set_origin(value):
    """(재실행) 납품 기준월을 바꾼다(기본 2026-07). 새 달 실제값을 받은 뒤 같은 절차로 다시 만들 때 쓴다."""
    global ORIGIN
    ORIGIN = str(pd.Period(str(value), freq="M"))


def train_separately(python, script, csv_path, output_base, extra=()):
    """다른 모델 계열을 별도 프로세스로 2026-07에 학습하고 run 폴더를 돌려준다."""
    result = subprocess.run([str(python), "-X", "utf8", "-B", str(HERE / script), "--csv", str(csv_path),
                             "--output-base", str(output_base), "--origins", ORIGIN, "--no-check", *extra],
                            capture_output=True, text=True, encoding="utf-8", cwd=ROOT)
    if result.returncode != 0:
        raise RuntimeError(result.stderr[-3000:])
    return result.stdout.strip().splitlines()[-1]


def check_source(name, predictions, expected_model=None):
    """재료 run이 2026-07 기준 24개월 예측인지(주면 모델 이름도 레시피와 같은지) 확인한다."""
    if set(predictions["origin"]) != {ORIGIN} or sorted(predictions["horizon"].unique()) != list(range(1, HORIZON + 1)):
        raise ValueError(f"{name} run이 {ORIGIN} 기준 {HORIZON}개월 예측이 아닙니다.")
    if expected_model is not None and set(predictions["model"]) != {expected_model}:
        raise ValueError(f"{name} run의 모델이 {sorted(set(predictions['model']))}로 레시피의 {expected_model}와 다릅니다.")


def run(tft_run, v2_run, csv_path=CSV_PATH, output_base=HERE / "production", recipe="ens11_npm",
        xgb_run=None, rich_run=None, ets_run=None, extra_run=None, seed_runs=(), extra2_run=None, more_runs=()):
    """재료 run을 주면 재사용하고(재학습 없음), 없으면 2026-07로 새로 학습한다."""
    input_hash = file_sha256(csv_path)
    data = baseline.read_sales_csv(guarded(csv_path))
    choice = read_json(guarded(v2_run) / "run_summary.json")["choice"]
    parameters = {country: item["parameters"] for country, item in choice.items()}
    xgb_run = xgb_run or forecast_xgboost.run(csv_path, output_base, 4, [ORIGIN], False, parameters=parameters,
                                              model_name="basic_xgboost_v2")
    rich_variant = RECIPES[recipe]["rich_variant"]
    rich_run = rich_run or train_separately(sys.executable, "forecast_rich49.py", csv_path, output_base,
                                            ("--variant", rich_variant))
    ets_run = ets_run or train_separately(ETS_PYTHON, "forecast_ets.py", csv_path, output_base, ("--workers", "4"))
    runs = {"tft": tft_run, "xgb": xgb_run, "rich": rich_run, "ets": ets_run}
    expected = {"tft": "tft_a", "xgb": RECIPES[recipe].get("xgb_model", "basic_xgboost_v2"),
                "rich": RICH_MODELS[rich_variant], "ets": "ets"}
    extra_model = RECIPES[recipe].get("extra_model")
    if extra_model:
        if extra_run is None:
            raise ValueError(f"{recipe} 레시피에는 {extra_model} run(--extra-run)이 필요합니다.")
        runs["extra"], expected["extra"] = extra_run, extra_model
    seed_models = RECIPES[recipe].get("seed_models", [])
    if len(seed_runs) != len(seed_models):
        raise ValueError(f"{recipe} 레시피에는 시드 run {len(seed_models)}개(--seed-runs)가 필요합니다.")
    for i, (path, model) in enumerate(zip(seed_runs, seed_models)):
        runs[f"seed{i}"], expected[f"seed{i}"] = path, model
    extra2_model = RECIPES[recipe].get("extra2_model")
    if extra2_model:
        if extra2_run is None:
            raise ValueError(f"{recipe} 레시피에는 {extra2_model} run(--extra2-run)이 필요합니다.")
        runs["extra2"], expected["extra2"] = extra2_run, extra2_model
    more_models = RECIPES[recipe].get("more_models", [])
    if len(more_runs) != len(more_models):
        raise ValueError(f"{recipe} 레시피에는 추가 재료 run {len(more_models)}개(--more-runs: {more_models})가 필요합니다.")
    for i, (path, model) in enumerate(zip(more_runs, more_models)):
        runs[f"more{i}"], expected[f"more{i}"] = path, model
    sources = {name: (read_run(path), read_statuses(path)) for name, path in runs.items()}
    for name, (predictions, _) in sources.items():
        check_source(name, predictions, expected[name])
    p, s = seasonal_naive(data, ORIGIN, HORIZON)
    naive = stack_outputs([p], [s], "seasonal_naive", pd.Period(ORIGIN, "M"))
    build, recipe_text = RECIPES[recipe]["build"], RECIPES[recipe]["text"]
    members = (sources["tft"], sources["xgb"], naive, sources["rich"], sources["ets"])
    seeds = [sources[f"seed{i}"] for i in range(len(seed_models))]
    more = [sources[f"more{i}"] for i in range(len(more_models))]
    if more_models:
        predictions, statuses = build(*members, sources["extra"], seeds, sources["extra2"], more)
    elif extra2_model:
        predictions, statuses = build(*members, sources["extra"], seeds, sources["extra2"])
    elif seed_models:
        predictions, statuses = build(*members, sources["extra"], seeds)
    else:
        predictions, statuses = build(*members, sources["extra"]) if extra_model else build(*members)

    out = new_run_dir(output_base, f"delivery_{recipe}")
    table = future_table(predictions, statuses, ORIGIN, HORIZON, out.name, recipe)
    ok = table["status"].eq("OK")
    if len(table) != len(statuses) * HORIZON or statuses.duplicated(KEY).any():
        raise ValueError("차종마다 24행이 아닙니다.")
    if not (np.isfinite(table.loc[ok, "prediction"]).all() and table.loc[ok, "prediction"].ge(0).all()):
        raise ValueError("OK 예측에 음수나 비유한 값이 있습니다.")
    if not table.loc[~ok, "prediction"].isna().all():
        raise ValueError("예측 불가 차종에 값이 들어 있습니다.")

    # 추적용: 차종·월별 재료 예측과 결과를 한 표에 모은다.
    combo = build_combo(sources["xgb"], naive)[0]
    xgb_name = expected["xgb"]  # 추적 표의 열 이름은 실제로 쓴 XGBoost 재료를 따른다
    parts = {"tft_a": sources["tft"][0], xgb_name: sources["xgb"][0], "seasonal_naive": naive[0],
             f"combo_horizon[{xgb_name}]": combo, RICH_MODELS[rich_variant]: sources["rich"][0], "ets": sources["ets"][0],
             **({extra_model: sources["extra"][0]} if extra_model else {}),
             **{model: sources[f"seed{i}"][0] for i, model in enumerate(seed_models)},
             **({extra2_model: sources["extra2"][0]} if extra2_model else {}),
             **{model: sources[f"more{i}"][0] for i, model in enumerate(more_models)}, recipe: predictions}
    components = None
    for name, frame in parts.items():
        column = frame[KEY + ["horizon", "prediction"]].rename(columns={"prediction": name})
        components = column if components is None else components.merge(column, on=KEY + ["horizon"], how="outer")

    require_unchanged(csv_path, input_hash)
    write_csv(out / f"{recipe}_future_24m.csv", table)
    write_csv(out / "components.csv", components.sort_values(KEY + ["horizon"]))
    write_json(out / "run_summary.json", {
        "algorithm": recipe, "recipe": recipe_text + "; combo_horizon_v2 = h1-12 XGBoost v2, "
                                                    "h13-24 0.5 XGBoost v2 + 0.5 seasonal naive",
        "train_end": ORIGIN, "origin": ORIGIN, "months": [table["Month"].min(), table["Month"].max()],
        "input_sha256": input_hash, "series": int(len(statuses)), "rows": int(len(table)),
        "status_counts": statuses["prediction_status"].value_counts().to_dict(),
        "sources": {name: str(guarded(path).relative_to(PROJECT_ROOT)) for name, path in runs.items()},
        "xgboost_v2_parameters": parameters,
        "backtest_champion_at_build": read_json(HERE / "champion.json")["current"]["model"]})
    print(out)
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tft-run", required=True)
    parser.add_argument("--v2-run", required=True)
    parser.add_argument("--csv", default=str(CSV_PATH))
    parser.add_argument("--recipe", default="ens11_npm", choices=sorted(RECIPES))
    parser.add_argument("--xgb-run")
    parser.add_argument("--rich-run")
    parser.add_argument("--ets-run")
    parser.add_argument("--extra-run", help="레시피의 다섯 번째 재료 run(ens5_tftgp: production tft_global_pt)")
    parser.add_argument("--seed-runs", nargs="*", default=[], help="TFT A 다른 시드 run들(ens5_tftgp_a3: s43, s44)")
    parser.add_argument("--extra2-run", help="레시피의 여섯 번째 재료 run(ens6_tw: production rich_tw)")
    parser.add_argument("--more-runs", nargs="*", default=[], help="추가 재료 run들(ens8_dn: nhits_global, deepar_global)")
    parser.add_argument("--origin", default=ORIGIN, help="(재실행) 납품 기준월(기본 2026-07); 재료 run도 같은 기준월이어야 함")
    args = parser.parse_args(argv)
    set_origin(args.origin)
    run(args.tft_run, args.v2_run, args.csv, recipe=args.recipe, xgb_run=args.xgb_run, rich_run=args.rich_run,
        ets_run=args.ets_run, extra_run=args.extra_run, seed_runs=args.seed_runs, extra2_run=args.extra2_run,
        more_runs=args.more_runs)


if __name__ == "__main__":
    main()
