"""rich49 계열의 기준월별 24개월 백테스트 예측. 반드시 별도 프로세스로 실행한다.

- rich_log_l2(모델 이름 rich49): seasonal_residual_v1의 rich_core 그대로(49입력, 7시장 공동, 로그 정답, L2)
- rich_raw(2026-10-01 추가, 점수를 보기 전에 고정): 같은 49입력·학습 행·XGBoost 설정(깊이 5, 학습률 0.1, 나무 200)에서
  정답만 판매 대수 그대로(L2). 예측은 역변환 없이 0 미만을 0으로 자른다. rich49의 로그 정답 과소예측(약 −8%)을 없애려는 가설.
- calendar_l2(모델 이름 rich_cal, 2026-10-01 사전 고정): calendar_exposure_v1의 calendar_core 그대로. rich49 49입력에
  대상월의 달력 일수·요일 7개 개수를 더한 57입력(공휴일 가정 없음), 같은 설정·로그 정답·학습 행.
- workday_l2(모델 이름 rich_wd, 2026-10-01 사전 고정): experiments/workdays_v1/workday_core. rich_cal 57입력에
  시장별 대상월 근무일(공휴일 반영, holidays 0.105로 만든 정적 표) 1개를 더한 58입력.
- policy_l2(모델 이름 rich_pol, 2026-10-01 사전 고정): experiments/policy_events_v1/policy_core. rich_cal 57입력에
  공표된 정책 마감 입력 4개(기준 시점까지 발표된 마감만; 학습 행은 대상월 − 1, 예측은 기준월 고정)를 더한 61입력.
- policy_pt_l2(모델 이름 rich_pol_pt, 2026-10-01 사전 고정): experiments/powertrain_v1/policy_pt_core. rich_pol에 동력원
  코드 1개를 더하고 전기차 범위 정책은 그 시점 플러그인 차종에만 적용한 62입력.
- tweedie_cal(모델 이름 rich_tw, 2026-10-02 사전 고정 (g)): experiments/tweedie_v1/tweedie_core. rich_cal 57입력·설정 그대로,
  정답은 판매 대수, objective reg:tweedie(power 1.5), 예측은 판매 대수(로그 역변환 없음).
- pol_pt_tw(모델 이름 rich_pol_pt_tw, 2026-10-02 사전 고정 (j)): experiments/tweedie_pol_pt_v1/pol_pt_tw_core. rich_pol_pt
  62입력·설정 그대로, 정답·손실만 tweedie_cal과 같이 바꿈.
- tweedie_lc(모델 이름 rich_tw_lc, 2026-10-02 사전 고정 (l2)): experiments/china_lifecycle_v1/lc_core. rich_tw에 중국 차종
  수명 특징 3개(공개된 단종·후속까지 개월, 신세대 이후 개월, 조사 대상 여부)를 더함.
- tweedie_mkt(모델 이름 rich_tw_mkt, 2026-10-02 사전 고정 (t)): experiments/market_momentum_v1/mkt_core. rich_tw에 시장 흐름
  2개(시장 합계의 최근 3·12개월 전년 대비 로그 변화, 기준 시점까지)를 더함.
- calendar_vw(모델 이름 rich_cal_vw, 2026-10-03 사전 고정 (z)): experiments/rich_vw_v1/richvw_core. rich_cal 그대로 + 학습
  표본 가중치(차종 평균 판매량 ÷ 차종 평균).
- policy_pt_vw(모델 이름 rich_pol_pt_vw, 2026-10-03 사전 고정 (aa)): experiments/rich_pol_pt_vw_v1/ppvw_core. rich_pol_pt
  그대로 + 같은 학습 가중치.
- calendar_vw_ent(모델 이름 rich_cal_vw_ent, 2026-10-03 사전 고정 (ay)): experiments/entrant_v1/ent_core. rich_cal_vw에
  시장 신차 비중 2개(entrant_features; 학습 행은 대상월 직전 달, 재귀 예측은 기준월 고정)를 더함.
- calendar_vw_brand(모델 이름 rich_cal_vw_brand, 2026-10-03 사전 고정 (bc)): experiments/brand_momentum_v1/brand_core.
  rich_cal_vw에 브랜드 흐름 2개((Country, Brand) 합계의 3·12개월 전년 대비)를 같은 방식으로 더함.
- calendar_vw_eb(모델 이름 rich_cal_vw_eb, 2026-10-03 사전 고정 (bf)): experiments/rich_eb_v1/eb_core.
  rich_cal_vw에 신차 비중 2개 + 브랜드 흐름 2개를 함께 더함.
- calendar_vw_bent(모델 이름 rich_cal_vw_bent, 2026-10-03 사전 고정 (bg)): experiments/rich_bent_v1/bent_core.
  rich_cal_vw에 브랜드 안 신차 비중 2개(같은 브랜드의 최근 출시 차종 판매 비중)를 더함.
- calendar_vw_lc(모델 이름 rich_cal_vw_lc, 2026-10-03 사전 고정 (bh)): experiments/rich_lcvw_v1/lcvw_core.
  rich_cal_vw에 (l2) 중국 수명 특징 3개(같은 사건 표)를 더함.
- calendar_vw_ebl(모델 이름 rich_cal_vw_ebl, 2026-10-03 사전 고정 (bi)): experiments/rich_ebl_v1/ebl_core.
  rich_cal_vw에 신차 비중 2개 + 브랜드 흐름 2개 + 수명 특징 3개를 함께 더함.
- tweedie_eb(모델 이름 rich_tw_eb, 2026-10-03 사전 고정 (bl)): experiments/rich_tweb_v1/tweb_core.
  rich_tw(Tweedie)에 신차 비중 2개 + 브랜드 흐름 2개를 더함.
- calendar_vw_org(모델 이름 rich_cal_vw_org, 2026-10-03 사전 고정 (bw)): experiments/origin_v1/org_core.
  rich_cal_vw에 브랜드 국적 무리의 시장 점유율·변화 2개(외부 표 brand_origin.csv)를 더함.
- calendar_vw_nev(모델 이름 rich_cal_vw_nev, 2026-10-03 사전 고정 (bx)): experiments/nev_v1/nev_core.
  rich_cal_vw에 자기 동력원 코드 + 시장 전기차 비중·변화(10/1 동력원 표)를 더함.
- calendar_vw_off(모델 이름 rich_cal_vw_off, 2026-10-03 사전 고정 (by)): experiments/official_totals_v1/off_core.
  rich_cal_vw에 공식 시장 판매의 3·12개월 변화와 우리 자료 비중(외부 공식 통계)을 더함.
- tweedie_all / calendar_vw_all(모델 이름 rich_tw_all / rich_cal_vw_all, 2026-10-03 사전 고정 (cc)·(cd)): experiments/rich_all_v1.
- calendar_all(모델 이름 rich_cal_all, 2026-10-03 사전 고정 (ci)): calendar_vw_all에서 학습 표본 가중치만 뺀 rich_all_v1/calall_core.
- catboost_all(모델 이름 rich_cb_all, 2026-10-03 사전 고정 (cl)): calendar_all에서 학습기만 CatBoost(catboost_v1/cball_core).
  catboost가 있는 experiments/catboost_v1/.venv로 실행하고, 모델 파일은 코어의 MODEL_FILE(model.cbm)을 쓴다.
- catboost_vw_all(모델 이름 rich_cb_vw_all, 2026-10-03 사전 고정 (cp)): catboost_all + 학습 표본 가중치(catboost_v1/cbvwall_core).
- catboost_tw_all(모델 이름 rich_cb_tw_all, 2026-10-03 사전 고정 (cr)): CatBoost Tweedie(판매 대수 정답, log 평균 baseline; catboost_v1/cbtwall_core).
- catboost_yoy_lcg(모델 이름 rich_cb_yoy_lcg, 2026-10-04 사전 고정 (dg)): 작년 대비 CatBoost + 전 시장 수명 특징(catboost_v1/cblcg_core).
- catboost_rel_all(모델 이름 rich_cb_rel, 2026-10-04 사전 고정 (dh)): 작년 대비 CatBoost에서 정답 기준만 직전 12개월 평균(catboost_v1/cbrelall_core).
- catboost_yoy_rw(모델 이름 rich_cb_yoy_rw, 2026-10-04 사전 고정 (di)): 작년 대비 CatBoost + 최근 자료 학습 가중치 반감기 24개월(catboost_v1/cbrw_core).
- catboost_yoy_sup(모델 이름 rich_cb_yoy_sup, 2026-10-05 사전 고정 (ek)): catboost_yoy_lcg + 공급 특징 2개(FRED AISRSA, catboost_v1/cbsup_core).
- catboost_covid(모델 이름 rich_cb_covid, 2026-10-08 사전 고정 (hh)): catboost_all + 공급난 기간 표시 2개(catboost_v1/cbcovid_core, shortage_features).
- catboost_covidw(모델 이름 rich_cb_covidw, 2026-10-08 사전 고정 (hh2)): catboost_all + 공급난 기간 대상 행 학습 비중 0.25(catboost_v1/cbcovidw_core).
- catboost_huber(모델 이름 rich_cb_huber, 2026-10-08 사전 고정 (hh3)): catboost_all에서 손실만 Huber(delta 0.5, 로그 정답; 정답에서 base_score를 빼고 학습)(catboost_v1/cbhuber_core).
- catboost_subsidy(모델 이름 rich_cb_subsidy, 2026-10-08 사전 고정 (hk)): catboost_all + 한국 전기차 국고보조금 2개(catboost_v1/cbsubsidy_core, subsidy_features).
- catboost_yoy_lcgw·catboost_yoyw_all·catboost_relw_all(rich_cb_yoy_lcgw·rich_cb_yoyw(_pm)·rich_cb_relw, 2026-10-08 사전 고정 (hh4)): 각 원래 코어 + 공급난 기간 대상 행 학습 비중 0.25(cblcgw·cbyoyw·cbrelw_core). --per-market: catboost_covidw → rich_cb_pm_covidw.
- catboost_trend(모델 이름 rich_cb_trend, 2026-10-08 사전 고정 (hj)): catboost_all + 검색 관심도 2개(Google Trends 미국 상위 50개, catboost_v1/cbtrend_core, trends_features).
- catboost_yoy_lcg3(모델 이름 rich_cb_yoy_lcgp, 2026-10-05 사전 고정 (ej)): catboost_yoy_lcg의 수명 모듈을 g3(가격 재설정 + 중국 조사 v2)로(catboost_v1/cblcg3_core). 이름 rich_cb_yoy_lcg3은 (dq) seed 평균 run이 이미 써서 lcgp.
- lightgbm_yoy(모델 이름 rich_lgb_yoy, 2026-10-05 사전 고정 (ei)): lightgbm_all에서 정답만 작년 대비 log 변화(lightgbm_v1/lgbyoy_core).
- catboost_rel_all_s43·s44(모델 이름 rich_cb_rel_s43·s44, 2026-10-05 사전 고정 (ee)): catboost_rel_all의 CatBoost seed만 43·44.
- catboost_yoy_lcg_big(모델 이름 rich_cb_yoy_lcg_big, 2026-10-05 사전 고정 (ds)): catboost_yoy_lcg의 CatBoost 용량을 라이브러리 기본(깊이 6·나무 1,000·학습률 자동)으로(catboost_v1/cbbig_core).
- catboost_yoy_lcg_s43·s44(모델 이름 rich_cb_yoy_lcg_s43·s44, 2026-10-05 사전 고정 (dq)): catboost_yoy_lcg의 CatBoost seed만 43·44.
- catboost_yoy_lcg_s45·s46(모델 이름 rich_cb_yoy_lcg_s45·s46, 2026-10-06 사전 고정 (ff)): 같은 방식으로 seed 45·46.
- catboost_yoy_seg(모델 이름 rich_cb_yoy_seg, 2026-10-04 사전 고정 (dp)): catboost_yoy_lcg + 세그먼트 흐름 2개·차체 원-핫 8개(catboost_v1/cbseg_core).
- catboost_yoy_lcg2(모델 이름 rich_cb_yoy_lcg2, 2026-10-04 사전 고정 (dm)): catboost_yoy_lcg와 같되 북미 확대 사건 표(catboost_v1/cblcg2_core).
- catboost_lcg_all(모델 이름 rich_cb_all_lcg, 2026-10-04 사전 고정 (dl)): 수준 CatBoost + 전 시장 수명 특징(catboost_v1/cblcgall_core).
- catboost_yoy_xm(모델 이름 rich_cb_yoy_xm, 2026-10-04 사전 고정 (dk)): 작년 대비 CatBoost + 같은 차종의 다른 시장 흐름 2개(catboost_v1/cbxm_core).
- calendar_vw_yoy(모델 이름 rich_cvw_yoy, 2026-10-04 사전 고정 (dd)): calendar_vw_all에서 정답만 작년 대비 log 변화(rich_all_v1/cvyoy_core).
- catboost_rate_all(모델 이름 rich_cb_rate, 2026-10-04 사전 고정 (db)): catboost_all + 시장 단기금리 2개(catboost_v1/cbrate_core, rate_features).
- catboost_yoy_all(모델 이름 rich_cb_yoy, 2026-10-04 사전 고정 (dc)): catboost_all에서 정답만 작년 대비 log 변화(catboost_v1/cbyoyall_core).
- lightgbm_all(모델 이름 rich_lgb_all, 2026-10-04 사전 고정 (cv)): calendar_all에서 학습기만 LightGBM(lightgbm_v1/lgball_core, model.txt). experiments/lightgbm_v1/.venv로 실행.
  rich_tw / rich_cal_vw에 신차 비중·브랜드 흐름·수명 사건·브랜드 국적 점유율·공식 시장 판매 12개를 모두 더함.
"""

import argparse
from pathlib import Path
import importlib.util
import json
import sys

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from common import (CHECK_HORIZON, CHECK_ORIGIN, CSV_PATH, HERE, HORIZON, KEY, ORIGINS, ROOT, file_sha256, guarded,
                    new_run_dir, require_unchanged, stack_outputs, write_csv, write_json)

sys.path.insert(0, str(ROOT / "experiments" / "seasonal_residual_v1"))
import rich_core as rc  # noqa: E402  (series_weight_v1의 baseline/xgboost_core 사본을 불러온다)
from baseline import read_sales_csv  # noqa: E402

MODEL_NAMES = {"rich_log_l2": "rich49", "rich_raw": "rich_raw", "calendar_l2": "rich_cal", "workday_l2": "rich_wd",
               "policy_l2": "rich_pol", "policy_pt_l2": "rich_pol_pt", "tweedie_cal": "rich_tw",
               "pol_pt_tw": "rich_pol_pt_tw", "tweedie_lc": "rich_tw_lc",
               "tweedie_mkt": "rich_tw_mkt", "calendar_vw": "rich_cal_vw",
               "policy_pt_vw": "rich_pol_pt_vw", "calendar_vw_ent": "rich_cal_vw_ent",
               "calendar_vw_brand": "rich_cal_vw_brand", "calendar_vw_eb": "rich_cal_vw_eb",
               "calendar_vw_bent": "rich_cal_vw_bent", "calendar_vw_lc": "rich_cal_vw_lc",
               "calendar_vw_ebl": "rich_cal_vw_ebl", "tweedie_eb": "rich_tw_eb",
               "calendar_vw_org": "rich_cal_vw_org", "calendar_vw_nev": "rich_cal_vw_nev",
               "calendar_vw_off": "rich_cal_vw_off", "tweedie_all": "rich_tw_all",
               "calendar_vw_all": "rich_cal_vw_all", "calendar_all": "rich_cal_all", "catboost_all": "rich_cb_all",
               "catboost_vw_all": "rich_cb_vw_all",
               "catboost_tw_all": "rich_cb_tw_all",
               "lightgbm_all": "rich_lgb_all",
               "catboost_yoy_all": "rich_cb_yoy",
               "catboost_rate_all": "rich_cb_rate",
               "calendar_vw_yoy": "rich_cvw_yoy",
               "catboost_yoy_lcg": "rich_cb_yoy_lcg",
               "catboost_rel_all": "rich_cb_rel",
               "catboost_yoy_rw": "rich_cb_yoy_rw",
               "catboost_yoy_xm": "rich_cb_yoy_xm",
               "catboost_lcg_all": "rich_cb_all_lcg",
               "catboost_yoy_lcg2": "rich_cb_yoy_lcg2",
               "catboost_yoy_seg": "rich_cb_yoy_seg",
               "catboost_yoy_lcg_s43": "rich_cb_yoy_lcg_s43",
               "catboost_yoy_lcg_s44": "rich_cb_yoy_lcg_s44",
               "catboost_yoy_lcg_s45": "rich_cb_yoy_lcg_s45", "catboost_yoy_lcg_s46": "rich_cb_yoy_lcg_s46",
               "catboost_yoy_lcg_big": "rich_cb_yoy_lcg_big",
               "catboost_rel_all_s43": "rich_cb_rel_s43",
               "catboost_rel_all_s44": "rich_cb_rel_s44",
               "lightgbm_yoy": "rich_lgb_yoy",
               "catboost_yoy_lcg3": "rich_cb_yoy_lcgp",
               "catboost_yoy_sup": "rich_cb_yoy_sup",
               "catboost_covid": "rich_cb_covid", "catboost_covidw": "rich_cb_covidw",
               "catboost_huber": "rich_cb_huber", "catboost_subsidy": "rich_cb_subsidy",
               "catboost_yoy_lcgw": "rich_cb_yoy_lcgw", "catboost_yoyw_all": "rich_cb_yoyw", "catboost_relw_all": "rich_cb_relw",
               "catboost_trend": "rich_cb_trend"}
EXTERNAL_CORES = {"calendar_l2": ("calendar_exposure_v1", "calendar_core"), "workday_l2": ("workdays_v1", "workday_core"),
                  "policy_l2": ("policy_events_v1", "policy_core"),
                  "policy_pt_l2": ("powertrain_v1", "policy_pt_core"),
                  "tweedie_cal": ("tweedie_v1", "tweedie_core"),
                  "pol_pt_tw": ("tweedie_pol_pt_v1", "pol_pt_tw_core"),
                  "tweedie_lc": ("china_lifecycle_v1", "lc_core"),
                  "tweedie_mkt": ("market_momentum_v1", "mkt_core"),
                  "calendar_vw": ("rich_vw_v1", "richvw_core"),
                  "policy_pt_vw": ("rich_pol_pt_vw_v1", "ppvw_core"),
                  "calendar_vw_ent": ("entrant_v1", "ent_core"),
                  "calendar_vw_brand": ("brand_momentum_v1", "brand_core"),
                  "calendar_vw_eb": ("rich_eb_v1", "eb_core"),
                  "calendar_vw_bent": ("rich_bent_v1", "bent_core"),
                  "calendar_vw_lc": ("rich_lcvw_v1", "lcvw_core"),
                  "calendar_vw_ebl": ("rich_ebl_v1", "ebl_core"),
                  "tweedie_eb": ("rich_tweb_v1", "tweb_core"),
                  "calendar_vw_org": ("origin_v1", "org_core"),
                  "calendar_vw_nev": ("nev_v1", "nev_core"),
                  "calendar_vw_off": ("official_totals_v1", "off_core"),
                  "tweedie_all": ("rich_all_v1", "twall_core"),
                  "calendar_vw_all": ("rich_all_v1", "cvwall_core"),
                  "calendar_all": ("rich_all_v1", "calall_core"),
                  "catboost_all": ("catboost_v1", "cball_core"),
                  "catboost_vw_all": ("catboost_v1", "cbvwall_core"),
                  "catboost_tw_all": ("catboost_v1", "cbtwall_core"),
                  "lightgbm_all": ("lightgbm_v1", "lgball_core"),
                  "catboost_yoy_all": ("catboost_v1", "cbyoyall_core"),
                  "catboost_rate_all": ("catboost_v1", "cbrate_core"),
                  "calendar_vw_yoy": ("rich_all_v1", "cvyoy_core"),
                  "catboost_yoy_lcg": ("catboost_v1", "cblcg_core"),
                  "catboost_rel_all": ("catboost_v1", "cbrelall_core"),
                  "catboost_yoy_rw": ("catboost_v1", "cbrw_core"),
                  "catboost_yoy_xm": ("catboost_v1", "cbxm_core"),
                  "catboost_lcg_all": ("catboost_v1", "cblcgall_core"),
                  "catboost_yoy_lcg2": ("catboost_v1", "cblcg2_core"),
                  "catboost_yoy_seg": ("catboost_v1", "cbseg_core"),
                  "catboost_yoy_lcg_s43": ("catboost_v1", "cblcg_s43_core"),
                  "catboost_yoy_lcg_s44": ("catboost_v1", "cblcg_s44_core"),
                  "catboost_yoy_lcg_s45": ("catboost_v1", "cblcg_s45_core"),
                  "catboost_yoy_lcg_s46": ("catboost_v1", "cblcg_s46_core"),
                  "catboost_yoy_lcg_big": ("catboost_v1", "cbbig_core"),
                  "catboost_rel_all_s43": ("catboost_v1", "cbrel_s43_core"),
                  "catboost_rel_all_s44": ("catboost_v1", "cbrel_s44_core"),
                  "lightgbm_yoy": ("lightgbm_v1", "lgbyoy_core"),
                  "catboost_yoy_lcg3": ("catboost_v1", "cblcg3_core"),
                  "catboost_yoy_sup": ("catboost_v1", "cbsup_core"),
                  "catboost_covid": ("catboost_v1", "cbcovid_core"),
                  "catboost_covidw": ("catboost_v1", "cbcovidw_core"),
                  "catboost_huber": ("catboost_v1", "cbhuber_core"),
                  "catboost_subsidy": ("catboost_v1", "cbsubsidy_core"),
                  "catboost_yoy_lcgw": ("catboost_v1", "cblcgw_core"), "catboost_yoyw_all": ("catboost_v1", "cbyoyw_core"),
                  "catboost_relw_all": ("catboost_v1", "cbrelw_core"),
                  "catboost_trend": ("catboost_v1", "cbtrend_core")}
RAW_SETTINGS = {"max_depth": 5, "learning_rate": .1, "n_estimators": 200, "n_jobs": 4, "objective": "reg:squarederror",
                "tree_method": "hist", "enable_categorical": True, "device": "cpu", "random_state": 42,
                "subsample": 1., "colsample_bytree": 1.}
PRED_COLUMNS = KEY + ["Month", "horizon", "prediction"]
STATE_COLUMNS = KEY + ["prediction_status", "reason"]
_EXTERNAL = {}


def external_core(variant):
    """달력 계열 변형의 코어를 그 폴더의 특징 모듈과 함께 파일 경로로 불러온다(한 프로세스에 한 변형)."""
    if variant not in _EXTERNAL:
        folder, name = EXTERNAL_CORES[variant]
        folder = ROOT / "experiments" / folder
        sys.path.insert(0, str(folder))
        spec = importlib.util.spec_from_file_location(name, folder / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _EXTERNAL[variant] = module
    return _EXTERNAL[variant]


def core_for(variant):
    return external_core(variant) if variant in EXTERNAL_CORES else rc


def raw_frame(rows, levels):
    frame = rows[rc.FEATURE_COLUMNS].copy()
    for name in rc.NUMERIC:
        frame[name] = frame[name].astype("int32" if name == "month" else "float64")
    for name in rc.gc.CATEGORICAL_FEATURES:
        frame[name] = pd.Categorical(frame[name], categories=levels[name])
    return frame


def fit_raw(rows, cutoff):
    """rich 학습 행으로 판매 대수를 그대로 맞추는 공동 모델."""
    cutoff = pd.Period(cutoff, freq="M")
    training = rows.loc[rows["Month"].le(cutoff)].sort_values(KEY + ["Month"]).reset_index(drop=True)
    levels = {name: sorted(training[name].unique().tolist()) for name in rc.gc.CATEGORICAL_FEATURES}
    model = XGBRegressor(**RAW_SETTINGS)
    model.fit(raw_frame(training, levels), training["Sales"].to_numpy(float))
    meta = {"levels": levels, "seen_series": training[KEY].drop_duplicates().values.tolist(), "train_end": str(cutoff),
            "training_rows": int(len(training)), "target": "sales", "settings": RAW_SETTINGS}
    return model, meta


def forecast_raw(model, meta, history, origin, horizon):
    """rc.forecast와 같은 흐름(최근 12개월 완전·학습에 나온 차종, 24개월 창을 자기 예측으로 잇기)에서 역변환만 없다."""
    origin = pd.Period(origin, freq="M")
    observed = rc.observations(history, origin, reject_future=True)
    seen = {tuple(key) for key in meta["seen_series"]}
    required = pd.period_range(origin - 11, origin, freq="M")
    extended = pd.period_range(origin - 23, origin, freq="M")
    keys, buffers, ages, states = [], [], [], []
    for key, group in observed.groupby(KEY, sort=True):
        monthly = group.set_index("Month")["Sales"]
        status = ("INSUFFICIENT_HISTORY" if monthly.index.min() > required[0]
                  else "INCOMPLETE_HISTORY" if monthly.reindex(required).isna().any() else "OK")
        if status == "OK" and key not in seen:
            status = "UNSEEN_SERIES"
        states.append({**dict(zip(KEY, key)), "prediction_status": status, "reason": "" if status == "OK" else status})
        if status == "OK":
            keys.append(key)
            buffers.append(monthly.reindex(extended).to_numpy(float))
            ages.append(origin.ordinal - monthly.index.min().ordinal + 2)
    rows = []
    if keys:
        buffers = np.asarray(buffers, float)
        for step in range(horizon):
            target = origin + step + 1
            batch = rc.make_features(buffers, keys, [target] * len(keys), [age + step for age in ages])
            prediction = np.maximum(0.0, np.asarray(model.predict(raw_frame(batch, meta["levels"])), float))
            if not np.isfinite(prediction).all():
                raise ValueError(f"PREDICTION_ERROR at {target}")
            rows += [{**dict(zip(KEY, key)), "Month": target, "horizon": step + 1, "prediction": value}
                     for key, value in zip(keys, prediction)]
            buffers[:, :-1] = buffers[:, 1:]
            buffers[:, -1] = prediction
    return pd.DataFrame(rows, columns=PRED_COLUMNS), pd.DataFrame(states, columns=STATE_COLUMNS)


def save_raw(model, meta, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(path)
    path.with_name("meta.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")


def load_raw(path):
    model = XGBRegressor()
    model.load_model(path)
    return model, json.loads(path.with_name("meta.json").read_text(encoding="utf-8"))


def model_file(variant):
    """저장 모델 파일 이름: 코어가 MODEL_FILE을 정하면 그것(CatBoost = model.cbm), 아니면 XGBoost UBJSON."""
    if variant == "rich_raw":
        return "model.ubj"
    return getattr(core_for(variant), "MODEL_FILE", "model.ubj")


# (cs) --per-market: 시장마다 그 시장 자료만으로 학습·예측하는 변형과 그 모델 이름
PER_MARKET_NAMES = {"catboost_all": "rich_cb_pm_all", "tweedie_all": "rich_tw_pm_all",
                    "catboost_yoy_all": "rich_cb_yoy_pm",  # (cs)·(cu)·(df)
                    "catboost_covidw": "rich_cb_pm_covidw", "catboost_yoyw_all": "rich_cb_yoyw_pm"}  # (hh4)


def forecast_per_market(data, origin, horizon, model_dir, variant):
    """시장마다 forecast_origin을 그 시장 자료로 따로 부르고(모델은 model_dir/<시장>/<기준월>/) 이름을 바꿔 합친다."""
    parts_p, parts_s = [], []
    for country in sorted(data["Country"].unique()):
        market = data.loc[data["Country"].eq(country)]
        p, s = forecast_origin(market, origin, horizon, None if model_dir is None else Path(model_dir) / country, variant)
        parts_p.append(p.assign(model=PER_MARKET_NAMES[variant]))
        parts_s.append(s.assign(model=PER_MARKET_NAMES[variant]))
    return pd.concat(parts_p, ignore_index=True), pd.concat(parts_s, ignore_index=True)


def forecast_origin(data, origin, horizon, model_dir=None, variant="rich_log_l2"):
    origin = pd.Period(origin, freq="M")
    history = data.loc[data["Month"].le(origin)].copy()
    module = core_for(variant)
    rows, _ = module.build_training_data(history, origin)
    path = guarded(model_dir) / str(origin) / model_file(variant) if model_dir is not None else None
    if variant == "rich_raw":
        model, meta = fit_raw(rows, origin)
        if path is not None:
            save_raw(model, meta, path)
        p, s = forecast_raw(model, meta, history, origin, horizon)
    else:
        model, schema = module.fit(rows, variant, origin)
        if path is not None:
            module.save(model, schema, path)
        p, s = module.forecast(model, history, origin, horizon, schema)
    return stack_outputs([p], [s], MODEL_NAMES[variant], origin)


def check_reload(data, model_dir, origin, horizon, expected, variant="rich_log_l2"):
    origin = pd.Period(origin, freq="M")
    path = guarded(model_dir) / str(origin) / model_file(variant)
    history = data.loc[data["Month"].le(origin)]
    if variant == "rich_raw":
        model, meta = load_raw(path)
        again, _ = forecast_raw(model, meta, history, origin, horizon)
    else:
        module = core_for(variant)
        model, schema = module.load(path)
        again, _ = module.forecast(model, history, origin, horizon, schema)
    saved = expected.loc[expected["origin"].eq(str(origin))]
    order = KEY + ["Month"]
    np.testing.assert_allclose(again.sort_values(order)["prediction"].to_numpy(float),
                               saved.sort_values(order)["prediction"].to_numpy(float), rtol=1e-12, atol=1e-9)


def run(csv_path=CSV_PATH, output_base=HERE / "reports", origins=ORIGINS, check=True, variant="rich_log_l2",
        per_market=False):
    if per_market and variant not in PER_MARKET_NAMES:
        raise ValueError(f"--per-market은 {sorted(PER_MARKET_NAMES)} 변형에만 씁니다.")
    input_hash = file_sha256(csv_path)
    data = read_sales_csv(guarded(csv_path))
    name = PER_MARKET_NAMES[variant] if per_market else MODEL_NAMES[variant]
    run_dir = new_run_dir(output_base, name)
    jobs = [(o, HORIZON) for o in origins] + ([(CHECK_ORIGIN, CHECK_HORIZON)] if check else [])
    frames_p, frames_s = [], []
    for origin, horizon in jobs:
        if per_market:
            p, s = forecast_per_market(data, origin, horizon, run_dir / "models", variant)
        else:
            p, s = forecast_origin(data, origin, horizon, run_dir / "models", variant)
        frames_p.append(p)
        frames_s.append(s)
        print(f"{name} {origin}: OK {int(s['prediction_status'].eq('OK').sum())} / {len(s)}", flush=True)
    predictions = pd.concat(frames_p, ignore_index=True)
    statuses = pd.concat(frames_s, ignore_index=True)
    if per_market:  # 첫 시장의 첫 기준월 모델을 다시 불러 그 시장 예측과 비교
        country = sorted(data["Country"].unique())[0]
        check_reload(data.loc[data["Country"].eq(country)], run_dir / "models" / country, jobs[0][0], jobs[0][1],
                     predictions.loc[predictions["Country"].eq(country)], variant)
    else:
        check_reload(data, run_dir / "models", jobs[0][0], jobs[0][1], predictions, variant)
    require_unchanged(csv_path, input_hash)
    write_csv(run_dir / "predictions.csv", predictions)
    write_csv(run_dir / "statuses.csv", statuses)
    write_json(run_dir / "run_summary.json", {
        "model": name, "variant": variant, "per_market": per_market, "input_sha256": input_hash, "origins": list(origins),
        "check_origin": CHECK_ORIGIN if check else None,
        "status_counts": statuses.groupby(["origin", "prediction_status"]).size().rename("n").reset_index().to_dict("records"),
        "reload_check": "PASS"})
    return run_dir


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default=str(CSV_PATH))
    parser.add_argument("--output-base", default=str(HERE / "reports"))
    parser.add_argument("--origins", nargs="*", default=ORIGINS)
    parser.add_argument("--no-check", action="store_true")
    parser.add_argument("--variant", default="rich_log_l2", choices=sorted(MODEL_NAMES))
    parser.add_argument("--per-market", action="store_true", help="(cs) 시장마다 따로 학습·예측(catboost_all → rich_cb_pm_all)")
    args = parser.parse_args(argv)
    print(run(args.csv, args.output_base, args.origins, not args.no_check, args.variant, args.per_market))


if __name__ == "__main__":
    main()
