"""가짜 해석 모델(MockAnalyticPredictor) 상수를 원본 데이터 수준에 맞춘다.

소유자: P4  (이 파일은 P4만 수정한다)

목적은 정확도가 아니라 "실제 문제와 비슷한 난이도"다. 결과는 config.yaml 의
predictor.mock 에 손으로 옮긴다 (자동 덮어쓰기 하지 않음).

    python -m module3_optimization.analysis.calibrate_mock

기록 (2026-10-11, 결측 없는 275행):
  처음 시도한 곱셈형(아레니우스 × 포화) 모델은 Ni 최대치가 91.7% 로 수렴하고
  RMSE 4.3%p 로 선형 기준선(2.1%p)보다 나빴다. 선형 잔차가 변수마다 가운데 +, 양 끝 − 인
  오목한 패턴이라 가산형 포화 모델로 바꿨고 RMSE Ni 1.62 / Co 1.84 로 개선됐다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from module3_optimization.analysis.load_raw import load_minimal
from module3_optimization.predictors import MockAnalyticPredictor

NEED = ["acid_l", "feed_kg", "temp_c", "time_hr", "cu_pct", "fe_pct", "al_pct"]
TARGETS = ["ni_recovery_pct", "co_recovery_pct"]
COEFS = ["b0", "acid", "temp", "time", "impurity"]

# θ = [acid_scale, temp_scale, time_scale, ni(b0, acid, temp, time, impurity), co(...)]
X0 = [6, 30, 4] + [70, 10, 10, 8, 0.9] * 2
LOWER = [0.5, 1, 0.3] + [0, 0, 0, 0, 0] * 2
UPPER = [100, 500, 50] + [120, 80, 80, 80, 5] * 2


def _unpack(theta) -> dict:
    return {
        "acid_scale": theta[0],
        "temp_scale": theta[1],
        "time_scale": theta[2],
        "ni": dict(zip(COEFS, theta[3:8])),
        "co": dict(zip(COEFS, theta[8:13])),
    }


def fit(df: pd.DataFrame):
    data = df.dropna(subset=NEED + TARGETS)
    y = data[TARGETS].to_numpy()

    def residuals(theta):
        pred = MockAnalyticPredictor(**_unpack(theta), cap=np.inf).predict(data)
        return (pred[["ni_pred", "co_pred"]].to_numpy() - y).ravel()

    result = least_squares(residuals, X0, bounds=(LOWER, UPPER))
    resid = residuals(result.x).reshape(-1, 2)
    params = _unpack(result.x)
    rmse = np.sqrt((resid**2).mean(axis=0))
    params["std"] = {"ni": rmse[0], "co": rmse[1]}
    return params, resid, data


def main():
    params, resid, data = fit(load_minimal())
    print(f"학습 행 수: {len(data)}")
    print(f"RMSE  Ni {params['std']['ni']:.2f}  Co {params['std']['co']:.2f}  (%p)")
    print("\n# config.yaml predictor.mock 에 옮길 값")
    for key in ("acid_scale", "temp_scale", "time_scale"):
        print(f"{key}: {params[key]:.4g}")
    for metal in ("ni", "co"):
        body = ", ".join(f"{k}: {v:.4g}" for k, v in params[metal].items())
        print(f"{metal}: {{ {body} }}")
    print(f"std: {{ ni: {params['std']['ni']:.3g}, co: {params['std']['co']:.3g} }}")

    print("\n# Ni 잔차 구간 평균 (5분위) — 0 근처면 구조가 맞다")
    res = pd.Series(resid[:, 0], index=data.index)
    for col in ("acid_per_feed", "temp_c", "time_hr"):
        means = res.groupby(pd.qcut(data[col], 5), observed=True).mean().round(2)
        print(f"  {col}: {means.tolist()}")


if __name__ == "__main__":
    main()
