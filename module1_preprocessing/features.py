"""Module 1 — 특성 공학과 효과 검증.

소유자: P2
P1 정제 CSV 로딩, 원료·공정 파생변수 생성, 시간순 모델 비교를 제공한다.
기본 후보는 acid_per_feed, impurity_total, temp_x_time, arrhenius,
ni_co_ratio이며 ALL_DERIVED_COLS로 총 11개를 선택할 수 있다.
Arrhenius의 Ea/R=3000 K는 가정값이다.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from module1_preprocessing.cleaning import FEATURE_COLS, TARGET_COLS
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from module1_preprocessing import cleaning as p1

DERIVED_COLS = [
    "acid_per_feed", "impurity_total", "temp_x_time", "arrhenius", "ni_co_ratio"
]

ADDITIONAL_COLS = [
    "ni_co_total", "impurity_to_target", "acid_per_target_kg",
    "acid_temp_interaction", "acid_time_interaction", "thermal_exposure",
]

ALL_DERIVED_COLS = DERIVED_COLS + ADDITIONAL_COLS

EA_OVER_R = 3000.0

def add_features(df: pd.DataFrame, feature_cols=None) -> pd.DataFrame:
    """정제된 프레임에 파생변수를 추가해 반환한다."""
    requested = list(DERIVED_COLS if feature_cols is None else feature_cols)
    unknown = set(requested) - set(ALL_DERIVED_COLS)
    if unknown:
        raise ValueError(f"알 수 없는 파생변수: {sorted(unknown)}")
    required = ["acid_l", "feed_kg", "cu_pct", "fe_pct", "al_pct",
                "temp_c", "time_hr", "ni_pct", "co_pct"]
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"파생변수 생성에 필요한 컬럼 없음: {sorted(missing)}")
    out = df.copy()
    values = df[required].apply(pd.to_numeric, errors="coerce")
    values = values.replace([np.inf, -np.inf], np.nan)
    out["acid_per_feed"] = values["acid_l"] / values["feed_kg"].where(values["feed_kg"] > 0)
    out["impurity_total"] = values[["cu_pct", "fe_pct", "al_pct"]].sum(axis=1, min_count=3)
    out["temp_x_time"] = values["temp_c"] * values["time_hr"]
    kelvin = (values["temp_c"] + 273.15).where(values["temp_c"] + 273.15 > 0)
    out["arrhenius"] = np.exp(-EA_OVER_R / kelvin)
    out["ni_co_ratio"] = values["ni_pct"] / values["co_pct"].where(values["co_pct"] > 0)
    out["ni_co_total"] = values[["ni_pct", "co_pct"]].sum(axis=1, min_count=2)
    target_content = out["ni_co_total"].where(out["ni_co_total"] > 0)
    out["impurity_to_target"] = out["impurity_total"] / target_content
    target_mass = (values["feed_kg"].where(values["feed_kg"] > 0) * target_content / 100)
    out["acid_per_target_kg"] = values["acid_l"] / target_mass
    out["acid_temp_interaction"] = out["acid_per_feed"] * values["temp_c"]
    out["acid_time_interaction"] = out["acid_per_feed"] * values["time_hr"]
    out["thermal_exposure"] = out["arrhenius"] * values["time_hr"]
    out[ALL_DERIVED_COLS] = out[ALL_DERIVED_COLS].replace([np.inf, -np.inf], np.nan)
    # 기본 호출은 설계 후보 5개를 생성한다. feature_cols로 후보를 선택할 수 있다.
    result = df.copy()
    for col in requested:
        result[col] = out[col]
    return result

def load_p1_output(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError('P1 노트북에서 SAVE_RESULTS=True 로 p1_processed.csv를 먼저 저장하세요.')
    frame = pd.read_csv(path, encoding='utf-8-sig')
    required = ['batch_id', 'analysis_date', 'grade'] + FEATURE_COLS + TARGET_COLS
    missing = set(required) - set(frame)
    if missing:
        raise ValueError(f'P1 CSV의 필수 열이 없습니다: {sorted(missing)}')
    # P1 CSV는 원본 16열 + sentinel_/imputed_ 플래그 17열 = 33열이다.
    # 플래그는 대치 이력 추적용이며 모델 입력에는 쓰지 않는다. 존재해도 무시한다.
    frame['analysis_date'] = pd.to_datetime(frame['analysis_date'], errors='raise')
    if frame['batch_id'].isna().any() or not frame['batch_id'].is_unique:
        raise ValueError('P1 CSV의 배치 ID는 비어 있지 않고 고유해야 합니다.')
    if not frame['grade'].isin(['A', 'B', 'C']).all():
        raise ValueError('P1 CSV의 등급을 확인하세요.')
    frame['grade'] = frame['grade'].astype('string').astype('category')
    for col in FEATURE_COLS + TARGET_COLS:
        frame[col] = pd.to_numeric(frame[col], errors='raise')
    if not np.isfinite(frame[FEATURE_COLS + TARGET_COLS].to_numpy(dtype=float)).all():
        raise ValueError('P1 데이터의 수치 결측·무한값을 확인하세요. P2는 추가 대체하지 않습니다.')
    return frame

def prepare_processed(frame):
    eligible = frame['analysis_date'].notna() & frame[TARGET_COLS].notna().all(axis=1)
    return (frame.loc[eligible].sort_values(['analysis_date', 'batch_id']).reset_index(drop=True),
            frame.loc[~eligible].copy())

def make_model(name):
    if name == "LinearRegression":
        return make_pipeline(StandardScaler(), LinearRegression())
    if name == "Ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    if name == "RandomForest":
        return RandomForestRegressor(n_estimators=120, min_samples_leaf=3, random_state=42, n_jobs=-1)
    if name == "ExtraTrees":
        return ExtraTreesRegressor(n_estimators=120, min_samples_leaf=3, random_state=42, n_jobs=2)
    if name == "LightGBM":
        from lightgbm import LGBMRegressor
        return MultiOutputRegressor(LGBMRegressor(
            n_estimators=200, learning_rate=.04, num_leaves=7,
            min_child_samples=10, reg_lambda=1., random_state=42,
            n_jobs=2, verbosity=-1), n_jobs=1)
    if name == "XGBoost":
        from xgboost import XGBRegressor
        return MultiOutputRegressor(XGBRegressor(
            n_estimators=200, learning_rate=.04, max_depth=3,
            min_child_weight=3, reg_lambda=1., objective="reg:squarederror",
            tree_method="hist", random_state=42, n_jobs=2), n_jobs=1)
    raise ValueError(name)

def metrics(y, prediction):
    result = {}
    for index, metal in enumerate(["Ni", "Co"]):
        result[f"RMSE_{metal}"] = float(np.sqrt(mean_squared_error(y.iloc[:, index], prediction[:, index])))
        result[f"MAE_{metal}"] = float(mean_absolute_error(y.iloc[:, index], prediction[:, index]))
        result[f"R2_{metal}"] = float(r2_score(y.iloc[:, index], prediction[:, index]))
    result["mean_RMSE"] = (result["RMSE_Ni"] + result["RMSE_Co"]) / 2
    return result

def variants(derived_cols=None):
    candidates = list(DERIVED_COLS if derived_cols is None else derived_cols)
    result = {"baseline": []}
    result.update({f"add:{col}": [col] for col in candidates})
    result["all"] = candidates.copy()
    result.update({f"without:{col}": [c for c in candidates if c != col] for col in candidates})
    if candidates != DERIVED_COLS:
        result["original_only"] = [c for c in DERIVED_COLS if c in candidates]
        result["new_only"] = [c for c in candidates if c not in DERIVED_COLS]
    return result

def evaluate_features(frame, n_splits=4, model_names=None, derived_cols=None, progress=None):
    """마지막 15% 관측일은 테스트로 격리하고 앞 구간의 시간순 CV로 후보 선택."""
    model_names = ["Ridge", "RandomForest"] if model_names is None else list(model_names)
    derived_cols = list(DERIVED_COLS if derived_cols is None else derived_cols)
    def feature_pair(train, other):
        for data in (train, other):
            if not np.isfinite(data[p1.FEATURE_COLS + p1.TARGET_COLS].to_numpy(dtype=float)).all():
                raise ValueError('P1 정제값을 그대로 쓰는 평가에는 수치 결측·무한값이 없어야 합니다.')
        return add_features(train, feature_cols=derived_cols), add_features(other, feature_cols=derived_cols)
    dates = frame["analysis_date"].drop_duplicates().to_numpy()
    cut = int(len(dates) * .85)
    if cut <= n_splits or cut == len(dates):
        raise ValueError("시간순 평가에 충분한 날짜가 없습니다.")
    test_start = dates[cut]
    dev = frame.loc[frame["analysis_date"] < test_start].copy()
    test = frame.loc[frame["analysis_date"] >= test_start].copy()
    dev_dates = dev["analysis_date"].drop_duplicates().to_numpy()
    configs = variants(derived_cols)
    scores, folds = [], []
    for fold, (train_idx, valid_idx) in enumerate(TimeSeriesSplit(n_splits=n_splits).split(dev_dates), 1):
        train = dev.loc[dev["analysis_date"].isin(dev_dates[train_idx])]
        valid = dev.loc[dev["analysis_date"].isin(dev_dates[valid_idx])]
        assert train["analysis_date"].max() < valid["analysis_date"].min()
        xtrain, xvalid = feature_pair(train, valid)
        folds.append({"fold": fold, "train_rows": len(train), "valid_rows": len(valid),
                      "train_end": train["analysis_date"].max(), "valid_start": valid["analysis_date"].min(),
                      "valid_end": valid["analysis_date"].max()})
        for model_name in model_names:
            for name, extra in configs.items():
                columns = p1.FEATURE_COLS + extra
                model = make_model(model_name)
                model.fit(xtrain[columns], xtrain[p1.TARGET_COLS])
                prediction = model.predict(xvalid[columns])
                scores.append({"model": model_name, "variant": name, "fold": fold,
                               **metrics(xvalid[p1.TARGET_COLS], prediction)})
        if progress is not None:
            progress(f"Fold {fold}/{n_splits} 완료")
    detail = pd.DataFrame(scores)
    summary = detail.groupby(["model", "variant"], sort=False).agg(
        mean_RMSE=("mean_RMSE", "mean"), std_RMSE=("mean_RMSE", "std"),
        RMSE_Ni=("RMSE_Ni", "mean"), RMSE_Co=("RMSE_Co", "mean"),
        MAE_Ni=("MAE_Ni", "mean"), MAE_Co=("MAE_Co", "mean"),
        R2_Ni=("R2_Ni", "mean"), R2_Co=("R2_Co", "mean")).reset_index()
    baseline = summary.loc[summary["variant"].eq("baseline")].set_index("model")["mean_RMSE"]
    summary["improvement_pct"] = 100 * (1 - summary["mean_RMSE"] / summary["model"].map(baseline))
    paired = detail.merge(detail.loc[detail["variant"].eq("baseline"), ["model", "fold", "mean_RMSE"]].rename(columns={"mean_RMSE": "baseline_RMSE"}), on=["model", "fold"])
    paired["delta_RMSE"] = paired["mean_RMSE"] - paired["baseline_RMSE"]
    wins = paired.assign(improved=paired["delta_RMSE"] < 0).groupby(["model", "variant"])["improved"].sum().rename("improved_folds").reset_index()
    summary = summary.merge(wins, on=["model", "variant"], how="left")
    selected = summary.loc[summary.groupby("model")["mean_RMSE"].idxmin()].copy()
    xdev, xtest = feature_pair(dev, test)
    test_scores = []
    for model_name in model_names:
        chosen = selected.loc[selected["model"].eq(model_name), "variant"].iloc[0]
        for name in dict.fromkeys(["baseline", chosen]):
            columns = p1.FEATURE_COLS + configs[name]
            model = make_model(model_name)
            model.fit(xdev[columns], xdev[p1.TARGET_COLS])
            test_scores.append({"model": model_name, "variant": name,
                                **metrics(xtest[p1.TARGET_COLS], model.predict(xtest[columns]))})
    return {"cv_detail": detail, "cv_summary": summary, "folds": pd.DataFrame(folds),
            "selected": selected, "test_scores": pd.DataFrame(test_scores), "paired": paired,
            "development": dev, "test": test, "test_start": pd.Timestamp(test_start)}
