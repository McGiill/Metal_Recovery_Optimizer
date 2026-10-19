# Module 2 — 회수율 예측 (P3)

> 소유자: P3. 기준일 2026-10-19, 모델 버전 `p3-v0.1`. 브랜치 `feat/p3-prediction`(로컬, `feat/p2-EDA_n_feateng` 기준).

## 실행

```bash
# 저장소 루트, data/raw/ 에 원본 xlsx 가 있어야 한다
python -m module1_preprocessing.cleaning        # Module 1 → data/processed/batches.csv
python -m module2_prediction.train              # 실험·선택·평가·저장 (약 35초)
python -m module2_prediction.inference          # predict_batch · 7:3 혼합 예시
python -m pytest module2_prediction/tests -q    # 8개 테스트
```

macOS 에서 LightGBM·XGBoost 는 `brew install libomp` 가 필요하다.

| 산출물 | 위치 |
|---|---|
| 저장 모델 (전체 420행 재학습) | `artifacts/recovery_model.joblib` |
| CV 비교표 · 하위군 오차 · 예측값 · 목표 영역 사례 · 중요도 · 학습곡선 · 요약 | `data/processed/p3_prediction/*.csv`, `summary.json` |
| 그림 | `docs/figures/p3_01~05_*.png` |

## Module 1 과의 연결

- 입력은 Module 1 `clean()` 결과(`batches.csv`, 420행×33열)다. P1 노트북의 `p1_proccessed.csv` 와 16열 값이 같음을 확인했다.
- Module 1 은 **전체 420행**의 등급별 중앙값으로 대치한다. 그대로 쓰면 검증 구간 정보가 학습 대치값에 섞인다.
  `imputed_<col>` 플래그가 대치 직전 결측 위치와 정확히 일치하므로(원본 대조 검증) 291칸을 NaN 으로 되돌리고,
  대치는 학습 Pipeline 안에서 fold 의 학습 데이터로만 다시 한다. 방법(등급별 중앙값)은 Module 1 과 같다.
- 파생 변수는 Module 1 `add_features()` 를 그대로 호출한다.
- 플래그 17개는 평가 하위군 구성에만 쓰고 모델 입력에 넣지 않는다. 배치ID·날짜·등급도 입력에서 제외한다.

## 평가 설계

- **분할:** 배치ID 순서 = 날짜 순서(역전 0건). P2 노트북이 선택에 쓰지 않은 2025-09-16 이후(BP-2025-0366~)를
  시험 구간 55행(공정 완전 49행)으로 두고, 앞 365행에서 확장 윈도 4-fold 전방 CV 로만 선택했다.
  P2 가 이 구간의 참고 점수를 이미 확인했으므로 **독립 외부 검증이 아닌 내부 검증**이다.
- **주 지표:** fold 별 *공정 4변수 완전 관측* 검증 행의 Ni·Co 평균 RMSE (Module 3 는 공정 값을 모두 지정해 호출).
- **공정 결측 정책:** A 등급별 중앙값 대치 · B NaN 유지(boosting 전용) · C 공정 완전 행만 학습, 결측 입력은 예측 거절.
- **채택 규칙:** 더 복잡한 후보(정책·특징·모델)는 평균 RMSE 가 낮고 4 fold 중 3개 이상에서 이길 때만 채택.
  모델은 Ridge → RF → XGBoost → LightGBM 순으로 비교한다.

## 결과

### 1. 공정 결측 정책 (같은 LightGBM, CV RMSE %p)

| 정책 | F0 | F1 (+acid_time) | 전체 검증행 예측 제공 |
|---|---:|---:|---:|
| A 대치 | 2.756 | 2.630 | 100% |
| B NaN 유지 | 2.650 | 2.569 | 100% |
| C 관측 행만 | 2.658 | **2.536** | 80% |
| 참고: Module 1 전역 대치값 그대로 | 2.691 | – | 100% |

B·C 모두 A 보다 낫다. 4개 모델 전부에서 주 평가군 기준 최선은 정책 C 였다(아래 표). C 는 공정 결측 과거 배치(검증 행의 약 20%)를 예측하지 않는다.

### 2. 모델별 최선 설정

| 모델 | 선택 설정 | CV RMSE | 시험 RMSE Ni / Co | 시험 R² Ni / Co |
|---|---|---:|---:|---:|
| Dummy 평균 | – | 6.200 | 6.04 / 7.25 | -0.09 / -0.06 |
| **Ridge** | **C · F4 (+arrhenius)** | **2.161 ± 0.064** | **1.85 / 1.98** | **0.90 / 0.92** |
| RandomForest | C · F3 | 2.840 | 2.22 / 2.57 | 0.85 / 0.87 |
| XGBoost | C · F3 | 2.478 | 2.07 / 2.37 | 0.87 / 0.89 |
| LightGBM | C · F2 | 2.481 | 2.17 / 2.24 | 0.86 / 0.90 |

- 최종 모델은 **Ridge, 정책 C, 원본 11개 + arrhenius**. 정책 C 는 같은 Ridge·F4 의 A 보다 4개 fold 모두에서 낮았다(2.208 → 2.161). 트리 모델 중 어느 것도 Ridge 를 3/4 fold 이상에서 이기지 못했다.
  P2 결과(선형 + arrhenius 최선)와 같은 방향이다. 시험 점수는 선택 후 1회 확인했고, 모델별 시험 점수는 참고용이다.
- Ridge 는 `RidgeCV` 가 학습 fold 안에서 alpha 를 고르므로 별도 튜닝은 하지 않았다.
- 파생 변수: 트리 모델은 `acid_time_interaction`·`acid_temp_interaction`(F3)에서, 선형 모델은 arrhenius(F4)에서만 개선됐다.
- 참고로 Module 1 전역 대치값을 그대로 쓴 같은 설정은 CV 2.161 ± 0.112 였다. 이 비교는 정책 C 의 학습 행 수도 달라지므로 누수 효과의 크기로 해석하지 않는다.

### 3. 하위군 · 목표 영역

| 집합 | 하위군 | n | RMSE Ni / Co |
|---|---|---:|---:|
| 개발 CV | 공정 완전 (주) | 233 | 2.02 / 2.31 |
| 개발 CV | 성분 대치 포함 | 113 (예측 88) | 2.13 / 2.52 |
| 개발 CV | 등급 C | 71 (예측 58) | 2.14 / 2.50 |
| 시험 | 공정 완전 (주) | 49 | 1.85 / 1.98 |
| 시험 | 등급 C | 19 (예측 17) | 2.21 / 2.34 |

- 온도·시간 결측 행(개발 58행, 시험 6행)은 정책 C 에 따라 예측하지 않는다.
- **목표 부근 과소 예측:** Ni≥95 또는 Co≥90 인 20개 배치(시험 구간 1개)에서 예측이 실제보다 낮은 경우가 많다(예: 실제 97.97 → 예측 93.47).
  두 목표 동시 달성 4개 중 예측 가능한 3개는 오차 1.7%p 이내였다. 표본이 적어 목표 달성 보장 근거로 쓰지 않는다.
- **학습곡선:** 고정 검증 56행에서 학습 72행 2.34 → 166행 2.15 → 240행 2.15. 같은 성격의 자료를 늘리는 효과는 작아지고 있다.
  추가 자료는 목표 부근·다양한 공정 조건 관측이 더 필요하다.

### 4. 중요도 (시험 공정 완전 49행, permutation, RMSE 증가 %p)

temp_c 2.54 > acid_l 2.32 > time_hr 1.44 ≈ al_pct 1.43 > cu_pct 1.03 ≈ feed_kg 1.03 > fe_pct 0.50 > 나머지(ni·co·li·mn) ≈ 0.
공정 3변수와 불순물이 핵심이다. `ni_pct` 의 상관(+0.42)은 불순물과 겹쳐 모델에서 별도 기여가 거의 없다. 인과 효과로 해석하지 않는다.

## Module 3 연결

```yaml
predictor:
  type: real
  path: artifacts/recovery_model.joblib
  version: p3-v0.1
```

`feat/p4_optimization` 의 `build_predictor()` → `RecoveryModelAdapter` 로 불러 예측되는 것을 확인했다.

- `RecoveryModel.predict(X) -> ndarray (n, 2)`, 열 순서 `[ni_recovery_pct, co_recovery_pct]`. 입력은 `FEATURE_COLS` 11개, `grade` 는 선택.
- 공정 변수(원료량·황산·온도·시간)가 빠진 행은 **NaN** 을 반환한다(정책 C). 추천 후보는 공정 값을 모두 채워 호출한다.
- `predict_std()` 는 개발 CV 공정 완전군 RMSE(Ni 2.02, Co 2.31)를 행마다 반환한다. **보정된 신뢰구간이 아니다.**
- Ridge 는 선형이라 학습 범위 밖에서 그대로 외삽한다(예: 황산 3000L → Ni 101.6%). 예측값을 자르지 않으므로
  Module 3 의 적용 범위 판정을 반드시 켜고, `inference.predict_batch()` 의 `out_of_domain` 상태를 참고한다.
- 목표 부근 과소 예측 경향 때문에 비용 최소 추천이 보수적으로 나올 수 있다.

## 한계 · 다음 작업

- 시험 55행·목표 동시 달성 6행으로 성능 추정의 분산이 크다. 독립 신규 배치 검증이 필요하다.
- 정책 C 는 공정 결측 과거 배치를 예측하지 않는다. 그 행 예측이 필요하면 B(LightGBM) 모델을 별도로 둔다.
- 남은 후보: Ridge 에 F3 상호작용 + arrhenius 조합, 금속별 다른 설정, conformal 구간, SHAP(추가구현 #8).
- 실측/합성 데이터 여부는 미확정이다.
