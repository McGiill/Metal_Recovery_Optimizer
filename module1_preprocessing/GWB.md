# GWB — P1·P2 실행 및 분석 안내

> 작성·관리: GWB · 기준일: 2026-10-07
> 경로는 `Metal_Recovery_Optimizer` 루트 기준이다.

## 실행 방법

1. 원본 엑셀을 `data/raw/영풍그룹_금속회수율최적화_RawMaterialData_학생배포용.xlsx`에 넣는다.
2. `module1_preprocessing/notebooks/01_p1_preprocessing_flow.ipynb`의 마지막 셀을 `SAVE_RESULTS=True`로 설정한다. 기본값은 `False`이다.
3. P1 노트북에서 **Restart Kernel and Run All**을 실행해 정제본을 저장한다.
4. `module1_preprocessing/notebooks/02_p2_feature_analysis.ipynb`에서 **Restart Kernel and Run All**을 실행한다. P2의 저장 기본값은 `True`이다.

노트북 커널에서 사용할 Python 환경에 의존성을 설치한다. 아래 명령은 프로젝트 루트에서 실행한다.

```bash
python -m pip install -r requirements.txt
```

노트북은 프로젝트 루트 또는 `module1_preprocessing/notebooks`에서 실행한다.
경로는 프로젝트 루트를 기준으로 계산한다. 원본·결과 CSV와 그림은 Git 제외 대상이다.

## P1 — 데이터 정제

`cleaning.py`와 `validate.py`를 사용해 다음 순서로 처리한다.

1. 원본 품질 진단 및 컬럼명 영문 통일
2. 완전 동일 중복 제거
3. 날짜 6종 형식 변환 및 등급 A/B/C 정규화
4. Ni 함량 999, 황산 15000, 온도 -5, 시간 0을 결측으로 변환
5. 등급별 중앙값으로 수치 결측 대체
6. 기본 16열 기준 `clean()` 결과와 일치 검증
7. 데이터사전 범위 위반 확인 및 시각화

원본은 424행 × 16열이며 정제본은 420행 × 16열이다.
수치 입력·타깃 결측은 0개이고 날짜 결측은 21행이다.
센티널 이외의 범위 밖 경계값은 유지하며 범위 위반은 54셀이다.
9절에서는 결측 위치·수와 `ni_pct`, `time_hr`, `temp_c`의 분포를 확인한다.

```text
data/processed/p1_flow_notebook/
  p1_proccessed.csv    기본 16열 정제 데이터
  range_summary.csv   정상 범위 위반 요약
```

## P2 — EDA와 파생변수 효과 검증

P2는 `p1_proccessed.csv`의 처리값을 그대로 사용한다.
흐름은 EDA → 파생변수 생성 → 모델별 효과 검증 → 결과 저장이다.
EDA에는 기본 통계, 등급별 평균, 회수율 분포, 공정 조건 구간별 추세,
원본 입력과 회수율의 상관행렬을 포함한다.
공정 추세는 입력값의 5개 분위수 구간별 평균 회수율이다.
상관과 구간 평균을 인과적 공정 효과로 해석하지 않는다.

### 파생변수의 계산식과 근거

| 후보 | 계산식 | 출처·근거 |
|---|---|---|
| `acid_per_feed` | 황산 L / 원료 kg | `features.py` 설계 후보 |
| `impurity_total` | Cu + Fe + Al 함량 | `features.py` 설계 후보 |
| `temp_x_time` | 온도 × 시간 | `features.py` 설계 후보 |
| `arrhenius` | exp(-3000 / 절대온도) | `features.py` 가정 Ea/R=3000 K |
| `ni_co_ratio` | Ni / Co 함량 | `features.py` 설계 후보 |
| `ni_co_total` | Ni + Co 함량 | GWB 가설: 대상 금속 총함량 |
| `impurity_to_target` | 불순물 합 / 대상 함량 | GWB 가설: 불순물 부담 |
| `acid_per_target_kg` | 황산 L / [원료 kg × 대상 함량/100] | GWB 가설: 금속 질량당 황산량 |
| `acid_temp_interaction` | 황산/원료 × 온도 | GWB 가설: 공정 상호작용 |
| `acid_time_interaction` | 황산/원료 × 시간 | GWB 가설: 공정 상호작용 |
| `thermal_exposure` | Arrhenius 항 × 시간 | GWB 가설: 온도 반응 표현과 시간 |

계산식은 `features.py`의 `add_features()`에 구현한다. 원본 열과 인덱스를 유지한다.
함수의 기본 호출은 기존 후보 5개를 생성하며, P2 노트북은 `feature_cols=ALL_DERIVED_COLS`로 11개 모두 생성한다.
분모 또는 절대온도가 0 이하이면 해당 파생값을 결측으로 처리한다.
Arrhenius 가정값은 측정한 반응속도나 추정한 활성화에너지가 아니다.
산 농도·전체 액량이 없어 황산량 비율을 몰비나 실제 액고비로 해석하지 않는다.
GWB 후보는 아래 연구의 공정 변수 선택을 참고한 가설이며 계산식의 효과는 데이터로 검증한다.

- [황산 침출 연구: 온도·시간·산 농도·펄프 밀도](https://www.sciencedirect.com/science/article/pii/S1385894715009110)
- [농황산 침출 연구: 온도·시간·고액비](https://www.sciencedirect.com/science/article/pii/S0925838821001821)

### 평가 설계

날짜가 있는 399행 중 개발 구간은 346행, 후기 참고 구간은 53행이다.
개발 구간에서 고유 관측일 기준 4-fold expanding-window CV를 사용한다.
모델은 선형회귀, Ridge, RandomForest, ExtraTrees, LightGBM, XGBoost이다.
원본 입력 11개에 기준선·단독 추가·전체 추가·하나씩 제외·후보 묶음의
총 26구성을 비교하며 624개 CV 점수를 생성한다.
선택 기준은 Ni·Co 평균 RMSE이다. RMSE·MAE 단위는 회수율 퍼센트포인트다.

P1 처리값을 그대로 사용하고 선형 모델의 스케일링은 학습 구간에서 계산한다.
P1이 전체 데이터의 중앙값을 사용했으므로 평가에는 그 정보가 반영되어 있다.
후기 구간은 참고 평가이며 독립적인 최종 성능 검증으로 보고하지 않는다.
고정 모델 설정·작은 표본·다수 후보 선택에 따른 한계를 고려한다.

### 저장된 결과 요약

아래 수치는 기준일 현재 `data/processed/p2_analysis/selected.csv`에 저장된 결과다.
각 모델에서 평균 RMSE가 가장 낮은 구성을 선택한 값이며, 모든 모델이 같은 파생변수 구성을 사용한 것은 아니다.
입력 데이터·모델 설정·판정 기준을 바꾸고 재실행하면 결과도 갱신되므로 최신 값은 노트북과 CSV에서 확인한다.

| 모델 | 선택 구성 | 평균 RMSE |
|---|---|---:|
| 선형회귀 | `add:arrhenius` | 2.4655 |
| Ridge | `add:arrhenius` | 2.4686 |
| RandomForest | `new_only` | 2.9974 |
| ExtraTrees | `without:impurity_total` | 2.8558 |
| LightGBM | `new_only` | 2.7726 |
| XGBoost | `without:impurity_total` | 2.8785 |

`add:`는 해당 후보 단독 추가, `new_only`는 추가 후보 6개 사용,
`without:`는 후보 11개 중 해당 후보만 제외한다. 모든 구성은 원본 입력 11개를 포함한다.

상세 구성·기준선 대비 개선·기간별 점수는 노트북과 CSV에서 확인한다.

### 피처별 판정 기준

노트북 4.2절은 각 후보를 원본 입력에 하나씩 추가한 CV 결과를 6모델에 걸쳐 집계한다.

- 개선 모델 수: Ni·Co 평균 RMSE가 기준선보다 감소한 모델 수
- 기준 충족 모델: 평균 RMSE 1% 이상 감소, 4fold 중 3개 이상 개선, Ni·Co 각각의 평균 RMSE 악화 없음
- 우선 검토: 기준 충족 모델 4개 이상
- 추가 확인: 개선 모델은 4개 이상이지만 기준 충족 모델 4개 미만
- 모델별 검토: 일부 모델에서만 개선
- 개선 근거 약함: 개선된 모델 없음

기준일 현재 저장된 `feature_verdicts.csv`에서 `acid_time_interaction`과 `acid_temp_interaction`이 우선 검토 대상이다.
두 후보 모두 RandomForest·ExtraTrees·LightGBM·XGBoost에서 기준을 만족한다.
Arrhenius 항은 선형회귀·Ridge에 적합한 모델별 검토 후보이다.

기준값은 셀의 상수로 조정할 수 있다. 실용적인 선별 기준이며 통계적 유의성을 뜻하지 않는다.
동일 데이터로 학습한 모델들은 독립적인 증거가 아니며,
단독 추가 판정과 조합 내 기여도도 구분해야 한다.
후기 점수는 이 판정에 사용하지 않는다.
`feature_verdicts.csv`, `feature_verdict_model_details.csv`, `feature_verdict_criteria.json`에 결과와 기준을 저장한다.

### 산출물

기본값 `SAVE_RESULTS=True`로 `data/processed/p2_analysis/`에 저장한다.
후보 정제본, EDA 통계, 원본·후보 상관계수, CV 요약·상세·fold,
선택 구성, 후기 참고 점수와 실험 메타데이터를 포함한다.
메타데이터에는 입력 구성, 모델 설정, 패키지 버전을 기록한다.
상관관계 그림은 노트북과 `docs/figures/gwb_p2_23_*.png`에서 확인한다.

## 파일 구성

아래 경로도 프로젝트 루트 기준이다.

- `module1_preprocessing/notebooks/01_p1_preprocessing_flow.ipynb`: P1 정제 흐름
- `module1_preprocessing/notebooks/02_p2_feature_analysis.ipynb`: P2 분석 전체 흐름
- `module1_preprocessing/cleaning.py`: P1 정제 함수
- `module1_preprocessing/validate.py`: P1 품질 진단과 시각화 함수
- `module1_preprocessing/features.py`: P1 정제 CSV 로딩·스키마 확인, 파생변수 생성, 모델·피처 구성 비교
- `requirements.txt`: 실행 의존성
