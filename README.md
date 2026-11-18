# 영풍그룹: 금속 회수율 최적화 (34_DT)

폐배터리 블랙파우더에서 니켈·코발트 회수율을 예측하고, 목표 회수율을 달성하는
최적 공정 레시피를 추천하는 시스템.

> **소유자: P5** — 이 파일은 P5만 수정한다. 섹션 기여는 PR로 제출.
> `TODO` 표시는 담당자가 채운다.

---

## 1. 프로젝트 개요

TODO (P5) — 과제 배경, 3개 모듈 구성, 최종 산출물 요약

## 2. 빠른 실행 — Docker

클론 직후 바로 된다. 별도 내려받을 것이 없다.

```bash
docker compose up --build
```

| | 주소 |
|---|---|
| 추천 데모 · 3D 디지털 트윈 | <http://localhost:8501> |
| 예측 API 문서 (Swagger) | <http://localhost:8000/docs> |

첫 빌드는 **5분 안팎**이다 — 의존성 설치에 더해 컨테이너 안에서
모듈 1 전처리와 모듈 2 학습(약 52초)을 끝내고 모델을 이미지에 굽기 때문이다.
두 번째부터는 레이어 캐시가 걸려 수십 초다.

멈출 때는 `Ctrl+C`, 정리는 `docker compose down`.

### 3D 화면만 따로 보기

디지털 트윈(P5)은 HTML 파일 하나라 파이썬 없이 열린다.
`docs/modeling/index.html` 을 Chrome 이나 Edge 로 연다 (인터넷 필요 — Three.js CDN).

주소 뒤에 옵션을 붙이면 원하는 장면을 바로 띄운다.

```text
index.html?cam=reactor&seek=14     침출 중인 반응조
index.html?cam=recovery&seek=27    결정이 쌓이는 배출 단계
index.html?debug=1                 개발용 슬라이더 (애니메이션만 바뀜, 예측값 아님)
```

> 단독으로 열면 데모 입력(BP-2025-0074 실측을 근거로 한 가정값)을 보여 준다.
> 실제 추천 결과는 Streamlit 데모에서 목표 달성 레시피가 나왔을 때 화면에 들어간다.
> GitHub Pages(Settings → Pages → `main` / `docs`)를 켜면 주소 하나로 열린다.

## 3. 직접 설치해서 쓰기

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt  # 컨테이너는 requirements-runtime.txt(부분집합)를 쓴다
pre-commit install
```

### 데이터 준비

**원본 데이터(59KB)는 저장소에 들어 있다.** 명세 `#2` 의 레포지토리 구조가
`data/` 에 원본 데이터를 두라고 요구하고, 개인정보가 없어서 그대로 커밋했다.
따로 받을 필요가 없다.

Unity 에셋 패키지(약 1GB)는 용량 때문에 제외한다. 3D 씬은 이 에셋 없이
절차적으로 만들어지므로 데모를 돌리는 데 필요하지 않다.

```bash
curl -O https://download.codyssey.kr/Youngpoong.zip   # 에셋이 필요할 때만
unzip Youngpoong.zip
```

### 모듈별 실행 방법

| 단계 | 명령 | 산출물 | 담당 |
|---|---|---|---|
| Module 1 전처리 | `python -m module1_preprocessing.cleaning` | `data/processed/batches.csv` | P1 |
| Module 1 특성공학 | `notebooks/02_p2_feature_analysis.ipynb` 실행 (파생 변수 비교) · `add_features()` 는 모듈 2 학습 때 자동 호출 | `data/processed/p2_analysis/` | P2 |
| Module 2 학습·평가 | `python -m module2_prediction.train` | `artifacts/recovery_model.joblib` | P3 |
| Module 3 추천 (1건) | `python -m module3_optimization.recommend --batch-id BP-2025-0033` | 레시피 카드 | P4 |
| Module 3 추천 (전체) | `python -m module3_optimization.recommend --all --out artifacts/recipes.csv` | CSV | P4 |
| 데모 화면 | `streamlit run app/streamlit_app.py` | <http://localhost:8501> | P5 |
| 예측 API | `uvicorn api.main:app --port 8000` | <http://localhost:8000/docs> | P5 |

모듈 3 는 모듈 2 의 모델을 읽으므로 **학습을 먼저 돌려야 한다.**

실행 예시:

```console
$ python -m module3_optimization.recommend --batch-id BP-2025-0033
...
$ python -m module3_optimization.recommend --all --out artifacts/recipes.csv
420건 저장 → artifacts/recipes.csv  {'infeasible': 381, 'ok': 39}  (입력 결측으로 건너뜀 0건)
```

420건 중 39건만 목표를 달성한다. 데이터의 성질이다 — 자세한 내용은
[`REPORT.md`](REPORT.md) 5.5 현장 적용 가능성 참고.

## 4. 데이터셋 설명

`영풍그룹_금속회수율최적화_RawMaterialData_학생배포용.xlsx`
— 시트 2개: `원료분석_원본데이터` (424행), `데이터사전`

| 컬럼 | 설명 | 단위 | 정상 범위 |
|---|---|---|---|
| 배치ID | 원료 배치(LOT) 고유번호 | - | `BP-2025-XXXX` |
| 분석일자 | 랩실 성분 분석 일자 | 날짜 | 2025년 |
| 원료등급 | 품질 등급 (A=고니켈·저불순물, C=저니켈·고불순물) | - | A / B / C |
| Ni_함량_pct | 니켈 함량 (회수 목표) | wt% | 8 ~ 30 |
| Co_함량_pct | 코발트 함량 (회수 목표) | wt% | 2 ~ 15 |
| Li_함량_pct | 리튬 함량 | wt% | 2 ~ 6 |
| Mn_함량_pct | 망간 함량 | wt% | 5 ~ 16 |
| Cu_불순물_pct | 구리 불순물 | wt% | 0.3 ~ 6 |
| Fe_불순물_pct | 철 불순물 | wt% | 0.2 ~ 6 |
| Al_불순물_pct | 알루미늄 불순물 | wt% | 0.5 ~ 7 |
| 원료투입량_kg | 1회 배치 원료 질량 | kg | 80 ~ 200 |
| 황산투입량_L | 침출용 황산 투입량 | L | 400 ~ 1500 |
| 침출온도_C | 침출 반응 온도 | ℃ | 40 ~ 95 |
| 침출시간_hr | 침출 반응 시간 | hr | 1 ~ 8 |
| Ni_회수율_pct | 니켈 회수율 (타깃) | % | - |
| Co_회수율_pct | 코발트 회수율 (타깃) | % | - |

### 사용 시 유의사항

- **결측**: 307셀. 424행 중 **224행이 결측 보유** (완전한 행 200개). 타깃 2개는 결측 없음
- **분석일자 포맷 6종 혼재** + 결측 22건 — `2025-01-02` · `2025/01/02` · `03.01.2025`(DD.MM.YYYY) · `20250103` · `Jan 03, 2025` · `01/03/25`(MM/DD/YY)
- **원료등급 표기 12종** (`A`, `A급`, `a`, 공백 패딩 `' A '` × 3등급)
- **센티널 이상치 9건**: `Ni_함량=999`(3) · `황산투입량=15000`(2) · `침출온도=-5`(2) · `침출시간=0`(2)
  - `BP-2025-0231` 은 침출시간 0hr 인데 Ni 회수율 95.56% — 물리적으로 불가능한 조합
- **배치ID 중복 4건** (`0018`, `0166`, `0180`, `0379`) — 전부 완전 동일 행이라 `drop_duplicates()` 로 420행이 된다
- **목표 달성 희소**: Ni≥95 & Co≥90 동시 달성이 420건 중 **6건**. Module 3 최적화는 외삽 위험이 있다

## 5. 저장소 구조

```
data/raw/              원본 .xlsx (git 포함 — 명세 #2)
data/processed/        정제 결과 (git 제외, 모듈 1 이 만든다)
data/fixtures/         계약 샘플 — 병렬 개발용 (git 포함)
module1_preprocessing/ 정제 · 특성공학          P1 / P2
module2_prediction/    예측 모델 · XAI          P3
module3_optimization/  최적화 · 파레토          P4
notebooks/             EDA · 모델링 과정        P2
docs/                  보고 자료 · 그림
docs/modeling/         3D 디지털 트윈 — 명세 #2-6 모델링 구현
app/                   Streamlit 데모 화면      P5
api/                   추론 API (명세 #8)       P5
artifacts/             학습된 모델 등 (git 제외)
Dockerfile             전처리→학습→데모를 한 번에 굽는다
compose.yaml           app(8501) + api(8000)
```

`docs/modeling/index.html` 은 3D 디지털 트윈(P5) 한 파일이다. 단독으로 열면 데모 입력을,
Streamlit 데모 안에서는 모듈 3 추천 결과(목표 달성일 때)를 받아 보여 준다.

## 6. 협업 규칙

- **한 파일은 한 사람만 수정한다.** 각 파일 상단에 소유자가 적혀 있다
- 브랜치: `feat/p1-cleaning` 형식. main 직접 push 금지
- `docs/INTERFACE.md` 는 **동결**. 변경은 주간 회의 승인 후에만
- 커밋 전 `pre-commit` 이 ruff·black 을 적용한다

## 7. 팀 구성

| | 필수 담당 | 추가구현 |
|---|---|---|
| P1 | Module 1 데이터 정제 + 품질 검증·시각화 | — |
| P2 | Module 1 특성공학 + `REPORT.md` | — |
| P3 | Module 2 예측 모델 | XAI (SHAP/LIME) |
| P4 | Module 3 최적화 | 다목적 최적화 + 파레토 |
| P5 | 모델링 구현(명세 `#2-6`) + `README.md` | 추론 API + 실시간 데모 |

> 명세 `#2-6` 은 **"모델링 구현"** 이라고만 쓰여 있고 도구를 지정하지 않는다.
> 영풍 배포 에셋이 `.unitypackage` 라 Unity 로 적어 왔으나 명세에 그런 요구는 없어서,
> 브라우저에서 바로 열리는 Three.js 로 구현했다 (`docs/modeling/`).
