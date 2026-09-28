# 영풍그룹: 금속 회수율 최적화 (34_DT)

폐배터리 블랙파우더에서 니켈·코발트 회수율을 예측하고, 목표 회수율을 달성하는
최적 공정 레시피를 추천하는 시스템.

> **소유자: P5** — 이 파일은 P5만 수정한다. 섹션 기여는 PR로 제출.
> `TODO` 표시는 담당자가 채운다.

---

## 1. 프로젝트 개요

TODO (P5) — 과제 배경, 3개 모듈 구성, 최종 산출물 요약

## 2. 환경 설정

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pre-commit install
```

### 데이터 준비

원본 데이터와 Unity 에셋은 용량 때문에 저장소에 포함하지 않는다.

```bash
# 에셋 패키지 (약 1GB) 다운로드 후 압축 해제
curl -O https://download.codyssey.kr/Youngpoong.zip
unzip Youngpoong.zip

# 원본 데이터를 data/raw/ 로 복사
cp Youngpoong/영풍그룹_금속회수율최적화_RawMaterialData_학생배포용.xlsx data/raw/
```

## 3. 모듈별 실행 방법

| 모듈 | 실행 | 담당 |
|---|---|---|
| Module 1 전처리 | TODO | P1 |
| Module 1 특성공학 | TODO | P2 |
| Module 2 예측 | TODO | P3 |
| Module 3 최적화 | TODO | P4 |
| 추가구현 API | TODO | P5 |

TODO (P5) — 각 모듈의 실행 예시 명령과 출력 샘플

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
- **분석일자 포맷 10종 혼재** + 결측 22건 (`2025-01-02`, `01/08/25`, `20250106`, `06.01.2025`, `Jan 08, 2025` 등)
- **원료등급 표기 12종** (`A`, `A급`, `a`, 공백 패딩 `' A '` × 3등급)
- **센티널 이상치 9건**: `Ni_함량=999`(3) · `황산투입량=15000`(2) · `침출온도=-5`(2) · `침출시간=0`(2)
  - `BP-2025-0231` 은 침출시간 0hr 인데 Ni 회수율 95.56% — 물리적으로 불가능한 조합
- **배치ID 중복 4건** (`0018`, `0166`, `0180`, `0379`) — 전부 완전 동일 행이라 `drop_duplicates()` 로 420행이 된다
- **목표 달성 희소**: Ni≥95 & Co≥90 동시 달성이 420건 중 **6건**. Module 3 최적화는 외삽 위험이 있다

## 5. 저장소 구조

```
data/raw/              원본 데이터 (git 제외)
data/processed/        정제 결과 (git 제외)
data/fixtures/         계약 샘플 — 병렬 개발용 (git 포함)
module1_preprocessing/ 정제 · 특성공학          P1 / P2
module2_prediction/    예측 모델 · XAI          P3
module3_optimization/  최적화 · 파레토          P4
notebooks/             EDA · 모델링 과정        P2
docs/                  계약 · 보고 자료
api/                   추가구현 추론 API        P5
unity/                 Unity 프로젝트           P5
artifacts/             학습된 모델 등 (git 제외)
```

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
| P5 | Unity 모델링 + `README.md` | 추론 API + 실시간 데모 |
