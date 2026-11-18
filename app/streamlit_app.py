"""추천 데모 + 디지털 트윈 — 명세 #8 '실시간 예측 시스템'.

    streamlit run app/streamlit_app.py

module3_optimization.recommend() 를 그대로 호출하므로 트리 앙상블 sigma,
두 개의 적용 범위 판정기, 비율 제약이 전부 실물이다.
3D 화면(docs/modeling/index.html, P5)은 아래에 끼워 넣는다. 목표 달성(ok) 추천이
나오면 Module 3 JSON 을 원료 정보와 묶어 window.__TWIN_INPUT__ 으로 넘긴다.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from module2_prediction import config as C  # noqa: E402
from module3_optimization.config import load_config  # noqa: E402
from module3_optimization.domain import build_domains  # noqa: E402
from module3_optimization.problem import InputError  # noqa: E402
from module3_optimization.recommend import feed_from_row, load_batches, recommend  # noqa: E402

SCENE = ROOT / "docs/modeling/index.html"
EVIDENCE_COLS = ("acid_l", "temp_c", "time_hr", "ni_recovery_pct", "co_recovery_pct")

st.set_page_config(page_title="금속 회수율 최적화", page_icon="⚗", layout="wide")


@st.cache_resource(show_spinner="모델과 적용 범위 판정기를 불러오는 중…")
def boot():
    cfg = load_config()
    batches = load_batches(cfg)
    process_dom, feed_dom = build_domains(cfg)
    return cfg, batches, process_dom, feed_dom


def twin_input(r: dict, row: pd.Series) -> dict:
    """3D 화면(P5)이 읽는 입력 묶음 — 원료 정보 + 실측 근거 + Module 3 JSON 그대로."""
    evidence = {c: float(row[c]) for c in EVIDENCE_COLS if pd.notna(row.get(c))}
    return {
        "scenario_name": f"{r['batch_id']} — Module 3 추천",
        "feed": feed_from_row(row),
        # 공정 값이 빈 배치는 실측 근거를 싣지 않는다 (화면이 빈 칸을 그대로 찍는다)
        "evidence": ({"source": "원본 데이터 실측", **evidence}
                     if len(evidence) == len(EVIDENCE_COLS) else None),
        "recipe": r,
    }


cfg, batches, process_dom, feed_dom = boot()

st.title("황산 침출 — 최적 공정 레시피 추천")
st.caption(
    "원료 성분과 목표 회수율을 받아 황산 투입량·침출 온도·침출 시간을 추천한다. "
    f"모듈 1 정제본 {len(batches)}건, 결정 변수 격자는 현장 단위로 93,240점."
)

# ── 입력 ─────────────────────────────────────────────────────────
left, right = st.columns([1, 1.6], gap="large")

with left:
    st.subheader("원료")
    bid = st.selectbox("배치", batches["batch_id"].tolist())
    row = batches[batches["batch_id"] == bid].iloc[0]

    imp = sum(float(row[c]) for c in ("cu_pct", "fe_pct", "al_pct"))
    m1, m2, m3 = st.columns(3)
    m1.metric("등급", f"{row.get('grade', '?')}급")
    m2.metric("Ni / Co", f"{row['ni_pct']:.1f} / {row['co_pct']:.1f} %")
    m3.metric("불순물 합", f"{imp:.1f} %")

    st.subheader("목표")
    ni_min = st.slider("Ni 회수율 하한 (%)", 80.0, 99.0,
                       float(cfg["targets"]["ni_recovery_pct"]), 0.5)
    co_min = st.slider("Co 회수율 하한 (%)", 70.0, 99.0,
                       float(cfg["targets"]["co_recovery_pct"]), 0.5)
    st.caption(
        f"금속별로 μ − {cfg['uncertainty']['chance_z']}σ ≥ 목표를 요구하는 보수적 판정이다. "
        "σ 는 검증 잔차 RMSE 와 ridge·rf·hgb 예측 편차를 합친 값으로, 보정된 확률이 아니므로 "
        "'90% 확률로 달성'을 뜻하지는 않는다."
    )
    run = st.button("최적 레시피 추천", type="primary", use_container_width=True)

# ── 추천 ─────────────────────────────────────────────────────────
if "recipe" not in st.session_state:
    st.session_state.recipe = None
    st.session_state.recipe_key = None

# 원료나 목표를 바꾸면 이전 추천은 현재 입력의 결과가 아니다
key = (bid, float(ni_min), float(co_min))

if run:
    with st.spinner("격자 93,240점을 평가하는 중… (앙상블 σ 때문에 1초쯤 걸린다)"):
        try:
            rec = recommend(
                feed_from_row(row),
                target={"ni_recovery_pct": ni_min, "co_recovery_pct": co_min},
                cfg=cfg, algo="grid", domain=process_dom, feed_domain=feed_dom,
            )
            st.session_state.recipe = rec.to_dict()
            st.session_state.recipe_key = key
        except InputError as exc:
            st.session_state.recipe = None
            st.session_state.recipe_key = None
            st.error(f"입력을 받을 수 없다: {exc}")

with right:
    st.subheader("추천 결과")
    r = st.session_state.recipe if st.session_state.recipe_key == key else None
    if r is None and st.session_state.recipe is not None:
        st.info("원료나 목표가 바뀌었다. **최적 레시피 추천** 을 다시 누른다.")
    elif r is None:
        st.info("왼쪽에서 원료와 목표를 고르고 **최적 레시피 추천** 을 누른다.")
    elif r["status"] == "ok":
        st.success(r["message"])
        c = st.columns(4)
        c[0].metric("황산", f"{r['recipe']['acid_l']:,} L")
        c[1].metric("온도", f"{r['recipe']['temp_c']} ℃")
        c[2].metric("시간", f"{r['recipe']['time_hr']} hr")
        c[3].metric("산 비율", f"{r['recipe']['acid_per_feed']:.2f} L/kg")
        p = r["prediction"]
        st.dataframe(pd.DataFrame({
            "예측 평균": [p["ni"]["mean"], p["co"]["mean"]],
            "구간 하한": [p["ni"]["low"], p["co"]["low"]],
            "구간 상한": [p["ni"]["high"], p["co"]["high"]],
        }, index=["Ni 회수율 %", "Co 회수율 %"]), use_container_width=True)
        st.caption(
            f"신뢰도 {r['confidence']} · 유사 배치 {r['similar_batches']}건 · "
            f"적용 범위 거리 {r['domain_distance']} · {r['meta'].get('model_version', '')}"
        )
    else:
        st.warning(r["message"])
        if r.get("shortfall"):
            st.caption(f"부족분 — Ni {r['shortfall'].get('ni', 0)}%p · Co {r['shortfall'].get('co', 0)}%p")

# ── 3D 디지털 트윈 ───────────────────────────────────────────────
st.divider()
st.subheader("공정 디지털 트윈")
st.caption(
    "명세 #2-6 모델링 구현 (P5) — 공장 25m×17m×6m · 맵 1 · 설비 1 · 애니메이션 3 · 광석 3. "
    "드래그로 회전, 휠로 확대, 설비를 누르면 정보가 나온다."
)

if not SCENE.exists():
    st.error("docs/modeling/index.html 이 없다.")
else:
    html = SCENE.read_text(encoding="utf-8")
    r = st.session_state.recipe if st.session_state.recipe_key == key else None
    if r and r["status"] == "ok":
        # </script> 가 문자열 안에 있어도 태그가 닫히지 않게 </ 를 이스케이프한다
        payload = json.dumps(twin_input(r, row), ensure_ascii=False).replace("</", "<\\/")
        html = html.replace("</head>", f"<script>window.__TWIN_INPUT__ = {payload};</script>\n</head>", 1)
        st.caption(f"위 추천 결과({bid})가 3D 화면에 들어가 있다.")
    else:
        st.caption(
            "3D 화면은 기본 데모 입력(BP-2025-0074 가정값)을 보여 준다. 목표 달성 레시피가 나오면 "
            "그 결과로 바뀐다. 목표 미달 · 범위 밖 결과의 화면 표시는 P5 작업 예정이다."
        )
    components.html(html, height=780, scrolling=False)

st.divider()
st.caption(
    "모듈 1 전처리 → 모듈 2 예측 → 모듈 3 최적화가 한 프로세스에서 이어진다. "
    f"목표 달성이 드문 것은 데이터의 성질이다 — 원본 420건 중 Ni≥{cfg['targets']['ni_recovery_pct']:.0f} 와 "
    f"Co≥{cfg['targets']['co_recovery_pct']:.0f} 를 동시에 만족한 실측은 6건뿐이다."
)
