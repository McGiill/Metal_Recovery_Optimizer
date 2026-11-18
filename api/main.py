"""실시간 회수율 예측 · 레시피 추천 API — 명세 #8 '모델을 API 형태로 배포'.

    uvicorn api.main:app --host 0.0.0.0 --port 8000
    http://localhost:8000/docs   (Swagger UI)

모델과 적용 범위 판정기는 프로세스 기동 때 한 번만 올린다.
요청마다 joblib.load() 를 다시 하면 7.8MB 를 매번 디스크에서 읽게 된다.
응답 본문은 module3_optimization.problem.Recipe.to_dict() 그대로다 —
스키마는 거기서 관리하고 여기서는 손대지 않는다.
"""

from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from module3_optimization.config import load_config  # noqa: E402
from module3_optimization.domain import build_domains  # noqa: E402
from module3_optimization.problem import InputError  # noqa: E402
from module3_optimization.recommend import feed_from_row, load_batches, recommend  # noqa: E402

S: dict = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    S["cfg"] = load_config()
    S["batches"] = load_batches(S["cfg"])
    S["domain"], S["feed_domain"] = build_domains(S["cfg"])
    yield
    S.clear()


app = FastAPI(
    title="금속 회수율 최적화 API",
    description="폐배터리 블랙파우더의 황산 침출 — 회수율 예측과 최적 레시피 추천.",
    version="0.1.0",
    lifespan=lifespan,
)


class Feed(BaseModel):
    """원료 1배치의 성분과 투입량. 범위는 데이터 사전의 정상 범위를 따른다."""

    ni_pct: float = Field(ge=0, le=40, examples=[24.3], description="니켈 함량 wt%")
    co_pct: float = Field(ge=0, le=20, examples=[9.1], description="코발트 함량 wt%")
    li_pct: float = Field(ge=0, le=12, examples=[3.8])
    mn_pct: float = Field(ge=0, le=25, examples=[9.4])
    cu_pct: float = Field(ge=0, le=12, examples=[1.1], description="구리 불순물 wt%")
    fe_pct: float = Field(ge=0, le=12, examples=[0.9], description="철 불순물 wt%")
    al_pct: float = Field(ge=0, le=12, examples=[1.4], description="알루미늄 불순물 wt%")
    feed_kg: float = Field(gt=0, le=400, examples=[140.0], description="1회 배치 원료 질량 kg")


class Target(BaseModel):
    ni_recovery_pct: float | None = Field(default=None, ge=0, le=100)
    co_recovery_pct: float | None = Field(default=None, ge=0, le=100)


class Request(BaseModel):
    feed: Feed
    target: Target | None = None


def _run(feed: dict, target: Target | None) -> dict:
    tgt = {k: v for k, v in (target.model_dump() if target else {}).items() if v is not None}
    try:
        rec = recommend(
            feed, target=tgt or None, cfg=S["cfg"], algo="grid",
            domain=S["domain"], feed_domain=S["feed_domain"],
        )
    except InputError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return rec.to_dict()


@app.get("/health", summary="기동 확인")
def health() -> dict:
    return {
        "status": "ok",
        "batches": int(len(S.get("batches", []))),
        "predictor": S.get("cfg", {}).get("predictor", {}).get("type"),
    }


@app.get("/batches", summary="정제본에 있는 배치 목록")
def list_batches() -> list[dict]:
    cols = [c for c in ("batch_id", "grade", "ni_pct", "co_pct", "feed_kg") if c in S["batches"]]
    return S["batches"][cols].to_dict("records")


@app.post("/recommend", summary="원료를 직접 넣어 레시피 추천")
def post_recommend(req: Request) -> dict:
    return _run(req.feed.model_dump(), req.target)


@app.post("/recommend/{batch_id}", summary="정제본의 배치 ID 로 레시피 추천")
def post_recommend_batch(batch_id: str, target: Target | None = None) -> dict:
    hit = S["batches"][S["batches"]["batch_id"] == batch_id]
    if hit.empty:
        raise HTTPException(status_code=404, detail=f"배치 ID 가 없다: {batch_id}")
    return _run(feed_from_row(hit.iloc[0]), target)
