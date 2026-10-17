"""Review Insights API.
Run from the project root:   uvicorn api.main:app --reload
Interactive docs:            http://127.0.0.1:8000/docs
"""
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from insights_service import InsightService  # noqa: E402

svc: InsightService | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load data + model once at startup, not on every request."""
    global svc
    svc = InsightService()
    svc.encode(["warm up"])          # loads the embedding model now
    yield


app = FastAPI(
    title="Review Insights API",
    description="Topics, complaint drivers, emerging trends, semantic search and summaries "
                "over 50K Amazon Fine Food reviews.",
    version="1.0.0",
    lifespan=lifespan,
)

Sentiment = Optional[Literal["positive", "neutral", "negative"]]


# ---------- request bodies ----------
class SummarizeRequest(BaseModel):
    query: Optional[str] = Field(None, examples=["why do customers complain about dog treats?"])
    topic_id: Optional[int] = None
    sentiment: Sentiment = None
    year_from: Optional[int] = None
    year_to: Optional[int] = None
    n_sentences: int = Field(5, ge=1, le=15)


class ClassifyRequest(BaseModel):
    text: str = Field(..., min_length=5,
                      examples=["The box arrived crushed and the chips tasted stale."])


# ---------- endpoints ----------
@app.get("/health")
def health():
    return {"status": "ok", "reviews_indexed": len(svc.df)}


@app.get("/overview")
def overview():
    return svc.overview()


@app.get("/topics")
def topics():
    return svc.topics


@app.get("/topics/{topic_id}")
def topic(topic_id: int):
    match = [t for t in svc.topics if t["id"] == topic_id]
    if not match:
        raise HTTPException(404, f"topic {topic_id} not found")
    return match[0]


@app.get("/complaints/themes")
def complaint_themes():
    return svc.complaints["complaint_themes"]


@app.get("/complaints/drivers")
def complaint_drivers():
    return svc.complaints["drivers"]


@app.get("/complaints/emerging")
def emerging(only_significant: bool = True):
    """Drivers whose share of complaints changed 2008-10 -> 2011-12 (two-proportion z-test)."""
    path = ROOT / "artifacts" / "driver_significance.csv"
    if not path.exists():
        raise HTTPException(404, "run the significance cell in 04_complaints.ipynb first")
    sig = pd.read_csv(path)
    if only_significant:
        sig = sig[sig.significant]
    return sig.sort_values("growth", ascending=False).to_dict(orient="records")


@app.get("/search")
def search(q: str = Query(..., min_length=2, examples=["coffee pods leaking"]),
           k: int = Query(10, ge=1, le=100),
           sentiment: Sentiment = None,
           topic_id: Optional[int] = None,
           year_from: Optional[int] = None,
           year_to: Optional[int] = None):
    return svc.search(q, k=k, sentiment=sentiment, topic_id=topic_id,
                      year_from=year_from, year_to=year_to)


@app.post("/summarize")
def summarize(req: SummarizeRequest):
    if not req.query and req.topic_id is None:
        raise HTTPException(422, "provide a query and/or a topic_id")
    return svc.summarize(query=req.query, topic_id=req.topic_id, n_sentences=req.n_sentences,
                         sentiment=req.sentiment, year_from=req.year_from, year_to=req.year_to)


@app.post("/classify")
def classify(req: ClassifyRequest):
    """Route a new review / support ticket to a topic and tag complaint drivers per sentence."""
    return svc.classify(req.text)