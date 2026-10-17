"""One service used by the notebook, the FastAPI app and the Streamlit dashboard."""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from aspects import ASPECTS, THRESHOLD
from nlp_utils import split_sentences

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
ART = Path(__file__).resolve().parents[1] / "artifacts"


class InsightService:
    def __init__(self, art: Path = ART):
        self.df = pd.read_pickle(art / "reviews_enriched.pkl")
        self.emb = np.load(art / "embeddings.npy")
        assert len(self.df) == len(self.emb), "reviews and embeddings out of sync"
        self.topics = json.loads((art / "topics.json").read_text())
        self.complaints = json.loads((art / "complaints.json").read_text())
        self.topic_centroids = np.load(art / "topic_centroids.npy")
        self._model = None
        self._aspect_centroids = None

    # ---------- model (loaded lazily: only needed for free-text queries) ----------
    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer("all-MiniLM-L6-v2")
            self._model.max_seq_length = 128
        return self._model

    def encode(self, texts):
        return self.model.encode(list(texts), normalize_embeddings=True, convert_to_numpy=True)

    # ---------- filters ----------
    def _mask(self, sentiment=None, topic_id=None, year_from=None, year_to=None):
        m = np.ones(len(self.df), dtype=bool)
        if sentiment:
            m &= (self.df.sentiment == sentiment).values
        if topic_id is not None:
            m &= (self.df.topic_id == topic_id).values
        if year_from:
            m &= (self.df.year >= year_from).values
        if year_to:
            m &= (self.df.year <= year_to).values
        return m

    # ---------- 1. semantic search ----------
    def search(self, query, k=10, **filters):
        q = self.encode([query])[0]
        idx = np.where(self._mask(**filters))[0]
        if len(idx) == 0:
            return []
        sims = self.emb[idx] @ q
        order = np.argsort(-sims)[:k]
        results = []
        for i, s in zip(idx[order], sims[order]):
            r = self.df.iloc[i]
            results.append({
                "row": int(i), "id": int(r.Id), "product_id": r.ProductId,
                "score": int(r.Score), "year": int(r.year), "topic": r.topic_label,
                "summary": r.Summary, "text": r.Text, "similarity": round(float(s), 3),
            })
        return results

    # ---------- 2. extractive summary (MMR) ----------
    def summarize(self, query=None, topic_id=None, n_sentences=5, k_reviews=40, lam=0.7, **filters):
        """Pick sentences that are relevant to the query/topic but not redundant with each other."""
        if query:
            hits = self.search(query, k=k_reviews, topic_id=topic_id, **filters)
            rows = [h["row"] for h in hits]
            anchor = self.encode([query])[0]
        else:
            idx = np.where(self._mask(topic_id=topic_id, **filters))[0]
            if len(idx) == 0:
                return {"summary": [], "n_reviews": 0}
            anchor = self.emb[idx].mean(0)
            anchor /= np.linalg.norm(anchor)
            rows = idx[np.argsort(-(self.emb[idx] @ anchor))[:k_reviews]].tolist()

        sents, src = [], []
        for i in rows:
            for s in split_sentences(self.df.Text.iloc[i], min_len=30, max_n=8):
                if len(s) < 260:
                    sents.append(s)
                    src.append(int(self.df.Id.iloc[i]))
        if not sents:
            return {"summary": [], "n_reviews": len(rows)}

        S = self.encode(sents)
        relevance = S @ anchor
        chosen = []
        while len(chosen) < min(n_sentences, len(sents)):
            redundancy = (S @ S[chosen].T).max(1) if chosen else np.zeros(len(sents))
            score = lam * relevance - (1 - lam) * redundancy
            score[chosen] = -np.inf
            chosen.append(int(score.argmax()))
        return {"summary": [{"sentence": sents[i], "review_id": src[i]} for i in chosen],
                "n_reviews": len(rows)}

    # ---------- 3. classify new text ----------
    def classify(self, text):
        if self._aspect_centroids is None:
            names = list(ASPECTS)
            c = np.vstack([self.encode(ASPECTS[n]).mean(0) for n in names])
            self._aspect_centroids = (names, c / np.linalg.norm(c, axis=1, keepdims=True))
        names, a_cent = self._aspect_centroids

        v = self.encode([text])[0]
        t_sims = self.topic_centroids @ v
        top3 = np.argsort(-t_sims)[:3]
        label = {t["id"]: t["label"] for t in self.topics}

        sents = split_sentences(text, min_len=5) or [text]
        s_sims = self.encode(sents) @ a_cent.T
        aspects = [{"sentence": s, "driver": names[row.argmax()], "similarity": round(float(row.max()), 3)}
                   for s, row in zip(sents, s_sims) if row.max() >= THRESHOLD]

        return {"topic": label[int(top3[0])],
                "top_topics": [{"topic": label[int(t)], "similarity": round(float(t_sims[t]), 3)} for t in top3],
                "drivers": aspects}

    # ---------- precomputed insights ----------
    def overview(self):
        d = self.df
        return {"n_reviews": int(len(d)), "avg_score": round(float(d.Score.mean()), 2),
                "pct_negative": round(float((d.Score <= 2).mean()), 3),
                "years": [int(d.year.min()), int(d.year.max())],
                "n_topics": len(self.topics), "n_complaint_themes": len(self.complaints["complaint_themes"])}