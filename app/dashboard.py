"""Review Insights dashboard.
Run from the project root:   streamlit run app/dashboard.py
"""
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from insights_service import InsightService  # noqa: E402

st.set_page_config(page_title="Review Insights", page_icon="🔎", layout="wide")


@st.cache_resource(show_spinner="Loading reviews, embeddings and model…")
def load_service():
    s = InsightService()
    s.encode(["warm up"])
    return s


svc = load_service()
df = svc.df
topics = pd.DataFrame(svc.topics)
drivers = pd.DataFrame(svc.complaints["drivers"])
themes = pd.DataFrame(svc.complaints["complaint_themes"])
sig_path = ROOT / "artifacts" / "driver_significance.csv"
sig = pd.read_csv(sig_path) if sig_path.exists() else None
neg = df[df.Score <= 2]

# ---------------- header + KPIs ----------------
st.title("Amazon Fine Food — Review Insights")
st.caption(f"{len(df):,} reviews (stratified sample) · {df.year.min()}–{df.year.max()} · "
           "embeddings: all-MiniLM-L6-v2")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Reviews analysed", f"{len(df):,}")
k2.metric("Average rating", f"{df.Score.mean():.2f} ★")
k3.metric("Complaints (1–2★)", f"{(df.Score <= 2).mean():.1%}")
k4.metric("Topics / complaint themes", f"{len(topics)} / {len(themes)}")

tab_topics, tab_complaints, tab_trends, tab_search, tab_classify = st.tabs(
    ["📚 Topics", "⚠️ Complaints", "📈 Emerging patterns", "🔍 Search & summarize", "🏷️ Classify text"])

# ---------------- 1. Topics ----------------
with tab_topics:
    st.subheader("What customers talk about")
    fig = px.scatter(topics, x="share", y="pct_negative", size="n", text="label",
                     hover_data={"keywords": True, "avg_score": True, "n": True},
                     labels={"share": "Share of all reviews", "pct_negative": "Share of 1–2★ reviews"})
    fig.update_traces(textposition="top center", marker=dict(opacity=0.6))
    fig.update_layout(height=560, xaxis_tickformat=".0%", yaxis_tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Top-right = big and unhappy → priority. Bubble size = number of reviews.")

    pick = st.selectbox("Inspect a topic", topics.sort_values("n", ascending=False).label)
    t = topics[topics.label == pick].iloc[0]
    c1, c2, c3 = st.columns(3)
    c1.metric("Reviews", f"{t.n:,}")
    c2.metric("Avg rating", f"{t.avg_score:.2f} ★")
    c3.metric("Negative", f"{t.pct_negative:.0%}")
    st.markdown("**Keywords:** " + ", ".join(t.keywords))

    yearly = (df[df.topic_label == pick].groupby("year").Score
              .agg(reviews="size", neg_rate=lambda s: (s <= 2).mean()).loc[2006:])
    st.plotly_chart(px.bar(yearly.reset_index(), x="year", y="reviews",
                           title=f"{pick}: reviews per year"), use_container_width=True)

    st.markdown("**What reviewers say (auto-summary of complaints in this topic):**")
    summ = svc.summarize(topic_id=int(t.id), sentiment="negative", n_sentences=4)
    for s in summ["summary"]:
        st.write(f"• {s['sentence']}  _(review {s['review_id']})_")

# ---------------- 2. Complaints ----------------
with tab_complaints:
    st.subheader("Why customers complain")
    left, right = st.columns([3, 2])
    with left:
        d = drivers.sort_values("pct_of_complaints")
        fig = px.bar(d, x="pct_of_complaints", y="driver", orientation="h",
                     labels={"pct_of_complaints": "Share of 1–2★ reviews mentioning", "driver": ""},
                     title="Complaint drivers (zero-shot, sentence-level)")
        fig.update_layout(height=520, xaxis_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        chosen = st.selectbox("Evidence for driver", drivers.sort_values("n_reviews", ascending=False).driver)
        st.markdown(f"**Example sentences tagged “{chosen}”**")
        for q in drivers[drivers.driver == chosen].iloc[0].quotes:
            st.info(q)

    st.subheader("Which problem dominates each complaint theme")
    ct = (neg[["complaint_label", "aspects"]].explode("aspects").dropna().reset_index(drop=True)
          .pipe(lambda x: pd.crosstab(x.complaint_label, x.aspects, normalize="index")))
    fig = px.imshow(ct, color_continuous_scale="Reds", aspect="auto", text_auto=".0%",
                    labels=dict(x="Driver", y="Complaint theme", color="Share"))
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Complaint themes (clusters of 1–2★ reviews)")
    st.dataframe(themes[["label", "n", "keywords"]].sort_values("n", ascending=False),
                 use_container_width=True, hide_index=True)

# ---------------- 3. Emerging patterns ----------------
with tab_trends:
    st.subheader("Which complaints are growing? (2011–12 vs 2008–10)")
    if sig is not None:
        show = sig.copy()
        show["change"] = show.share_base.map("{:.1%}".format) + " → " + show.share_recent.map("{:.1%}".format)
        show["status"] = show.significant.map({True: "🔺 Emerging (p<0.05)", False: "Stable / monitor"})
        st.dataframe(show[["driver", "change", "growth", "p_value", "status"]],
                     use_container_width=True, hide_index=True)

    trend_rows = [{"driver": d["driver"], "year": int(y), "share": v}
                  for d in svc.complaints["drivers"] for y, v in d.get("trend", {}).items()]
    tr = pd.DataFrame(trend_rows)
    default = (sig[sig.significant].driver.tolist() if sig is not None
               else drivers.sort_values("growth", ascending=False).driver.head(4).tolist())
    sel = st.multiselect("Drivers to plot", drivers.driver.tolist(), default=default)
    if sel and not tr.empty:
        fig = px.line(tr[tr.driver.isin(sel) & (tr.year >= 2007)], x="year", y="share",
                      color="driver", markers=True, labels={"share": "Share of complaints"})
        fig.update_layout(yaxis_tickformat=".1%", height=460)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Complaint rate by topic over time")
    rate = (df[df.year.between(2008, 2012)].groupby(["topic_label", "year"]).Score
            .apply(lambda s: (s <= 2).mean()).unstack())
    fig = px.imshow(rate, color_continuous_scale="Reds", aspect="auto", text_auto=".0%",
                    labels=dict(x="Year", y="Topic", color="% 1–2★"))
    fig.update_layout(height=650)
    st.plotly_chart(fig, use_container_width=True)

# ---------------- 4. Search & summarize ----------------
with tab_search:
    st.subheader("Ask a question about the reviews")
    q = st.text_input("Question or keywords", "coffee pods leaking in the machine")
    c1, c2, c3 = st.columns(3)
    sentiment = c1.selectbox("Sentiment", ["any", "negative", "neutral", "positive"])
    years = c2.slider("Years", 2000, 2012, (2006, 2012))
    k = c3.slider("Number of reviews", 5, 30, 10)
    filters = dict(sentiment=None if sentiment == "any" else sentiment,
                   year_from=years[0], year_to=years[1])
    if q:
        with st.spinner("Searching…"):
            summ = svc.summarize(query=q, n_sentences=5, **filters)
            hits = svc.search(q, k=k, **filters)
        st.markdown("#### Summary")
        for s in summ["summary"]:
            st.write(f"• {s['sentence']}  _(review {s['review_id']})_")
        st.markdown("#### Most relevant reviews")
        for h in hits:
            with st.expander(f"{'★' * h['score']}  {h['summary']}  ·  {h['year']}  ·  {h['topic']}  ·  sim {h['similarity']}"):
                st.write(h["text"])

# ---------------- 5. Classify ----------------
with tab_classify:
    st.subheader("Route a new review or support ticket")
    txt = st.text_area("Paste text", "The box arrived crushed and two bags were open. "
                                     "Also the chips tasted stale and old.", height=120)
    if st.button("Classify", type="primary"):
        res = svc.classify(txt)
        st.success(f"**Topic:** {res['topic']}")
        st.dataframe(pd.DataFrame(res["top_topics"]), hide_index=True)
        if res["drivers"]:
            st.markdown("**Complaint drivers found:**")
            st.dataframe(pd.DataFrame(res["drivers"]), hide_index=True, use_container_width=True)
        else:
            st.info("No complaint drivers detected.")