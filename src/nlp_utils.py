"""Reusable NLP helpers for the project."""
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS

DOMAIN_STOP = {"br", "product", "products", "just", "like", "good", "great", "really", "would",
               "one", "get", "got", "also", "much", "even", "make", "made", "use", "used", "buy",
               "bought", "try", "tried", "love", "best", "well", "think", "know", "lot", "little",
               "bit", "time", "ve", "don", "didn", "doesn", "amazon", "taste", "tastes", "flavor",
               "flavors", "delicious", "tasty", "stuff", "better"}
STOP = list(ENGLISH_STOP_WORDS | DOMAIN_STOP)


def ctfidf_keywords(texts, labels, top_n=10):
    """Class-based TF-IDF: words that are distinctive for each cluster."""
    docs = pd.Series(texts).groupby(labels).apply(" ".join)
    cv = CountVectorizer(stop_words=STOP, ngram_range=(1, 2), min_df=5,
                         max_features=40000, token_pattern=r"(?u)\b[a-zA-Z]{2,}\b")
    tf = cv.fit_transform(docs.values).astype(float)
    tf = tf.multiply(1 / tf.sum(axis=1)).tocsr()
    idf = np.log(1 + tf.shape[0] / np.asarray((tf > 0).sum(axis=0)).ravel())
    scores = tf.multiply(idf).tocsr()
    vocab = np.array(cv.get_feature_names_out())
    out = {}
    for row, lab in enumerate(docs.index):
        r = scores[row].toarray().ravel()
        words = []
        for w in vocab[np.argsort(-r)]:
            if any(w in x.split() or x in w.split() for x in words):
                continue
            words.append(w)
            if len(words) == top_n:
                break
        out[int(lab)] = words
    return out


_SPLIT = re.compile(r"(?<=[.!?])\s+")

def split_sentences(text, min_len=12, max_n=12):
    """Simple sentence splitter, good enough for reviews."""
    sents = [s.strip() for s in _SPLIT.split(str(text)) if len(s.strip()) >= min_len]
    return sents[:max_n]


def share_by_year(df, col, years=range(2006, 2013)):
    """Share of rows in each group, per year (rows=group, cols=year)."""
    d = df[df.year.isin(years)]
    return pd.crosstab(d[col], d.year, normalize="columns").reindex(columns=list(years), fill_value=0)


def growth(share, recent=(2011, 2012), base=(2008, 2009, 2010)):
    """Relative change in share: recent window vs baseline window."""
    r, b = share[list(recent)].mean(axis=1), share[list(base)].mean(axis=1)
    return ((r - b) / b.clip(lower=1e-6)).round(3)