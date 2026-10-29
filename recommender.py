from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = Path(__file__).resolve().parent


def load_movies(path=None):
    path = Path(path) if path else BASE_DIR / "movies.csv"
    df = pd.read_csv(path)
    for col in ["title", "genres"]:
        if col not in df.columns:
            raise ValueError(f"movies.csv must contain a '{col}' column")
    if "year" not in df.columns:
        df["year"] = ""
    if "overview" not in df.columns:
        df["overview"] = ""
    if "industry" not in df.columns:
        df["industry"] = "MovieLens"
    df["title"] = df["title"].fillna("").astype(str)
    df["genres"] = df["genres"].fillna("Unknown").astype(str)
    df["overview"] = df["overview"].fillna("").astype(str)
    df["tags"] = ((df["genres"].str.replace("|", " ", regex=False) + " ") * 3) + df["overview"]
    return df.reset_index(drop=True)


def build_similarity(df):
    tfidf = TfidfVectorizer(stop_words="english")
    matrix = tfidf.fit_transform(df["tags"])
    return cosine_similarity(matrix)


def recommend(title, df, sim, n=5):
    matches = df.index[df["title"].str.casefold() == title.casefold()]
    if len(matches) == 0:
        raise ValueError(f"Movie not found: {title}")
    idx = int(matches[0])
    scores = sorted(enumerate(sim[idx]), key=lambda x: x[1], reverse=True)
    scores = [s for s in scores if s[0] != idx][:n]
    out = df.iloc[[i for i, _ in scores]][["title", "year", "genres", "industry"]].copy()
    out["similarity"] = [round(float(s), 3) for _, s in scores]
    return out.reset_index(drop=True)
