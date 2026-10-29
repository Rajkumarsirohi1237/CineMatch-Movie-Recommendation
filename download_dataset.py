"""Download MovieLens movie titles and merge a curated set of Bollywood/Hollywood films."""
from pathlib import Path
from io import BytesIO
from urllib.request import Request, urlopen
import zipfile
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
OUTPUT = BASE_DIR / "movies.csv"
URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"

# Extra well-known movies (including Indian cinema) that may be missing from MovieLens.
# Genres are used by the content-based recommender; overview is optional.
EXTRA_MOVIES = [
    ("Inception", 2010, "Action|Adventure|Sci-Fi|Thriller", "Hollywood"),
    ("Interstellar", 2014, "Adventure|Drama|Sci-Fi", "Hollywood"),
    ("The Dark Knight", 2008, "Action|Crime|Drama|Thriller", "Hollywood"),
    ("Avengers: Endgame", 2019, "Action|Adventure|Sci-Fi", "Hollywood"),
    ("Titanic", 1997, "Drama|Romance", "Hollywood"),
    ("Avatar", 2009, "Action|Adventure|Sci-Fi", "Hollywood"),
    ("The Shawshank Redemption", 1994, "Drama", "Hollywood"),
    ("The Godfather", 1972, "Crime|Drama", "Hollywood"),
    ("The Matrix", 1999, "Action|Sci-Fi", "Hollywood"),
    ("Oppenheimer", 2023, "Biography|Drama|History", "Hollywood"),
    ("3 Idiots", 2009, "Comedy|Drama", "Bollywood"),
    ("Dangal", 2016, "Action|Biography|Drama|Sport", "Bollywood"),
    ("Lagaan", 2001, "Drama|Musical|Sport", "Bollywood"),
    ("Taare Zameen Par", 2007, "Drama|Family", "Bollywood"),
    ("Sholay", 1975, "Action|Adventure|Drama", "Bollywood"),
    ("Dilwale Dulhania Le Jayenge", 1995, "Drama|Romance", "Bollywood"),
    ("Zindagi Na Milegi Dobara", 2011, "Adventure|Comedy|Drama", "Bollywood"),
    ("Queen", 2013, "Adventure|Comedy|Drama", "Bollywood"),
    ("Bajrangi Bhaijaan", 2015, "Action|Comedy|Drama", "Bollywood"),
    ("PK", 2014, "Comedy|Drama|Sci-Fi", "Bollywood"),
    ("Jawan", 2023, "Action|Thriller", "Bollywood"),
    ("Pathaan", 2023, "Action|Thriller", "Bollywood"),
    ("Drishyam", 2015, "Crime|Drama|Thriller", "Bollywood"),
    ("Andhadhun", 2018, "Crime|Thriller|Comedy", "Bollywood"),
    ("Gully Boy", 2019, "Drama|Musical", "Bollywood"),
    ("Kahaani", 2012, "Mystery|Thriller", "Bollywood"),
    ("Swades", 2004, "Drama", "Bollywood"),
    ("Chak De! India", 2007, "Drama|Sport", "Bollywood"),
]

def main():
    print("Downloading the official MovieLens small dataset...")
    request = Request(URL, headers={"User-Agent": "Mozilla/5.0 MovieRecommendationML"})
    with urlopen(request, timeout=60) as response:
        archive_bytes = response.read()
    with zipfile.ZipFile(BytesIO(archive_bytes)) as archive:
        movies_member = next(n for n in archive.namelist() if n.endswith("/movies.csv"))
        from io import TextIOWrapper
        with archive.open(movies_member) as stream:
            ml = pd.read_csv(TextIOWrapper(stream, encoding="utf-8"))

    ml = ml.rename(columns={"title": "title", "genres": "genres"})
    ml["year"] = ml["title"].str.extract(r"\((\d{4})\)\s*$")[0]
    ml["year"] = pd.to_numeric(ml["year"], errors="coerce").astype("Int64")
    ml["title"] = ml["title"].str.replace(r"\s*\(\d{4}\)\s*$", "", regex=True).str.strip()
    ml["overview"] = ""
    ml["industry"] = "MovieLens"
    ml = ml[["title", "year", "genres", "overview", "industry"]]

    extras = pd.DataFrame(EXTRA_MOVIES, columns=["title", "year", "genres", "industry"])
    extras["overview"] = ""
    extras = extras[["title", "year", "genres", "overview", "industry"]]

    # Prefer curated labels for duplicate title/year entries.
    combined = pd.concat([ml, extras], ignore_index=True)
    combined["year_key"] = combined["year"].astype("string").fillna("")
    combined = combined.drop_duplicates(subset=["title", "year_key"], keep="last")
    combined = combined.drop(columns=["year_key"])
    combined["title"] = combined["title"].fillna("").astype(str)
    combined["genres"] = combined["genres"].fillna("Unknown").astype(str).str.replace("|", " ", regex=False)
    combined["overview"] = combined["overview"].fillna("").astype(str)
    combined["industry"] = combined["industry"].fillna("MovieLens")
    combined = combined.sort_values("title", key=lambda s: s.str.lower()).reset_index(drop=True)
    combined.to_csv(OUTPUT, index=False, encoding="utf-8")
    print(f"Success! Saved {len(combined):,} movies to: {OUTPUT}")
    print("Note: MovieLens supplies titles and genres, not full plot summaries or guaranteed industry labels.")

if __name__ == "__main__":
    main()
