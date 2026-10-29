
import streamlit as st
import pandas as pd

from recommender import load_movies, build_similarity, recommend

st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="wide"
)


@st.cache_data
def prepare():
    df = load_movies()
    similarity = build_similarity(df)
    return df, similarity


# Load movie dataset and similarity matrix
df, sim = prepare()

st.title("🎬 Movie Recommendation System")

st.write(
    "Search real movie titles and get similar recommendations "
    "using TF-IDF and cosine similarity."
)

st.caption(
    f"Dataset loaded: {len(df):,} movies. "
    "Includes MovieLens titles/genres and curated popular films."
)

# Sidebar menu
page = st.sidebar.radio(
    "Menu",
    ["Get Recommendations", "Dataset"]
)


if page == "Get Recommendations":

    search = st.text_input(
        "Search a movie",
        placeholder="e.g. Inception, 3 Idiots, Dangal, Holy Word"
    )

    candidates = df

    # Search movie by title
    if search.strip():
        candidates = df[
            df["title"].str.contains(
                search.strip(),
                case=False,
                regex=False,
                na=False
            )
        ]

    if candidates.empty:

        st.warning(
            "No matching title found. Try another spelling "
            "or refresh the dataset."
        )

    else:

        options = (
            candidates.sort_values("title")["title"]
            .drop_duplicates()
            .tolist()
        )

        title = st.selectbox(
            "Select a movie",
            options
        )

        n = st.slider(
            "Number of recommendations",
            min_value=1,
            max_value=10,
            value=5
        )

        if st.button("Recommend", type="primary"):

            selected = df[df["title"] == title].iloc[0]

            year = selected.get("year")

            # Safely format the release year
            if pd.isna(year) or str(year).strip() == "":
                year_text = ""
            else:
                try:
                    year_text = str(int(float(year)))
                except (ValueError, TypeError):
                    year_text = str(year)

            genres = selected.get("genres", "Unknown")
            industry = selected.get("industry", "Unknown")

            year_display = (
                f" ({year_text})" if year_text else ""
            )

            st.info(
                f"{selected['title']}{year_display} | "
                f"{genres} | {industry}"
            )

            st.subheader("🍿 You may also like")

            try:
                recommendations = recommend(
                    title, df, sim, n
                )

                if recommendations is None or recommendations.empty:
                    st.warning(
                        "No recommendations found for this movie."
                    )
                else:
                    st.dataframe(
                        recommendations,
                        use_container_width=True,
                        hide_index=True
                    )

            except Exception as error:
                st.error(
                    f"Recommendation error: {error}"
                )


else:

    st.subheader("📚 Movie Dataset")

    search = st.text_input(
        "Search dataset",
        placeholder="Enter a movie title"
    )

    industry = st.selectbox(
        "Dataset source",
        [
            "All",
            "Bollywood",
            "Hollywood",
            "MovieLens",
            "Other"
        ]
    )

    view = df.copy()

    # Filter by title
    if search.strip():
        view = view[
            view["title"].str.contains(
                search.strip(),
                case=False,
                regex=False,
                na=False
            )
        ]

    # Filter by industry
    if industry != "All":
        view = view[
            view["industry"].str.casefold()
            == industry.casefold()
        ]

    st.write(f"**{len(view):,} movies found**")

    columns = [
        column
        for column in ["title", "year", "genres", "industry"]
        if column in view.columns
    ]

    st.dataframe(
        view[columns],
        use_container_width=True,
        hide_index=True
    )


st.caption(
    "MovieLens provides movie titles and genres. "
    "Plot summaries and industry labels may not be complete "
    "for every movie."
)
