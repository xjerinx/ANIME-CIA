import html
import re
import requests

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from difflib import SequenceMatcher


# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="Anime Recommender", page_icon="🎌", layout="wide")


# -----------------------------
# Theme toggle (Main page can be dark/light)
# Sidebar ALWAYS dark, search input ALWAYS black
# -----------------------------
dark_mode = st.sidebar.checkbox("🌙 Dark Mode", value=True)

BASE_SIDEBAR_CSS = """
<style>
  /* Sidebar always dark */
  section[data-testid="stSidebar"]{
    background:#0a0e16 !important;
    border-right:1px solid rgba(255,255,255,.12) !important;
  }
  section[data-testid="stSidebar"] *{ color:#e5e7eb !important; }
  section[data-testid="stSidebar"] .stCaption,
  section[data-testid="stSidebar"] p,
  section[data-testid="stSidebar"] small{
    color:#9ca3af !important;
  }

  /* Sidebar input always black */
  section[data-testid="stSidebar"] div[data-baseweb="input"] input{
    background-color:#111827 !important;
    color:#e5e7eb !important;
    border:1px solid rgba(255,255,255,0.16) !important;
    border-radius:12px !important;
  }
  section[data-testid="stSidebar"] div[data-baseweb="input"] input::placeholder{
    color:#9ca3af !important;
  }

  /* Compact trending cards */
  .trend-card{
    background: rgba(17,24,39,0.42);
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 12px;
    padding: 6px;
    margin-bottom: 8px;
  }
  .trend-img{
    width:100%;
    height:110px;
    border-radius:10px;
    overflow:hidden;
    background: rgba(255,255,255,0.06);
  }
  .trend-img img{
    width:100%;
    height:110px;
    object-fit:cover;
    display:block;
  }
  .trend-title{
    font-size: 0.78rem;
    line-height: 1.1;
    color: #e5e7eb;
    text-align:center;
    white-space: nowrap;
    overflow:hidden;
    text-overflow: ellipsis;
    margin-top: 4px;
    margin-bottom: 2px;
  }
  /* Smaller checkbox area in sidebar */
  section[data-testid="stSidebar"] div[data-baseweb="checkbox"]{
    transform: scale(0.92);
    transform-origin: center top;
  }

  /* Explanation box */
  .explain-box{
    background: rgba(17,24,39,0.55);
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 14px;
    padding: 10px 12px;
    color: #9ca3af;
    font-size: 0.92rem;
  }
  
  /* Video container styling */
  .video-container {
    border-radius: 12px;
    overflow: hidden;
    box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    margin: 15px 0;
  }
</style>
"""

if dark_mode:
    st.markdown(
        """
        <style>
          :root{
            --bg:#0b0f19;
            --text:#e5e7eb;
            --muted:#9ca3af;
            --border:rgba(255,255,255,.12);
          }

          .stApp{background:var(--bg); color:var(--text);}
          .block-container{padding-top:1.2rem;padding-bottom:2rem;}

          h1,h2,h3,h4,h5,h6{color:var(--text);}
          .stCaption,p,li{color:var(--muted);}
          a{color:#93c5fd;}

          /* Metrics cards */
          div[data-testid="stMetric"]{
            background:rgba(17,24,39,.9);
            border:1px solid var(--border);
            border-radius:16px;
            padding:12px;
            box-shadow:0 10px 30px rgba(0,0,0,0.35);
          }

          /* Dataframe */
          .stDataFrame, [data-testid="stDataFrame"]{
            border-radius: 14px;
            overflow: hidden;
            border: 1px solid var(--border);
            background: rgba(17,24,39,0.6);
          }
        </style>
        """,
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        """
        <style>
          :root{
            --bg:#f2f6ff;
            --text:#0f172a;
            --muted:#475569;
            --border:rgba(15,23,42,.10);
          }

          .stApp{background:var(--bg); color:var(--text);}
          .block-container{padding-top:1.2rem;padding-bottom:2rem;}

          h1,h2,h3,h4,h5,h6{color:var(--text);}
          .stCaption,p,li{color:var(--muted);}
          a{color:#2563eb;}

          div[data-testid="stMetric"]{
            background:rgba(255,255,255,0.92);
            border:1px solid var(--border);
            border-radius:16px;
            padding:12px;
            box-shadow:0 10px 24px rgba(15,23,42,.06);
          }

          .stDataFrame, [data-testid="stDataFrame"]{
            border-radius: 14px;
            overflow: hidden;
            border: 1px solid var(--border);
            background: rgba(255,255,255,0.95);
          }
        </style>
        """,
        unsafe_allow_html=True,
    )

# Apply always-on sidebar/trending css
st.markdown(BASE_SIDEBAR_CSS, unsafe_allow_html=True)


# -----------------------------
# Header
# -----------------------------
st.title("🎌 Anime Recommendation System + Analytics (TF-IDF)")
st.caption("Content-based recommendation using TF-IDF + Cosine Similarity on genre (plus EDA dashboard).")


# -----------------------------
# Media fetch (Poster + MAL + Trailer) via Jikan API
# -----------------------------
@st.cache_data(ttl=24 * 3600)
def fetch_media(anime_name: str):
    """
    Returns: (poster_url, mal_url, trailer_url)
    - Searches top 5 matches
    - Picks the closest title match that has a trailer (if any)
    - Ensures we get actual YouTube trailer URL for embedding
    """
    try:
        url = "https://api.jikan.moe/v4/anime"
        r = requests.get(url, params={"q": anime_name, "limit": 5}, timeout=10)
        if r.status_code != 200:
            return None, None, None

        data = r.json().get("data", [])
        if not data:
            return None, None, None

        def title_score(item):
            title = (item.get("title") or "").lower()
            return SequenceMatcher(None, anime_name.lower(), title).ratio()

        # Prefer items that HAVE trailer, then best title match
        candidates = []
        for item in data:
            trailer = item.get("trailer") or {}
            has_trailer = bool(trailer.get("youtube_id") or trailer.get("url"))
            candidates.append((has_trailer, title_score(item), item))

        # Sort: trailer first, then best match
        candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
        item = candidates[0][2]

        poster = item.get("images", {}).get("jpg", {}).get("image_url")
        mal_url = item.get("url")

        # Get the trailer URL properly for embedding
        trailer_url = None
        trailer = item.get("trailer") or {}
        youtube_id = trailer.get("youtube_id")
        
        if youtube_id:
            # This format works perfectly with st.video
            trailer_url = f"https://www.youtube.com/watch?v={youtube_id}"
        else:
            # Fallback to the URL if no youtube_id
            trailer_url = trailer.get("url")

        return poster, mal_url, trailer_url

    except Exception as e:
        print(f"Error fetching media: {e}")
        return None, None, None


@st.cache_data(ttl=24 * 3600)
def fetch_detailed_trailer(anime_name: str):
    """
    Secondary function to fetch trailer with more detailed search
    Returns YouTube URL specifically for embedding
    """
    try:
        # First search for the anime to get MAL ID
        search_url = "https://api.jikan.moe/v4/anime"
        search_params = {"q": anime_name, "limit": 3}
        search_response = requests.get(search_url, params=search_params, timeout=10)
        
        if search_response.status_code == 200:
            search_data = search_response.json().get("data", [])
            if search_data:
                # Find best title match
                best_match = None
                best_score = 0
                
                for item in search_data:
                    title = item.get("title", "").lower()
                    score = SequenceMatcher(None, anime_name.lower(), title).ratio()
                    if score > best_score:
                        best_score = score
                        best_match = item
                
                if best_match and best_score > 0.6:  # Reasonable match threshold
                    mal_id = best_match.get("mal_id")
                    
                    # Get full anime details including trailer
                    detail_url = f"https://api.jikan.moe/v4/anime/{mal_id}/full"
                    detail_response = requests.get(detail_url, timeout=10)
                    
                    if detail_response.status_code == 200:
                        detail_data = detail_response.json().get("data", {})
                        trailer = detail_data.get("trailer", {})
                        
                        youtube_id = trailer.get("youtube_id")
                        if youtube_id:
                            return f"https://www.youtube.com/watch?v={youtube_id}"
                        else:
                            return trailer.get("url")
        return None
    except Exception as e:
        print(f"Error in detailed trailer fetch: {e}")
        return None


# -----------------------------
# Load + clean data
# -----------------------------
def generate_sample_anime_data(n: int = 700, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    types = np.array(["TV", "Movie", "OVA", "ONA", "Special"])
    genre_pool = np.array(
        ["Action", "Adventure", "Comedy", "Drama", "Fantasy", "Romance", "Sci-Fi", "Slice of Life",
         "Mystery", "Horror", "Sports", "Supernatural", "Psychological", "Thriller", "Mecha"]
    )

    names = [f"Anime {i+1:04d}" for i in range(n)]
    t = rng.choice(types, size=n, p=[0.55, 0.15, 0.12, 0.10, 0.08])

    episodes = np.where(
        t == "Movie", rng.integers(1, 2, size=n),
        np.where(t == "TV", rng.integers(12, 75, size=n), rng.integers(1, 26, size=n))
    ).astype(float)

    members = (10 ** rng.normal(5.2, 0.6, size=n)).astype(int)
    members = np.clip(members, 500, 2_500_000)

    base = rng.normal(7.1, 0.7, size=n)
    pop_boost = (np.log10(members) - np.log10(members).mean()) * 0.18
    rating = np.clip(base + pop_boost, 4.0, 9.8)

    genres = []
    for _ in range(n):
        k = rng.integers(2, 5)
        g = rng.choice(genre_pool, size=k, replace=False)
        genres.append(", ".join(g.tolist()))

    df = pd.DataFrame(
        {"name": names, "genre": genres, "type": t, "episodes": episodes, "members": members.astype(float), "rating": rating.astype(float)}
    )

    for col in ["rating", "members", "episodes"]:
        mask = rng.random(n) < 0.03
        df.loc[mask, col] = np.nan

    return df


@st.cache_data
def load_data():
    try:
        df = pd.read_csv("data/anime.csv")
        df = df.dropna(subset=["name"]).copy()

        df["name"] = df["name"].astype(str).str.strip()
        df["name"] = df["name"].apply(lambda x: html.unescape(html.unescape(x)))
        df["name"] = df["name"].str.replace(r'^\s*["\']+|["\']+\s*$', "", regex=True)
        df["name"] = df["name"].apply(lambda x: re.sub(r"[\x00-\x1f\x7f-\x9f]", "", x))
        df["name"] = df["name"].str.replace(r"\s+", " ", regex=True).str.strip()

        df = df[df["name"].str.len() > 2]
        df = df.drop_duplicates(subset=["name"]).reset_index(drop=True)

        required = {"name", "genre", "type", "episodes", "members", "rating"}
        missing = sorted(list(required - set(df.columns)))
        if missing:
            st.warning(f"data/anime.csv missing columns: {missing}. Using generated sample dataset.")
            return generate_sample_anime_data()

        df["genre"] = df["genre"].fillna("")
        df["type"] = df["type"].fillna("Unknown")
        df["episodes"] = pd.to_numeric(df["episodes"], errors="coerce")
        df["members"] = pd.to_numeric(df["members"], errors="coerce")
        df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
        return df
    except Exception:
        st.info("data/anime.csv not found (or failed). Using generated sample dataset.")
        return generate_sample_anime_data()


anime = load_data()


# -----------------------------
# FAST TF-IDF (no NxN matrix)
# -----------------------------
@st.cache_data
def build_tfidf_matrix(genres: pd.Series):
    tfidf = TfidfVectorizer(stop_words="english", max_features=6000)
    return tfidf.fit_transform(genres)


with st.spinner("Building TF-IDF vectors…"):
    X = build_tfidf_matrix(anime["genre"])


def recommend(anime_name: str, top_n: int = 10) -> pd.DataFrame:
    matches = anime[anime["name"].str.lower() == anime_name.lower()]
    if matches.empty:
        return pd.DataFrame({"Message": ["Anime not found."]})

    idx = matches.index[0]
    sims = cosine_similarity(X[idx], X).flatten()
    sims[idx] = -1
    top_idx = np.argsort(sims)[::-1][:top_n]
    sim_vals = sims[top_idx]

    recs = anime.loc[top_idx, ["name", "genre", "type", "rating", "episodes", "members"]].copy()
    recs["similarity"] = np.round(sim_vals, 3)
    return recs.sort_values("similarity", ascending=False)


def top_trending(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    temp = df.copy()
    temp["rating_f"] = temp["rating"].fillna(0.0)
    temp["members_f"] = temp["members"].fillna(0.0)
    temp["trend_score"] = temp["rating_f"] * np.log10(temp["members_f"] + 1)
    temp = temp.sort_values("trend_score", ascending=False).head(n)
    temp["trend_score"] = temp["trend_score"].round(3)
    return temp[["name", "type", "rating", "members", "trend_score"]]


def shorten_title(title: str, max_len: int = 20) -> str:
    t = str(title).strip()
    return t if len(t) <= max_len else (t[: max_len - 1] + "…")


# -----------------------------
# Session state (SAFE clearing)
# -----------------------------
if "selected_anime" not in st.session_state:
    st.session_state.selected_anime = None

if "clear_search" not in st.session_state:
    st.session_state.clear_search = False

if "clear_trending" not in st.session_state:
    st.session_state.clear_trending = False

if "trend_selected_idx" not in st.session_state:
    st.session_state.trend_selected_idx = None


def request_clear_all():
    st.session_state.selected_anime = None
    st.session_state.clear_search = True
    st.session_state.clear_trending = True
    st.session_state.trend_selected_idx = None
    st.rerun()


def request_clear_trending_only():
    st.session_state.clear_trending = True
    st.session_state.trend_selected_idx = None


def request_clear_search_only():
    st.session_state.clear_search = True


# -----------------------------
# Sidebar: Search (typeahead)
# -----------------------------
st.sidebar.header("🔎 Search Anime")

if st.session_state.clear_search:
    st.session_state["anime_select"] = None
    st.session_state.clear_search = False

anime_names = anime["name"].astype(str).tolist()

chosen_name = st.sidebar.selectbox(
    label="",
    options=anime_names,
    index=None,
    placeholder="Type anime name...",
    key="anime_select",
    label_visibility="collapsed",
)

top_n = st.sidebar.slider("Number of recommendations", 5, 20, 10)

if chosen_name:
    poster_url, mal_url, trailer_url = fetch_media(chosen_name)

    if poster_url:
        st.sidebar.image(poster_url, width=140)

    if mal_url:
        st.sidebar.markdown(f"[Open on MAL]({mal_url})")

    if trailer_url:
        st.sidebar.markdown(f"[▶️ Watch Trailer]({trailer_url})")

    if st.sidebar.button("✅ Use this anime", use_container_width=True):
        request_clear_trending_only()
        request_clear_search_only()
        st.session_state.selected_anime = chosen_name
        st.rerun()


# -----------------------------
# Sidebar: Top 6 Trending (compact, aligned, checkbox below poster)
# -----------------------------
st.sidebar.divider()
st.sidebar.subheader("🔥 Top Trending (Top 6)")

trending6 = top_trending(anime, n=min(6, len(anime)))

if st.session_state.clear_trending:
    for i in range(6):
        st.session_state[f"trend_cb_{i}"] = False
    st.session_state.clear_trending = False

cols = st.sidebar.columns(3, gap="small")


def on_trend_select(i: int, name: str):
    for j in range(6):
        st.session_state[f"trend_cb_{j}"] = (j == i)
    st.session_state.trend_selected_idx = i
    st.session_state.selected_anime = name


for i, row in enumerate(trending6.itertuples(index=False)):
    name = str(row.name)
    poster_url, _, _ = fetch_media(name)
    poster_url = poster_url or "https://via.placeholder.com/300x450?text=No+Img"
    short_name = shorten_title(name, 20)

    with cols[i % 3]:
        st.markdown('<div class="trend-card">', unsafe_allow_html=True)

        st.markdown(
            f"""
            <div class="trend-img" title="{html.escape(name)}">
              <img src="{poster_url}" alt="{html.escape(name)}"/>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div class="trend-title" title="{html.escape(name)}">
              {html.escape(short_name)}
            </div>
            """,
            unsafe_allow_html=True,
        )

        k = f"trend_cb_{i}"
        if k not in st.session_state:
            st.session_state[k] = (st.session_state.trend_selected_idx == i)

        st.checkbox(
            " ",
            key=k,
            on_change=on_trend_select,
            args=(i, name),
            label_visibility="collapsed",
        )

        st.markdown("</div>", unsafe_allow_html=True)


# -----------------------------
# EDA filter (Dashboard only)
# -----------------------------
st.sidebar.divider()
st.sidebar.subheader("📊 Dashboard / EDA filter")
st.sidebar.caption("Affects Dashboard tab only (NOT recommendations).")

type_filter = st.sidebar.multiselect(
    "Filter by type",
    sorted(anime["type"].dropna().unique()),
    default=[]
)

eda_min_rating = st.sidebar.slider("EDA Min rating", 0.0, 10.0, 0.0, 0.1)
eda_max_rating = st.sidebar.slider("EDA Max rating", 0.0, 10.0, 10.0, 0.1)
if eda_min_rating > eda_max_rating:
    eda_min_rating, eda_max_rating = eda_max_rating, eda_min_rating

eda_df = anime.copy()
if type_filter:
    eda_df = eda_df[eda_df["type"].isin(type_filter)]
eda_df = eda_df[eda_df["rating"].notna()]
eda_df = eda_df[(eda_df["rating"] >= eda_min_rating) & (eda_df["rating"] <= eda_max_rating)]


# -----------------------------
# Tabs
# -----------------------------
tab1, tab2 = st.tabs(["✅ Recommend", "📊 Dashboard / EDA"])


# -----------------------------
# Recommend tab
# -----------------------------
with tab1:
    st.subheader("🔥 Top Trending (dataset-based)")
    st.caption("Trending score = rating × log10(members + 1)")
    st.dataframe(top_trending(anime, n=10), use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("📌 Selected Anime + Recommendations")

    if st.session_state.selected_anime is not None:
        if st.button("❌ Clear Selected Anime", use_container_width=True):
            request_clear_all()

    if st.session_state.selected_anime is None:
        st.info("Use sidebar search or pick from Top Trending.")
    else:
        selected_name = st.session_state.selected_anime
        row = anime[anime["name"] == selected_name].iloc[0]

        left, right = st.columns([1, 2])

        with left:
            with st.spinner("Fetching media…"):
                poster_url, mal_url, trailer_url = fetch_media(selected_name)
                
                # If no trailer found, try detailed search
                if not trailer_url:
                    trailer_url = fetch_detailed_trailer(selected_name)

            if poster_url:
                st.image(poster_url, width=260)
                if mal_url:
                    st.markdown(f"[📝 Open on MyAnimeList]({mal_url})")
            else:
                st.info("Poster not found.")

        with right:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Rating", f"{row['rating']:.2f}" if pd.notna(row["rating"]) else "N/A")
            c2.metric("Type", str(row["type"]))
            c3.metric("Episodes", int(row["episodes"]) if pd.notna(row["episodes"]) else "N/A")
            c4.metric("Members", f"{int(row['members']):,}" if pd.notna(row["members"]) else "N/A")
            st.write(f"**Genre:** {row['genre'] if row['genre'] else 'N/A'}")

        # ✅ Video Trailer Section - Now properly embedded!
        st.divider()
        st.subheader("🎬 Trailer / Preview")
        
        # Create a container for the video with some styling
        st.markdown('<div class="video-container">', unsafe_allow_html=True)
        
        if trailer_url:
            # Check if it's a YouTube URL
            if "youtube.com" in trailer_url or "youtu.be" in trailer_url:
                # st.video works perfectly with YouTube URLs
                st.video(trailer_url)
                
                # Add a caption with source
                st.caption("🎥 Official trailer from MyAnimeList")
                
                # Add a direct link as backup
                st.markdown(f"[🔗 Direct YouTube Link]({trailer_url})")
            else:
                # For non-YouTube videos, still try to embed
                st.video(trailer_url)
        else:
            # No trailer found - offer alternatives
            st.info("📺 No official trailer available for this anime")
            
            # Provide YouTube search link
            search_query = f"{selected_name} anime official trailer"
            youtube_search = f"https://www.youtube.com/results?search_query={search_query.replace(' ', '+')}"
            st.markdown(f"[🔍 Search for trailer on YouTube]({youtube_search})")
            
            # Also try to find on MAL
            if mal_url:
                st.markdown(f"[🔗 Check on MyAnimeList for videos]({mal_url})")
        
        st.markdown('</div>', unsafe_allow_html=True)

        st.divider()
        st.subheader("✨ Recommendations (TF-IDF similarity)")
        with st.spinner("Generating recommendations…"):
            recs = recommend(selected_name, top_n=top_n)

        st.dataframe(
            recs,
            use_container_width=True,
            hide_index=True,
            column_config={
                "similarity": st.column_config.ProgressColumn(
                    "Confidence", min_value=0.0, max_value=1.0, format="%.3f"
                )
            },
        )


# -----------------------------
# Dashboard / EDA tab (5 clean graphs + short explanation)
# -----------------------------
with tab2:
    st.subheader("📊 Dashboard / EDA (Clean)")
    st.caption(
        f"Type: {', '.join(type_filter) if type_filter else 'All'} | "
        f"Rating: {eda_min_rating:.1f}–{eda_max_rating:.1f}"
    )

    total_anime = len(eda_df)
    avg_rating = pd.to_numeric(eda_df["rating"], errors="coerce").dropna().mean()
    avg_members = pd.to_numeric(eda_df["members"], errors="coerce").dropna().mean()

    m1, m2, m3 = st.columns(3)
    m1.metric("Anime count", f"{total_anime:,}")
    m2.metric("Avg rating", f"{avg_rating:.2f}" if not np.isnan(avg_rating) else "N/A")
    m3.metric("Avg members", f"{int(avg_members):,}" if not np.isnan(avg_members) else "N/A")

    st.divider()

    # 1) Rating Distribution
    g1, e1 = st.columns([3, 1])
    with g1:
        st.markdown("### Rating Distribution")
        fig = plt.figure(figsize=(8, 4))
        sns.histplot(pd.to_numeric(eda_df["rating"], errors="coerce").dropna(), bins=20, kde=True)
        plt.xlabel("Rating")
        plt.ylabel("Count")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    with e1:
        st.markdown(
            """
            <div class="explain-box">
              Shows how ratings are distributed.<br/>
              The peak is the most common rating range.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.divider()

    # 2) Avg Rating by Type
    g2, e2 = st.columns([3, 1])
    with g2:
        st.markdown("### Average Rating by Type")
        temp = eda_df.dropna(subset=["rating"]).groupby("type")["rating"].mean().sort_values(ascending=False)
        fig = plt.figure(figsize=(8, 4))
        plt.bar(temp.index.astype(str), temp.values)
        plt.xticks(rotation=35, ha="right")
        plt.ylabel("Avg Rating")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    with e2:
        st.markdown(
            """
            <div class="explain-box">
              Compares average rating across formats.<br/>
              Useful to justify type-wise differences.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.divider()

    # 3) Rating vs Members
    g3, e3 = st.columns([3, 1])
    with g3:
        st.markdown("### Rating vs Members (log scale)")
        temp = eda_df.dropna(subset=["rating", "members"]).copy()
        fig = plt.figure(figsize=(8, 4))
        plt.scatter(temp["members"], temp["rating"], alpha=0.55)
        plt.xscale("log")
        plt.xlabel("Members (log scale)")
        plt.ylabel("Rating")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    with e3:
        st.markdown(
            """
            <div class="explain-box">
              Shows popularity vs rating pattern.<br/>
              Log scale helps compare small & huge anime.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.divider()

    # 4) Correlation Heatmap
    g4, e4 = st.columns([3, 1])
    with g4:
        st.markdown("### Correlation Heatmap")
        corr_df = eda_df[["rating", "members", "episodes"]].copy().dropna()
        if len(corr_df) >= 3:
            fig = plt.figure(figsize=(6.5, 4))
            sns.heatmap(corr_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        else:
            st.info("Not enough data after filtering to compute correlations.")
    with e4:
        st.markdown(
            """
            <div class="explain-box">
              Measures relationships between numeric fields.<br/>
              Values near ±1 mean stronger association.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.divider()

    # 5) Top Genres
    g5, e5 = st.columns([3, 1])
    with g5:
        st.markdown("### Top Genres (Count)")
        genres = (
            eda_df["genre"]
            .fillna("")
            .astype(str)
            .str.split(",")
            .explode()
            .str.strip()
        )
        genres = genres[genres != ""]
        top_g = genres.value_counts().head(15)

        if not top_g.empty:
            fig = plt.figure(figsize=(8, 4))
            plt.barh(top_g.index[::-1], top_g.values[::-1])
            plt.xlabel("Count")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        else:
            st.info("No genres available after filtering.")
    with e5:
        st.markdown(
            """
            <div class="explain-box">
              Shows most common genres in the dataset.<br/>
              Helps explain dataset bias toward certain genres.
            </div>
            """,
            unsafe_allow_html=True,
        )
