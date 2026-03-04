import html
import re
import requests
import os
import random
from collections import Counter

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# YouTube API imports
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError


# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="AnimeVerse Recommender",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)


# -----------------------------
# YouTube API Setup (unchanged)
# -----------------------------
def get_youtube_api_key():
    try:
        return st.secrets["youtube_api_key"]
    except:
        return None

YOUTUBE_API_KEY = get_youtube_api_key()


@st.cache_data(ttl=24 * 3600)
def search_youtube_trailer(anime_name):
    if not YOUTUBE_API_KEY:
        return None
    try:
        youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY, cache_discovery=False)
        search_queries = [
            f"{anime_name} official trailer",
            f"{anime_name} anime trailer",
            f"{anime_name} PV",
            anime_name
        ]
        for query in search_queries:
            search_response = youtube.search().list(
                q=query,
                part='id',
                type='video',
                maxResults=1,
                videoEmbeddable='true'
            ).execute()
            items = search_response.get('items', [])
            if items:
                video_id = items[0]['id']['videoId']
                return f"https://www.youtube.com/watch?v={video_id}"
        return None
    except Exception as e:
        print(f"YouTube search error: {e}")
        return None


@st.cache_data(ttl=24 * 3600)
def fetch_youtube_trailer_fallback(anime_name):
    try:
        clean_name = re.sub(r'[^\w\s-]', '', anime_name)
        clean_name = clean_name.strip().replace(' ', '+')
        search_patterns = [
            f"{clean_name}+official+trailer",
            f"{clean_name}+anime+trailer",
            f"{clean_name}+pv",
            clean_name
        ]
        headers = {'User-Agent': 'Mozilla/5.0'}
        for pattern in search_patterns:
            search_url = f"https://www.youtube.com/results?search_query={pattern}"
            response = requests.get(search_url, headers=headers, timeout=10)
            if response.status_code == 200:
                video_pattern = r'"videoId":"([a-zA-Z0-9_-]{11})"'
                matches = re.findall(video_pattern, response.text)
                if matches:
                    return f"https://www.youtube.com/watch?v={matches[0]}"
        return None
    except Exception as e:
        print(f"Fallback error: {e}")
        return None


# -----------------------------
# ENHANCED CREATIVE CSS (with animations and neon accents)
# -----------------------------
def apply_custom_css(dark_mode):
    base_css = """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700;800&family=Righteous&display=swap');
        
        * { font-family: 'Poppins', sans-serif; }
        
        /* Animated gradient background */
        .stApp {
            background: linear-gradient(-45deg, #ee7752, #e73c7e, #23a6d5, #23d5ab);
            background-size: 400% 400%;
            animation: gradient 15s ease infinite;
            position: relative;
        }
        .stApp::before {
            content: "";
            position: absolute;
            top: 0; left: 0; right: 0; bottom: 0;
            background: rgba(0,0,0,0.6);
            pointer-events: none;
        }
        @keyframes gradient {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }
        
        /* Glassmorphism containers */
        .glass-card {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border: 1px solid rgba(255,255,255,0.2);
            border-radius: 30px;
            padding: 1.5rem;
            box-shadow: 0 15px 35px rgba(0,0,0,0.3);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        .glass-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 20px 40px rgba(0,0,0,0.4);
            border-color: rgba(255,215,0,0.5);
        }
        
        /* Neon text effect */
        .neon-text {
            text-shadow: 0 0 5px #fff, 0 0 10px #fff, 0 0 20px #ff00de, 0 0 30px #ff00de;
        }
        
        /* Title with glow */
        h1 {
            font-family: 'Righteous', cursive;
            font-size: 4rem !important;
            background: linear-gradient(135deg, #ffe6b0, #ffb347, #ff6b6b);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            animation: titleGlow 3s ease-in-out infinite;
        }
        @keyframes titleGlow {
            0% { filter: drop-shadow(0 0 5px #ffb347); }
            50% { filter: drop-shadow(0 0 20px #ff6b6b); }
            100% { filter: drop-shadow(0 0 5px #ffb347); }
        }
        
        /* Sidebar glass */
        section[data-testid="stSidebar"] {
            background: rgba(10, 14, 23, 0.8) !important;
            backdrop-filter: blur(16px) saturate(180%);
            -webkit-backdrop-filter: blur(16px) saturate(180%);
            border-right: 1px solid rgba(255,215,0,0.2);
        }
        
        /* Cards for recommendations */
        .rec-card {
            background: rgba(20, 25, 45, 0.7);
            backdrop-filter: blur(10px);
            border-radius: 24px;
            padding: 15px;
            border: 1px solid rgba(255,215,0,0.2);
            transition: all 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275);
            height: 100%;
            display: flex;
            flex-direction: column;
        }
        .rec-card:hover {
            transform: scale(1.03) translateY(-8px);
            border-color: gold;
            box-shadow: 0 20px 30px -10px rgba(255,215,0,0.3);
        }
        .rec-img {
            width: 100%;
            height: 200px;
            object-fit: cover;
            border-radius: 18px;
            margin-bottom: 12px;
            border: 2px solid rgba(255,215,0,0.3);
            transition: border 0.3s;
        }
        .rec-card:hover .rec-img {
            border-color: gold;
        }
        .rec-title {
            font-weight: 700;
            font-size: 1rem;
            color: white;
            margin-bottom: 5px;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .rec-similarity {
            color: #ffd966;
            font-size: 0.9rem;
            font-weight: 600;
            margin-top: auto;
        }
        
        /* Button styling */
        .stButton > button {
            background: linear-gradient(135deg, #667eea, #764ba2);
            border: none;
            border-radius: 50px;
            color: white;
            font-weight: 600;
            padding: 10px 30px;
            transition: all 0.3s;
            box-shadow: 0 10px 20px rgba(0,0,0,0.2);
        }
        .stButton > button:hover {
            transform: translateY(-3px);
            box-shadow: 0 15px 30px rgba(102,126,234,0.4);
        }
        
        /* Metric cards */
        div[data-testid="stMetric"] {
            background: rgba(20, 25, 45, 0.6);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,215,0,0.2);
            border-radius: 30px !important;
            padding: 15px !important;
            transition: transform 0.3s;
        }
        div[data-testid="stMetric"]:hover {
            transform: translateY(-5px);
            border-color: gold;
        }
        
        /* Progress column styling */
        .stProgress > div > div {
            background-color: gold !important;
        }
    </style>
    """
    st.markdown(base_css, unsafe_allow_html=True)


# -----------------------------
# Data loading (unchanged)
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
        st.info("data/anime.csv not found. Using generated sample dataset.")
        return generate_sample_anime_data()


anime = load_data()


# -----------------------------
# TF-IDF (unchanged)
# -----------------------------
@st.cache_data
def build_tfidf_matrix(genres: pd.Series):
    tfidf = TfidfVectorizer(stop_words="english", max_features=6000)
    return tfidf.fit_transform(genres)


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
# Media fetch (unchanged)
# -----------------------------
@st.cache_data(ttl=24 * 3600)
def fetch_media(anime_name):
    poster = None
    mal_url = None
    trailer_url = None
    
    try:
        url = "https://api.jikan.moe/v4/anime"
        r = requests.get(url, params={"q": anime_name, "limit": 3}, timeout=10)
        
        if r.status_code == 200:
            data = r.json().get("data", [])
            if data:
                poster = data[0].get("images", {}).get("jpg", {}).get("image_url")
                mal_url = data[0].get("url")
                
                for item in data:
                    trailer = item.get("trailer") or {}
                    youtube_id = trailer.get("youtube_id")
                    if youtube_id:
                        trailer_url = f"https://www.youtube.com/watch?v={youtube_id}"
                        break
                    elif not trailer_url:
                        trailer_url = trailer.get("url")
        
        if not trailer_url and YOUTUBE_API_KEY:
            trailer_url = search_youtube_trailer(anime_name)
        
        if not trailer_url:
            trailer_url = fetch_youtube_trailer_fallback(anime_name)
        
        return poster, mal_url, trailer_url
        
    except Exception as e:
        print(f"Error fetching media: {e}")
        return poster, mal_url, fetch_youtube_trailer_fallback(anime_name)


# -----------------------------
# Session state
# -----------------------------
if "selected_anime" not in st.session_state:
    st.session_state.selected_anime = None
if "clear_search" not in st.session_state:
    st.session_state.clear_search = False
if "clear_trending" not in st.session_state:
    st.session_state.clear_trending = False
if "trend_selected_idx" not in st.session_state:
    st.session_state.trend_selected_idx = None
if "watchlist" not in st.session_state:
    st.session_state.watchlist = []


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
# Apply CSS based on dark mode toggle (we keep dark mode as default but can be toggled)
# -----------------------------
dark_mode = st.sidebar.checkbox("🌙 Dark Mode", value=True)
apply_custom_css(dark_mode)


# -----------------------------
# Header with floating animation
# -----------------------------
st.markdown(
    """
    <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 20px;">
        <h1>✨ AnimeVerse</h1>
        <div style="font-size: 1.5rem; background: rgba(255,255,255,0.1); padding: 8px 25px; border-radius: 60px; backdrop-filter: blur(10px); border:1px solid gold;">
            🎌 content‑based recommender
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# -----------------------------
# Sidebar: Search + Trending + New Random button
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

# NEW: Random anime picker
if st.sidebar.button("🎲 Random Anime", use_container_width=True):
    random_anime = random.choice(anime_names)
    st.session_state.selected_anime = random_anime
    st.session_state.clear_trending = True
    st.rerun()

if chosen_name:
    poster_url, mal_url, trailer_url = fetch_media(chosen_name)

    if poster_url:
        st.sidebar.image(poster_url, width=140)
    if mal_url:
        st.sidebar.markdown(f"[![MAL](https://img.icons8.com/color/24/000000/myanimelist.png) Open on MAL]({mal_url})")
    if trailer_url:
        st.sidebar.markdown(f"[▶️ Watch Trailer]({trailer_url})")

    if st.sidebar.button("✅ Use this anime", use_container_width=True):
        request_clear_trending_only()
        request_clear_search_only()
        st.session_state.selected_anime = chosen_name
        st.rerun()


# -----------------------------
# Sidebar: Top 6 Trending (cards)
# -----------------------------
st.sidebar.divider()
st.sidebar.subheader("🔥 Top Trending")

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
        st.markdown(
            f"""
            <div class="trend-card" style="cursor:pointer;" onclick="document.getElementById('trend_cb_{i}').click()">
                <div class="trend-img">
                    <img src="{poster_url}" alt="{html.escape(name)}">
                </div>
                <div class="trend-title">{html.escape(short_name)}</div>
                <div style="font-size:0.7rem; color:gold;">⭐ {row.rating:.2f}</div>
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


# -----------------------------
# Sidebar: EDA filters (unchanged)
# -----------------------------
st.sidebar.divider()
st.sidebar.subheader("📊 Dashboard Filters")

type_filter = st.sidebar.multiselect(
    "Type",
    sorted(anime["type"].dropna().unique()),
    default=[]
)

eda_min_rating = st.sidebar.slider("Min rating", 0.0, 10.0, 0.0, 0.1)
eda_max_rating = st.sidebar.slider("Max rating", 0.0, 10.0, 10.0, 0.1)
if eda_min_rating > eda_max_rating:
    eda_min_rating, eda_max_rating = eda_max_rating, eda_min_rating

eda_df = anime.copy()
if type_filter:
    eda_df = eda_df[eda_df["type"].isin(type_filter)]
eda_df = eda_df[eda_df["rating"].notna()]
eda_df = eda_df[(eda_df["rating"] >= eda_min_rating) & (eda_df["rating"] <= eda_max_rating)]


# -----------------------------
# NEW: Watchlist feature
# -----------------------------
st.sidebar.divider()
st.sidebar.subheader("📋 My Watchlist")
if st.session_state.watchlist:
    for anime_name in st.session_state.watchlist:
        col1, col2 = st.sidebar.columns([3, 1])
        col1.write(f"• {anime_name}")
        if col2.button("❌", key=f"remove_{anime_name}"):
            st.session_state.watchlist.remove(anime_name)
            st.rerun()
else:
    st.sidebar.caption("Empty. Add from recommendations.")


# -----------------------------
# Tabs: Recommend, Dashboard, Explore (new)
# -----------------------------
tab1, tab2, tab3 = st.tabs(["✨ Recommend", "📈 Analytics", "🔍 Genre Explorer"])


# -----------------------------
# Tab 1: Recommend (enhanced)
# -----------------------------
with tab1:
    st.subheader("🔥 Top Trending (dataset-based)")
    st.caption("Trending score = rating × log10(members + 1)")
    st.dataframe(top_trending(anime, n=10), use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("📌 Selected Anime")

    if st.session_state.selected_anime is not None:
        if st.button("❌ Clear Selection", use_container_width=True):
            request_clear_all()

    if st.session_state.selected_anime is None:
        st.info("👀 Use sidebar search, pick from Trending, or click **Random Anime**.")
    else:
        selected_name = st.session_state.selected_anime
        row = anime[anime["name"] == selected_name].iloc[0]

        # Display selected anime in a glass card
        with st.container():
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            left, right = st.columns([1, 2])

            with left:
                with st.spinner("Fetching media…"):
                    poster_url, mal_url, trailer_url = fetch_media(selected_name)

                if poster_url:
                    st.image(poster_url, width=260, caption=selected_name)
                else:
                    st.info("Poster not found.")

                if mal_url:
                    st.markdown(f"[📝 Open on MyAnimeList]({mal_url})")

                # Add to watchlist button
                if st.button("➕ Add to Watchlist", key="add_watchlist"):
                    if selected_name not in st.session_state.watchlist:
                        st.session_state.watchlist.append(selected_name)
                        st.success("Added!")
                        st.rerun()

            with right:
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Rating", f"{row['rating']:.2f}" if pd.notna(row["rating"]) else "N/A")
                c2.metric("Type", str(row["type"]))
                c3.metric("Episodes", int(row["episodes"]) if pd.notna(row["episodes"]) else "N/A")
                c4.metric("Members", f"{int(row['members']):,}" if pd.notna(row["members"]) else "N/A")
                st.write(f"**Genre:** {row['genre'] if row['genre'] else 'N/A'}")

                # Mini genre distribution (placeholder)
                if row['genre']:
                    genres = [g.strip() for g in row['genre'].split(',')]
                    st.markdown("**Genre tags:**")
                    cols = st.columns(len(genres))
                    for i, g in enumerate(genres):
                        cols[i].markdown(f"<span style='background:rgba(255,215,0,0.2); padding:3px 10px; border-radius:30px;'>{g}</span>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # Trailer section
        st.divider()
        st.subheader("🎬 Trailer / Preview")
        st.markdown('<div class="video-container">', unsafe_allow_html=True)
        if trailer_url and ("youtube.com" in trailer_url or "youtu.be" in trailer_url):
            st.video(trailer_url)
            st.caption("🎥 Found via YouTube API" if YOUTUBE_API_KEY else "🎥 Found via YouTube search")
        else:
            st.info("📺 No trailer automatically found")
            col1, col2 = st.columns(2)
            with col1:
                search_query = f"{selected_name} anime official trailer"
                youtube_search = f"https://www.youtube.com/results?search_query={search_query.replace(' ', '+')}"
                st.markdown(f"[🔍 Search YouTube]({youtube_search})")
            with col2:
                if mal_url:
                    st.markdown(f"[📝 Check MyAnimeList]({mal_url})")
        st.markdown('</div>', unsafe_allow_html=True)

        # Recommendations as beautiful cards
        st.divider()
        st.subheader("✨ Recommendations (TF-IDF similarity)")
        with st.spinner("Generating recommendations…"):
            recs = recommend(selected_name, top_n=top_n)

        # Display recs as a grid of cards
        cols_per_row = 3
        for i in range(0, len(recs), cols_per_row):
            cols = st.columns(cols_per_row)
            for j in range(cols_per_row):
                idx = i + j
                if idx < len(recs):
                    rec = recs.iloc[idx]
                    with cols[j]:
                        rec_poster, _, _ = fetch_media(rec['name'])
                        rec_poster = rec_poster or "https://via.placeholder.com/300x450?text=No+Img"
                        st.markdown(
                            f"""
                            <div class="rec-card">
                                <img src="{rec_poster}" class="rec-img">
                                <div class="rec-title">{html.escape(shorten_title(rec['name'], 30))}</div>
                                <div style="display:flex; justify-content:space-between; margin:8px 0;">
                                    <span>⭐ {rec['rating']:.2f}</span>
                                    <span>{rec['type']}</span>
                                </div>
                                <div class="rec-similarity">✨ {rec['similarity']:.2f} match</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                        if st.button("Select", key=f"select_rec_{idx}"):
                            st.session_state.selected_anime = rec['name']
                            st.rerun()


# -----------------------------
# Tab 2: Dashboard (enhanced with wordcloud)
# -----------------------------
with tab2:
    st.subheader("📊 Anime Analytics")
    st.caption(f"Type: {', '.join(type_filter) if type_filter else 'All'} | Rating: {eda_min_rating:.1f}–{eda_max_rating:.1f}")

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
        fig, ax = plt.subplots(facecolor='none')
        ax.set_facecolor('none')
        sns.histplot(pd.to_numeric(eda_df["rating"], errors="coerce").dropna(), bins=20, kde=True, color='gold', ax=ax)
        ax.set_xlabel("Rating", color='white')
        ax.set_ylabel("Count", color='white')
        ax.tick_params(colors='white')
        st.pyplot(fig)
    with e1:
        st.markdown('<div class="explain-box">🌟 Shows how ratings are distributed. The peak is the most common rating range.</div>', unsafe_allow_html=True)

    st.divider()

    # 2) Avg Rating by Type
    g2, e2 = st.columns([3, 1])
    with g2:
        st.markdown("### Average Rating by Type")
        temp = eda_df.dropna(subset=["rating"]).groupby("type")["rating"].mean().sort_values(ascending=False)
        fig, ax = plt.subplots(facecolor='none')
        ax.set_facecolor('none')
        ax.bar(temp.index.astype(str), temp.values, color='#ffb347', edgecolor='gold')
        ax.set_xticklabels(temp.index, rotation=35, ha="right", color='white')
        ax.set_ylabel("Avg Rating", color='white')
        ax.tick_params(colors='white')
        st.pyplot(fig)
    with e2:
        st.markdown('<div class="explain-box">📺 Compares average rating across formats. Useful to see which types are best rated.</div>', unsafe_allow_html=True)

    st.divider()

    # 3) Rating vs Members
    g3, e3 = st.columns([3, 1])
    with g3:
        st.markdown("### Rating vs Members (log scale)")
        temp = eda_df.dropna(subset=["rating", "members"]).copy()
        fig, ax = plt.subplots(facecolor='none')
        ax.set_facecolor('none')
        ax.scatter(temp["members"], temp["rating"], alpha=0.65, color='#6a5acd', edgecolor='white', s=30)
        ax.set_xscale("log")
        ax.set_xlabel("Members (log scale)", color='white')
        ax.set_ylabel("Rating", color='white')
        ax.tick_params(colors='white')
        st.pyplot(fig)
    with e3:
        st.markdown('<div class="explain-box">📈 Shows popularity vs rating pattern. Log scale helps compare small & huge anime.</div>', unsafe_allow_html=True)

    st.divider()

    # 4) Correlation Heatmap
    g4, e4 = st.columns([3, 1])
    with g4:
        st.markdown("### Correlation Heatmap")
        corr_df = eda_df[["rating", "members", "episodes"]].copy().dropna()
        if len(corr_df) >= 3:
            fig, ax = plt.subplots(facecolor='none')
            ax.set_facecolor('none')
            sns.heatmap(corr_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", cbar_kws={'label': 'Correlation'}, ax=ax)
            ax.tick_params(colors='white')
            st.pyplot(fig)
        else:
            st.info("Not enough data after filtering to compute correlations.")
    with e4:
        st.markdown('<div class="explain-box">🔗 Measures relationships between numeric fields. Values near ±1 mean stronger association.</div>', unsafe_allow_html=True)

    st.divider()

    # 5) Top Genres (now with word cloud)
    g5, e5 = st.columns([3, 1])
    with g5:
        st.markdown("### Genre Word Cloud")
        genres_series = (
            eda_df["genre"]
            .fillna("")
            .astype(str)
            .str.split(",")
            .explode()
            .str.strip()
        )
        genres_series = genres_series[genres_series != ""]
        if not genres_series.empty:
            wordcloud = WordCloud(
                width=800, height=400,
                background_color=None,
                mode="RGBA",
                colormap='viridis',
                prefer_horizontal=0.5
            ).generate(' '.join(genres_series))
            fig, ax = plt.subplots(facecolor='none')
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis("off")
            st.pyplot(fig)
        else:
            st.info("No genres available after filtering.")
    with e5:
        st.markdown('<div class="explain-box">🎭 Word cloud of most frequent genres. Bigger words appear more often.</div>', unsafe_allow_html=True)


# -----------------------------
# Tab 3: Genre Explorer (NEW)
# -----------------------------
with tab3:
    st.subheader("🔍 Explore by Genre")
    st.caption("Select multiple genres to discover top anime")

    # Get all unique genres
    all_genres = sorted(set(
        genre.strip() for sublist in anime["genre"].dropna().str.split(',') for genre in sublist
    ))

    selected_genres = st.multiselect("Choose genres", all_genres, default=["Action", "Comedy"])

    if selected_genres:
        # Filter anime that contain ALL selected genres (strict) or ANY? Let's do ANY for broader results
        mask = anime["genre"].apply(
            lambda x: any(genre in str(x) for genre in selected_genres) if pd.notna(x) else False
        )
        filtered = anime[mask].copy()
        filtered = filtered.dropna(subset=["rating"]).sort_values("rating", ascending=False).head(20)

        if not filtered.empty:
            st.dataframe(
                filtered[["name", "type", "rating", "members", "genre"]],
                use_container_width=True,
                hide_index=True
            )
            # Show a small bar chart of ratings
            fig, ax = plt.subplots(figsize=(10, 5), facecolor='none')
            ax.set_facecolor('none')
            ax.barh(filtered["name"].apply(lambda x: shorten_title(x, 30)), filtered["rating"], color='gold')
            ax.set_xlabel("Rating", color='white')
            ax.tick_params(colors='white')
            st.pyplot(fig)
        else:
            st.warning("No anime found with selected genres.")


# -----------------------------
# Footer
# -----------------------------
st.divider()
st.markdown(
    "<p style='text-align: center; color: #9ca3af; font-size: 0.9rem;'>✨ crafted with 💜 for anime lovers • TF‑IDF + Cosine Similarity • now with extra sparkle ✨</p>",
    unsafe_allow_html=True,
)
