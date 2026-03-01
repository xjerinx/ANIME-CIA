import html
import re
import requests
import os

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from difflib import SequenceMatcher

# YouTube API imports
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError


# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="Anime Recommender", page_icon="🎌", layout="wide")


# -----------------------------
# YouTube API Setup
# -----------------------------
def get_youtube_api_key():
    """Get YouTube API key from Streamlit secrets"""
    try:
        return st.secrets["youtube_api_key"]
    except:
        return None

# Initialize YouTube API key
YOUTUBE_API_KEY = get_youtube_api_key()


# -----------------------------
# YouTube API Functions
# -----------------------------
@st.cache_data(ttl=24 * 3600)
def search_youtube_trailer(anime_name):
    """Search YouTube for anime trailer using API"""
    if not YOUTUBE_API_KEY:
        return None
    
    try:
        # Build YouTube API client
        youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY, cache_discovery=False)
        
        # Try multiple search queries
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
    """Fallback method using web scraping (no API key needed)"""
    try:
        # Clean the anime name for URL
        clean_name = re.sub(r'[^\w\s-]', '', anime_name)
        clean_name = clean_name.strip().replace(' ', '+')
        
        # Try different search patterns
        search_patterns = [
            f"{clean_name}+official+trailer",
            f"{clean_name}+anime+trailer",
            f"{clean_name}+pv",
            clean_name
        ]
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        for pattern in search_patterns:
            search_url = f"https://www.youtube.com/results?search_query={pattern}"
            response = requests.get(search_url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                # Look for video IDs in the response
                video_pattern = r'"videoId":"([a-zA-Z0-9_-]{11})"'
                matches = re.findall(video_pattern, response.text)
                
                if matches:
                    return f"https://www.youtube.com/watch?v={matches[0]}"
        
        return None
    except Exception as e:
        print(f"Fallback error: {e}")
        return None


# -----------------------------
# Media fetch (Poster + MAL + Trailer) via Jikan API
# -----------------------------
@st.cache_data(ttl=24 * 3600)
def fetch_media(anime_name):
    """
    Returns: (poster_url, mal_url, trailer_url)
    - First tries MAL, then YouTube API, then fallback
    """
    poster = None
    mal_url = None
    trailer_url = None
    
    try:
        # Try MAL first
        url = "https://api.jikan.moe/v4/anime"
        r = requests.get(url, params={"q": anime_name, "limit": 3}, timeout=10)
        
        if r.status_code == 200:
            data = r.json().get("data", [])
            if data:
                # Get poster and MAL URL from first result
                poster = data[0].get("images", {}).get("jpg", {}).get("image_url")
                mal_url = data[0].get("url")
                
                # Check for trailer in all results
                for item in data:
                    trailer = item.get("trailer") or {}
                    youtube_id = trailer.get("youtube_id")
                    if youtube_id:
                        trailer_url = f"https://www.youtube.com/watch?v={youtube_id}"
                        break
                    elif not trailer_url:
                        trailer_url = trailer.get("url")
        
        # If MAL didn't have trailer, try YouTube API
        if not trailer_url and YOUTUBE_API_KEY:
            trailer_url = search_youtube_trailer(anime_name)
        
        # If still no trailer, try fallback
        if not trailer_url:
            trailer_url = fetch_youtube_trailer_fallback(anime_name)
        
        return poster, mal_url, trailer_url
        
    except Exception as e:
        print(f"Error fetching media: {e}")
        # Last resort - try fallback
        return poster, mal_url, fetch_youtube_trailer_fallback(anime_name)


# -----------------------------
# CLEAN UI THEMES - Professional, No Excessive Colors
# -----------------------------
dark_mode = st.sidebar.checkbox("🌙 Dark Mode", value=True)

# Base CSS that applies to both themes
BASE_CSS = """
<style>
  /* Import Google Fonts */
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
  
  * {
    font-family: 'Inter', sans-serif;
  }
  
  /* Smooth scrolling */
  html {
    scroll-behavior: smooth;
  }
  
  /* Sidebar styling */
  section[data-testid="stSidebar"]{
    background: #0a0e1a !important;
    border-right: 1px solid rgba(255,255,255,0.05) !important;
  }
  section[data-testid="stSidebar"] *{ color:#e5e7eb !important; }
  section[data-testid="stSidebar"] .stCaption,
  section[data-testid="stSidebar"] p,
  section[data-testid="stSidebar"] small{
    color:#9ca3af !important;
  }

  /* Sidebar input styling */
  section[data-testid="stSidebar"] div[data-baseweb="input"] input{
    background-color: #1f2937 !important;
    color:#e5e7eb !important;
    border: 1px solid #374151 !important;
    border-radius: 8px !important;
    padding: 8px 12px !important;
    font-size: 14px !important;
  }
  section[data-testid="stSidebar"] div[data-baseweb="input"] input:focus{
    border-color: #60a5fa !important;
    box-shadow: 0 0 0 2px rgba(96, 165, 250, 0.2) !important;
  }

  /* Trending cards */
  .trend-card{
    background: #1f2937;
    border: 1px solid #374151;
    border-radius: 8px;
    padding: 8px;
    margin-bottom: 12px;
    transition: all 0.2s;
  }
  .trend-card:hover{
    background: #2d3748;
  }
  
  .trend-img{
    width:100%;
    height:120px;
    border-radius:6px;
    overflow:hidden;
    background: #2d3748;
  }
  .trend-img img{
    width:100%;
    height:120px;
    object-fit:cover;
    display:block;
  }
  .trend-title{
    font-size: 0.8rem;
    font-weight: 500;
    color: #e5e7eb;
    text-align:center;
    white-space: nowrap;
    overflow:hidden;
    text-overflow: ellipsis;
    margin-top: 8px;
  }
  
  /* Video container */
  .video-container {
    border-radius: 8px;
    overflow: hidden;
    margin: 20px 0;
  }
  
  /* Explanation box */
  .explain-box{
    background: #f3f4f6;
    border: 1px solid #e5e7eb;
    border-radius: 8px;
    padding: 16px;
    color: #4b5563;
    font-size: 0.9rem;
    height: 100%;
  }
  
  /* Genre tags */
  .genre-tag {
    display: inline-block;
    background: #f3f4f6;
    border: 1px solid #e5e7eb;
    border-radius: 16px;
    padding: 4px 12px;
    margin: 0 4px 4px 0;
    font-size: 0.8rem;
    color: #4b5563;
  }
  
  /* Divider styling */
  .custom-divider {
    margin: 30px 0;
    border: none;
    height: 1px;
    background: #e5e7eb;
  }
  
  /* Button styling */
  .stButton button {
    border-radius: 8px !important;
    border: 1px solid #d1d5db !important;
    background: white !important;
    color: #1f2937 !important;
    font-weight: 500 !important;
    font-size: 14px !important;
    padding: 8px 16px !important;
    transition: all 0.2s !important;
  }
  .stButton button:hover {
    background: #f9fafb !important;
    border-color: #9ca3af !important;
  }
  
  /* Selectbox styling */
  div[data-baseweb="select"] div {
    border-radius: 8px !important;
    border: 1px solid #d1d5db !important;
  }
  
  /* Slider styling */
  div[data-testid="stSlider"] label {
    color: #6b7280 !important;
  }
  
  /* Tabs styling */
  .stTabs [data-baseweb="tab-list"] {
    gap: 8px;
  }
  .stTabs [data-baseweb="tab"] {
    border-radius: 8px !important;
    padding: 8px 16px !important;
    font-weight: 500 !important;
  }
  
  /* DataFrame styling */
  .stDataFrame, [data-testid="stDataFrame"] {
    border-radius: 8px !important;
    border: 1px solid #e5e7eb !important;
  }
</style>
"""

# Dark Mode Theme
if dark_mode:
    st.markdown(
        """
        <style>
          .stApp {
            background: #111827;
            color: #e5e7eb;
          }
          
          .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
          }

          h1, h2, h3, h4, h5, h6 {
            color: #e5e7eb;
            font-weight: 600;
          }
          
          h1 {
            color: #60a5fa !important;
          }
          
          .stCaption, p, li {
            color: #9ca3af;
          }
          
          a {
            color: #60a5fa;
            text-decoration: none;
          }
          
          a:hover {
            text-decoration: underline;
          }
          
          div[data-testid="stMetric"] {
            background: #1f2937;
            border: 1px solid #374151;
            border-radius: 8px;
            padding: 16px;
          }
          
          div[data-testid="stMetric"] label {
            color: #9ca3af !important;
          }
          
          div[data-testid="stMetric"] div {
            color: #e5e7eb !important;
            font-weight: 600 !important;
          }
          
          /* Light mode overrides for dark mode */
          .explain-box {
            background: #1f2937;
            border-color: #374151;
            color: #9ca3af;
          }
          
          .genre-tag {
            background: #1f2937;
            border-color: #374151;
            color: #9ca3af;
          }
          
          .custom-divider {
            background: #374151;
          }
          
          .stButton button {
            background: #1f2937 !important;
            border-color: #374151 !important;
            color: #e5e7eb !important;
          }
          .stButton button:hover {
            background: #2d3748 !important;
            border-color: #4b5563 !important;
          }
          
          .stDataFrame, [data-testid="stDataFrame"] {
            border-color: #374151 !important;
            background: #1f2937 !important;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )

# Light Mode Theme - Clean and Professional
else:
    st.markdown(
        """
        <style>
          .stApp {
            background: #ffffff;
            color: #1f2937;
          }
          
          .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
          }

          h1, h2, h3, h4, h5, h6 {
            color: #1f2937;
            font-weight: 600;
          }
          
          h1 {
            color: #2563eb !important;
          }
          
          .stCaption, p, li {
            color: #6b7280;
          }
          
          a {
            color: #2563eb;
            text-decoration: none;
          }
          
          a:hover {
            text-decoration: underline;
          }
          
          div[data-testid="stMetric"] {
            background: #f9fafb;
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            padding: 16px;
          }
          
          div[data-testid="stMetric"] label {
            color: #6b7280 !important;
          }
          
          div[data-testid="stMetric"] div {
            color: #1f2937 !important;
            font-weight: 600 !important;
          }
          
          /* Explanation box in light mode */
          .explain-box {
            background: #f9fafb;
            border-color: #e5e7eb;
            color: #6b7280;
          }
          
          /* Genre tags in light mode */
          .genre-tag {
            background: #f9fafb;
            border-color: #e5e7eb;
            color: #6b7280;
          }
          
          /* Custom scrollbar */
          ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
          }
          
          ::-webkit-scrollbar-track {
            background: #f3f4f6;
          }
          
          ::-webkit-scrollbar-thumb {
            background: #9ca3af;
            border-radius: 4px;
          }
          
          ::-webkit-scrollbar-thumb:hover {
            background: #6b7280;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )

# Apply base CSS
st.markdown(BASE_CSS, unsafe_allow_html=True)

# Show API status in sidebar
if YOUTUBE_API_KEY:
    st.sidebar.success("✅ YouTube API Connected")
else:
    st.sidebar.warning("⚠️ YouTube API key not found. Using basic search.")


# -----------------------------
# Header
# -----------------------------
st.markdown("""
    <div style='text-align: center; margin-bottom: 30px;'>
        <h1 style='font-size: 2.5rem; margin-bottom: 5px;'>🎌 ANIME RECOMMENDER</h1>
        <p style='color: #6b7280; font-size: 1rem;'>Discover your next favorite anime</p>
    </div>
""", unsafe_allow_html=True)


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
# FAST TF-IDF
# -----------------------------
@st.cache_data
def build_tfidf_matrix(genres: pd.Series):
    tfidf = TfidfVectorizer(stop_words="english", max_features=6000)
    return tfidf.fit_transform(genres)


with st.spinner("Building recommendation engine..."):
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
# Sidebar: Search
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
    poster_url, _, _ = fetch_media(chosen_name)

    if poster_url:
        st.sidebar.image(poster_url, width=140)

    if st.sidebar.button("✅ Use this anime", use_container_width=True):
        request_clear_trending_only()
        request_clear_search_only()
        st.session_state.selected_anime = chosen_name
        st.rerun()


# -----------------------------
# Sidebar: Top 6 Trending
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
# EDA filter
# -----------------------------
st.sidebar.divider()
st.sidebar.subheader("📊 Dashboard Filters")
st.sidebar.caption("Affects Analytics tab only")

type_filter = st.sidebar.multiselect(
    "Filter by type",
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
# Tabs
# -----------------------------
tab1, tab2 = st.tabs(["🎯 Recommendations", "📊 Analytics"])


# -----------------------------
# Recommendations Tab
# -----------------------------
with tab1:
    # Trending section
    st.subheader("🔥 Trending Now")
    st.caption("Based on rating × popularity score")
    
    trending_df = top_trending(anime, n=10)
    st.dataframe(
        trending_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "name": "Anime Title",
            "type": "Format",
            "rating": st.column_config.NumberColumn("Rating", format="%.2f"),
            "members": st.column_config.NumberColumn("Members", format="%d"),
            "trend_score": st.column_config.NumberColumn("Trend Score", format="%.3f")
        }
    )

    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)
    
    st.subheader("📌 Selected Anime")

    if st.session_state.selected_anime is not None:
        if st.button("❌ Clear Selection", use_container_width=True):
            request_clear_all()

    if st.session_state.selected_anime is None:
        st.info("👆 Select an anime from the sidebar to get recommendations")
    else:
        selected_name = st.session_state.selected_anime
        row = anime[anime["name"] == selected_name].iloc[0]

        # Display anime title
        st.markdown(f"<h3>{selected_name}</h3>", unsafe_allow_html=True)
        
        # Create two columns for poster and info
        left, right = st.columns([1, 2])

        with left:
            with st.spinner("Loading..."):
                poster_url, mal_url, trailer_url = fetch_media(selected_name)

            if poster_url:
                st.image(poster_url, width=260)
            else:
                st.info("📷 Poster not available")

        with right:
            # Metrics display
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Rating", f"{row['rating']:.2f}" if pd.notna(row["rating"]) else "N/A")
            
            with col2:
                st.metric("Type", str(row["type"]))
            
            with col3:
                ep_count = int(row["episodes"]) if pd.notna(row["episodes"]) else "N/A"
                st.metric("Episodes", ep_count)
            
            with col4:
                members_count = f"{int(row['members']):,}" if pd.notna(row["members"]) else "N/A"
                st.metric("Members", members_count)
            
            # Genre tags
            st.markdown("**Genres**")
            if row['genre'] and row['genre'] != "":
                genres = row['genre'].split(', ')
                genre_html = ""
                for genre in genres:
                    genre_html += f"<span class='genre-tag'>{genre}</span>"
                st.markdown(genre_html, unsafe_allow_html=True)
            else:
                st.markdown("*No genre data available*")
            
            # Links
            if mal_url:
                st.markdown(f"[View on MyAnimeList]({mal_url})")

        # Video Trailer Section
        st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)
        st.subheader("🎬 Trailer")
        
        if trailer_url and ("youtube.com" in trailer_url or "youtu.be" in trailer_url):
            st.video(trailer_url)
            if YOUTUBE_API_KEY:
                st.caption("via YouTube API")
        else:
            st.info("📺 No trailer found")
            
            col1, col2 = st.columns(2)
            with col1:
                search_query = f"{selected_name} anime official trailer"
                youtube_search = f"https://www.youtube.com/results?search_query={search_query.replace(' ', '+')}"
                st.markdown(f"[Search YouTube]({youtube_search})")
            with col2:
                if mal_url:
                    st.markdown(f"[MyAnimeList]({mal_url})")

        st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)
        st.subheader("✨ Similar Anime")
        st.caption(f"Based on genre similarity")
        
        with st.spinner("Generating recommendations..."):
            recs = recommend(selected_name, top_n=top_n)

        st.dataframe(
            recs,
            use_container_width=True,
            hide_index=True,
            column_config={
                "name": "Anime Title",
                "genre": "Genres",
                "type": "Format",
                "rating": st.column_config.NumberColumn("Rating", format="%.2f"),
                "episodes": "Episodes",
                "members": st.column_config.NumberColumn("Members", format="%d"),
                "similarity": st.column_config.ProgressColumn(
                    "Match", 
                    min_value=0.0, 
                    max_value=1.0, 
                    format="%.3f"
                )
            },
        )


# -----------------------------
# Analytics Dashboard Tab
# -----------------------------
with tab2:
    st.subheader("📊 Analytics Dashboard")
    st.caption(
        f"Type: {', '.join(type_filter) if type_filter else 'All'} | "
        f"Rating: {eda_min_rating:.1f}–{eda_max_rating:.1f}"
    )

    # Key metrics
    total_anime = len(eda_df)
    avg_rating = pd.to_numeric(eda_df["rating"], errors="coerce").dropna().mean()
    avg_members = pd.to_numeric(eda_df["members"], errors="coerce").dropna().mean()

    m1, m2, m3 = st.columns(3)
    m1.metric("Total Anime", f"{total_anime:,}")
    m2.metric("Avg Rating", f"{avg_rating:.2f}" if not np.isnan(avg_rating) else "N/A")
    m3.metric("Avg Members", f"{int(avg_members):,}" if not np.isnan(avg_members) else "N/A")

    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

    # 1) Rating Distribution
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("#### Rating Distribution")
        fig = plt.figure(figsize=(8, 4))
        ratings = pd.to_numeric(eda_df["rating"], errors="coerce").dropna()
        sns.histplot(ratings, bins=20, kde=True, color='#2563eb', alpha=0.6)
        plt.xlabel("Rating")
        plt.ylabel("Count")
        st.pyplot(fig)
        plt.close()
    with col2:
        st.markdown(
            """
            <div class="explain-box">
                <strong>Distribution</strong><br>
                Shows how ratings are spread.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

    # 2) Avg Rating by Type
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("#### Average Rating by Type")
        temp = eda_df.dropna(subset=["rating"]).groupby("type")["rating"].mean().sort_values(ascending=False)
        fig = plt.figure(figsize=(8, 4))
        plt.bar(temp.index.astype(str), temp.values, color='#2563eb', alpha=0.6)
        plt.xticks(rotation=35, ha='right')
        plt.ylabel("Avg Rating")
        st.pyplot(fig)
        plt.close()
    with col2:
        st.markdown(
            """
            <div class="explain-box">
                <strong>Format Comparison</strong><br>
                Average rating by anime type.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

    # 3) Rating vs Members
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("#### Rating vs Popularity")
        temp = eda_df.dropna(subset=["rating", "members"]).copy()
        fig = plt.figure(figsize=(8, 4))
        plt.scatter(temp["members"], temp["rating"], alpha=0.5, color='#2563eb', s=20)
        plt.xscale("log")
        plt.xlabel("Members (log scale)")
        plt.ylabel("Rating")
        st.pyplot(fig)
        plt.close()
    with col2:
        st.markdown(
            """
            <div class="explain-box">
                <strong>Popularity vs Quality</strong><br>
                Log scale shows all data.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

    # 4) Correlation Heatmap
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("#### Correlation Matrix")
        corr_df = eda_df[["rating", "members", "episodes"]].copy().dropna()
        if len(corr_df) >= 3:
            fig = plt.figure(figsize=(6.5, 4))
            corr = corr_df.corr()
            sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=1)
            st.pyplot(fig)
            plt.close()
        else:
            st.info("Not enough data")
    with col2:
        st.markdown(
            """
            <div class="explain-box">
                <strong>Relationships</strong><br>
                Values near ±1 = strong link.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

    # 5) Top Genres
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("#### Top Genres")
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
            plt.barh(top_g.index[::-1], top_g.values[::-1], color='#2563eb', alpha=0.6)
            plt.xlabel("Count")
            st.pyplot(fig)
            plt.close()
        else:
            st.info("No genres available")
    with col2:
        st.markdown(
            """
            <div class="explain-box">
                <strong>Genre Popularity</strong><br>
                Most common genres.
            </div>
            """,
            unsafe_allow_html=True,
        )
