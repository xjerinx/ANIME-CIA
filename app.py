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
st.set_page_config(page_title="Anime Recommender", page_icon="🎌", layout="wide", initial_sidebar_state="expanded")


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

# Debug: Show API key status in sidebar
if YOUTUBE_API_KEY:
    st.sidebar.success("✅ YouTube API Connected")
else:
    st.sidebar.warning("⚠️ YouTube API key not found. Using basic search.")


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
            f"{anime_name} PV",  # PV = Promotional Video
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
# Theme toggle + Global Styling
# -----------------------------
dark_mode = st.sidebar.checkbox("🌙 Dark Mode", value=True)

# ======================
# ENHANCED CREATIVE CSS
# ======================
BASE_SIDEBAR_CSS = """
<style>
  /* Import anime-style font */
  @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&family=Righteous&display=swap');

  /* Global font override */
  * { font-family: 'Poppins', sans-serif; }

  /* Sidebar always dark with glass effect */
  section[data-testid="stSidebar"]{
    background: rgba(10, 14, 22, 0.95) !important;
    backdrop-filter: blur(12px) saturate(180%);
    -webkit-backdrop-filter: blur(12px) saturate(180%);
    border-right: 1px solid rgba(255, 255, 255, 0.15) !important;
    box-shadow: 4px 0 25px rgba(0,0,0,0.5);
  }
  section[data-testid="stSidebar"] *{ color:#f0f3fa !important; }
  section[data-testid="stSidebar"] .stCaption,
  section[data-testid="stSidebar"] p,
  section[data-testid="stSidebar"] small{
    color: #b0b8cc !important;
  }

  /* Sidebar input with glow */
  section[data-testid="stSidebar"] div[data-baseweb="input"] input{
    background-color: #1e2638 !important;
    color: #ffffff !important;
    border: 1px solid rgba(255,215,0,0.3) !important;
    border-radius: 30px !important;
    padding: 12px 18px !important;
    font-weight: 400;
    transition: all 0.3s ease;
    box-shadow: 0 4px 12px rgba(0,0,0,0.3);
  }
  section[data-testid="stSidebar"] div[data-baseweb="input"] input:focus{
    border-color: gold !important;
    box-shadow: 0 0 0 3px rgba(255,215,0,0.2) !important;
  }
  section[data-testid="stSidebar"] div[data-baseweb="input"] input::placeholder{
    color: #6a7299 !important;
    font-style: italic;
  }

  /* Sidebar buttons */
  .stButton > button {
    background: linear-gradient(135deg, #2a2f45, #1a1f30);
    border: 1px solid rgba(255,215,0,0.3);
    border-radius: 40px;
    color: #f0f3fa;
    font-weight: 600;
    padding: 10px 24px;
    transition: all 0.2s ease;
    box-shadow: 0 6px 14px rgba(0,0,0,0.3);
  }
  .stButton > button:hover {
    transform: translateY(-2px);
    background: linear-gradient(135deg, #3a405a, #2a2f45);
    border-color: gold;
    box-shadow: 0 12px 20px rgba(0,0,0,0.4);
  }

  /* Sidebar sliders / selects */
  div[data-testid="stSlider"] {
    padding: 10px 0;
  }
  div[data-testid="stMultiSelect"] > div {
    background-color: #1e2638 !important;
    border-radius: 30px;
    border: 1px solid rgba(255,215,0,0.2);
  }

  /* Compact trending cards - anime vibe */
  .trend-card{
    background: rgba(20, 25, 40, 0.7);
    backdrop-filter: blur(8px);
    -webkit-backdrop-filter: blur(8px);
    border: 1px solid rgba(255,215,0,0.15);
    border-radius: 24px;
    padding: 10px 8px 12px 8px;
    margin-bottom: 16px;
    transition: all 0.3s cubic-bezier(0.2, 0.9, 0.3, 1.2);
    box-shadow: 0 10px 20px -5px rgba(0,0,0,0.5);
  }
  .trend-card:hover{
    transform: scale(1.02) translateY(-5px);
    border-color: rgba(255,215,0,0.5);
    box-shadow: 0 18px 30px -5px rgba(255,215,0,0.2);
    background: rgba(30, 35, 55, 0.9);
  }
  .trend-img{
    width:100%;
    height:130px;
    border-radius:18px;
    overflow:hidden;
    background: rgba(0,0,0,0.3);
    border: 1px solid rgba(255,255,255,0.1);
  }
  .trend-img img{
    width:100%;
    height:130px;
    object-fit:cover;
    display:block;
    transition: transform 0.5s ease;
  }
  .trend-card:hover .trend-img img{
    transform: scale(1.1);
  }
  .trend-title{
    font-size: 0.82rem;
    line-height: 1.2;
    font-weight: 600;
    color: #f0e6d0;
    text-align:center;
    white-space: nowrap;
    overflow:hidden;
    text-overflow: ellipsis;
    margin: 10px 4px 6px 4px;
    letter-spacing: 0.3px;
    text-shadow: 0 2px 4px rgba(0,0,0,0.5);
  }
  
  /* Video container styling */
  .video-container {
    border-radius: 24px;
    overflow: hidden;
    box-shadow: 0 20px 40px rgba(0,0,0,0.5), 0 0 0 2px rgba(255,215,0,0.2);
    margin: 25px 0;
    transition: box-shadow 0.3s;
  }
  .video-container:hover {
    box-shadow: 0 25px 50px rgba(255,215,0,0.2), 0 0 0 3px gold;
  }
  
  /* Explanation box with anime quote feel */
  .explain-box{
    background: rgba(20, 25, 45, 0.75);
    backdrop-filter: blur(6px);
    border: 1px solid rgba(255,215,0,0.25);
    border-radius: 24px;
    padding: 16px 20px;
    color: #d0d8f0;
    font-size: 0.95rem;
    font-style: italic;
    box-shadow: 0 10px 20px rgba(0,0,0,0.3);
    border-left: 5px solid gold;
  }

  /* Metrics cards with glass */
  div[data-testid="stMetric"] {
    background: rgba(20, 25, 45, 0.7) !important;
    backdrop-filter: blur(8px);
    border: 1px solid rgba(255,215,0,0.2);
    border-radius: 28px !important;
    padding: 16px 10px !important;
    box-shadow: 0 15px 30px rgba(0,0,0,0.4);
    transition: transform 0.2s;
  }
  div[data-testid="stMetric"]:hover {
    transform: translateY(-5px);
    border-color: gold;
  }

  /* DataFrames with rounded glass */
  .stDataFrame, [data-testid="stDataFrame"]{
    border-radius: 24px !important;
    overflow: hidden;
    border: 1px solid rgba(255,215,0,0.2);
    background: rgba(20, 25, 45, 0.6) !important;
    backdrop-filter: blur(4px);
    box-shadow: 0 15px 30px rgba(0,0,0,0.3);
  }

  /* Custom header glow */
  h1, h2, h3 {
    text-shadow: 0 2px 10px rgba(255,215,0,0.3);
    letter-spacing: 0.5px;
  }
  h1 {
    font-family: 'Righteous', cursive;
    font-size: 3rem !important;
    background: linear-gradient(135deg, #fff8e7, #ffd966);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.2rem;
  }
  .stCaption {
    font-size: 1rem;
    color: #a0a8c0 !important;
  }
</style>
"""

if dark_mode:
    st.markdown(
        """
        <style>
          :root{
            --bg: radial-gradient(circle at 30% 10%, #141b2b, #0a0f1a);
            --text: #f0f3fa;
            --muted: #b0b8cc;
            --border: rgba(255,215,0,0.2);
            --card-bg: rgba(20, 25, 45, 0.6);
          }

          .stApp{
            background: var(--bg);
            color: var(--text);
          }
          .block-container{
            padding-top: 1.8rem;
            padding-bottom: 3rem;
            backdrop-filter: blur(2px);
          }

          h1,h2,h3,h4,h5,h6{color: var(--text);}
          .stCaption,p,li{color: var(--muted);}
          a{color: #ffd966; font-weight: 500;}

          /* Tabs styling */
          button[data-baseweb="tab"] {
            background: rgba(20,25,45,0.5) !important;
            border-radius: 40px !important;
            margin: 0 5px;
            border: 1px solid rgba(255,215,0,0.2) !important;
            color: #d0d8f0 !important;
            font-weight: 600;
            backdrop-filter: blur(8px);
          }
          button[data-baseweb="tab"][aria-selected="true"] {
            background: rgba(255,215,0,0.2) !important;
            border-color: gold !important;
            color: white !important;
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
            --bg: linear-gradient(145deg, #f8f3ff, #e6ecff);
            --text: #1e2438;
            --muted: #3f4a6b;
            --border: rgba(106, 90, 205, 0.2);
            --card-bg: rgba(255,255,255,0.7);
          }

          .stApp{
            background: var(--bg);
            color: var(--text);
          }
          .block-container{
            padding-top: 1.8rem;
            padding-bottom: 3rem;
            backdrop-filter: blur(2px);
          }

          h1,h2,h3,h4,h5,h6{color: #1e1a36;}
          .stCaption,p,li{color: #4a5180;}
          a{color: #6a4fe0; font-weight: 600;}

          /* Light mode metric cards */
          div[data-testid="stMetric"]{
            background: rgba(255,255,255,0.7) !important;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(106,90,205,0.2);
            border-radius: 28px !important;
            box-shadow: 0 15px 30px rgba(60,50,100,0.1);
          }

          /* Light mode tabs */
          button[data-baseweb="tab"] {
            background: rgba(255,255,255,0.6) !important;
            border-radius: 40px !important;
            margin: 0 5px;
            border: 1px solid rgba(106,90,205,0.3) !important;
            color: #2d2f6e !important;
            font-weight: 600;
            backdrop-filter: blur(8px);
          }
          button[data-baseweb="tab"][aria-selected="true"] {
            background: rgba(106,90,205,0.2) !important;
            border-color: #6a4fe0 !important;
            color: #2b1b6e !important;
          }

          /* DataFrames light */
          .stDataFrame, [data-testid="stDataFrame"]{
            background: rgba(255,255,255,0.7) !important;
            border: 1px solid rgba(106,90,205,0.2);
          }
          .explain-box{
            background: rgba(255,255,255,0.7);
            border-left: 5px solid #6a4fe0;
            color: #2b1b6e;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )

# Apply always-on sidebar/trending css
st.markdown(BASE_SIDEBAR_CSS, unsafe_allow_html=True)


# -----------------------------
# Header with anime vibe
# -----------------------------
st.markdown(
    """
    <div style="display: flex; align-items: center; gap: 12px; margin-bottom: -10px;">
        <h1>🎌 ANIME RECOMMENDER</h1>
        <span style="font-size: 1.2rem; background: rgba(255,215,0,0.2); padding: 5px 15px; border-radius: 60px; border:1px solid gold;">✨ content-based</span>
    </div>
    """,
    unsafe_allow_html=True,
)
st.caption("Discover your next favorite anime with TF-IDF magic + interactive dashboard.")


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
    poster_url, mal_url, trailer_url = fetch_media(chosen_name)

    if poster_url:
        st.sidebar.image(poster_url, width=140, output_format="auto")

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
# Sidebar: Top 6 Trending
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
# EDA filter
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
tab1, tab2 = st.tabs(["✨ Recommend", "📈 Dashboard / EDA"])


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
        st.info("👀 Use sidebar search or pick from Top Trending.")
    else:
        selected_name = st.session_state.selected_anime
        row = anime[anime["name"] == selected_name].iloc[0]

        left, right = st.columns([1, 2])

        with left:
            with st.spinner("Fetching media…"):
                poster_url, mal_url, trailer_url = fetch_media(selected_name)

            if poster_url:
                st.image(poster_url, width=260, caption=selected_name)
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

        # ✅ Video Trailer Section
        st.divider()
        st.subheader("🎬 Trailer / Preview")
        
        # Create a container for the video
        st.markdown('<div class="video-container">', unsafe_allow_html=True)
        
        if trailer_url and ("youtube.com" in trailer_url or "youtu.be" in trailer_url):
            # Play the video directly in the app
            st.video(trailer_url)
            
            # Show source info
            if YOUTUBE_API_KEY and "googleapis" not in str(trailer_url):
                st.caption("🎥 Found via YouTube API")
            else:
                st.caption("🎥 Found via YouTube search")
        else:
            # No trailer found - show options
            st.info("📺 No trailer automatically found")
            
            # Create columns for action buttons
            col1, col2 = st.columns(2)
            
            with col1:
                # YouTube search link
                search_query = f"{selected_name} anime official trailer"
                youtube_search = f"https://www.youtube.com/results?search_query={search_query.replace(' ', '+')}"
                st.markdown(f"[🔍 Search YouTube]({youtube_search})")
            
            with col2:
                if mal_url:
                    st.markdown(f"[📝 Check MyAnimeList]({mal_url})")
        
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
# Dashboard / EDA tab
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
        fig = plt.figure(figsize=(8, 4), facecolor='none')
        ax = fig.add_subplot(111)
        ax.set_facecolor('none')
        sns.histplot(pd.to_numeric(eda_df["rating"], errors="coerce").dropna(), bins=20, kde=True, color='gold')
        plt.xlabel("Rating", color='#b0b8cc')
        plt.ylabel("Count", color='#b0b8cc')
        plt.tick_params(colors='#b0b8cc')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    with e1:
        st.markdown(
            """
            <div class="explain-box">
              🌟 Shows how ratings are distributed.<br/>
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
        fig = plt.figure(figsize=(8, 4), facecolor='none')
        ax = fig.add_subplot(111)
        ax.set_facecolor('none')
        plt.bar(temp.index.astype(str), temp.values, color='#ffb347', edgecolor='gold')
        plt.xticks(rotation=35, ha="right", color='#b0b8cc')
        plt.ylabel("Avg Rating", color='#b0b8cc')
        plt.tick_params(colors='#b0b8cc')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    with e2:
        st.markdown(
            """
            <div class="explain-box">
              📺 Compares average rating across formats.<br/>
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
        fig = plt.figure(figsize=(8, 4), facecolor='none')
        ax = fig.add_subplot(111)
        ax.set_facecolor('none')
        plt.scatter(temp["members"], temp["rating"], alpha=0.65, color='#6a5acd', edgecolor='white', s=30)
        plt.xscale("log")
        plt.xlabel("Members (log scale)", color='#b0b8cc')
        plt.ylabel("Rating", color='#b0b8cc')
        plt.tick_params(colors='#b0b8cc')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    with e3:
        st.markdown(
            """
            <div class="explain-box">
              📈 Shows popularity vs rating pattern.<br/>
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
            fig = plt.figure(figsize=(6.5, 4), facecolor='none')
            ax = fig.add_subplot(111)
            ax.set_facecolor('none')
            sns.heatmap(corr_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", 
                        cbar_kws={'label': 'Correlation'}, ax=ax)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        else:
            st.info("Not enough data after filtering to compute correlations.")
    with e4:
        st.markdown(
            """
            <div class="explain-box">
              🔗 Measures relationships between numeric fields.<br/>
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
            fig = plt.figure(figsize=(8, 4), facecolor='none')
            ax = fig.add_subplot(111)
            ax.set_facecolor('none')
            plt.barh(top_g.index[::-1], top_g.values[::-1], color='#77dd77', edgecolor='white')
            plt.xlabel("Count", color='#b0b8cc')
            plt.tick_params(colors='#b0b8cc')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        else:
            st.info("No genres available after filtering.")
    with e5:
        st.markdown(
            """
            <div class="explain-box">
              🎭 Shows most common genres in the dataset.<br/>
              Helps explain dataset bias toward certain genres.
            </div>
            """,
            unsafe_allow_html=True,
        )

# -----------------------------
# Footer
# -----------------------------
st.divider()
st.markdown(
    "<p style='text-align: center; color: #9ca3af;'>✨ crafted with 💜 for anime lovers • TF‑IDF + Cosine Similarity</p>",
    unsafe_allow_html=True,
)
