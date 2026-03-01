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
# ENHANCED UI THEMES - Dark & Light Mode with Modern Design
# -----------------------------
dark_mode = st.sidebar.checkbox("🌙 Dark Mode", value=True)

# Base CSS that applies to both themes
BASE_CSS = """
<style>
  /* Import Google Fonts */
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
  
  * {
    font-family: 'Inter', sans-serif;
  }
  
  /* Smooth scrolling */
  html {
    scroll-behavior: smooth;
  }
  
  /* Sidebar always dark with glass effect */
  section[data-testid="stSidebar"]{
    background: linear-gradient(180deg, #0a0e1a 0%, #111827 100%) !important;
    border-right: 1px solid rgba(255,255,255,0.08) !important;
    backdrop-filter: blur(10px);
  }
  section[data-testid="stSidebar"] *{ color:#e5e7eb !important; }
  section[data-testid="stSidebar"] .stCaption,
  section[data-testid="stSidebar"] p,
  section[data-testid="stSidebar"] small{
    color:#9ca3af !important;
  }

  /* Sidebar input styling */
  section[data-testid="stSidebar"] div[data-baseweb="input"] input{
    background-color: rgba(17,24,39,0.8) !important;
    color:#e5e7eb !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 30px !important;
    padding: 12px 16px !important;
    font-size: 14px !important;
    backdrop-filter: blur(5px);
    transition: all 0.3s;
  }
  section[data-testid="stSidebar"] div[data-baseweb="input"] input:focus{
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.2) !important;
  }
  section[data-testid="stSidebar"] div[data-baseweb="input"] input::placeholder{
    color:#6b7280 !important;
  }

  /* Trending cards */
  .trend-card{
    background: linear-gradient(135deg, rgba(17,24,39,0.5) 0%, rgba(17,24,39,0.3) 100%);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 8px;
    margin-bottom: 12px;
    backdrop-filter: blur(10px);
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
  }
  .trend-card:hover{
    transform: translateY(-4px);
    box-shadow: 0 20px 30px -10px rgba(0,0,0,0.5);
    border-color: rgba(99, 102, 241, 0.3);
    background: linear-gradient(135deg, rgba(17,24,39,0.7) 0%, rgba(17,24,39,0.5) 100%);
  }
  
  .trend-img{
    width:100%;
    height:120px;
    border-radius:12px;
    overflow:hidden;
    background: rgba(255,255,255,0.03);
    box-shadow: 0 8px 16px rgba(0,0,0,0.2);
  }
  .trend-img img{
    width:100%;
    height:120px;
    object-fit:cover;
    display:block;
    transition: transform 0.5s;
  }
  .trend-img img:hover{
    transform: scale(1.08);
  }
  .trend-title{
    font-size: 0.8rem;
    font-weight: 600;
    line-height: 1.2;
    color: #e5e7eb;
    text-align:center;
    white-space: nowrap;
    overflow:hidden;
    text-overflow: ellipsis;
    margin-top: 8px;
    margin-bottom: 4px;
    letter-spacing: 0.3px;
  }
  
  /* Video container */
  .video-container {
    border-radius: 20px;
    overflow: hidden;
    box-shadow: 0 30px 50px -20px rgba(0,0,0,0.5);
    margin: 25px 0;
    border: 1px solid rgba(255,255,255,0.1);
    transition: all 0.3s;
  }
  .video-container:hover {
    box-shadow: 0 40px 60px -20px rgba(99, 102, 241, 0.3);
    border-color: rgba(99, 102, 241, 0.3);
  }
  
  /* Explanation box */
  .explain-box{
    background: linear-gradient(135deg, rgba(17,24,39,0.7) 0%, rgba(17,24,39,0.5) 100%);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 20px;
    padding: 16px 18px;
    color: #9ca3af;
    font-size: 0.9rem;
    backdrop-filter: blur(10px);
    box-shadow: 0 10px 25px -10px rgba(0,0,0,0.3);
    height: 100%;
  }
  
  /* Custom metric cards */
  .custom-metric {
    background: linear-gradient(135deg, rgba(17,24,39,0.8) 0%, rgba(17,24,39,0.6) 100%);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 20px;
    padding: 20px;
    text-align: center;
    backdrop-filter: blur(10px);
    box-shadow: 0 15px 30px -15px rgba(0,0,0,0.3);
    transition: all 0.3s;
  }
  .custom-metric:hover {
    transform: translateY(-5px);
    border-color: rgba(99, 102, 241, 0.3);
    box-shadow: 0 25px 40px -20px rgba(99, 102, 241, 0.4);
  }
  
  .metric-label {
    font-size: 0.8rem;
    color: #9ca3af;
    text-transform: uppercase;
    letter-spacing: 1px;
    font-weight: 600;
  }
  .metric-value {
    font-size: 2rem;
    font-weight: 800;
    color: #e5e7eb;
    line-height: 1.2;
    background: linear-gradient(135deg, #fff 0%, #e5e7eb 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
  }
  
  /* Anime title styling */
  .anime-title {
    font-size: 2.5rem;
    font-weight: 800;
    background: linear-gradient(135deg, #fff 0%, #cbd5e1 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 5px;
    letter-spacing: -0.02em;
  }
  .anime-subtitle {
    font-size: 0.95rem;
    color: #9ca3af;
    font-style: italic;
  }
  
  /* Genre tags */
  .genre-tag {
    display: inline-block;
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.15) 0%, rgba(139, 92, 246, 0.15) 100%);
    border: 1px solid rgba(99, 102, 241, 0.3);
    border-radius: 30px;
    padding: 6px 16px;
    margin: 0 6px 8px 0;
    font-size: 0.85rem;
    font-weight: 500;
    color: #a5b4fc;
    backdrop-filter: blur(5px);
    transition: all 0.2s;
  }
  .genre-tag:hover {
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.25) 0%, rgba(139, 92, 246, 0.25) 100%);
    transform: translateY(-2px);
    border-color: #6366f1;
  }
  
  /* Divider styling */
  .custom-divider {
    margin: 35px 0;
    border: none;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
  }
  
  /* Button styling */
  .stButton button {
    border-radius: 40px !important;
    border: none !important;
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
    color: white !important;
    font-weight: 600 !important;
    font-size: 16px !important;
    padding: 12px 24px !important;
    box-shadow: 0 10px 20px -8px rgba(99, 102, 241, 0.3) !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    letter-spacing: 0.5px;
  }
  .stButton button:hover {
    transform: scale(1.02) translateY(-2px) !important;
    box-shadow: 0 20px 30px -10px rgba(99, 102, 241, 0.5) !important;
  }
  .stButton button:active {
    transform: scale(0.98) !important;
  }
  
  /* Selectbox styling */
  div[data-baseweb="select"] div {
    border-radius: 30px !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    background: rgba(17,24,39,0.8) !important;
    backdrop-filter: blur(5px);
  }
  
  /* Slider styling */
  div[data-testid="stSlider"] div[data-baseweb="slider"] div {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
  }
  
  /* Tabs styling */
  button[data-baseweb="tab"] {
    border-radius: 30px !important;
    margin: 0 5px !important;
    font-weight: 600 !important;
    transition: all 0.2s !important;
  }
  button[data-baseweb="tab"]:hover {
    background: rgba(99, 102, 241, 0.1) !important;
  }
  button[aria-selected="true"] {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
    color: white !important;
  }
  
  /* DataFrame styling */
  .stDataFrame, [data-testid="stDataFrame"] {
    border-radius: 20px !important;
    overflow: hidden !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    background: rgba(17,24,39,0.4) !important;
    backdrop-filter: blur(5px);
    box-shadow: 0 20px 40px -20px rgba(0,0,0,0.4) !important;
  }
  
  /* Progress bar styling */
  .stProgress div div {
    background: linear-gradient(90deg, #6366f1, #8b5cf6) !important;
    border-radius: 30px !important;
  }
  
  /* Success/Warning/Info messages */
  .stAlert {
    border-radius: 30px !important;
    border: none !important;
    backdrop-filter: blur(5px);
  }
  
  /* Checkbox styling */
  .stCheckbox {
    display: flex;
    justify-content: center;
  }
</style>
"""

# Dark Mode Theme
if dark_mode:
    st.markdown(
        """
        <style>
          :root{
            --bg: linear-gradient(135deg, #0b0f1a 0%, #111827 100%);
            --card-bg: rgba(17,24,39,0.5);
            --text: #e5e7eb;
            --text-light: #9ca3af;
            --primary: #6366f1;
            --primary-light: #818cf8;
            --secondary: #8b5cf6;
            --accent: #f59e0b;
            --border: rgba(99, 102, 241, 0.2);
            --shadow: 0 25px 50px -12px rgba(0,0,0,0.5);
          }

          .stApp {
            background: var(--bg);
            color: var(--text);
          }
          
          .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
          }

          h1, h2, h3, h4, h5, h6 {
            color: var(--text);
            font-weight: 700;
            letter-spacing: -0.02em;
          }
          
          h1 {
            background: linear-gradient(135deg, #818cf8 0%, #c084fc 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 3.2rem !important;
          }
          
          .stCaption, p, li {
            color: var(--text-light);
          }
          
          a {
            color: #818cf8;
            text-decoration: none;
            font-weight: 500;
            transition: all 0.2s;
            border-bottom: 1px dotted transparent;
          }
          
          a:hover {
            color: #a78bfa;
            border-bottom-color: #a78bfa;
          }
          
          div[data-testid="stMetric"] {
            background: linear-gradient(135deg, rgba(17,24,39,0.8) 0%, rgba(17,24,39,0.6) 100%);
            border: 1px solid var(--border);
            border-radius: 20px;
            padding: 20px;
            box-shadow: var(--shadow);
            backdrop-filter: blur(10px);
            transition: all 0.3s;
          }
          div[data-testid="stMetric"]:hover {
            transform: translateY(-5px);
            border-color: #6366f1;
            box-shadow: 0 30px 60px -15px rgba(99, 102, 241, 0.3);
          }
          
          div[data-testid="stMetric"] label {
            color: #9ca3af !important;
          }
          
          div[data-testid="stMetric"] div {
            color: #e5e7eb !important;
            font-weight: 700 !important;
          }
          
          /* Custom scrollbar */
          ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
          }
          
          ::-webkit-scrollbar-track {
            background: rgba(17,24,39,0.4);
            border-radius: 10px;
          }
          
          ::-webkit-scrollbar-thumb {
            background: linear-gradient(135deg, #6366f1, #8b5cf6);
            border-radius: 10px;
          }
          
          ::-webkit-scrollbar-thumb:hover {
            background: linear-gradient(135deg, #818cf8, #a78bfa);
          }
        </style>
        """,
        unsafe_allow_html=True,
    )

# Enhanced Light Mode Theme
else:
    st.markdown(
        """
        <style>
          :root{
            --bg: linear-gradient(135deg, #f8faff 0%, #f0f4ff 100%);
            --card-bg: rgba(255, 255, 255, 0.85);
            --text: #1e293b;
            --text-light: #475569;
            --primary: #4f46e5;
            --primary-light: #818cf8;
            --secondary: #7c3aed;
            --accent: #f59e0b;
            --border: rgba(79, 70, 229, 0.15);
            --shadow: 0 25px 50px -12px rgba(79, 70, 229, 0.15);
          }

          .stApp {
            background: var(--bg);
            color: var(--text);
          }
          
          .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
          }

          h1, h2, h3, h4, h5, h6 {
            color: var(--text);
            font-weight: 700;
            letter-spacing: -0.02em;
          }
          
          h1 {
            background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 3.2rem !important;
          }
          
          .stCaption, p, li {
            color: var(--text-light);
          }
          
          a {
            color: #4f46e5;
            text-decoration: none;
            font-weight: 500;
            transition: all 0.2s;
            border-bottom: 1px dotted transparent;
          }
          
          a:hover {
            color: #7c3aed;
            border-bottom-color: #7c3aed;
          }
          
          div[data-testid="stMetric"] {
            background: rgba(255, 255, 255, 0.9);
            border: 1px solid var(--border);
            border-radius: 20px;
            padding: 20px;
            box-shadow: var(--shadow);
            backdrop-filter: blur(10px);
            transition: all 0.3s;
          }
          div[data-testid="stMetric"]:hover {
            transform: translateY(-5px);
            border-color: #4f46e5;
            box-shadow: 0 30px 60px -15px rgba(79, 70, 229, 0.2);
          }
          
          div[data-testid="stMetric"] label {
            color: #64748b !important;
          }
          
          div[data-testid="stMetric"] div {
            color: #1e293b !important;
            font-weight: 700 !important;
          }
          
          /* Dataframe in light mode */
          .stDataFrame, [data-testid="stDataFrame"] {
            background: rgba(255, 255, 255, 0.9) !important;
            border: 1px solid rgba(79, 70, 229, 0.1) !important;
          }
          
          .stDataFrame td, .stDataFrame th {
            color: #1e293b !important;
          }
          
          /* Genre tags in light mode */
          .genre-tag {
            background: linear-gradient(135deg, rgba(79, 70, 229, 0.08) 0%, rgba(124, 58, 237, 0.08) 100%);
            border: 1px solid rgba(79, 70, 229, 0.2);
            color: #4f46e5;
          }
          .genre-tag:hover {
            background: linear-gradient(135deg, rgba(79, 70, 229, 0.15) 0%, rgba(124, 58, 237, 0.15) 100%);
          }
          
          /* Explanation box in light mode */
          .explain-box {
            background: rgba(255, 255, 255, 0.9);
            border: 1px solid rgba(79, 70, 229, 0.1);
            color: #475569;
          }
          
          /* Custom scrollbar for light mode */
          ::-webkit-scrollbar-track {
            background: rgba(79, 70, 229, 0.05);
          }
          
          ::-webkit-scrollbar-thumb {
            background: linear-gradient(135deg, #4f46e5, #7c3aed);
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
# Header with animated gradient
# -----------------------------
st.markdown("""
    <div style='text-align: center; margin-bottom: 30px;'>
        <h1 style='font-size: 3.5rem; margin-bottom: 5px;'>
            🎌 ANIME RECOMMENDER
        </h1>
        <p style='font-size: 1.2rem; opacity: 0.8; letter-spacing: 0.5px;'>
            Discover your next favorite anime with AI-powered recommendations
        </p>
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
# Tabs with icons
# -----------------------------
tab1, tab2 = st.tabs(["🎯 Recommendations", "📊 Analytics Dashboard"])


# -----------------------------
# Background Music Setup (Optional)
# -----------------------------
def setup_background_music():
    """
    Sets up background music with controls in sidebar
    """
    # Initialize session state
    if 'music_playing' not in st.session_state:
        st.session_state.music_playing = False
    if 'music_volume' not in st.session_state:
        st.session_state.music_volume = 0.2
    
    # Direct audio URL (replace with your actual download link from Pixabay)
    MUSIC_URL = "YOUR_PIXABAY_DIRECT_DOWNLOAD_LINK.mp3"
    
    # Only proceed if URL is set
    if MUSIC_URL != "YOUR_PIXABAY_DIRECT_DOWNLOAD_LINK.mp3":
        # Audio player HTML
        audio_html = f"""
        <audio id="anime-bg-music" loop preload="auto">
            <source src="{MUSIC_URL}" type="audio/mpeg">
        </audio>
        
        <script>
            (function() {{
                const audio = document.getElementById('anime-bg-music');
                if (!audio) return;
                
                audio.volume = {st.session_state.music_volume};
                
                if ({'true' if st.session_state.music_playing else 'false'}) {{
                    const playPromise = audio.play();
                    if (playPromise !== undefined) {{
                        playPromise.catch(e => {{
                            console.log("Autoplay prevented:", e);
                        }});
                    }}
                }}
                
                window.animeAudio = audio;
            }})();
        </script>
        """
        
        st.markdown(audio_html, unsafe_allow_html=True)
        
        # Add controls to sidebar
        with st.sidebar:
            st.divider()
            st.subheader("🎵 Background Music")
            
            col1, col2 = st.columns(2)
            with col1:
                play_label = "⏸️ Pause" if st.session_state.music_playing else "▶️ Play"
                if st.button(play_label, key="music_play_btn"):
                    st.session_state.music_playing = not st.session_state.music_playing
                    st.rerun()
            
            with col2:
                new_volume = st.slider("🔊", 0.0, 1.0, 
                                      st.session_state.music_volume, 0.1,
                                      key="music_vol_slider",
                                      label_visibility="collapsed")
                if new_volume != st.session_state.music_volume:
                    st.session_state.music_volume = new_volume
                    st.rerun()
            
            st.caption("🎌 Background music by Pixabay")

# Uncomment the line below and add your Pixabay audio URL to enable background music
# setup_background_music()


# -----------------------------
# Recommendations Tab
# -----------------------------
with tab1:
    # Trending section
    st.markdown("### 🔥 Trending Now")
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
    
    st.markdown("### 📌 Selected Anime")

    if st.session_state.selected_anime is not None:
        if st.button("❌ Clear Selection", use_container_width=True):
            request_clear_all()

    if st.session_state.selected_anime is None:
        st.info("👆 Select an anime from the sidebar to get recommendations")
    else:
        selected_name = st.session_state.selected_anime
        row = anime[anime["name"] == selected_name].iloc[0]

        # Display anime title
        st.markdown(f"<div class='anime-title'>{selected_name}</div>", unsafe_allow_html=True)
        
        # Create two columns for poster and info
        left, right = st.columns([1, 2])

        with left:
            with st.spinner("Loading..."):
                poster_url, mal_url, trailer_url = fetch_media(selected_name)

            if poster_url:
                st.image(poster_url, width=260, caption="Poster")
            else:
                st.info("📷 Poster not available")

        with right:
            # Metrics display
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("⭐ Rating", f"{row['rating']:.2f}" if pd.notna(row["rating"]) else "N/A")
            
            with col2:
                st.metric("📺 Type", str(row["type"]))
            
            with col3:
                ep_count = int(row["episodes"]) if pd.notna(row["episodes"]) else "N/A"
                st.metric("📼 Episodes", ep_count)
            
            with col4:
                members_count = f"{int(row['members']):,}" if pd.notna(row["members"]) else "N/A"
                st.metric("👥 Members", members_count)
            
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
                st.markdown(f"[📝 View on MyAnimeList]({mal_url})")

        # Video Trailer Section
        st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)
        st.markdown("### 🎬 Trailer")
        
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

        st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)
        st.markdown("### ✨ Similar Anime Recommendations")
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
                    format="%.3f",
                    help="Similarity score based on genres"
                )
            },
        )


# -----------------------------
# Analytics Dashboard Tab
# -----------------------------
with tab2:
    st.markdown("### 📊 Analytics Dashboard")
    st.caption(
        f"**Active filters:** Type: {', '.join(type_filter) if type_filter else 'All'} | "
        f"Rating: {eda_min_rating:.1f}–{eda_max_rating:.1f}"
    )

    # Key metrics
    total_anime = len(eda_df)
    avg_rating = pd.to_numeric(eda_df["rating"], errors="coerce").dropna().mean()
    avg_members = pd.to_numeric(eda_df["members"], errors="coerce").dropna().mean()
    total_members = pd.to_numeric(eda_df["members"], errors="coerce").dropna().sum()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("📊 Total Anime", f"{total_anime:,}")
    m2.metric("⭐ Avg Rating", f"{avg_rating:.2f}" if not np.isnan(avg_rating) else "N/A")
    m3.metric("👥 Avg Members", f"{int(avg_members):,}" if not np.isnan(avg_members) else "N/A")
    m4.metric("💰 Total Members", f"{int(total_members):,}" if not np.isnan(total_members) else "N/A")

    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

    # 1) Rating Distribution
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("#### 📈 Rating Distribution")
        fig = plt.figure(figsize=(8, 4), facecolor='none')
        ax = fig.add_subplot(111)
        ratings = pd.to_numeric(eda_df["rating"], errors="coerce").dropna()
        sns.histplot(ratings, bins=20, kde=True, color='#6366f1', alpha=0.7)
        plt.xlabel("Rating", color='#9ca3af')
        plt.ylabel("Count", color='#9ca3af')
        plt.tick_params(colors='#9ca3af')
        plt.grid(alpha=0.2)
        st.pyplot(fig)
        plt.close()
    with col2:
        st.markdown(
            """
            <div class="explain-box">
                <strong>📊 Distribution</strong><br>
                Shows how ratings are spread.<br><br>
                <strong>Peak:</strong> Most common rating<br>
                <strong>Shape:</strong> Normal distribution expected
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

    # 2) Avg Rating by Type
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("#### 📊 Average Rating by Format")
        temp = eda_df.dropna(subset=["rating"]).groupby("type")["rating"].mean().sort_values(ascending=False)
        fig = plt.figure(figsize=(8, 4), facecolor='none')
        ax = fig.add_subplot(111)
        bars = plt.bar(temp.index.astype(str), temp.values, color='#8b5cf6', alpha=0.8)
        plt.xticks(rotation=35, ha='right', color='#9ca3af')
        plt.ylabel("Avg Rating", color='#9ca3af')
        plt.tick_params(colors='#9ca3af')
        plt.grid(axis='y', alpha=0.2)
        st.pyplot(fig)
        plt.close()
    with col2:
        st.markdown(
            """
            <div class="explain-box">
                <strong>📊 Format Comparison</strong><br>
                Average rating by anime type.<br><br>
                <strong>Movies</strong> often rate higher<br>
                <strong>TV series</strong> have more variance
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

    # 3) Rating vs Members
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("#### 📉 Rating vs Popularity")
        temp = eda_df.dropna(subset=["rating", "members"]).copy()
        fig = plt.figure(figsize=(8, 4), facecolor='none')
        ax = fig.add_subplot(111)
        plt.scatter(temp["members"], temp["rating"], alpha=0.5, c='#10b981', s=20)
        plt.xscale("log")
        plt.xlabel("Members (log scale)", color='#9ca3af')
        plt.ylabel("Rating", color='#9ca3af')
        plt.tick_params(colors='#9ca3af')
        plt.grid(alpha=0.2)
        st.pyplot(fig)
        plt.close()
    with col2:
        st.markdown(
            """
            <div class="explain-box">
                <strong>📉 Popularity vs Quality</strong><br>
                Log scale helps visualize all data.<br><br>
                <strong>Trend:</strong> Popular shows tend to rate higher
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

    # 4) Correlation Heatmap
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("#### 🔗 Correlation Matrix")
        corr_df = eda_df[["rating", "members", "episodes"]].copy().dropna()
        if len(corr_df) >= 3:
            fig = plt.figure(figsize=(6.5, 4), facecolor='none')
            ax = fig.add_subplot(111)
            corr = corr_df.corr()
            sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", 
                       linewidths=1, cbar_kws={'label': 'Correlation'})
            plt.tick_params(colors='#9ca3af')
            st.pyplot(fig)
            plt.close()
        else:
            st.info("Not enough data for correlation")
    with col2:
        st.markdown(
            """
            <div class="explain-box">
                <strong>🔗 Relationships</strong><br>
                Values near ±1 = strong link.<br><br>
                <strong>Rating vs Members:</strong> Weak positive<br>
                <strong>Episodes vs Members:</strong> Weak correlation
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

    # 5) Top Genres
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("#### 🏷️ Top Genres")
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
            plt.barh(top_g.index[::-1], top_g.values[::-1], color='#f59e0b', alpha=0.8)
            plt.xlabel("Count", color='#9ca3af')
            plt.tick_params(colors='#9ca3af')
            plt.grid(axis='x', alpha=0.2)
            st.pyplot(fig)
            plt.close()
        else:
            st.info("No genres available")
    with col2:
        st.markdown(
            """
            <div class="explain-box">
                <strong>🏷️ Genre Popularity</strong><br>
                Most common genres in dataset.<br><br>
                <strong>Top genres</strong> indicate market trends
            </div>
            """,
            unsafe_allow_html=True,
        )
