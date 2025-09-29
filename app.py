# app.py
import os
import re
import time
import requests
import pandas as pd
import googlemaps
import openai
import streamlit as st
from dotenv import load_dotenv

# Load .env locally (optional)
load_dotenv()

# Read keys from environment or Streamlit secrets
MAPS_KEY = os.getenv("GOOGLE_MAPS_KEY", "")
OPENAI_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = "gpt-3.5-turbo"  # النسخة الأقل تكلفة

st.set_page_config(page_title="Restaurant Classifier", layout="wide")
st.title("🍽️ Restaurant Classifier — Step by Step (gpt-3.5-turbo)")

# Sidebar keys override
with st.sidebar:
    st.header("API Keys / Settings")
    maps_input = st.text_input("Google Maps API Key", value=MAPS_KEY, type="password")
    openai_input = st.text_input("OpenAI API Key", value=OPENAI_KEY, type="password")
    if st.button("Use these keys"):
        MAPS_KEY = maps_input.strip()
        OPENAI_KEY = openai_input.strip()
        st.success("Keys updated (in-memory)")

st.write("Maps key loaded:", bool(MAPS_KEY), " — OpenAI key loaded:", bool(OPENAI_KEY))

# Helpers
def expand_short_url(url):
    try:
        r = requests.get(url, allow_redirects=True, timeout=4)
        return r.url
    except Exception:
        return url

def extract_coordinates(url):
    u = url.strip()
    if "maps.app.goo.gl" in u or "goo.gl" in u:
        u = expand_short_url(u)
    m = re.search(r'@([-+]?\d+\.\d+),([-+]?\d+\.\d+)', u)
    if m:
        return float(m.group(1)), float(m.group(2))
    return None, None

def fetch_restaurants(lat, lng, maps_key, radius=3000):
    client = googlemaps.Client(key=maps_key)
    places = client.places_nearby(location=(lat,lng), radius=radius, type="restaurant")
    results = places.get("results", [])
    rows = []
    for r in results:
        rows.append({
            "name": r.get("name",""),
            "address": r.get("vicinity",""),
            "types": ", ".join(r.get("types", [])),
            "place_id": r.get("place_id",""),
            "map_url": f"https://www.google.com/maps/place/?q=place_id:{r.get('place_id','')}"
        })
    return pd.DataFrame(rows)

# Categories
CATEGORIES_AR = [
    "مطاعم هندية",
    "مطاعم شاورما",
    "مطاعم لبنانية",
    "مطاعم خليجية",
    "مطاعم أسماك",
    "مطاعم برجر",
    "أخرى"
]

# Classifier using new OpenAI interface
def classify_restaurant(name, address, types):
    openai.api_key = OPENAI_KEY
    try:
        resp = openai.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "صنّف المطعم إلى أحد التصنيفات التالية بدقة: " + ", ".join(CATEGORIES_AR) + ". أجب فقط بالكلمة العربية المطابقة."},
                {"role": "user", "content": f"Name: {name}\nAddress: {address}\nTypes: {types}"}
            ],
            max_tokens=20,
            temperature=0
        )
        text = resp.choices[0].message.content.strip()
        if text in CATEGORIES_AR:
            return text
        return "أخرى"
    except Exception as e:
        return f"❌ Error: {e}"

# Streamlit session state
if "coords" not in st.session_state: st.session_state.coords = None
if "restaurants" not in st.session_state: st.session_state.restaurants = None
if "classified" not in st.session_state: st.session_state.classified = []
if "index" not in st.session_state: st.session_state.index = 0

# Step 1: Paste URL
st.markdown("### 1) Paste Google Maps URL")
url = st.text_input("Google Maps URL")
if st.button("▶️ Start — Extract Coordinates"):
    lat, lng = extract_coordinates(url)
    if lat is None:
        st.error("Could not extract coordinates. Use full Maps URL.")
    else:
        st.session_state.coords = (lat, lng)
        st.success(f"Coordinates: {lat}, {lng}")

# Step 2: Fetch restaurants
st.markdown("### 2) Fetch nearby restaurants")
if st.button("➡️ Fetch Restaurants"):
    if st.session_state.coords is None:
        st.error("Run Start first to extract coordinates.")
    else:
        lat, lng = st.session_state.coords
        df = fetch_restaurants(lat, lng, MAPS_KEY)
        st.session_state.restaurants = df
        st.session_state.index = 0
        st.session_state.classified = []
        st.success(f"Fetched {len(df)} restaurants")
        st.dataframe(df[["name","address","types"]])

# Step 3: Classify one-by-one
st.markdown("### 3) Classify restaurants one-by-one")
if st.session_state.restaurants is not None:
    if st.button("➡️ Classify Next"):
        idx = st.session_state.index
        if idx >= len(st.session_state.restaurants):
            st.success("All restaurants classified.")
        else:
            row = st.session_state.restaurants.iloc[idx]
            category = classify_restaurant(row["name"], row["address"], row["types"])
            st.session_state.classified.append({
                "name": row["name"], "address": row["address"], "category": category, "map_url": row["map_url"]
            })
            st.session_state.index += 1
            st.success(f"{row['name']} → {category}")

# Show classified so far
if st.session_state.classified:
    st.subheader("Classified restaurants so far")
    st.dataframe(pd.DataFrame(st.session_state.classified))
