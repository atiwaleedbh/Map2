# app.py
import os
import re
import time
import json
import requests
import pandas as pd
import streamlit as st
import googlemaps
import openai
from dotenv import load_dotenv

# Load .env if exists
load_dotenv()

# Load API keys from environment
MAPS_KEY = os.getenv("GOOGLE_MAPS_KEY", "").strip()
OPENAI_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

st.set_page_config(page_title="Restaurant Classifier", layout="wide")
st.title("🍽️ Restaurant Classifier — Streamlit + GPT (Step-by-step)")

# Sidebar for keys
with st.sidebar:
    st.header("API keys / settings")
    maps_key_input = st.text_input("Google Maps API Key", value=MAPS_KEY, type="password")
    openai_key_input = st.text_input("OpenAI API Key", value=OPENAI_KEY, type="password")
    model_input = st.text_input("OpenAI model (optional)", value=OPENAI_MODEL)
    if st.button("Use these keys"):
        MAPS_KEY = maps_key_input.strip()
        OPENAI_KEY = openai_key_input.strip()
        OPENAI_MODEL = model_input.strip()
        st.success("Keys updated (in-memory)")

# Helpers
def expand_short_url_once(url: str, timeout=4):
    try:
        r = requests.get(url, allow_redirects=True, timeout=timeout)
        return r.url
    except Exception:
        return url

def extract_coordinates(url: str):
    start = time.time()
    if not url:
        return None, None, round(time.time()-start,3)
    u = url.strip()
    if "maps.app.goo.gl" in u or "goo.gl" in u:
        u = expand_short_url_once(u)
    m = re.search(r'@([-+]?\d+\.\d+),([-+]?\d+\.\d+)', u)
    if m:
        return float(m.group(1)), float(m.group(2)), round(time.time()-start,3)
    m = re.search(r'!3d([-+]?\d+\.\d+)!4d([-+]?\d+\.\d+)', u)
    if m:
        return float(m.group(1)), float(m.group(2)), round(time.time()-start,3)
    m = re.search(r'([-+]?\d{1,3}\.\d+)[, ]+([-+]?\d{1,3}\.\d+)', u)
    if m:
        lat, lng = float(m.group(1)), float(m.group(2))
        if -90 <= lat <= 90 and -180 <= lng <= 180:
            return lat, lng, round(time.time()-start,3)
    return None, None, round(time.time()-start,3)

def fetch_restaurants_places(lat, lng, maps_key, radius=3000, max_pages=3):
    client = googlemaps.Client(key=maps_key)
    all_results = []
    places = client.places_nearby(location=(lat,lng), radius=radius, type="restaurant")
    all_results.extend(places.get("results", []))
    pages = 0
    while places.get("next_page_token") and pages < max_pages:
        pages += 1
        time.sleep(2)
        places = client.places_nearby(page_token=places["next_page_token"])
        all_results.extend(places.get("results", []))
    rows = []
    for r in all_results:
        rows.append({
            "name": r.get("name",""),
            "address": r.get("vicinity",""),
            "rating": r.get("rating",""),
            "types": ", ".join(r.get("types", [])),
            "place_id": r.get("place_id",""),
            "map_url": f"https://www.google.com/maps/place/?q=place_id:{r.get('place_id','')}"
        })
    return pd.DataFrame(rows)

def classify_all_with_gpt(df):
    """Send all restaurants at once to GPT for custom classification."""
    if df.empty:
        return pd.DataFrame()
    openai.api_key = OPENAI_KEY
    system_msg = "أنت مساعد لتصنيف أسماء المطاعم. صنّف هذه المطاعم بشكل مناسب استنادًا إلى أسمائها وتعليقات المستخدمين. أجب JSON array: [{\"name\": ..., \"category\": ...}]"
    restaurants_list = df[["name","address","types"]].to_dict(orient="records")
    user_msg = f"Restaurants:\n{json.dumps(restaurants_list, ensure_ascii=False)}"
    messages = [{"role":"system","content":system_msg}, {"role":"user","content":user_msg}]
    try:
        resp = openai.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            temperature=0,
            max_tokens=2000
        )
        text = resp.choices[0].message.content
        # clean code fences
        clean_text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE)
        return pd.DataFrame(json.loads(clean_text))
    except Exception as e:
        st.error(f"Error calling GPT: {e}")
        return pd.DataFrame()

# Session state
if "coords" not in st.session_state: st.session_state["coords"] = None
if "restaurants" not in st.session_state: st.session_state["restaurants"] = None
if "classified" not in st.session_state: st.session_state["classified"] = None

# UI
st.markdown("### 1) Paste Google Maps URL (short or long) and press **Start**")
url = st.text_input("Google Maps URL")
if st.button("▶️ Start — Extract Coordinates"):
    lat, lng, t = extract_coordinates(url)
    if lat is None:
        st.error(f"Could not extract coordinates (took {t}s).")
    else:
        st.session_state["coords"] = (lat,lng)
        st.success(f"Coordinates: {lat}, {lng} (extraction {t}s)")

st.markdown("### 2) Fetch nearby restaurants")
if st.button("➡️ Fetch Restaurants"):
    if not st.session_state["coords"]:
        st.error("No coordinates found.")
    else:
        lat,lng = st.session_state["coords"]
        try:
            df = fetch_restaurants_places(lat,lng, MAPS_KEY)
            st.session_state["restaurants"] = df
            st.dataframe(df[["name","address","rating","types"]].head(50))
        except Exception as e:
            st.error(f"Places API error: {e}")

st.markdown("### 3) Classify all restaurants with GPT")
if st.button("➡️ Classify All"):
    if st.session_state["restaurants"] is None:
        st.error("No restaurants loaded.")
    else:
        st.session_state["classified"] = classify_all_with_gpt(st.session_state["restaurants"])
        if not st.session_state["classified"].empty:
            st.subheader("Classified Restaurants")
            st.dataframe(st.session_state["classified"])
