# app.py
import os
import re
import time
import json
import requests
import pandas as pd
import streamlit as st
import googlemaps
from dotenv import load_dotenv
from openai import OpenAI

# ---------------- Config / Load env ----------------
load_dotenv()
MAPS_KEY = os.getenv("GOOGLE_MAPS_KEY", "").strip()
OPENAI_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")

client = None
if OPENAI_KEY:
    client = OpenAI(api_key=OPENAI_KEY)

# ---------------- Streamlit page ----------------
st.set_page_config(page_title="Restaurant Classifier (Smart Cuisine)", layout="wide")
st.title("📍 Smart Restaurant Classifier — Cuisine-based (Google Maps → GPT)")

# Sidebar
with st.sidebar:
    st.header("API Keys & Settings")
    maps_key_input = st.text_input("Google Maps API Key", value=MAPS_KEY, type="password")
    openai_key_input = st.text_input("OpenAI API Key", value=OPENAI_KEY, type="password")
    model_input = st.text_input("OpenAI Model", value=OPENAI_MODEL)
    rating_threshold = st.number_input("Minimum rating to include (>=)", 0.0, 5.0, 4.0, 0.1)
    max_places = st.number_input("Max places to fetch (per page)", 10, 60, 40, 5)
    show_raw = st.checkbox("Show raw GPT output", value=False)
    if st.button("Apply keys/settings"):
        MAPS_KEY = maps_key_input.strip()
        OPENAI_KEY = openai_key_input.strip()
        OPENAI_MODEL = model_input.strip() or OPENAI_MODEL
        client = OpenAI(api_key=OPENAI_KEY) if OPENAI_KEY else None
        st.success("Settings applied.")

st.write("Maps key loaded:", bool(MAPS_KEY), " — OpenAI key loaded:", bool(OPENAI_KEY))
st.write("Model:", OPENAI_MODEL, " — Rating threshold:", rating_threshold)

# ---------------- Helpers ----------------
def expand_short_url_once(url: str, timeout=4):
    try:
        r = requests.get(url, allow_redirects=True, timeout=timeout)
        return r.url
    except Exception:
        return url

def extract_coordinates(url: str):
    if not url: return None, None, 0
    u = url.strip()
    if "maps.app.goo.gl" in u or "goo.gl" in u:
        u = expand_short_url_once(u)
    m = re.search(r'@([-+]?\d+\.\d+),([-+]?\d+\.\d+)', u)
    if m: return float(m.group(1)), float(m.group(2)), 0
    m = re.search(r'!3d([-+]?\d+\.\d+)!4d([-+]?\d+\.\d+)', u)
    if m: return float(m.group(1)), float(m.group(2)), 0
    m = re.search(r'([-+]?\d{1,3}\.\d+)[, ]+([-+]?\d{1,3}\.\d+)', u)
    if m:
        lat, lng = float(m.group(1)), float(m.group(2))
        if -90 <= lat <= 90 and -180 <= lng <= 180:
            return lat, lng, 0
    return None, None, 0

def fetch_restaurants_with_reviews(lat, lng, maps_key, radius_m=3000, max_pages=1, max_reviews=3, max_places=40):
    client_maps = googlemaps.Client(key=maps_key)
    all_results = []
    places = client_maps.places_nearby(location=(lat,lng), radius=radius_m, type="restaurant")
    all_results.extend(places.get("results", []))
    pages = 0
    while places.get("next_page_token") and pages < max_pages:
        pages += 1
        time.sleep(2)
        places = client_maps.places_nearby(page_token=places["next_page_token"])
        all_results.extend(places.get("results", []))

    all_results = all_results[:max_places]
    rows = []
    for r in all_results:
        place_id = r.get("place_id", "")
        reviews_texts = []
        try:
            details = client_maps.place(place_id=place_id, fields=["review","name","formatted_address"])
            for rev in details.get("result", {}).get("reviews", [])[:max_reviews]:
                reviews_texts.append(rev.get("text",""))
        except Exception:
            reviews_texts = []
        rows.append({
            "name": r.get("name",""),
            "address": r.get("vicinity") or details.get("result",{}).get("formatted_address",""),
            "rating": r.get("rating", None),
            "map_url": f"https://www.google.com/maps/place/?q=place_id:{place_id}",
            "reviews": reviews_texts
        })
    return pd.DataFrame(rows)

def clean_and_parse_json(raw_text: str):
    text = raw_text.strip()
    text = re.sub(r"^```(?:json)?", "", text)
    text = re.sub(r"```$", "", text)
    text = text.strip()
    text = re.sub(r",\s*([\]}])", r"\1", text)
    return json.loads(text)

# ---------------- Session ----------------
if "coords" not in st.session_state: st.session_state["coords"] = None
if "restaurants" not in st.session_state: st.session_state["restaurants"] = None
if "classified" not in st.session_state: st.session_state["classified"] = None
if "raw_gpt" not in st.session_state: st.session_state["raw_gpt"] = ""

# ---------------- UI ----------------
st.markdown("### 1) Paste Google Maps URL")
maps_url = st.text_input("Paste Google Maps URL", "")
if st.button("📍 Extract Coordinates"):
    lat,lng,_ = extract_coordinates(maps_url)
    if lat is None:
        st.error("Could not extract coordinates from URL.")
    else:
        st.session_state["coords"] = (lat,lng)
        st.success(f"Coordinates extracted: {lat}, {lng}")

radius_km = st.number_input("Search radius (km)", 0.5, 10.0, 3.0, 0.5)
radius_m = int(radius_km * 1000)

st.markdown("### 2) Fetch nearby restaurants")
if st.button("🍴 Fetch Restaurants"):
    if not st.session_state["coords"]:
        st.error("No coordinates extracted.")
    elif not MAPS_KEY:
        st.error("Google Maps API key missing.")
    else:
        lat,lng = st.session_state["coords"]
        with st.spinner("Fetching restaurants..."):
            df_places = fetch_restaurants_with_reviews(lat, lng, MAPS_KEY, radius_m, 1, 3, int(max_places))
            df_filtered = df_places[df_places["rating"].notna() & (df_places["rating"] >= float(rating_threshold))]
            st.session_state["restaurants"] = df_filtered.reset_index(drop=True)
            st.success(f"Found {len(df_places)} total, {len(df_filtered)} with rating >= {rating_threshold}.")

if st.session_state["restaurants"] is not None:
    st.dataframe(st.session_state["restaurants"][["name","address","rating","map_url"]])

st.markdown("### 3) Classify with GPT")
if st.button("🤖 Classify All with GPT"):
    if st.session_state["restaurants"] is None or st.session_state["restaurants"].empty:
        st.error("No restaurants loaded.")
    elif not client:
        st.error("OpenAI API key missing.")
    else:
        df_to_send = st.session_state["restaurants"].copy()
        records = df_to_send[["name","rating","map_url"]].to_dict(orient="records")
        prompt_user = (
            "You are a smart assistant that classifies restaurants by cuisine type.\n"
            "For each restaurant entry (only name), guess the cuisine (e.g. Indian, Asian, Khaleeji, Egyptian, Lebanese, American, Burger, Pizza, Seafood, Fast Food, Cafe, Bakery, etc.).\n"
            "Return STRICT JSON array with: name, cuisine, confidence (High/Medium/Low), rating, map_url.\n\n"
            "Restaurants:\n" + json.dumps(records, ensure_ascii=False)
        )
        try:
            with st.spinner("Calling GPT..."):
                resp = client.chat.completions.create(
                    model=OPENAI_MODEL,
                    messages=[
                        {"role":"system","content":"Return only valid JSON."},
                        {"role":"user","content":prompt_user}
                    ],
                    temperature=0.2,
                    max_tokens=1500
                )
                raw_output = resp.choices[0].message.content
                st.session_state["raw_gpt"] = raw_output
                if show_raw: st.text(raw_output)
                parsed = clean_and_parse_json(raw_output)
                df_classified = pd.DataFrame(parsed)
                st.session_state["classified"] = df_classified
                st.success("Classification completed.")
                st.dataframe(df_classified)
        except Exception as e:
            st.error(f"Error: {e}")
            st.text(st.session_state["raw_gpt"])