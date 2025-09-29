import os, time, re, json
import pandas as pd
import requests
import openai
import googlemaps
import streamlit as st
from dotenv import load_dotenv

# Load env if exists
load_dotenv()

# Read keys
OPENAI_KEY = os.getenv("OPENAI_API_KEY", "").strip()
MAPS_KEY = os.getenv("GOOGLE_MAPS_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
openai.api_key = OPENAI_KEY

st.set_page_config(page_title="Google Maps Restaurant Classifier", layout="wide")
st.title("🍽️ Google Maps Restaurant Classifier with GPT (Batch)")

# Sidebar: keys override
with st.sidebar:
    st.header("API Keys / Settings")
    maps_key_input = st.text_input("Google Maps API Key", value=MAPS_KEY, type="password")
    openai_key_input = st.text_input("OpenAI API Key", value=OPENAI_KEY, type="password")
    model_input = st.text_input("OpenAI Model", value=OPENAI_MODEL)
    if st.button("Update Keys"):
        MAPS_KEY = maps_key_input.strip()
        OPENAI_KEY = openai_key_input.strip()
        OPENAI_MODEL = model_input.strip()
        openai.api_key = OPENAI_KEY
        st.success("Keys updated in memory")

# Session state
if "coords" not in st.session_state:
    st.session_state["coords"] = None
if "restaurants" not in st.session_state:
    st.session_state["restaurants"] = None
if "classified" not in st.session_state:
    st.session_state["classified"] = None

# Helpers
def expand_short_url(url, timeout=4):
    try:
        r = requests.get(url, allow_redirects=True, timeout=timeout)
        return r.url
    except Exception:
        return url

def extract_coordinates(url):
    start = time.time()
    if not url:
        return None, None, 0
    u = url.strip()
    if "maps.app.goo.gl" in u or "goo.gl" in u:
        u = expand_short_url(u)
    m = re.search(r'@([-+]?\d+\.\d+),([-+]?\d+\.\d+)', u)
    if m:
        return float(m.group(1)), float(m.group(2)), round(time.time()-start,3)
    return None, None, round(time.time()-start,3)

def fetch_restaurants(lat, lng, maps_key, radius=3000, max_pages=2):
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
        # fetch reviews
        try:
            details = client.place(place_id=r["place_id"], fields=["review"])
            reviews = details.get("result", {}).get("reviews", [])
            reviews_text = " | ".join([rev.get("text","") for rev in reviews[:3]])
        except:
            reviews_text = ""
        rows.append({
            "name": r.get("name",""),
            "address": r.get("vicinity",""),
            "types": ", ".join(r.get("types", [])),
            "place_id": r.get("place_id",""),
            "reviews": reviews_text
        })
    return pd.DataFrame(rows)

# Step 1: input Google Maps URL
st.subheader("Step 1: Paste Google Maps URL (short or long)")
maps_url = st.text_input("Google Maps URL", placeholder="https://www.google.com/maps/place/...")
if st.button("▶️ Start - Extract Coordinates"):
    lat, lng, t = extract_coordinates(maps_url)
    if lat is None:
        st.error(f"Could not extract coordinates (took {t}s). Try full URL or long link.")
    else:
        st.session_state["coords"] = (lat, lng)
        st.success(f"Coordinates: {lat}, {lng} (extraction {t}s)")

# Step 2: fetch restaurants & reviews
st.subheader("Step 2: Fetch Nearby Restaurants & Reviews")
if st.button("➡️ Fetch Restaurants & Reviews"):
    if not st.session_state["coords"]:
        st.error("No coordinates. Run Step 1 first.")
    elif not MAPS_KEY:
        st.error("Set Google Maps API key.")
    else:
        lat,lng = st.session_state["coords"]
        with st.spinner("Fetching restaurants and reviews..."):
            df = fetch_restaurants(lat, lng, MAPS_KEY)
            st.session_state["restaurants"] = df
            st.success(f"Fetched {len(df)} restaurants")
            st.dataframe(df[["name","address","types","reviews"]].head(50))

# Step 3: classify all at once via GPT
st.subheader("Step 3: Classify All Restaurants with GPT")
if st.button("➡️ Classify All Restaurants"):
    df = st.session_state["restaurants"]
    if df is None or df.empty:
        st.error("No restaurants to classify. Run Step 2 first.")
    else:
        prompt = "You are an assistant. Categorize the following restaurants based on name, address, types, and reviews. " \
                 "Return JSON array of {name, category}.\n"
        for _, row in df.iterrows():
            prompt += f"- Name: {row['name']}, Address: {row['address']}, Types: {row['types']}, Reviews: {row['reviews']}\n"
        try:
            with st.spinner("Classifying with GPT..."):
                resp = openai.ChatCompletion.create(
                    model=OPENAI_MODEL,
                    messages=[
                        {"role":"system","content":"You categorize restaurants dynamically."},
                        {"role":"user","content":prompt}
                    ],
                    max_tokens=1500,
                    temperature=0.7
                )
                text = resp["choices"][0]["message"]["content"]
                try:
                    categorized = json.loads(text)
                    st.session_state["classified"] = pd.DataFrame(categorized)
                    st.subheader("Categorized Restaurants")
                    st.dataframe(st.session_state["classified"])
                except Exception:
                    st.warning("Could not parse GPT output as JSON. Raw output:")
                    st.text(text)
        except Exception as e:
            st.error(f"Error calling GPT: {e}")
