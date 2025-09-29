import os, time, json, re
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

st.set_page_config(page_title="Batch Restaurant Classifier", layout="wide")
st.title("🍽️ Batch Restaurant Classifier with Reviews & GPT")

# Sidebar keys override
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
if "restaurants" not in st.session_state:
    st.session_state["restaurants"] = pd.DataFrame([
        {"name": "مطاعم فلفلة", "address": "Dammam"},
        {"name": "Burger King", "address": "Riyadh"},
        {"name": "Indian Palace", "address": "Jeddah"}
    ])
if "classified" not in st.session_state:
    st.session_state["classified"] = None

# Functions
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

def fetch_reviews(place_id, client, max_reviews=3):
    try:
        details = client.place(place_id=place_id, fields=["review"])
        reviews = details.get("result", {}).get("reviews", [])
        return " | ".join(r.get("text","") for r in reviews[:max_reviews])
    except:
        return ""

def enrich_restaurants(df, maps_key):
    client = googlemaps.Client(key=maps_key)
    enriched = []
    for _, row in df.iterrows():
        try:
            result = client.find_place(input=row["name"], input_type="textquery", fields=["place_id","formatted_address","types"])
            candidates = result.get("candidates", [])
            if not candidates:
                enriched.append({**row, "place_id":"", "reviews":""})
                continue
            place = candidates[0]
            reviews = fetch_reviews(place["place_id"], client)
            enriched.append({
                "name": row["name"],
                "address": place.get("formatted_address", row["address"]),
                "types": ",".join(place.get("types", [])),
                "place_id": place["place_id"],
                "reviews": reviews
            })
        except:
            enriched.append({**row, "place_id":"", "reviews":""})
    return pd.DataFrame(enriched)

# Step 1: show restaurants
st.subheader("Restaurants")
st.dataframe(st.session_state["restaurants"])

# Step 2: fetch reviews
if st.button("➡️ Fetch Reviews from Google Maps"):
    if not MAPS_KEY:
        st.error("Set Google Maps API key first")
    else:
        with st.spinner("Fetching reviews..."):
            df_enriched = enrich_restaurants(st.session_state["restaurants"], MAPS_KEY)
            st.session_state["restaurants"] = df_enriched
            st.success("Reviews fetched")
            st.dataframe(df_enriched)

# Step 3: batch classify with GPT
if st.button("➡️ Classify All Restaurants with GPT"):
    df = st.session_state["restaurants"]
    if df.empty or "reviews" not in df.columns:
        st.error("No restaurant data with reviews. Run previous step first.")
    else:
        prompt = "You are an assistant that categorizes restaurants based on names, addresses, types, and reviews. " \
                 "Create dynamic categories and return a JSON array [{name, category}].\n"
        for _, row in df.iterrows():
            prompt += f"- Name: {row['name']}, Address: {row.get('address','')}, Types: {row.get('types','')}, Reviews: {row.get('reviews','')}\n"

        try:
            with st.spinner("Classifying..."):
                t0 = time.time()
                resp = openai.ChatCompletion.create(
                    model=OPENAI_MODEL,
                    messages=[
                        {"role":"system","content":"You categorize restaurants dynamically."},
                        {"role":"user","content":prompt}
                    ],
                    max_tokens=1500,
                    temperature=0.7
                )
                elapsed = round(time.time()-t0,2)
                text = resp["choices"][0]["message"]["content"]

                try:
                    categorized = json.loads(text)
                    categorized_df = pd.DataFrame(categorized)
                    st.subheader("Categorized Restaurants")
                    st.dataframe(categorized_df)
                except Exception:
                    st.warning("Could not parse GPT response as JSON. Raw output:")
                    st.text(text)
                st.success(f"Classification completed in {elapsed} seconds")
        except Exception as e:
            st.error(f"Error calling OpenAI API: {e}")
