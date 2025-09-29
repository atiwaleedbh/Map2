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

# Load environment variables
load_dotenv()

MAPS_KEY = os.getenv("GOOGLE_MAPS_KEY", "").strip()
OPENAI_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

st.set_page_config(page_title="Restaurant Classifier", layout="wide")
st.title("🍽️ Smart Restaurant Classifier — Cuisine-based")

# Sidebar
with st.sidebar:
    st.header("🔑 API Keys")
    maps_key_input = st.text_input("Google Maps API Key", value=MAPS_KEY, type="password")
    openai_key_input = st.text_input("OpenAI API Key", value=OPENAI_KEY, type="password")
    model_input = st.text_input("OpenAI model", value=OPENAI_MODEL)
    if st.button("Use these keys"):
        MAPS_KEY = maps_key_input.strip()
        OPENAI_KEY = openai_key_input.strip()
        OPENAI_MODEL = model_input.strip()
        st.success("✅ Keys updated (in-memory)")

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

def fetch_restaurants_with_reviews(lat, lng, maps_key, radius=3000, max_pages=1, max_reviews=3):
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
        place_id = r.get("place_id", "")
        # Fetch details with reviews
        details = client.place(place_id=place_id, fields=["review"])
        reviews = []
        for review in details.get("result", {}).get("reviews", [])[:max_reviews]:
            reviews.append(review.get("text", ""))
        rows.append({
            "name": r.get("name",""),
            "address": r.get("vicinity",""),
            "rating": r.get("rating",""),
            "types": ", ".join(r.get("types", [])),
            "reviews": " | ".join(reviews),
            "map_url": f"https://www.google.com/maps/place/?q=place_id:{place_id}"
        })
    return pd.DataFrame(rows)

def classify_cuisine_with_gpt(df):
    """Classify restaurants by cuisine using GPT with smart dish-based reasoning."""
    if df.empty:
        return pd.DataFrame()
    openai.api_key = OPENAI_KEY

    system_msg = """
    أنت مساعد متخصص في تصنيف المطاعم بدقة عالية.
    هدفك: تحديد نوع المطبخ (Cuisine) الحقيقي للمطعم استناداً إلى:
    - اسم المطعم
    - نوعه (Google types)
    - المراجعات (ابحث عن أطباق وأكلات مذكورة)

    يجب أن يكون التصنيف محدداً: مثل (هندي، باكستاني، خليجي/سعودي، لبناني، سوري، تركي،
    أمريكي، مكسيكي، إيطالي، ياباني، صيني، كوري، برجر، بيتزا، حلويات/مخبز، مقهى).

    إذا لم يتضح النوع 100%، اختر الأقرب بناءً على الأطباق المذكورة.

    أرجع النتيجة كـ JSON Array بالصيغة:
    [
      {
        "name": "...",
        "cuisine": "...",
        "confidence": "High / Medium / Low",
        "evidence": "سبب التصنيف (أسماء أطباق أو كلمات من المراجعات)",
        "rating": "...",
        "map_url": "..."
      }
    ]
    """

    restaurants_list = df[["name","types","reviews","rating","map_url"]].to_dict(orient="records")
    user_msg = f"Here are restaurant data:\n{json.dumps(restaurants_list, ensure_ascii=False)}"

    messages = [
        {"role":"system","content":system_msg},
        {"role":"user","content":user_msg}
    ]

    try:
        resp = openai.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            temperature=0,
            max_tokens=3500
        )
        text = resp.choices[0].message.content.strip()
        clean_text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE)
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
        st.error(f"❌ Could not extract coordinates (took {t}s).")
    else:
        st.session_state["coords"] = (lat,lng)
        st.success(f"✅ Coordinates: {lat}, {lng} (extraction {t}s)")

st.markdown("### 2) Fetch nearby restaurants")
if st.button("➡️ Fetch Restaurants"):
    if not st.session_state["coords"]:
        st.error("⚠️ No coordinates found.")
    else:
        lat,lng = st.session_state["coords"]
        try:
            df = fetch_restaurants_with_reviews(lat,lng, MAPS_KEY)
            st.session_state["restaurants"] = df
            st.subheader("📋 Restaurants Found")
            st.dataframe(df[["name","address","rating","types","reviews","map_url"]].head(30))
        except Exception as e:
            st.error(f"Places API error: {e}")

st.markdown("### 3) Classify restaurants by cuisine")
if st.button("➡️ Classify All"):
    if st.session_state["restaurants"] is None:
        st.error("⚠️ No restaurants loaded.")
    else:
        st.session_state["classified"] = classify_cuisine_with_gpt(st.session_state["restaurants"])
        if not st.session_state["classified"].empty:
            st.subheader("🍴 Classified Restaurants by Cuisine")
            st.dataframe(st.session_state["classified"])