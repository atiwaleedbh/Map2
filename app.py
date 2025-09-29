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
from openai import OpenAI  # واجهة OpenAI الحديثة

# ---------------- Config / Load env ----------------
load_dotenv()
MAPS_KEY = os.getenv("GOOGLE_MAPS_KEY", "").strip()
OPENAI_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")  # انصح gpt-3.5-turbo أو gpt-4o-mini إن متاح

# instantiate OpenAI client (new SDK)
client = None
if OPENAI_KEY:
    client = OpenAI(api_key=OPENAI_KEY)

# ---------------- Streamlit page ----------------
st.set_page_config(page_title="Restaurant Classifier (Smart Cuisine)", layout="wide")
st.title("📍 Smart Restaurant Classifier — Cuisine-based (Google Maps → GPT)")

# Sidebar: keys & settings
with st.sidebar:
    st.header("API Keys & Settings")
    maps_key_input = st.text_input("Google Maps API Key", value=MAPS_KEY, type="password")
    openai_key_input = st.text_input("OpenAI API Key", value=OPENAI_KEY, type="password")
    model_input = st.text_input("OpenAI Model", value=OPENAI_MODEL)
    rating_threshold = st.number_input("Minimum rating to include (>=)", min_value=0.0, max_value=5.0, value=4.0, step=0.1)
    max_places = st.number_input("Max places to fetch (per page)", min_value=10, max_value=60, value=40, step=5)
    show_raw = st.checkbox("Show raw GPT output (for debugging)", value=False)
    if st.button("Apply keys/settings"):
        MAPS_KEY = maps_key_input.strip()
        OPENAI_KEY = openai_key_input.strip()
        OPENAI_MODEL = model_input.strip() or OPENAI_MODEL
        client = OpenAI(api_key=OPENAI_KEY) if OPENAI_KEY else None
        st.success("Keys/settings applied (in-memory).")

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
    """Try several patterns to extract lat,lng from a Google Maps URL (short or long)."""
    start = time.time()
    if not url:
        return None, None, round(time.time()-start,3)
    u = url.strip()
    if "maps.app.goo.gl" in u or "goo.gl" in u:
        u = expand_short_url_once(u)
    # @lat,lng pattern
    m = re.search(r'@([-+]?\d+\.\d+),([-+]?\d+\.\d+)', u)
    if m:
        return float(m.group(1)), float(m.group(2)), round(time.time()-start,3)
    # !3dLAT!4dLNG pattern
    m = re.search(r'!3d([-+]?\d+\.\d+)!4d([-+]?\d+\.\d+)', u)
    if m:
        return float(m.group(1)), float(m.group(2)), round(time.time()-start,3)
    # fallback any lat,lng pair in URL
    m = re.search(r'([-+]?\d{1,3}\.\d+)[, ]+([-+]?\d{1,3}\.\d+)', u)
    if m:
        lat, lng = float(m.group(1)), float(m.group(2))
        if -90 <= lat <= 90 and -180 <= lng <= 180:
            return lat, lng, round(time.time()-start,3)
    return None, None, round(time.time()-start,3)

def fetch_restaurants_with_reviews(lat, lng, maps_key, radius_m=3000, max_pages=1, max_reviews=3, max_places=40):
    """
    Use Google Places 'nearbysearch' then 'place' details for reviews.
    Returns DataFrame with name, address, rating, types, reviews (first 3), map_url.
    """
    if not maps_key:
        raise ValueError("Google Maps API key missing.")
    client_maps = googlemaps.Client(key=maps_key)
    all_results = []
    places = client_maps.places_nearby(location=(lat,lng), radius=radius_m, type="restaurant", page_token=None)
    all_results.extend(places.get("results", []))
    pages = 0
    # fetch next pages up to max_pages
    while places.get("next_page_token") and pages < max_pages:
        pages += 1
        time.sleep(2)  # required wait
        places = client_maps.places_nearby(page_token=places["next_page_token"])
        all_results.extend(places.get("results", []))

    # limit total places
    all_results = all_results[:max_places]

    rows = []
    for r in all_results:
        place_id = r.get("place_id", "")
        # fetch place details -> reviews
        reviews_texts = []
        try:
            details = client_maps.place(place_id=place_id, fields=["review","name","formatted_address"])
            for rev in details.get("result", {}).get("reviews", [])[:max_reviews]:
                reviews_texts.append(rev.get("text",""))
        except Exception:
            # ignore details errors, keep empty reviews
            reviews_texts = []
        rows.append({
            "name": r.get("name",""),
            "address": r.get("vicinity") or details.get("result",{}).get("formatted_address",""),
            "rating": r.get("rating", None),
            "types": ", ".join(r.get("types", [])),
            "reviews": reviews_texts,
            "map_url": f"https://www.google.com/maps/place/?q=place_id:{place_id}"
        })
    df = pd.DataFrame(rows)
    return df

# Robust JSON cleaner/repair before json.loads
def clean_and_parse_json(raw_text: str):
    """
    Clean common GPT output issues:
    - Remove surrounding ``` or ```json fences
    - Replace single quotes with double quotes (careful)
    - Remove trailing commas before } or ]
    - Try to extract the first JSON array/object in the text
    Returns python object or raises JSONDecodeError.
    """
    if raw_text is None:
        raise json.JSONDecodeError("Empty text", raw_text or "", 0)
    text = raw_text.strip()

    # remove code fences
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)

    # try to find first JSON array or object
    # find substring starting at first '[' or '{' and ending at matching bracket
    first_bracket = None
    for i,ch in enumerate(text):
        if ch in ('[','{'):
            first_bracket = i
            break
    if first_bracket is not None:
        text_sub = text[first_bracket:]
    else:
        text_sub = text

    # remove trailing commas like ,] or ,}
    text_sub = re.sub(r",\s*([\]}])", r"\1", text_sub)

    # replace smart quotes and single quotes to double quotes cautiously
    # First normalize unicode quotes
    text_sub = text_sub.replace("“","\"").replace("”","\"").replace("’","'")
    # Attempt to fix single-quoted JSON keys/values by replacing only when safe:
    # (best-effort) replace '([^']*)' -> "..."
    # but avoid changing apostrophes within words — use a regex that matches keys/values patterns.
    try:
        # Replace keys 'key':  OR 'key' : with "key":
        text_sub = re.sub(r"(?<=^|[\s\[,])'([A-Za-z0-9_\- ]+?)'\s*:", r'"\1":', text_sub)
        # Replace simple 'value' occurrences that are followed by comma or closing bracket/braces
        text_sub = re.sub(r":\s*'([^']*?)'(?=\s*[,\]}])", lambda m: ': "' + m.group(1).replace('"','\\"') + '"', text_sub)
    except Exception:
        pass

    # final replace of any remaining single quotes that look like they delimit items (best-effort)
    # only as last resort:
    if text_sub.count('"') < text_sub.count("'"):
        text_sub = text_sub.replace("'", '"')

    # Trim to balanced JSON (try to find matching closing bracket for first bracket)
    # If starts with [, find matching ] ; if {, find matching }.
    def find_matching(s, start_idx):
        stack = []
        opens = {'[':']', '{':'}'}
        open_ch = s[start_idx]
        close_ch = opens[open_ch]
        for i in range(start_idx, len(s)):
            ch = s[i]
            if ch == open_ch:
                stack.append(ch)
            elif ch == close_ch:
                stack.pop()
                if not stack:
                    return i
        return None

    if text_sub and text_sub[0] in ('[','{'):
        end_idx = find_matching(text_sub, 0)
        if end_idx:
            text_sub = text_sub[:end_idx+1]

    # final strip
    text_sub = text_sub.strip()

    # Attempt parse
    return json.loads(text_sub)

# ---------------- Session state ----------------
if "coords" not in st.session_state:
    st.session_state["coords"] = None
if "restaurants" not in st.session_state:
    st.session_state["restaurants"] = None
if "classified" not in st.session_state:
    st.session_state["classified"] = None
if "raw_gpt" not in st.session_state:
    st.session_state["raw_gpt"] = ""

# ---------------- UI steps ----------------
st.markdown("## Steps")
st.markdown("1. Paste a Google Maps link (place / neighborhood / point) -> Extract coordinates.")
st.markdown("2. Choose search radius (km) and press Fetch Restaurants (gets first 3 reviews + rating + link).")
st.markdown("3. Press Classify to send filtered restaurants (rating>=threshold) to GPT for smart cuisine classification.")

# Step 1: Paste link and extract coords
st.markdown("### 1) Paste Google Maps URL")
maps_url = st.text_input("Paste Google Maps URL (short or long)", value="")
if st.button("📍 Extract Coordinates"):
    lat,lng,t = extract_coordinates(maps_url)
    if lat is None:
        st.error(f"Could not extract coordinates (took {t}s). Make sure the URL contains coordinates (e.g. @26.123,50.123).")
    else:
        st.session_state["coords"] = (lat,lng)
        st.success(f"Coordinates extracted: {lat}, {lng}")

# radius input
radius_km = st.number_input("Search radius (km)", min_value=0.5, max_value=10.0, value=3.0, step=0.5)
radius_m = int(radius_km * 1000)

# Step 2: Fetch restaurants
st.markdown("### 2) Fetch nearby restaurants")
col1, col2 = st.columns([1,3])
with col1:
    if st.button("🍴 Fetch Restaurants"):
        if not st.session_state["coords"]:
            st.error("No coordinates extracted. Run step 1 first.")
        elif not MAPS_KEY:
            st.error("Google Maps API key missing. Set it in the sidebar or environment.")
        else:
            lat,lng = st.session_state["coords"]
            with st.spinner("Fetching restaurants and reviews... (this may take a few seconds)"):
                try:
                    df_places = fetch_restaurants_with_reviews(lat, lng, MAPS_KEY, radius_m, max_pages=1, max_reviews=3, max_places=int(max_places))
                    # filter by rating threshold
                    df_filtered = df_places[df_places["rating"].notna() & (df_places["rating"] >= float(rating_threshold))]
                    st.session_state["restaurants"] = df_filtered.reset_index(drop=True)
                    st.success(f"Found {len(df_places)} total, {len(df_filtered)} with rating >= {rating_threshold}.")
                except Exception as e:
                    st.error(f"Error fetching places: {e}")
with col2:
    st.write("Results will show in the table below. You can adjust radius and rating threshold in the sidebar.")

if st.session_state["restaurants"] is not None:
    st.subheader("📋 Restaurants (filtered)")
    # show first 50 rows
    display_df = st.session_state["restaurants"].copy()
    # convert reviews list to multiline string for display if needed
    display_df["reviews_preview"] = display_df["reviews"].apply(lambda x: "\n---\n".join(x) if isinstance(x, list) else x)
    st.dataframe(display_df[["name","address","rating","types","reviews_preview","map_url"]].rename(columns={"reviews_preview":"first_reviews"}).head(100))

# Step 3: Classify with GPT
st.markdown("### 3) Classify (smart cuisine) with GPT")
if st.button("🤖 Classify All with GPT"):
    if st.session_state["restaurants"] is None or st.session_state["restaurants"].empty:
        st.error("No restaurants loaded. Run Fetch Restaurants first.")
    elif not OPENAI_KEY or not client:
        st.error("OpenAI API key missing. Set it in the sidebar or environment.")
    else:
        df_to_send = st.session_state["restaurants"].copy()
        # prepare list of dicts for GPT (limit to 50 records to avoid huge prompts)
        records = df_to_send[["name","types","reviews","rating","map_url"]].to_dict(orient="records")[:50]
        prompt_user = (
            "You are a smart assistant that classifies restaurants by cuisine.\n"
            "For each restaurant entry (name, types, up to 3 reviews, rating, map_url), "
            "determine the most appropriate cuisine category (e.g. Indian, Asian, Khaleeji, Egyptian, Lebanese, American, Burger, Pizza, Seafood, Fast Food, Cafe, Bakery, etc.).\n"
            "Return STRICT JSON: an array of objects with keys: name, cuisine, confidence (High/Medium/Low), evidence (short phrase quoting review or types), rating, map_url.\n\n"
            "Restaurants:\n" + json.dumps(records, ensure_ascii=False)
        )

        system_msg = "You are a helpful assistant that MUST return valid JSON (double quotes) only. No extra text."
        try:
            with st.spinner("Calling GPT to classify..."):
                resp = client.chat.completions.create(
                    model=OPENAI_MODEL,
                    messages=[
                        {"role":"system", "content": system_msg},
                        {"role":"user", "content": prompt_user}
                    ],
                    temperature=0.2,
                    max_tokens=2000
                )
                raw_output = resp.choices[0].message.content
                st.session_state["raw_gpt"] = raw_output
                if show_raw:
                    st.subheader("Raw GPT output")
                    st.text(raw_output)

                # clean & parse
                try:
                    parsed = clean_and_parse_json(raw_output)
                    df_classified = pd.DataFrame(parsed)
                    # Ensure all columns exist
                    expected_cols = ["name","cuisine","confidence","evidence","rating","map_url"]
                    for c in expected_cols:
                        if c not in df_classified.columns:
                            df_classified[c] = None
                    st.session_state["classified"] = df_classified
                    st.success("Classification completed.")
                    st.subheader("🍴 Classified Restaurants")
                    st.dataframe(df_classified[expected_cols])
                except Exception as e:
                    st.error(f"Failed to parse GPT JSON: {e}")
                    st.subheader("Raw GPT output for inspection")
                    st.text(raw_output)
        except Exception as e:
            st.error(f"Error calling OpenAI: {e}")

# Show classified if available
if st.session_state.get("classified") is not None:
    st.markdown("### Final classified table")
    st.dataframe(st.session_state["classified"])

# Footer / tips
st.markdown("---")
st.write("Tips: if JSON parsing fails, tick 'Show raw GPT output' in the sidebar to inspect. "
         "You can reduce number of places fetched or lower model tokens to avoid truncation.")