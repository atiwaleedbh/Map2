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
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")  # غيّر إلى نموذجك إن رغبت

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
    max_places = st.number_input("Max places to fetch", min_value=5, max_value=60, value=40, step=5)
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
    if not url:
        return None, None, 0.0
    u = url.strip()
    if "maps.app.goo.gl" in u or "goo.gl" in u:
        u = expand_short_url_once(u)
    # @lat,lng pattern
    m = re.search(r'@([-+]?\d+\.\d+),([-+]?\d+\.\d+)', u)
    if m:
        return float(m.group(1)), float(m.group(2)), 0.0
    # !3dLAT!4dLNG pattern
    m = re.search(r'!3d([-+]?\d+\.\d+)!4d([-+]?\d+\.\d+)', u)
    if m:
        return float(m.group(1)), float(m.group(2)), 0.0
    # fallback any lat,lng pair in URL
    m = re.search(r'([-+]?\d{1,3}\.\d+)[, ]+([-+]?\d{1,3}\.\d+)', u)
    if m:
        lat, lng = float(m.group(1)), float(m.group(2))
        if -90 <= lat <= 90 and -180 <= lng <= 180:
            return lat, lng, 0.0
    return None, None, 0.0

def fetch_restaurants_with_reviews(lat, lng, maps_key, radius_m=3000, max_pages=1, max_reviews=3, max_places=40):
    """
    Use googlemaps client to call Places Nearby and Place Details for reviews (first N).
    Returns DataFrame with name, address, rating, types, reviews(list), map_url.
    """
    if not maps_key:
        raise ValueError("Google Maps API key missing.")
    client_maps = googlemaps.Client(key=maps_key)
    all_results = []
    places = client_maps.places_nearby(location=(lat,lng), radius=radius_m, type="restaurant", page_token=None)
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
        addr = r.get("vicinity", "")
        try:
            details = client_maps.place(place_id=place_id, fields=["review","formatted_address"])
            # first max_reviews texts
            for rev in details.get("result", {}).get("reviews", [])[:max_reviews]:
                # keep full review but we'll show truncated later
                reviews_texts.append(rev.get("text",""))
            # use formatted address if available
            addr = details.get("result", {}).get("formatted_address", addr)
        except Exception:
            # ignore details errors
            pass
        rows.append({
            "name": r.get("name",""),
            "address": addr,
            "rating": r.get("rating", None),
            "types": ", ".join(r.get("types", [])),
            "reviews": reviews_texts,  # list
            "map_url": f"https://www.google.com/maps/place/?q=place_id:{place_id}"
        })
    df = pd.DataFrame(rows)
    return df

# Robust JSON cleaner/repair before json.loads
def clean_and_parse_json(raw_text: str):
    """
    Try multiple heuristics to clean & repair common GPT JSON issues:
    - remove code fences
    - remove trailing commas
    - normalize quotes
    - if missing closing brackets/quotes, attempt to append closers
    - as last resort, extract individual {...} objects and build array
    Returns python object or raises JSONDecodeError.
    """
    if raw_text is None:
        raise json.JSONDecodeError("Empty text", raw_text or "", 0)
    text = raw_text.strip()

    # remove leading/trailing code fences
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text, flags=re.IGNORECASE)

    # find first JSON start
    first_bracket = None
    for i,ch in enumerate(text):
        if ch in ('[','{'):
            first_bracket = i
            break
    if first_bracket is None:
        raise json.JSONDecodeError("No JSON start found", text, 0)
    sub = text[first_bracket:].strip()

    # remove trailing commas before ] or }
    sub = re.sub(r",\s*([\]}])", r"\1", sub)

    # normalize smart quotes
    sub = sub.replace("“", '"').replace("”", '"').replace("„", '"').replace("’", "'").replace("‘","'")

    # try direct load first
    try:
        return json.loads(sub)
    except json.JSONDecodeError as e:
        last_error = e

    # Attempt 1: if missing closing brackets, append the needed closing brackets
    if sub and sub[0] in ('[','{'):
        opens = {'[':']','{':'}'}
        stack = []
        for ch in sub:
            if ch in opens:
                stack.append(opens[ch])
            elif stack and ch == stack[-1]:
                stack.pop()
        if stack:
            sub2 = sub + ''.join(reversed(stack))
            try:
                return json.loads(sub2)
            except json.JSONDecodeError as e:
                last_error = e
                sub = sub2  # continue with sub2 for next heuristics

    # Attempt 2: if odd number of double quotes, append closing quote and closers
    if sub.count('"') % 2 == 1:
        sub2 = sub + '"'
        # append closers if needed
        opens = {'[':']','{':'}'}
        stack = []
        for ch in sub2:
            if ch in opens:
                stack.append(opens[ch])
            elif stack and ch == stack[-1]:
                stack.pop()
        sub2 = sub2 + ''.join(reversed(stack))
        try:
            return json.loads(sub2)
        except json.JSONDecodeError as e:
            last_error = e
            # fall through

    # Attempt 3: if it's an array but last element incomplete, try to strip last comma+fragment and close array
    if sub.startswith('['):
        # try to find last complete object '}' and cut there
        last_close = sub.rfind('}')
        if last_close != -1:
            candidate = sub[:last_close+1]
            # ensure it starts with [ and ends with ]
            if not candidate.strip().endswith(']'):
                candidate = candidate + ']'
            try:
                return json.loads(candidate)
            except json.JSONDecodeError as e:
                last_error = e

    # Attempt 4: extract all {...} objects and form an array (best-effort)
    objs = re.findall(r'\{[^{}]*\}', sub)
    if objs:
        arr_text = '[' + ','.join(objs) + ']'
        try:
            return json.loads(arr_text)
        except json.JSONDecodeError as e:
            last_error = e

    # If all attempts fail, raise the most recent JSON error with raw for debugging
    raise last_error

# ---------------- Session state ----------------
if "coords" not in st.session_state: st.session_state["coords"] = None
if "restaurants" not in st.session_state: st.session_state["restaurants"] = None
if "classified" not in st.session_state: st.session_state["classified"] = None
if "raw_gpt" not in st.session_state: st.session_state["raw_gpt"] = ""

# ---------------- UI steps ----------------
st.markdown("## Steps")
st.markdown("1. Paste a Google Maps link (place / neighborhood / point) → Extract coordinates.")
st.markdown("2. Choose search radius (km) and press Fetch Restaurants (gets first 3 reviews + rating + link).")
st.markdown("3. Press Classify to send filtered restaurants (rating >= threshold) to GPT for smart cuisine classification (names only).")

# Step 1: Paste link and extract coords
st.markdown("### 1) Paste Google Maps URL")
maps_url = st.text_input("Paste Google Maps URL (short or long)", value="")
if st.button("📍 Extract Coordinates"):
    lat,lng,_ = extract_coordinates(maps_url)
    if lat is None:
        st.error("Could not extract coordinates. Make sure URL contains coordinates like @26.39,50.05 or use a place URL.")
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
            st.error("Google Maps API key missing. Add it in the sidebar or environment.")
        else:
            lat,lng = st.session_state["coords"]
            with st.spinner("Fetching restaurants and reviews... (this may take a few seconds)"):
                try:
                    df_places = fetch_restaurants_with_reviews(lat, lng, MAPS_KEY, radius_m=radius_m, max_pages=1, max_reviews=3, max_places=int(max_places))
                    # filter by rating threshold
                    df_filtered = df_places[df_places["rating"].notna() & (df_places["rating"] >= float(rating_threshold))]
                    # show only first 100 to UI
                    st.session_state["restaurants"] = df_filtered.reset_index(drop=True)
                    st.success(f"Found {len(df_places)} total, {len(df_filtered)} with rating >= {rating_threshold}.")
                except Exception as e:
                    st.error(f"Error fetching places: {e}")
with col2:
    st.write("Results will show in the table below. Adjust radius and rating threshold in the sidebar.")

# Show fetched restaurants (with first 3 reviews displayed in UI only)
if st.session_state["restaurants"] is not None:
    st.subheader("📋 Restaurants (filtered)")
    display_df = st.session_state["restaurants"].copy()
    # show reviews as truncated multiline for UI, but we will NOT send reviews to GPT
    def preview_reviews(revs):
        if not isinstance(revs, list) or len(revs) == 0:
            return ""
        # show up to 3 reviews, each truncated to 200 chars
        return "\n---\n".join([r[:200].replace("\n"," ") + ("..." if len(r)>200 else "") for r in revs[:3]])
    display_df["first_reviews"] = display_df["reviews"].apply(preview_reviews)
    st.dataframe(display_df[["name","address","rating","types","first_reviews","map_url"]].head(200))

# Step 3: Classify with GPT (names only)
st.markdown("### 3) Classify (smart cuisine) with GPT — Names ONLY")
if st.button("🤖 Classify All with GPT"):
    if st.session_state["restaurants"] is None or st.session_state["restaurants"].empty:
        st.error("No restaurants loaded. Run Fetch Restaurants first.")
    elif not OPENAI_KEY or not client:
        st.error("OpenAI API key missing. Add it in the sidebar or environment.")
    else:
        df_to_send = st.session_state["restaurants"].copy()
        # only names (limit to 50)
        names = df_to_send["name"].tolist()[:50]
        if not names:
            st.error("No restaurant names to send.")
        else:
            # build prompt: send only names, ask for strict JSON
            names_block = "\n".join([f"- {n}" for n in names])
            system_msg = "You are a smart assistant. Return ONLY a VALID JSON array (no extra commentary). Use double quotes only."
            user_prompt = (
                "Classify each of the following restaurant NAMES into a cuisine category (examples: Asian, Indian, Khaleeji, Egyptian, Lebanese, American, Burger, Pizza, Seafood, Fast Food, Cafe, Bakery, etc.).\n"
                "You must return STRICT JSON array of objects with keys: name, cuisine, confidence (High/Medium/Low), rating (if known else null), map_url (if known else null).\n\n"
                "Restaurant NAMES:\n" + names_block
            )

            try:
                with st.spinner("Calling GPT to classify (may take a few seconds)..."):
                    resp = client.chat.completions.create(
                        model=OPENAI_MODEL,
                        messages=[
                            {"role":"system","content": system_msg},
                            {"role":"user","content": user_prompt}
                        ],
                        temperature=0.2,
                        max_tokens=2000
                    )
                    raw_output = resp.choices[0].message.content
                    st.session_state["raw_gpt"] = raw_output
                    if show_raw:
                        st.subheader("Raw GPT output")
                        st.text(raw_output)

                    # Try parsing / cleaning
                    try:
                        parsed = clean_and_parse_json(raw_output)
                    except Exception as parse_err:
                        # parsing failed — attempt one retry asking GPT to return only JSON (shorter)
                        st.warning(f"Initial parse failed: {parse_err}. Attempting a single retry asking for JUST JSON.")
                        retry_system = "You must return only a valid JSON array (no explanation). Use double quotes."
                        retry_user = "Previous response was invalid or truncated. Please return ONLY the JSON array for these names (no text):\n" + names_block
                        try:
                            resp2 = client.chat.completions.create(
                                model=OPENAI_MODEL,
                                messages=[
                                    {"role":"system","content": retry_system},
                                    {"role":"user","content": retry_user}
                                ],
                                temperature=0.0,
                                max_tokens=2000
                            )
                            raw_output2 = resp2.choices[0].message.content
                            st.session_state["raw_gpt"] = raw_output2
                            if show_raw:
                                st.subheader("Raw GPT output (retry)")
                                st.text(raw_output2)
                            parsed = clean_and_parse_json(raw_output2)
                        except Exception as e2:
                            st.error(f"Retry also failed to produce valid JSON: {e2}")
                            st.subheader("Last raw GPT output for inspection")
                            st.text(raw_output if not raw_output else (raw_output2 if 'raw_output2' in locals() else raw_output))
                            parsed = None

                    if parsed is None:
                        st.error("Could not parse GPT output into JSON. See raw output (enable Show raw GPT output).")
                    else:
                        # parsed -> DataFrame
                        df_classified = pd.DataFrame(parsed)
                        # ensure columns exist
                        expected_cols = ["name","cuisine","confidence","rating","map_url"]
                        for c in expected_cols:
                            if c not in df_classified.columns:
                                df_classified[c] = None
                        # merge ratings/map_url from original df if missing
                        merged = df_classified.merge(df_to_send[["name","rating","map_url"]], on="name", how="left", suffixes=("","_orig"))
                        # prefer GPT rating if present else original
                        merged["rating"] = merged["rating"].combine_first(merged["rating_orig"])
                        merged = merged.drop(columns=[c for c in merged.columns if c.endswith("_orig")])
                        st.session_state["classified"] = merged[expected_cols]
                        st.success("Classification completed.")
                        st.subheader("🍴 Classified Restaurants")
                        st.dataframe(st.session_state["classified"].head(200))
            except Exception as e:
                st.error(f"Error calling OpenAI: {e}")
                if st.session_state.get("raw_gpt"):
                    st.subheader("Raw GPT output")
                    st.text(st.session_state["raw_gpt"])

# Show classified if available
if st.session_state.get("classified") is not None:
    st.markdown("### Final classified table")
    st.dataframe(st.session_state["classified"].head(500))

# Footer / tips
st.markdown("---")
st.write("Tips: If parsing fails, enable 'Show raw GPT output' to inspect. You can reduce number of places fetched or limit names sent to GPT to avoid token truncation.")