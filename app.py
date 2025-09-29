import streamlit as st
import re
import requests
import openai

# ================= SETTINGS =================
st.set_page_config(page_title="🍽 Smart Restaurant Classifier", layout="wide")

st.title("📍 Smart Restaurant Classifier — Cuisine-based (Google Maps → GPT)")

# Sidebar API keys
st.sidebar.header("🔑 API Keys & Settings")
gmaps_key = st.sidebar.text_input("Google Maps API Key", type="password")
openai_key = st.sidebar.text_input("OpenAI API Key", type="password")
model_name = st.sidebar.selectbox("OpenAI Model", ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"])
rating_threshold = st.sidebar.slider("Minimum rating to include (>=)", 0.0, 5.0, 4.0, 0.1)
max_places = st.sidebar.number_input("Max places to fetch (per page)", min_value=1, max_value=60, value=20)

if gmaps_key:
    st.sidebar.success("Maps key loaded ✅")
if openai_key:
    st.sidebar.success("OpenAI key loaded ✅")

openai.api_key = openai_key

# =============== Extract coordinates from Google Maps URL =================
def extract_coordinates(url):
    match = re.search(r"@([-.\d]+),([-.\d]+)", url)
    if match:
        return float(match.group(1)), float(match.group(2))
    return None, None

# =============== Google Places API =================
def get_nearby_places(lat, lng, radius_km, keyword="restaurant"):
    url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
    params = {
        "location": f"{lat},{lng}",
        "radius": int(radius_km * 1000),
        "type": "restaurant",
        "keyword": keyword,
        "key": gmaps_key
    }
    results = []
    while True:
        res = requests.get(url, params=params).json()
        if "results" not in res:
            break
        results.extend(res["results"])
        if "next_page_token" in res:
            params["pagetoken"] = res["next_page_token"]
        else:
            break
    return results

def get_place_details(place_id):
    url = "https://maps.googleapis.com/maps/api/place/details/json"
    params = {
        "place_id": place_id,
        "fields": "name,rating,url,reviews",
        "key": gmaps_key
    }
    res = requests.get(url, params=params).json()
    return res.get("result", {})

# =============== GPT Classification =================
def classify_restaurants(restaurant_names):
    prompt = """
    You are a smart assistant. 
    Classify each restaurant into a cuisine category (Asian, Khaleeji, Egyptian, Indian, Burger, Pizza, Lebanese, etc.).
    Respond in strict JSON format as a list of objects:
    [
      {"name": "Restaurant A", "cuisine": "Indian"},
      {"name": "Restaurant B", "cuisine": "Pizza"}
    ]
    Only use the restaurant name as input, do not add extra commentary.
    """

    names_text = "\n".join(restaurant_names)
    full_prompt = prompt + "\n\nRestaurants:\n" + names_text

    response = openai.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": full_prompt}],
        temperature=0
    )

    return response.choices[0].message.content.strip()

# ================= Streamlit Workflow =================
st.subheader("1) Paste Google Maps URL")
maps_url = st.text_input("Paste Google Maps URL (short or long)")
radius_km = st.number_input("Search radius (km)", min_value=0.5, max_value=20.0, value=3.0, step=0.5)

restaurants_data = []
if st.button("🔍 Fetch nearby restaurants"):
    if not gmaps_key:
        st.error("Please enter Google Maps API key in the sidebar.")
    else:
        lat, lng = extract_coordinates(maps_url)
        if not lat or not lng:
            st.error("Could not extract coordinates from URL.")
        else:
            with st.spinner("Fetching restaurants..."):
                places = get_nearby_places(lat, lng, radius_km)
                for p in places[:max_places]:
                    details = get_place_details(p["place_id"])
                    if not details:
                        continue
                    rating = details.get("rating", 0)
                    if rating >= rating_threshold:
                        reviews = details.get("reviews", [])[:3]
                        review_texts = [r["text"] for r in reviews]
                        restaurants_data.append({
                            "name": details.get("name"),
                            "rating": rating,
                            "map_url": details.get("url"),
                            "reviews": review_texts
                        })
            st.success(f"Fetched {len(restaurants_data)} restaurants")

if restaurants_data:
    st.subheader("📋 Restaurants (filtered)")
    for r in restaurants_data:
        st.markdown(f"**[{r['name']}]({r['map_url']})** — ⭐ {r['rating']}")
        for i, rev in enumerate(r["reviews"]):
            st.write(f"_{i+1}. {rev[:200]}..._")
        st.write("---")

    if st.button("🤖 Classify (smart cuisine) with GPT"):
        names_only = [r["name"] for r in restaurants_data]
        with st.spinner("Classifying cuisines..."):
            try:
                gpt_output = classify_restaurants(names_only)
                st.subheader("🍴 Cuisine Classification")
                st.json(gpt_output)
            except Exception as e:
                st.error(f"GPT classification failed: {e}")