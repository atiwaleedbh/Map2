import requests
import json
import streamlit as st
from openai import OpenAI

# إعداد OpenAI
client = OpenAI(api_key="YOUR_API_KEY")

# استدعاء Google Places API
GOOGLE_API_KEY = "YOUR_GOOGLE_API_KEY"
PLACES_URL = "https://maps.googleapis.com/maps/api/place/textsearch/json"
DETAILS_URL = "https://maps.googleapis.com/maps/api/place/details/json"

# 1. جلب قائمة المطاعم
def fetch_restaurants(location="Dammam", query="restaurants"):
    params = {"query": query + " in " + location, "key": GOOGLE_API_KEY}
    response = requests.get(PLACES_URL, params=params)
    results = response.json().get("results", [])

    restaurants = []
    for place in results:
        place_id = place.get("place_id")
        details_params = {"place_id": place_id, "key": GOOGLE_API_KEY, "language": "ar"}
        details_response = requests.get(DETAILS_URL, params=details_params)
        details = details_response.json().get("result", {})

        reviews = [r.get("text", "") for r in details.get("reviews", [])]

        restaurants.append({
            "name": place.get("name"),
            "description": details.get("editorial_summary", {}).get("overview", ""),
            "reviews": reviews,
            "link": f"https://www.google.com/maps/place/?q=place_id:{place_id}"
        })

    return restaurants

# 2. إرسال للموديل GPT لتصنيف المطاعم
def classify_restaurants(restaurants):
    restaurants_text = "\n\n".join([
        f"Name: {r['name']}\nDescription: {r.get('description', 'N/A')}\nReviews: {', '.join(r.get('reviews', []))}\nLink: {r.get('link','')}"
        for r in restaurants
    ])

    prompt = f"""
You are a smart restaurant classification assistant.
You will receive a list of restaurants with their name, description, user reviews, and Google Maps link.
For each restaurant, analyze the information and classify it into a cuisine type 
(like: Indian, Asian, Gulf, Lebanese, American, Burger, Pizza, Seafood, Fast Food, Bakery, Cafe, etc).
Be as accurate as possible, even if cuisine type is implicit from the name or reviews.

Return ONLY valid JSON array. Each item must have:
- name
- cuisine
- link

Here are the restaurants:

{restaurants_text}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a JSON generator for restaurant classification."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )

    output = response.choices[0].message.content.strip()

    try:
        data = json.loads(output)
    except json.JSONDecodeError:
        cleaned = output.replace("```json", "").replace("```", "").strip()
        data = json.loads(cleaned)

    return data

# 3. واجهة Streamlit
def main():
    st.title("Restaurant Classifier 🍴")
    st.write("أدخل مدينة للبحث عن المطاعم وتصنيفها حسب نوع المطبخ (Cuisine).")

    location = st.text_input("الموقع:", "Dammam")
    if st.button("Fetch Restaurants"):
        with st.spinner("جاري جلب المطاعم من خرائط قوقل..."):
            restaurants = fetch_restaurants(location)
            st.success(f"تم العثور على {len(restaurants)} مطاعم.")

            if restaurants:
                st.write("### المطاعم المستخرجة من خرائط قوقل:")
                st.json(restaurants)

                with st.spinner("جاري تحليل المطاعم وتصنيفها..."):
                    classified = classify_restaurants(restaurants)

                st.write("### جدول التصنيف النهائي")
                st.table(classified)

if __name__ == "__main__":
    main()