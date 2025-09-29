import re
import requests
import json
import streamlit as st
from openai import OpenAI

# إعداد مفاتيح API
client = OpenAI(api_key="YOUR_OPENAI_API_KEY")
GOOGLE_API_KEY = "YOUR_GOOGLE_API_KEY"

NEARBY_URL = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
DETAILS_URL = "https://maps.googleapis.com/maps/api/place/details/json"


# 🔹 استخراج الإحداثيات من رابط خرائط قوقل
def extract_coordinates_from_url(url):
    match = re.search(r'@([-0-9.]+),([-0-9.]+)', url)
    if match:
        return float(match.group(1)), float(match.group(2))
    return None, None


# 🔹 جلب قائمة المطاعم بالقرب من إحداثيات معينة
def fetch_restaurants_nearby(lat, lng, radius=3000):
    params = {"location": f"{lat},{lng}", "radius": radius, "type": "restaurant", "key": GOOGLE_API_KEY}
    response = requests.get(NEARBY_URL, params=params).json()
    results = response.get("results", [])

    restaurants = []
    for place in results:
        place_id = place.get("place_id")
        details_params = {"place_id": place_id, "key": GOOGLE_API_KEY, "language": "ar"}
        details = requests.get(DETAILS_URL, params=details_params).json().get("result", {})

        reviews = [r.get("text", "") for r in details.get("reviews", [])]

        restaurants.append({
            "name": place.get("name"),
            "description": details.get("editorial_summary", {}).get("overview", ""),
            "reviews": reviews,
            "link": f"https://www.google.com/maps/place/?q=place_id:{place_id}"
        })

    return restaurants


# 🔹 GPT لتصنيف المطاعم حسب نوع المطبخ (Cuisine)
def classify_restaurants(restaurants):
    restaurants_text = "\n\n".join([
        f"Name: {r['name']}\nDescription: {r.get('description', 'N/A')}\nReviews: {', '.join(r.get('reviews', []))}\nLink: {r.get('link','')}"
        for r in restaurants
    ])

    prompt = f"""
You are a smart restaurant classification assistant.
Analyze the following restaurants (name + description + reviews) and classify each into cuisine type 
(like: Indian, Asian, Gulf, Lebanese, American, Burger, Pizza, Seafood, Fast Food, Bakery, Cafe, etc).
Return ONLY valid JSON array with:
- name
- cuisine
- link

Restaurants data:

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


# 🔹 واجهة Streamlit
def main():
    st.title("📍 مطاعم قريبة مع التصنيف الذكي")
    st.write("ألصق رابط خرائط قوقل (Google Maps) لأي موقع، وسيتم البحث عن المطاعم القريبة (٣ كم) وتصنيفها حسب نوع المطبخ.")

    url = st.text_input("ألصق رابط خرائط قوقل هنا:")

    if st.button("Fetch Restaurants"):
        lat, lng = extract_coordinates_from_url(url)
        if not lat or not lng:
            st.error("تعذر استخراج الإحداثيات من الرابط. تأكد أن الرابط يحتوي على إحداثيات (مثل ...@26.12345,50.12345...).")
            return

        with st.spinner("جاري جلب المطاعم من خرائط قوقل..."):
            restaurants = fetch_restaurants_nearby(lat, lng)
            st.success(f"✅ تم العثور على {len(restaurants)} مطاعم.")

            if restaurants:
                st.write("### 📌 جدول بيانات المطاعم من Google Maps")
                st.json(restaurants)

                with st.spinner("🤖 جاري تحليل المطاعم وتصنيفها..."):
                    classified = classify_restaurants(restaurants)

                st.write("### 🍽️ جدول التصنيف النهائي")
                st.table(classified)


if __name__ == "__main__":
    main()