import asyncio
import json
import pandas as pd
import streamlit as st
import requests
from kafka import KafkaProducer

from features.feedback import build_feedback

producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

st.title("Product Recommendation System")

async def main():
    await asyncio.gather(recommend_items())

async def recommend_items():
    def engage_item(user_id):
        def send(item):
            interaction = build_feedback(item, user_id)
            producer.send("feedback_update", interaction)
            producer.flush()
        return send

    while True:
        try:
            if st.button('Search'):
                response = requests.post(
                    f"http://{args.host}:8000/recommend",
                    json={"user_id": user_id}, 
                    headers={'Content-Type': 'application/json'}
                )
    
                if response.status_code == 200:
                    results = response.json().get('items', [])
                    if results:
                        item = st.selectbox('Recommended Items', results, index=None, on_change=engage_item(user_id), placeholder="Select to engage")
                        st.write("You selected:", item)
                    else:
                        st.write("No items found.")
                else:
                    st.write("Failed to fetch results:", response.status_code)
        except KeyboardInterrupt:
            print("Canceled by user")
            producer.close()


asyncio.run(main())