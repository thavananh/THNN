import streamlit as st
import requests

API_URL = "http://localhost:5000/predict"

@st.cache_resource(show_spinner=False)
def get_prediction(text: str, model: str = "cnn") -> str:
    """
    Gọi API Flask để phân loại email.
    Trả về 'Spam' hoặc 'Ham'.
    """
    try:
        print(model)
        print(text)
        response = requests.post(API_URL, json={"text": text, "model": model})
        response.raise_for_status()
        data = response.json()
        return data.get("prediction", "")
    except Exception as e:
        st.error(f"Lỗi khi gọi API: {e}")
        return ""
