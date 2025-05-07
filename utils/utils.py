import streamlit as st
import requests

API_URL = "http://localhost:5000/predict"

model_mapping = {
    "CNN_LSTM_Attention_English": "cnn_lstm_attention",
    "XLMRoBerta_CNN_LSTM_Vietnamese": "xlm_roberta_cnn_lstm",
    "Logistic Regression": "logistic_regression",
    "Random Forest": "random_forest",
    "XGBoost": "xgboost",
    "Naive Bayes": "naive_bayes",
    "Extra Trees": "extra_trees",
    "LightGBM": "lightgbm",
    "Voting Classifier": "voting"
}

@st.cache_resource(show_spinner=False)
def get_prediction(text: str, model: str = "cnn") -> str:
    """
    Gọi API Flask để phân loại email.
    Trả về 'Spam' hoặc 'Ham'.
    """
    try:
        print(model)
        print(text)
        if model in ["XLMRoBerta_CNN_LSTM_Vietnamese"]:
            # text = text.replace("\n", " ")
            API_URL = "http://localhost:5000/predict/phobert"
        elif model in ["CNN_LSTM_Attention_English",]:
            API_URL = "http://localhost:5000/predict/dl"
        response = requests.post(API_URL, json={"text": text, "model": model_mapping[model]})
        response.raise_for_status()
        data = response.json()
        return data.get("prediction", "")
    except Exception as e:
        st.error(f"Lỗi khi gọi API: {e}")
        return ""
