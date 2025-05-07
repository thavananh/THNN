from flask import Flask, request, jsonify
import torch
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Tham số giống Streamlit app
MODEL_NAME = "vinai/phobert-base-v2"
BEST_MODEL = "best_vinai_phobert-base-v2_aspect_cateogry_analysis_sigmoid_prob.pth"
LABEL_MAP  = "label_map.json"
MAX_LEN    = 128
THRESHOLD  = 0.48

# load label map và đảo chỉ mục
with open(LABEL_MAP, "r", encoding="utf-8") as f:
    raw_map = json.load(f)
    label_map = {tuple(k.split("###")): v for k, v in raw_map.items()}
    idx2label = {v: k for k, v in label_map.items()}

# khởi tạo tokenizer + model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=len(idx2label)
).to(device)
model.load_state_dict(torch.load(BEST_MODEL, map_location=device))
model.eval()

# Hàm inference
def predict_pairs(sentence: str):
    inputs = tokenizer(
        sentence,
        padding="max_length",
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.sigmoid(logits).squeeze(0).cpu()
    pairs = [idx2label[i] for i, p in enumerate(probs) if p > THRESHOLD]
    return [{"aspect": asp, "sentiment": sen} for asp, sen in pairs]

# Khởi tạo Flask app
app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict_endpoint():
    """
    Nhận JSON: { "text": "Một câu tiếng Việt ..." }
    Trả về JSON: { "predictions": [ { "aspect": ..., "sentiment": ... }, ... ] }
    """
    data = request.get_json(force=True)
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "Missing 'text' in request"}), 400

    preds = predict_pairs(text)
    return jsonify({"predictions": preds})

if __name__ == "__main__":
    # debug=True chỉ để dev; khi deploy dùng gunicorn hoặc uWSGI
    app.run(host="0.0.0.0", port=5000, debug=True)
