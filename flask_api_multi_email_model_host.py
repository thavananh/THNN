# flask_api.py
import os
import re
import unicodedata
import joblib
import torch
import torch.nn as nn
from flask import Flask, request, jsonify
from gensim.models import KeyedVectors
import nltk
from nltk.tokenize import word_tokenize

# --- Hyperparameters & Config ---
EMBEDDING_DIM       = 200
HIDDEN_DIM          = 128
NUM_HEADS           = 4
NUM_FILTERS         = 200
KERNEL_SIZES        = [2]
DROPOUT             = 0.2
MAX_LEN             = 256
PAD_TOKEN           = "<pad>"
UNK_TOKEN           = "<unk>"
DL_ARTIFACTS_DIR    = "cnn_lstm_attention_component"
ML_ARCHIVE_DIR      = "ML_archive"
BEST_MODEL_FILENAME = "email_cnn_lstm_attention_word2vec.pt"

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Ensure NLTK tokenizer data
nltk.download("punkt", quiet=True)

# --- Model Definition ---
class CNNBiLSTM_MHA(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_filters, kernel_sizes,
                 hidden_dim, num_heads, output_dim, dropout, pad_idx,
                 pretrained_matrix=None, freeze_embeddings=False):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        if pretrained_matrix is not None:
            self.embedding.weight.data.copy_(pretrained_matrix)
            self.embedding.weight.requires_grad = not freeze_embeddings
        self.convs = nn.ModuleList([
            nn.Conv1d(embedding_dim, num_filters, ks) for ks in kernel_sizes
        ])
        self.lstm = nn.LSTM(num_filters, hidden_dim, batch_first=True, bidirectional=True)
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim*2, num_heads=num_heads,
                                          dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim*2, output_dim)
        self.pad_idx = pad_idx

    def forward(self, text):
        mask = (text == self.pad_idx)
        emb = self.dropout(self.embedding(text))         # [B, L, E]
        x = emb.permute(0, 2, 1)                         # [B, E, L]
        c = torch.relu(self.convs[0](x)).permute(0, 2, 1)  # [B, L, F]
        out, _ = self.lstm(self.dropout(c))              # [B, L, 2*H]
        # adjust mask length if needed
        if out.size(1) != mask.size(1):
            diff = out.size(1) - mask.size(1)
            if diff > 0:
                mask = torch.cat([mask, mask.new_ones(mask.size(0), diff)], dim=1)
            else:
                mask = mask[:, :out.size(1)]
        attn_out, _ = self.attn(out, out, out, key_padding_mask=mask)
        attn_out = attn_out.masked_fill(mask.unsqueeze(-1), 0.0)
        summed = attn_out.sum(1)
        cnt = (~mask).sum(1).clamp(min=1).unsqueeze(1)
        pooled = summed / cnt
        return self.fc(self.dropout(pooled))  # [B, output_dim]

# --- Utilities ---
def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

# --- Artifact Loading ---
def load_dl_artifacts(dir_path=DL_ARTIFACTS_DIR):
    vocab = joblib.load(os.path.join(dir_path, "vocab.joblib"))
    idx_to_label = joblib.load(os.path.join(dir_path, "idx_to_label.joblib"))
    emb_mat = joblib.load(os.path.join(dir_path, "embedding_matrix.joblib"))
    _ = KeyedVectors.load(os.path.join(dir_path, "word2vec_avg.kv"), mmap="r")
    model = CNNBiLSTM_MHA(
        vocab_size=len(vocab), embedding_dim=EMBEDDING_DIM,
        num_filters=NUM_FILTERS, kernel_sizes=KERNEL_SIZES,
        hidden_dim=HIDDEN_DIM, num_heads=NUM_HEADS,
        output_dim=len(idx_to_label), dropout=DROPOUT,
        pad_idx=vocab[PAD_TOKEN], pretrained_matrix=emb_mat
    )
    model.load_state_dict(torch.load(os.path.join(dir_path, BEST_MODEL_FILENAME),
                                      map_location=device))
    model.to(device).eval()
    return model, vocab, idx_to_label


def load_ml_artifacts(dir_path=ML_ARCHIVE_DIR):
    tfidf = joblib.load(os.path.join(dir_path, 'tfidf_vectorizer.joblib'))
    le = joblib.load(os.path.join(dir_path, 'label_encoder.joblib'))
    names = ['logistic','random_forest','xgboost','naive_bayes',
             'extra_trees','lightgbm','voting']
    models = {}
    for n in names:
        p = os.path.join(dir_path, f"{n}.joblib")
        if os.path.exists(p): models[n] = joblib.load(p)
    return tfidf, le, models

# Load artifacts once
dl_model, dl_vocab, dl_idx2label = load_dl_artifacts()
tfidf_vect, label_enc, ml_models = load_ml_artifacts()

# --- Prediction Functions ---
def predict_deep(text: str, model: str) -> str:
    tokens = word_tokenize(normalize_text(text))
    ids = [dl_vocab.get(w, dl_vocab[UNK_TOKEN]) for w in tokens]
    if len(ids) < MAX_LEN:
        ids += [dl_vocab[PAD_TOKEN]] * (MAX_LEN - len(ids))
    else:
        ids = ids[:MAX_LEN]
    tensor = torch.tensor([ids], dtype=torch.long).to(device)
    with torch.no_grad():
        logits = dl_model(tensor)
    idx = logits.argmax(dim=1).item()
    label = dl_idx2label[idx]
    return 'Spam' if label in {1,'1'} else 'Ham'


def predict_ml(text: str, model_name: str) -> str:
    vec = tfidf_vect.transform([normalize_text(text)])
    model = ml_models.get(model_name)
    if not model:
        return "Model không tồn tại"
    raw = model.predict(vec)[0]
    lbl = label_enc.inverse_transform([raw])[0]
    return 'Spam' if lbl in {1,'1'} else 'Ham'

# --- Flask App ---
app = Flask(__name__)

@app.route('/predict/dl', methods=['POST'])
def predict_dl():
    data = request.get_json(force=True)
    text = data.get('text','').strip()
    model = data.get('model','').lower()
    if not text:
        return jsonify({'error':'Missing text'}), 400
    result = predict_deep(text)
    return jsonify({'prediction': result})

@app.route('/predict/ml', methods=['POST'])
def predict_ml_endpoint():
    data = request.get_json(force=True)
    text = data.get('text','').strip()
    model = data.get('model','').lower()
    if not text:
        return jsonify({'error':'Missing text'}), 400
    result = predict_ml(text, model)
    return jsonify({'model': model, 'prediction': result})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
