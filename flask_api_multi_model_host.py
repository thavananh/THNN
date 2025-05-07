# flask_api.py
from flask import Flask, request, jsonify
import os
import json
import torch
import torch.nn as nn
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from gensim.models import KeyedVectors

# --- Common setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- PhoBERT-based model setup ---
MODEL_NAME_PHO   = "vinai/phobert-base-v2"
BEST_MODEL_PHO   = "best_vinai_phobert-base-v2_aspect_cateogry_analysis_sigmoid_prob.pth"
LABEL_MAP_PHO    = "label_map.json"
MAX_LEN_PHO      = 128
THRESHOLD_PHO    = 0.48

# load label map và đảo chỉ mục
with open(LABEL_MAP_PHO, "r", encoding="utf-8") as f:
    raw_map_pho = json.load(f)
pho_label_map  = {tuple(k.split("###")): v for k, v in raw_map_pho.items()}
pho_idx2label  = {v: k for k, v in pho_label_map.items()}

# khởi tạo tokenizer + model
tokenizer_pho = AutoTokenizer.from_pretrained(MODEL_NAME_PHO)
model_pho     = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME_PHO, num_labels=len(pho_idx2label)
).to(device)
model_pho.load_state_dict(torch.load(BEST_MODEL_PHO, map_location=device))
model_pho.eval()

def predict_pho(text: str):
    inputs = tokenizer_pho(
        text,
        padding="max_length",
        truncation=True,
        max_length=MAX_LEN_PHO,
        return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        logits = model_pho(**inputs).logits
        probs  = torch.sigmoid(logits).squeeze(0).cpu()
    pairs = [pho_idx2label[i] for i, p in enumerate(probs) if p > THRESHOLD_PHO]
    return [{"aspect": asp, "sentiment": sen} for asp, sen in pairs]


# --- CNN–LSTM–Attention model setup ---
# Hyperparameters must match those used in training
EMBEDDING_DIM      = 100
HIDDEN_DIM         = 128
NUM_HEADS          = 4
NUM_FILTERS        = 100
KERNEL_SIZES       = [2, 3, 4]
DROPOUT            = 0.4
ARTIFACTS_DIR_CNN  = "cnn_lstm_attention_component"
PAD_TOKEN          = "<pad>"
UNK_TOKEN          = "<unk>"
MAX_LEN_CNN        = 80
THRESHOLD_CNN      = 0.5

class CNNBiLSTM_MHA_ACSA(nn.Module):
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
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim*2,
                                          num_heads=num_heads,
                                          dropout=dropout,
                                          batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_dim*2, output_dim)
        self.pad_idx = pad_idx

    def forward(self, text):
        mask = (text == self.pad_idx)
        emb  = self.dropout(self.embedding(text))
        x    = emb.permute(0,2,1)
        c    = torch.relu(self.convs[0](x)).permute(0,2,1)
        out, _ = self.lstm(self.dropout(c))
        # adjust mask length if needed
        if out.size(1) != mask.size(1):
            diff = out.size(1) - mask.size(1)
            if diff > 0:
                mask = torch.cat([mask, mask.new_ones(mask.size(0), diff)], dim=1)
            else:
                mask = mask[:, :out.size(1)]
        attn_out, _ = self.attn(out, out, out, key_padding_mask=mask)
        attn_out    = attn_out.masked_fill(mask.unsqueeze(-1), 0.0)
        summed      = attn_out.sum(1)
        cnt_nonpad  = (~mask).sum(1).clamp(min=1).unsqueeze(1)
        pooled      = summed / cnt_nonpad
        return self.fc(self.dropout(pooled))

def load_cnn_artifacts(artifacts_dir=ARTIFACTS_DIR_CNN):
    # vocab
    with open(os.path.join(artifacts_dir, 'vocab.json'), 'r', encoding='utf-8') as f:
        vocab = json.load(f)
    # label_map
    raw_lm = json.load(open(os.path.join(artifacts_dir, 'label_map.json'), 'r', encoding='utf-8'))
    label_map = {tuple(key.split('|||')): idx for key, idx in raw_lm.items()}
    # idx_to_label
    raw_itl = json.load(open(os.path.join(artifacts_dir, 'idx_to_label.json'), 'r', encoding='utf-8'))
    idx_to_label = {int(k): tuple(v) for k, v in raw_itl.items()}
    # Word2Vec
    kv = KeyedVectors.load(os.path.join(artifacts_dir, 'word2vec.kv'), mmap='r')

    # rebuild model
    vocab_size = len(vocab)
    num_classes= len(label_map)
    pad_idx    = vocab[PAD_TOKEN]
    model = CNNBiLSTM_MHA_ACSA(
        vocab_size=vocab_size,
        embedding_dim=EMBEDDING_DIM,
        num_filters=NUM_FILTERS,
        kernel_sizes=KERNEL_SIZES,
        hidden_dim=HIDDEN_DIM,
        num_heads=NUM_HEADS,
        output_dim=num_classes,
        dropout=DROPOUT,
        pad_idx=pad_idx,
        pretrained_matrix=None,
        freeze_embeddings=False
    ).to(device)
    model.load_state_dict(torch.load(os.path.join(artifacts_dir, 'best_model.pt'),
                                     map_location=device))
    model.eval()
    return model, vocab, idx_to_label

cnn_model, cnn_vocab, cnn_idx2label = load_cnn_artifacts()

def predict_cnn(text: str):
    # tokenize & pad
    tokens = text.lower().split()
    idxs = [cnn_vocab.get(w, cnn_vocab.get(UNK_TOKEN)) for w in tokens]
    if len(idxs) < MAX_LEN_CNN:
        idxs += [cnn_vocab[PAD_TOKEN]] * (MAX_LEN_CNN - len(idxs))
    else:
        idxs = idxs[:MAX_LEN_CNN]
    tensor = torch.tensor([idxs], dtype=torch.long).to(device)
    with torch.no_grad():
        logits = cnn_model(tensor)
        probs  = torch.sigmoid(logits).squeeze(0).cpu().numpy()
    pairs = [cnn_idx2label[i] for i, p in enumerate(probs) if p > THRESHOLD_CNN]
    return [{"aspect": asp, "sentiment": sen} for asp, sen in pairs]


# --- Flask App ---
app = Flask(__name__)

@app.route("/predict_pho", methods=["POST"])
def predict_pho_endpoint():
    data = request.get_json(force=True)
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "Missing 'text'"}), 400
    preds = predict_pho(text)
    return jsonify({"predictions": preds})

@app.route("/predict_cnn", methods=["POST"])
def predict_cnn_endpoint():
    data = request.get_json(force=True)
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "Missing 'text'"}), 400
    preds = predict_cnn(text)
    return jsonify({"predictions": preds})

if __name__ == "__main__":
    # khi deploy, cân nhắc dùng gunicorn/uWSGI thay debug=True
    app.run(host="0.0.0.0", port=5000, debug=True)
