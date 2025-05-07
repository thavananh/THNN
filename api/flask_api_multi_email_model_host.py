import os
from pyexpat import model
import re
import unicodedata
import json
import joblib
from matplotlib.dates import FR
import torch
import torch.nn as nn
import torch.nn.functional as F
from flask import Flask, request, jsonify
from gensim.models import KeyedVectors
import nltk
from nltk.tokenize import word_tokenize
from transformers import AutoTokenizer, AutoModel

# --- Hyperparameters & config ---
EMBEDDING_DIM        = 200
HIDDEN_DIM           = 128
NUM_HEADS            = 4
NUM_FILTERS          = 200
KERNEL_SIZES         = [2]
DROPOUT              = 0.2
MAX_LEN              = 256
PAD_TOKEN            = "<pad>"
UNK_TOKEN            = "<unk>"

# DL artifacts (original CNNBiLSTM_MHA)
DL_ARTIFACTS_DIR     = "../cnn_lstm_attention_component"
BEST_DL_MODEL        = "email_cnn_lstm_attention_word2vec.pt"

# ML artifacts (TF-IDF + traditional models)
ML_ARCHIVE_DIR       = "../ML_archive"

# PhoBERT + CNN-LSTM config
XLM_ROBERTA_CNN_LSTM_MODEL_NAME   = 'FacebookAI/xlm-roberta-base'
XLM_ROBERTA_CNN_LSTM_ARTIFACTS_DIR= "../xlm_roberta_cnn_lstm"
FREEZE_BERT          = False    

PHOBERT_CNN_LSTM_MODEL_NAME = 'vinai/phobert-base-v2'
PHOBERT_CNN_LSTM_ARTIFACTS_DIR = "../phobert_cnn_lstm"
FREEZE_BERT = False

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# NLTK
nltk.download("punkt", quiet=True)

# --- Model classes ---

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
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim * 2,
                                          num_heads=num_heads,
                                          dropout=dropout,
                                          batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.pad_idx = pad_idx

    def forward(self, text):
        mask = (text == self.pad_idx)
        emb = self.dropout(self.embedding(text))         # [B, L, E]
        x = emb.permute(0, 2, 1)                         # [B, E, L]
        c = torch.relu(self.convs[0](x)).permute(0, 2, 1)  # [B, L, F]
        out, _ = self.lstm(self.dropout(c))              # [B, L, 2*H]
        # adjust mask if out length ≠ mask length
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
        return self.fc(self.dropout(pooled))


class BertCNNLSTMClassifier(nn.Module):
    def __init__(
        self,
        model_name,
        num_labels,
        freeze_bert=False,
        cnn_filters=128,
        cnn_kernel_size=3,
        lstm_hidden_dim=128,
        lstm_layers=1,
        bidirectional=True,
        dropout=0.3
    ):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        if freeze_bert:
            for p in self.bert.parameters():
                p.requires_grad = False

        hidden_size = self.bert.config.hidden_size

        self.conv = nn.Conv1d(
            in_channels=hidden_size,
            out_channels=cnn_filters,
            kernel_size=cnn_kernel_size,
            padding=cnn_kernel_size // 2
        )
        self.lstm = nn.LSTM(
            input_size=cnn_filters,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=bidirectional
        )
        lstm_out_dim = lstm_hidden_dim * (2 if bidirectional else 1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(lstm_out_dim, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        seq_out = outputs.last_hidden_state          # [B, L, H]
        x = seq_out.permute(0, 2, 1)                 # [B, H, L]
        x = F.relu(self.conv(x)).permute(0, 2, 1)    # [B, L, F]
        out, (h_n, _) = self.lstm(x)
        if self.lstm.bidirectional:
            h_fwd = h_n[-2]
            h_bwd = h_n[-1]
            h = torch.cat((h_fwd, h_bwd), dim=1)
        else:
            h = h_n[-1]
        h = self.dropout(h)
        return self.classifier(h)


# --- Utils ---

def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


# --- Artifact loading ---

def load_dl_artifacts():
    vocab      = joblib.load(os.path.join(DL_ARTIFACTS_DIR, "vocab.joblib"))
    idx2label  = joblib.load(os.path.join(DL_ARTIFACTS_DIR, "idx_to_label.joblib"))
    emb_mat    = joblib.load(os.path.join(DL_ARTIFACTS_DIR, "embedding_matrix.joblib"))
    _          = KeyedVectors.load(os.path.join(DL_ARTIFACTS_DIR, "word2vec_avg.kv"), mmap="r")
    model      = CNNBiLSTM_MHA(
        vocab_size=len(vocab), embedding_dim=EMBEDDING_DIM,
        num_filters=NUM_FILTERS, kernel_sizes=KERNEL_SIZES,
        hidden_dim=HIDDEN_DIM, num_heads=NUM_HEADS,
        output_dim=len(idx2label), dropout=DROPOUT,
        pad_idx=vocab[PAD_TOKEN], pretrained_matrix=emb_mat
    )
    state = torch.load(os.path.join(DL_ARTIFACTS_DIR, BEST_DL_MODEL), map_location=device)
    model.load_state_dict(state)
    return model.to(device).eval(), vocab, idx2label


def load_ml_artifacts():
    tfidf      = joblib.load(os.path.join(ML_ARCHIVE_DIR, 'tfidf_vectorizer.joblib'))
    le         = joblib.load(os.path.join(ML_ARCHIVE_DIR, 'label_encoder.joblib'))
    names      = ['logistic','random_forest','xgboost','naive_bayes','extra_trees','lightgbm','voting']
    models     = {}
    for n in names:
        p = os.path.join(ML_ARCHIVE_DIR, f"{n}.joblib")
        if os.path.exists(p):
            models[n] = joblib.load(p)
    return tfidf, le, models


def load_phobert_artifacts(artifacts_dir=XLM_ROBERTA_CNN_LSTM_ARTIFACTS_DIR):
    tokenizer = AutoTokenizer.from_pretrained(XLM_ROBERTA_CNN_LSTM_MODEL_NAME)
    label_map = joblib.load(os.path.join(XLM_ROBERTA_CNN_LSTM_ARTIFACTS_DIR, 'label_map.joblib'))
    idx2lab   = {v: k for k, v in label_map.items()}
    model     = BertCNNLSTMClassifier(
        model_name=XLM_ROBERTA_CNN_LSTM_MODEL_NAME,
        num_labels=len(label_map),
        freeze_bert=FREEZE_BERT,
        cnn_filters=128,
        cnn_kernel_size=3,
        lstm_hidden_dim=128,
        lstm_layers=1,
        bidirectional=True,
        dropout=0.3
    )
    state = torch.load(os.path.join(XLM_ROBERTA_CNN_LSTM_ARTIFACTS_DIR, 'best_model.pt'), map_location=device)
    model.load_state_dict(state)
    return tokenizer, model.to(device).eval(), idx2lab


# --- Load all artifacts once ---

dl_model, dl_vocab, dl_idx2label   = load_dl_artifacts()
tfidf_vect, label_enc, ml_models   = load_ml_artifacts()
tokenizer_xlm_roberta_cnn_lstm, xlm_roberta_cnn_lstm, xlm_roberta_cnn_lstm_idx2label = load_phobert_artifacts()
tokenizer_phobert_cnn_lstm, phobert_cnn_lstm, phobert_cnn_lstm_idx2label = load_phobert_artifacts(PHOBERT_CNN_LSTM_ARTIFACTS_DIR)

# --- Prediction fns ---

def predict_deep(text: str, model_name: str) -> str:
    tokens = word_tokenize(normalize_text(text))
    ids    = [dl_vocab.get(w, dl_vocab[UNK_TOKEN]) for w in tokens]
    if len(ids) < MAX_LEN:
        ids += [dl_vocab[PAD_TOKEN]] * (MAX_LEN - len(ids))
    else:
        ids = ids[:MAX_LEN]
    tensor = torch.tensor([ids], dtype=torch.long).to(device)
    with torch.no_grad():
        logits = dl_model(tensor)
    idx = logits.argmax(dim=1).item()
    lbl = dl_idx2label[idx]
    return 'Spam' if lbl in {1,'1'} else 'Ham'


def predict_ml(text: str, model_name: str) -> str:
    vec = tfidf_vect.transform([normalize_text(text)])
    model = ml_models.get(model_name)
    if not model:
        return "Model không tồn tại"
    raw = model.predict(vec)[0]
    lbl = label_enc.inverse_transform([raw])[0]
    return 'Spam' if lbl in {1,'1'} else 'Ham'
    # return label_enc.inverse_transform([raw])[0]


def predict_phobert(text: str, model_name: str="xlm_roberta_cnn_lstm") -> str:
    lbl = None
    if model_name == 'xlm_roberta_cnn_lstm':
        enc = tokenizer_xlm_roberta_cnn_lstm(
            text,
            add_special_tokens=True,
            max_length=MAX_LEN,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        input_ids     = enc['input_ids'].to(device)
        attention_mask= enc['attention_mask'].to(device)
        with torch.no_grad():
            logits = xlm_roberta_cnn_lstm(input_ids, attention_mask)
        idx = logits.argmax(dim=1).item()
        lbl =  xlm_roberta_cnn_lstm_idx2label.get(idx, 'Unknown')
    elif model_name == 'phobert_cnn_lstm':
        enc = tokenizer_phobert_cnn_lstm(
            text,
            add_special_tokens=True,
            max_length=MAX_LEN,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        input_ids     = enc['input_ids'].to(device)
        attention_mask= enc['attention_mask'].to(device)
        with torch.no_grad():
            logits = phobert_cnn_lstm(input_ids, attention_mask)
        idx = logits.argmax(dim=1).item()
        lbl =  phobert_cnn_lstm_idx2label.get(idx, 'Unknown')
    return 'Spam' if lbl in {1,'1'} else 'Ham'


# --- Flask app ---

app = Flask(__name__)

@app.route('/predict/dl', methods=['POST'])
def predict_dl():
    data = request.get_json(force=True)
    text = data.get('text', '').strip()
    model = data.get('model', '').lower()
    if not text:
        return jsonify({'error': 'Missing text'}), 400
    return jsonify({'prediction': predict_deep(text, model)})


@app.route('/predict/ml', methods=['POST'])
def predict_ml_endpoint():
    data  = request.get_json(force=True)
    text  = data.get('text', '').strip()
    model = data.get('model', '').lower()
    if not text:
        return jsonify({'error': 'Missing text'}), 400
    return jsonify({'model': model, 'prediction': predict_ml(text, model)})


@app.route('/predict/phobert', methods=['POST'])
def predict_phobert_endpoint():
    data = request.get_json(force=True)
    text = data.get('text', '').strip()
    model = data.get('model', '').lower()
    if not text:
        return jsonify({'error': 'Missing text'}), 400
    return jsonify({'prediction': predict_phobert(text, model_name=model)})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
