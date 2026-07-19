"""
==============================================================================
 Attention Mekanizması — BiLSTM + Bahdanau (Additive) Attention ile
 Oyun Yorumu Duygu Analizi (OYUN VERSİYONU)
==============================================================================

ÖĞRENME NOTU (yönlendirme):
  Bu dosyanın amacı, BiLSTM + Bahdanau (Additive) Attention ile ikili
  duygu analizi yapan bir mimari inşa etmek ve attention ağırlıklarını
  ısı haritası olarak görselleştirmektir.

  VERI SETI NOTU: Paylasilan 9 Kaggle veri setinden hicbiri UZUN FORMLU
  (birkac cumlelik) ham oyun yorumu metni icermiyor (bkz.
  01-tf-idf ve MachineLearning/08-naive-bayes README'lerindeki ayni
  tespit). Bu yuzden gorev tanimindaki istisna uygulanarak, benzer
  olcekte (12.000 ornek) ve benzer uzunlukta (birden fazla cumle)
  SENTETIK oyun yorumu paragraflari uretiliyor.

RNN/LSTM projelerinden farkı: model artık cümleyi tek bir sabit vektöre
sıkıştırmıyor, her kelimeye ne kadar "dikkat" edeceğine karar veriyor.
==============================================================================
"""

import os
import re
import time
import random
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# ──────────────────────────────────────────────────────────────────────────
# 0) Sabitler / Hiperparametreler
# ──────────────────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

DATA_DIR = "data"; FIG_DIR = "figures"
os.makedirs(DATA_DIR, exist_ok=True); os.makedirs(FIG_DIR, exist_ok=True)

REVIEWS_PATH = os.path.join(DATA_DIR, "game_reviews_long.csv")

N_SAMPLES = 12000
MAX_VOCAB = 15000
MAX_LEN = 150
EMB_DIM = 100
HIDDEN_SIZE = 96
BATCH_SIZE = 128
EPOCHS = 6
LR = 1e-3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Cihaz: {device} | CPU çekirdek: {torch.get_num_threads()}")


# ──────────────────────────────────────────────────────────────────────────
# 1) Sentetik uzun formlu oyun yorumu üretimi
# ──────────────────────────────────────────────────────────────────────────
POS_SENTENCES = [
    "This game absolutely blew me away from the very first hour.",
    "The graphics are stunning and run smoothly even on lower settings.",
    "I love how the story keeps you engaged with unexpected twists.",
    "The combat system feels responsive and rewarding to master.",
    "Multiplayer matches are balanced and genuinely fun with friends.",
    "The developers clearly listen to feedback and patch issues fast.",
    "Sound design and music create an incredible atmosphere throughout.",
    "For the price, the amount of content here is outstanding.",
    "Every boss fight feels creative and never gets repetitive.",
    "The open world is full of secrets that reward exploration.",
    "Character progression feels meaningful and well paced.",
    "I have sunk over a hundred hours and still enjoy every session.",
    "Controls are intuitive, I picked it up within minutes.",
    "The art style gives the game a unique and memorable identity.",
    "Regular free updates keep adding fresh content to enjoy.",
    "This is easily one of the best games I have played this year.",
]

NEG_SENTENCES = [
    "The game constantly crashes and I have lost progress multiple times.",
    "Performance is terrible, even on a high end system it stutters.",
    "The story feels rushed and none of the characters are memorable.",
    "Combat is clunky and inputs often do not register correctly.",
    "Servers are unstable, making multiplayer nearly unplayable.",
    "Support never responds to tickets about game breaking bugs.",
    "The monetization is aggressive and ruins the overall balance.",
    "For the price charged there is barely any content included.",
    "Every encounter feels like a repeat of the last one.",
    "The world feels empty and exploration is rarely rewarded.",
    "Progression grinds to a halt after just a few hours.",
    "I regret this purchase, it was a complete waste of money.",
    "Controls are unresponsive and the tutorial explains almost nothing.",
    "The visuals look dated compared to other games at this price.",
    "Updates are rare and never fix the most reported issues.",
    "This is easily one of the most disappointing releases this year.",
]

NEUTRAL_FILLERS = [
    "I bought this game during a seasonal sale a few weeks ago.",
    "A friend recommended this title so I decided to give it a try.",
    "I have been playing on PC with a mid range setup.",
    "I usually enjoy this genre so I had high expectations.",
    "I have put in around twenty hours so far.",
    "This was my first time trying a game from this studio.",
]


def make_review(is_positive):
    pool = POS_SENTENCES if is_positive else NEG_SENTENCES
    n_main = random.randint(5, 9)
    sentences = random.choices(pool, k=n_main)
    if random.random() < 0.15:
        other_pool = NEG_SENTENCES if is_positive else POS_SENTENCES
        sentences.append(random.choice(other_pool))
    if random.random() < 0.5:
        sentences.insert(0, random.choice(NEUTRAL_FILLERS))
    random.shuffle(sentences)
    return " ".join(sentences)


def generate_reviews():
    if os.path.exists(REVIEWS_PATH):
        print(f"Sentetik yorum veri seti zaten mevcut: {REVIEWS_PATH}")
        return pd.read_csv(REVIEWS_PATH)
    print("Sentetik uzun formlu oyun yorumu veri seti üretiliyor...")
    records = []
    for _ in range(N_SAMPLES):
        is_pos = random.random() < 0.5
        records.append({
            "review": make_review(is_pos),
            "sentiment": "positive" if is_pos else "negative",
        })
    df = pd.DataFrame(records)
    df.to_csv(REVIEWS_PATH, index=False)
    print(f"Kaydedildi: {REVIEWS_PATH} ({len(df)} satır)")
    return df


df = generate_reviews()
df = df.sample(n=N_SAMPLES, random_state=SEED).reset_index(drop=True)
print(f"\nToplam örnek: {len(df)} | Sınıf dağılımı:\n{df['sentiment'].value_counts()}")


# ──────────────────────────────────────────────────────────────────────────
# 2) Metin temizleme + tokenizasyon
# ──────────────────────────────────────────────────────────────────────────
def clean_and_tokenize(text):
    text = re.sub(r"<br\s*/?>", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    return text.lower().split()


t0 = time.time()
df["tokens"] = df["review"].apply(clean_and_tokenize)
df["label"] = (df["sentiment"] == "positive").astype(np.float32)
print(f"Tokenizasyon: {time.time()-t0:.1f}s")


# ──────────────────────────────────────────────────────────────────────────
# 3) Kelime dağarcığı + dizilere çevirme
# ──────────────────────────────────────────────────────────────────────────
n_train = int(len(df) * 0.85)
train_df = df.iloc[:n_train].reset_index(drop=True)
test_df = df.iloc[n_train:].reset_index(drop=True)

counter = Counter(w for toks in train_df["tokens"] for w in toks)
vocab_words = [w for w, c in counter.most_common(MAX_VOCAB - 2)]
word2id = {"<pad>": 0, "<unk>": 1}
for w in vocab_words:
    word2id[w] = len(word2id)
VOCAB_SIZE = len(word2id)
print(f"Vocab boyutu: {VOCAB_SIZE:,} | Eğitim: {len(train_df)} | Test: {len(test_df)}")


def encode(tokens, max_len=MAX_LEN):
    ids = [word2id.get(w, 1) for w in tokens[:max_len]]
    length = len(ids)
    ids = ids + [0] * (max_len - length)
    return ids, length


train_df["ids"], train_df["length"] = zip(*train_df["tokens"].apply(encode))
test_df["ids"], test_df["length"] = zip(*test_df["tokens"].apply(encode))


class ReviewDataset(Dataset):
    def __init__(self, frame):
        self.ids = torch.tensor(np.stack(frame["ids"].values), dtype=torch.long)
        self.lengths = torch.tensor(frame["length"].values, dtype=torch.long)
        self.labels = torch.tensor(frame["label"].values, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.ids[idx], self.lengths[idx], self.labels[idx]


train_loader = DataLoader(ReviewDataset(train_df), batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(ReviewDataset(test_df), batch_size=BATCH_SIZE, shuffle=False)


# ──────────────────────────────────────────────────────────────────────────
# 4) Model — BiLSTM + Bahdanau (Additive) Attention (sıfırdan)
# ──────────────────────────────────────────────────────────────────────────
class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.W = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, lstm_out, mask):
        scores = self.v(torch.tanh(self.W(lstm_out))).squeeze(-1)
        scores = scores.masked_fill(mask == 0, -1e9)
        weights = torch.softmax(scores, dim=1)
        context = torch.bmm(weights.unsqueeze(1), lstm_out).squeeze(1)
        return context, weights


class SentimentAttentionModel(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(emb_dim, hidden_size, batch_first=True, bidirectional=True)
        self.attention = BahdanauAttention(hidden_size * 2)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_size * 2, 1)

    def forward(self, x, mask):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        context, attn_weights = self.attention(lstm_out, mask)
        logits = self.fc(self.dropout(context)).squeeze(-1)
        return logits, attn_weights


model = SentimentAttentionModel(VOCAB_SIZE, EMB_DIM, HIDDEN_SIZE).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.BCEWithLogitsLoss()

print(f"\nModel parametre sayısı: {sum(p.numel() for p in model.parameters()):,}")


def make_mask(lengths, max_len=MAX_LEN):
    idx = torch.arange(max_len).unsqueeze(0)
    return (idx < lengths.unsqueeze(1)).float()


# ──────────────────────────────────────────────────────────────────────────
# 5) Eğitim döngüsü
# ──────────────────────────────────────────────────────────────────────────
def run_epoch(loader, train=True):
    model.train() if train else model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []
    with torch.set_grad_enabled(train):
        for ids, lengths, labels in loader:
            ids, lengths, labels = ids.to(device), lengths.to(device), labels.to(device)
            mask = make_mask(lengths).to(device)
            if train:
                optimizer.zero_grad()
            logits, _ = model(ids, mask)
            loss = criterion(logits, labels)
            if train:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
            total_loss += loss.item() * len(labels)
            all_preds.extend((torch.sigmoid(logits) > 0.5).float().cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    return total_loss / len(loader.dataset), acc, f1


print("\n── Eğitim başlıyor ──")
history = {"train_loss": [], "test_loss": [], "train_acc": [], "test_acc": []}
t_start = time.time()
for epoch in range(1, EPOCHS + 1):
    tr_loss, tr_acc, tr_f1 = run_epoch(train_loader, train=True)
    te_loss, te_acc, te_f1 = run_epoch(test_loader, train=False)
    history["train_loss"].append(tr_loss); history["test_loss"].append(te_loss)
    history["train_acc"].append(tr_acc); history["test_acc"].append(te_acc)
    elapsed = time.time() - t_start
    print(f"Epoch {epoch}/{EPOCHS} | train_loss={tr_loss:.4f} acc={tr_acc:.3f} | "
          f"test_loss={te_loss:.4f} acc={te_acc:.3f} f1={te_f1:.3f} | {elapsed:.0f}s")

print(f"Eğitim tamamlandı: {time.time()-t_start:.0f}s")


# ──────────────────────────────────────────────────────────────────────────
# 6) Değerlendirme görselleri
# ──────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
axes[0].plot(history["train_loss"], label="Eğitim", color="#534AB7")
axes[0].plot(history["test_loss"], label="Test", color="#E24B4A")
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Kayıp"); axes[0].set_title("Eğitim/Test Kaybı")
axes[0].legend()
axes[1].plot(history["train_acc"], label="Eğitim", color="#534AB7")
axes[1].plot(history["test_acc"], label="Test", color="#E24B4A")
axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Doğruluk"); axes[1].set_title("Eğitim/Test Doğruluğu")
axes[1].legend()
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "01_training_curves.png"), dpi=130)
plt.close(fig)

model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for ids, lengths, labels in test_loader:
        mask = make_mask(lengths)
        logits, _ = model(ids, mask)
        all_preds.extend((torch.sigmoid(logits) > 0.5).float().tolist())
        all_labels.extend(labels.tolist())

cm = confusion_matrix(all_labels, all_preds)
fig, ax = plt.subplots(figsize=(5, 4.5))
im = ax.imshow(cm, cmap="Purples")
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)
ax.set_xticks([0, 1]); ax.set_xticklabels(["Negatif", "Pozitif"])
ax.set_yticks([0, 1]); ax.set_yticklabels(["Negatif", "Pozitif"])
ax.set_xlabel("Tahmin"); ax.set_ylabel("Gerçek")
ax.set_title(f"Karmaşıklık Matrisi (Doğruluk: %{100*accuracy_score(all_labels, all_preds):.1f})")
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "02_confusion_matrix.png"), dpi=130)
plt.close(fig)

final_acc = accuracy_score(all_labels, all_preds)
final_f1 = f1_score(all_labels, all_preds)
print(f"\n── Test Sonuçları ── Doğruluk: %{100*final_acc:.1f} | F1: {final_f1:.3f}")


# ──────────────────────────────────────────────────────────────────────────
# 7) ASIL GÖSTERİ — Attention Ağırlıklarını Görselleştirme
# ──────────────────────────────────────────────────────────────────────────
def get_attention_for_review(review_text, true_label=None):
    tokens = clean_and_tokenize(review_text)
    ids, length = encode(tokens)
    ids_t = torch.tensor([ids])
    mask = make_mask(torch.tensor([length]))
    model.eval()
    with torch.no_grad():
        logits, weights = model(ids_t, mask)
    prob = torch.sigmoid(logits).item()
    words = tokens[:length]
    attn = weights[0, :length].numpy()
    return words, attn, prob


sample_reviews = [
    "This game absolutely blew me away, the combat feels rewarding and the story kept me engaged the whole time",
    "This was a complete waste of money, the game constantly crashes and support never responds to tickets",
    "The game started great but after a few hours the progression grinds to a halt and it becomes repetitive",
]

print("\n── Örnek Attention Analizi ──")
attention_examples = []
for review in sample_reviews:
    words, attn, prob = get_attention_for_review(review)
    attention_examples.append((words, attn, prob))
    top_idx = np.argsort(-attn)[:5]
    top_words = [(words[i], float(attn[i])) for i in top_idx]
    label = "POZİTİF" if prob > 0.5 else "NEGATİF"
    print(f"\n  Yorum: \"{review[:70]}...\"")
    print(f"  Tahmin: {label} (p={prob:.2f})")
    print(f"  En çok dikkat edilen kelimeler: " + ", ".join(f"{w}({s:.2f})" for w, s in top_words))

fig, axes = plt.subplots(len(sample_reviews), 1, figsize=(14, 3.2 * len(sample_reviews)))
for ax, (words, attn, prob) in zip(axes, attention_examples):
    attn_norm = attn / (attn.max() + 1e-9)
    colors = plt.cm.Purples(0.15 + 0.8 * attn_norm)
    ax.bar(range(len(words)), [1] * len(words), color=colors, width=1.0)
    ax.set_xticks(range(len(words)))
    ax.set_xticklabels(words, rotation=60, ha="right", fontsize=8)
    ax.set_yticks([])
    label = "POZİTİF" if prob > 0.5 else "NEGATİF"
    ax.set_title(f"Tahmin: {label} (p={prob:.2f}) — koyu renk = yüksek attention", fontsize=10, loc="left")
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "03_attention_heatmap.png"), dpi=130)
plt.close(fig)

print(f"\nGörseller '{FIG_DIR}/' klasörüne kaydedildi.")

# ──────────────────────────────────────────────────────────────────────────
# 8) Modeli kaydet
# ──────────────────────────────────────────────────────────────────────────
torch.save({
    "model_state": model.state_dict(),
    "word2id": word2id,
    "max_len": MAX_LEN,
    "hidden_size": HIDDEN_SIZE,
    "emb_dim": EMB_DIM,
}, os.path.join(DATA_DIR, "attention_model.pt"))

print("\n── Özet ──")
print(f"Vocab: {VOCAB_SIZE:,} | Eğitim örneği: {len(train_df)} | Test örneği: {len(test_df)}")
print(f"Test doğruluğu: %{100*final_acc:.1f} | F1: {final_f1:.3f}")
print("Tamamlandı.")
