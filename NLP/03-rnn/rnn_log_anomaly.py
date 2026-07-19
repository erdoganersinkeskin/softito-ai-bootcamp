"""
Oyun Sunucusu Oturum Logu Anomali Tespiti — Vanilla RNN (PyTorch)
(OYUN VERSİYONU)
===================================================================

ÖĞRENME NOTU (yönlendirme):
  Bu dosyanın amacı, olay dizilerinden (event sequence) ikili sınıflandırma
  yapan bir Vanilla RNN (nn.RNN, embedding, pos_weight ile dengesiz sınıf
  telafisi, ROC/confusion matrix) inşa etmektir.

  VERI SETI NOTU: Oturum bazlı olay dizisi + başarı/hata etiketi içeren
  gerçek bir veri seti, paylaşılan 9 Kaggle veri setinin hiçbirinde yok
  (hepsi satır-bazlı katalog/değerlendirme verisi, oturum/dizi verisi
  değil). Bu yüzden görev tanımındaki istisna uygulanarak, E<n> etiketli
  olay token dizileri + Normal/Anomali etiketi yapısını koruyan, ama
  olayları GERÇEKÇİ bir oyun sunucusu OTURUM akışına (giriş, eşleşme,
  oyun başı, satın alma, seviye atlama, sohbet, çökme, hile bayrağı,
  bağlantı kopması, oyun sonu) göre isimlendiren SENTETİK bir veri seti
  üretiliyor.
"""

import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score, roc_curve, auc
)

# ─────────────────────────────────────────────
# 0. Ayarlar
# ─────────────────────────────────────────────
SEED           = 42
FIGURES_DIR    = "figures"
BATCH_SIZE     = 128
EPOCHS         = 20
LR             = 1e-3
HIDDEN_SIZE    = 64
NUM_LAYERS     = 1
MAX_SEQ_LEN    = 50
TEST_SIZE      = 0.2
N_SESSIONS     = 60000        # sentetik oturum (session) sayisi
ANOMALY_RATE   = 0.12         # cokme/hile bayrakli "anormal" oturum orani

os.makedirs(FIGURES_DIR, exist_ok=True)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Cihaz: {DEVICE}")

# ─────────────────────────────────────────────
# 1. Sentetik Oyun Sunucusu Oturum Logu Üretimi
# ─────────────────────────────────────────────
# Her oturum, bir oyuncunun sunucuyla etkilesim boyunca ürettigi bir olay
# dizisidir. Normal oturumlar duzenli bir akis izler (giris -> eslesme ->
# oyun -> cikis); anormal oturumlar cokme/hile/baglanti kopmasi gibi
# beklenmedik olaylar icerir - tipki HDFS'teki "Fail" bloklarinin normalden
# sapan olay oruntuleri gibi.
NORMAL_EVENTS = [
    "LOGIN", "AUTH_OK", "MATCHMAKING_START", "MATCHMAKING_FOUND",
    "GAME_START", "LEVEL_UP", "ITEM_PICKUP", "CHAT_MSG", "ACHIEVEMENT",
    "ITEM_PURCHASE", "GAME_SAVE", "GAME_END", "LOGOUT",
]
ANOMALY_EVENTS = [
    "CONN_TIMEOUT", "PACKET_LOSS", "CLIENT_CRASH", "CHEAT_FLAG",
    "SPEED_HACK_DETECTED", "SERVER_ERROR_500", "FORCED_DISCONNECT",
    "DESYNC_WARNING", "REPEATED_LOGIN_FAIL",
]
ALL_EVENTS = NORMAL_EVENTS + ANOMALY_EVENTS
event2idx = {e: i + 1 for i, e in enumerate(sorted(ALL_EVENTS))}  # 0 = padding
VOCAB_SIZE = len(event2idx) + 1
print(f"Olay turu sayisi: {len(event2idx)}")


def make_normal_session():
    length = random.randint(6, MAX_SEQ_LEN)
    seq = ["LOGIN", "AUTH_OK", "MATCHMAKING_START", "MATCHMAKING_FOUND", "GAME_START"]
    body_pool = ["LEVEL_UP", "ITEM_PICKUP", "CHAT_MSG", "ACHIEVEMENT", "ITEM_PURCHASE", "GAME_SAVE"]
    while len(seq) < length - 2:
        seq.append(random.choice(body_pool))
    seq += ["GAME_END", "LOGOUT"]
    return seq[:MAX_SEQ_LEN]


def make_anomalous_session():
    length = random.randint(6, MAX_SEQ_LEN)
    seq = ["LOGIN", "AUTH_OK", "MATCHMAKING_START", "MATCHMAKING_FOUND", "GAME_START"]
    body_pool = ["LEVEL_UP", "ITEM_PICKUP", "CHAT_MSG", "ACHIEVEMENT"]
    n_anomaly_events = random.randint(1, 4)
    while len(seq) < length - 2:
        if random.random() < 0.35:
            seq.append(random.choice(ANOMALY_EVENTS))
        else:
            seq.append(random.choice(body_pool))
    # en az bir anormal olay garanti et
    if not any(e in ANOMALY_EVENTS for e in seq):
        seq.insert(random.randint(len(seq) // 2, len(seq) - 1), random.choice(ANOMALY_EVENTS))
    seq += [random.choice(["CLIENT_CRASH", "FORCED_DISCONNECT", "GAME_END"]), "LOGOUT"]
    return seq[:MAX_SEQ_LEN]


print("Sentetik oyun sunucusu oturum logu uretiliyor...")
records = []
for _ in range(N_SESSIONS):
    is_anomaly = random.random() < ANOMALY_RATE
    tokens = make_anomalous_session() if is_anomaly else make_normal_session()
    records.append({"tokens": tokens, "label_int": int(is_anomaly)})

df = pd.DataFrame(records)
print(f"  Toplam oturum : {len(df):,}")
print(f"  Normal        : {(df['label_int']==0).sum():,}")
print(f"  Anomali       : {(df['label_int']==1).sum():,}")

# ─────────────────────────────────────────────
# 2. Dizi Kodlama
# ─────────────────────────────────────────────
def encode_sequence(tokens, max_len=MAX_SEQ_LEN):
    ids = [event2idx.get(t, 0) for t in tokens[:max_len]]
    pad_len = max_len - len(ids)
    return ids + [0] * pad_len

df["encoded"] = df["tokens"].apply(encode_sequence)

# ─────────────────────────────────────────────
# 3. EDA Görselleri
# ─────────────────────────────────────────────
print("\nEDA görselleri üretiliyor...")

fig, axes = plt.subplots(1, 3, figsize=(16, 4))
fig.suptitle("Oyun Sunucusu Oturum Logu — Keşifsel Analiz", fontsize=14, fontweight="bold")

counts = df["label_int"].value_counts()
axes[0].bar(["Normal", "Anomali"], [counts.get(0, 0), counts.get(1, 0)],
            color=["#4C72B0", "#DD8452"])
axes[0].set_title("Sınıf Dağılımı")
axes[0].set_ylabel("Oturum Sayısı")
for i, v in enumerate([counts.get(0, 0), counts.get(1, 0)]):
    axes[0].text(i, v + 500, f"{v:,}", ha="center", fontsize=9)

seq_lens = df["tokens"].apply(len)
axes[1].hist(seq_lens, bins=40, color="#4C72B0", edgecolor="white")
axes[1].axvline(MAX_SEQ_LEN, color="red", linestyle="--", label=f"MAX_SEQ_LEN={MAX_SEQ_LEN}")
axes[1].set_title("Oturum Dizi Uzunluğu Dağılımı")
axes[1].set_xlabel("Olay Sayısı")
axes[1].set_ylabel("Oturum Sayısı")
axes[1].legend()

from collections import Counter
all_tokens = []
for t in df["tokens"]:
    all_tokens.extend(t)
freq = Counter(all_tokens).most_common(15)
labels_ev, vals = zip(*freq)
axes[2].barh(labels_ev[::-1], vals[::-1], color="#4C72B0")
axes[2].set_title("En Sık Görülen 15 Olay")
axes[2].set_xlabel("Frekans")

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "01_eda.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  -> figures/01_eda.png kaydedildi")

# ─────────────────────────────────────────────
# 4. Dataset & DataLoader
# ─────────────────────────────────────────────
class SessionDataset(Dataset):
    def __init__(self, encodings, labels):
        self.X = torch.tensor(encodings, dtype=torch.long)
        self.y = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

X = np.array(df["encoded"].tolist())
y = df["label_int"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y
)

train_loader = DataLoader(SessionDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(SessionDataset(X_test,  y_test),  batch_size=BATCH_SIZE, shuffle=False)

# ─────────────────────────────────────────────
# 5. Vanilla RNN Modeli
# ─────────────────────────────────────────────
class VanillaRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn = nn.RNN(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            nonlinearity="tanh"
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        embedded = self.embedding(x)
        _, hidden = self.rnn(embedded)
        out = self.fc(hidden[-1])
        return out.squeeze(1)

EMBED_DIM = 32
model = VanillaRNN(VOCAB_SIZE, EMBED_DIM, HIDDEN_SIZE, NUM_LAYERS).to(DEVICE)
print(f"\nModel:\n{model}")
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Eğitilebilir parametre: {total_params:,}")

pos_weight = torch.tensor([(y_train == 0).sum() / (y_train == 1).sum()]).to(DEVICE)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# ─────────────────────────────────────────────
# 6. Eğitim
# ─────────────────────────────────────────────
print("\nEğitim başlıyor...")
train_losses, val_accs = [], []

for epoch in range(1, EPOCHS + 1):
    model.train()
    epoch_loss = 0.0
    for X_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}", leave=False):
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        epoch_loss += loss.item() * len(y_batch)

    avg_loss = epoch_loss / len(train_loader.dataset)
    train_losses.append(avg_loss)

    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(DEVICE)
            logits = model(X_batch)
            preds = (torch.sigmoid(logits) >= 0.5).cpu().long().tolist()
            all_preds.extend(preds)
            all_labels.extend(y_batch.long().tolist())

    acc = accuracy_score(all_labels, all_preds) * 100
    val_accs.append(acc)
    print(f"  Epoch {epoch:>2}/{EPOCHS}  Loss: {avg_loss:.4f}  Val Accuracy: {acc:.2f}%")

# ─────────────────────────────────────────────
# 7. Eğitim Eğrisi (koyu arka plan)
# ─────────────────────────────────────────────
BG      = "#1e1e2e"
GRID    = "#3a3a5c"
BLUE    = "#5c7cfa"
GREEN   = "#40c057"

with plt.style.context("dark_background"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor(BG)
    for ax in (ax1, ax2):
        ax.set_facecolor(BG)

    ax1.plot(range(EPOCHS), train_losses, color=BLUE, linewidth=2)
    ax1.set_title("Eğitim Kaybı (BCEWithLogitsLoss)", color="white", fontsize=12)
    ax1.set_xlabel("Epoch", color="white")
    ax1.set_ylabel("Kayıp", color="white")
    ax1.tick_params(colors="white")
    ax1.grid(color=GRID, linewidth=0.7)
    for spine in ax1.spines.values():
        spine.set_edgecolor(GRID)

    baseline = (y_test == 0).sum() / len(y_test) * 100
    ax2.plot(range(EPOCHS), val_accs, color=GREEN, linewidth=2)
    ax2.axhline(baseline, color="gray", linestyle="--",
                label=f"Çoğunluk tahmini ({baseline:.0f}%)")
    ax2.set_title("Eğitim Doğruluğu (%)", color="white", fontsize=12)
    ax2.set_xlabel("Epoch", color="white")
    ax2.set_ylabel("Doğruluk (%)", color="white")
    ax2.set_ylim(0, 100)
    ax2.tick_params(colors="white")
    ax2.grid(color=GRID, linewidth=0.7)
    ax2.legend(facecolor=BG, edgecolor=GRID, labelcolor="white")
    for spine in ax2.spines.values():
        spine.set_edgecolor(GRID)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "02_training_curves.png"),
                dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
print("\n  -> figures/02_training_curves.png kaydedildi")

# ─────────────────────────────────────────────
# 8. Final Değerlendirme
# ─────────────────────────────────────────────
model.eval()
all_preds, all_labels, all_probs = [], [], []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(DEVICE)
        logits = model(X_batch)
        probs = torch.sigmoid(logits).cpu().tolist()
        preds = [1 if p >= 0.5 else 0 for p in probs]
        all_preds.extend(preds)
        all_labels.extend(y_batch.long().tolist())
        all_probs.extend(probs)

acc  = accuracy_score(all_labels, all_preds)
f1   = f1_score(all_labels, all_preds, average="macro", zero_division=0)
f1_a = f1_score(all_labels, all_preds, pos_label=1, zero_division=0)

print("\n─── Test Sonuçları ───────────────────────────────")
print(f"  Accuracy       : {acc:.4f}")
print(f"  F1 (macro)     : {f1:.4f}")
print(f"  F1 (anomali)   : {f1_a:.4f}")
print("\n" + classification_report(all_labels, all_preds,
      target_names=["Normal", "Anomali"], zero_division=0))

# ─────────────────────────────────────────────
# 9. Confusion Matrix
# ─────────────────────────────────────────────
cm = confusion_matrix(all_labels, all_preds)
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Normal", "Anomali"],
            yticklabels=["Normal", "Anomali"], ax=ax)
ax.set_title("Confusion Matrix", fontsize=13, fontweight="bold")
ax.set_ylabel("Gerçek")
ax.set_xlabel("Tahmin")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "03_confusion_matrix.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  -> figures/03_confusion_matrix.png kaydedildi")

# ─────────────────────────────────────────────
# 10. ROC Eğrisi
# ─────────────────────────────────────────────
fpr, tpr, _ = roc_curve(all_labels, all_probs)
roc_auc = auc(fpr, tpr)

fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(fpr, tpr, color="#4C72B0", lw=2, label=f"ROC AUC = {roc_auc:.4f}")
ax.plot([0, 1], [0, 1], color="gray", linestyle="--")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Eğrisi — Vanilla RNN", fontsize=13, fontweight="bold")
ax.legend(loc="lower right")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "04_roc_curve.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  -> figures/04_roc_curve.png kaydedildi")

# ─────────────────────────────────────────────
# 11. Örnek Tahminler
# ─────────────────────────────────────────────
print("\n─── Örnek Tahminler ─────────────────────────────")
sample_df = df.sample(5, random_state=SEED).reset_index(drop=True)
model.eval()
with torch.no_grad():
    for _, row in sample_df.iterrows():
        seq_tensor = torch.tensor([row["encoded"]], dtype=torch.long).to(DEVICE)
        prob = torch.sigmoid(model(seq_tensor)).item()
        pred = "Anomali" if prob >= 0.5 else "Normal"
        gercek = "Anomali" if row["label_int"] == 1 else "Normal"
        flag = "✓" if pred == gercek else "✗"
        print(f"  {flag}  Gerçek: {gercek:<8}  Tahmin: {pred:<8}  Olasılık: {prob:.3f}")

print("\nTamamlandı. Görseller 'figures/' klasöründe.")
