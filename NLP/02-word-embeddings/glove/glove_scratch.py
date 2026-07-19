"""
==============================================================================
 GloVe (Global Vectors) — Sıfırdan PyTorch İmplementasyonu (OYUN VERSİYONU)
==============================================================================

ÖĞRENME NOTU (yönlendirme):
  Bu dosyanın amacı, count-based matrix factorization'a dayanan GloVe
  (Global Vectors) yöntemini sıfırdan PyTorch ile kurup bir metin
  korpusu üzerinde eğitmektir.

  VERI SETI NOTU: word2vec projesiyle AYNI gerekce - `fronkongames/
  steam-games-dataset` katalogundaki "About the game" aciklamalari
  birlestirilerek olusturulan GERCEK oyun-domaini metin korpusu
  kullaniliyor (adil kiyas icin word2vec projesiyle AYNI korpus).
  Degerlendirme yine Mikolov'un resmi (genel Ingilizce) analoji test
  setiyle yapiliyor.

Veri seti: Kaggle fronkongames/steam-games-dataset — "About the game"
kolonlarinin birlestirilmesiyle olusturulan ~12M karakterlik alt kume
(word2vec projesindeki AYNI korpus, adil kiyas icin).
==============================================================================
"""

import os
import re
import time
import random
import urllib.request

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.manifold import TSNE
from scipy.sparse import coo_matrix
from collections import Counter

import kagglehub

# ──────────────────────────────────────────────────────────────────────────
# 0) Sabitler / Hiperparametreler
# ──────────────────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

DATA_DIR = "data"
FIG_DIR = "figures"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

QUESTIONS_URL = "https://raw.githubusercontent.com/tmikolov/word2vec/master/questions-words.txt"
CORPUS_PATH = os.path.join(DATA_DIR, "game_corpus.txt")
QUESTIONS_PATH = os.path.join(DATA_DIR, "questions-words.txt")

SUBSET_CHARS = 12_000_000     # Word2Vec projesiyle AYNI alt küme (adil kıyas)
MIN_COUNT = 5
WINDOW_SIZE = 5
EMB_DIM = 100
X_MAX = 100.0                 # ağırlıklandırma fonksiyonu doygunluk noktası
ALPHA = 0.75                  # ağırlıklandırma fonksiyonu üsteli
BATCH_SIZE = 65536
EPOCHS = 35                   # kayıp epoch ~30 civarında yakınsıyor
LR = 0.05                     # Adagrad (orijinal makaledeki optimizer)

N_ANALOGY_PER_CATEGORY = 150

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Cihaz: {device} | CPU çekirdek: {torch.get_num_threads()}")


# ──────────────────────────────────────────────────────────────────────────
# 1) Oyun aciklamasi korpusunu olustur (Kaggle) + analoji test setini indir
# ──────────────────────────────────────────────────────────────────────────
def build_game_corpus():
    if os.path.exists(CORPUS_PATH):
        print(f"Oyun aciklamasi korpusu zaten mevcut: {CORPUS_PATH}"); return
    print("Steam oyun katalogu indiriliyor (Kaggle: fronkongames/steam-games-dataset)...")
    dataset_path = kagglehub.dataset_download("fronkongames/steam-games-dataset")
    csv_files = [f for f in os.listdir(dataset_path) if f.lower().endswith(".csv")]
    catalog_path = os.path.join(dataset_path, csv_files[0])

    header = pd.read_csv(catalog_path, nrows=0).columns
    about_candidates = ["About the game", "about_the_game", "detailed_description", "short_description"]
    about_col = next((c for c in about_candidates if c in header), None)
    if about_col is None:
        raise KeyError(f"Aciklama kolonu bulunamadi. Mevcut kolonlar: {list(header)}")

    print(f"Aciklama kolonu: '{about_col}' - metinler birlestiriliyor...")
    parts, total_chars = [], 0
    for chunk in pd.read_csv(catalog_path, usecols=[about_col], chunksize=5000):
        for text in chunk[about_col].dropna().astype(str):
            parts.append(text)
            total_chars += len(text)
        if total_chars > SUBSET_CHARS * 2:
            break

    raw_text = " ".join(parts)
    raw_text = re.sub(r"<[^>]+>", " ", raw_text)
    raw_text = re.sub(r"[^a-zA-Z\s]", " ", raw_text).lower()
    raw_text = re.sub(r"\s+", " ", raw_text)

    with open(CORPUS_PATH, "w") as f:
        f.write(raw_text)
    print(f"Korpus olusturuldu: {len(raw_text):,} karakter -> {CORPUS_PATH}")


def download_questions():
    if os.path.exists(QUESTIONS_PATH):
        print(f"Analoji test seti zaten mevcut: {QUESTIONS_PATH}"); return
    print("Google analoji test seti indiriliyor...")
    urllib.request.urlretrieve(QUESTIONS_URL, QUESTIONS_PATH)


build_game_corpus()
download_questions()

with open(CORPUS_PATH) as f:
    raw_text = f.read(SUBSET_CHARS)
tokens = raw_text.split()
print(f"\nAlt küme boyutu: {len(raw_text):,} karakter → {len(tokens):,} kelime (token)")


# ──────────────────────────────────────────────────────────────────────────
# 2) Kelime dağarcığı
# ──────────────────────────────────────────────────────────────────────────
counts = Counter(tokens)
vocab_words = [w for w, c in counts.items() if c >= MIN_COUNT]
vocab_words.sort(key=lambda w: -counts[w])
word2id = {w: i for i, w in enumerate(vocab_words)}
id2word = {i: w for w, i in word2id.items()}
VOCAB_SIZE = len(vocab_words)
print(f"Vocab boyutu (min_count={MIN_COUNT}): {VOCAB_SIZE:,}")

token_ids = np.array([word2id[w] for w in tokens if w in word2id], dtype=np.int64)


# ──────────────────────────────────────────────────────────────────────────
# 3) Global birlikte-geçme (co-occurrence) matrisi — vektörize kurulum
# ──────────────────────────────────────────────────────────────────────────
t0 = time.time()
rows, cols, vals = [], [], []
for d in range(1, WINDOW_SIZE + 1):
    ci, cj = token_ids[:-d], token_ids[d:]
    w = 1.0 / d
    rows.append(ci); cols.append(cj); vals.append(np.full(len(ci), w, dtype=np.float32))
    rows.append(cj); cols.append(ci); vals.append(np.full(len(ci), w, dtype=np.float32))
rows = np.concatenate(rows); cols = np.concatenate(cols); vals = np.concatenate(vals)

cooc = coo_matrix((vals, (rows, cols)), shape=(VOCAB_SIZE, VOCAB_SIZE))
cooc.sum_duplicates()
print(f"Co-occurrence matrisi: {cooc.nnz:,} sıfır olmayan çift ({time.time()-t0:.1f}s)")

i_arr = torch.from_numpy(cooc.row.astype(np.int64))
j_arr = torch.from_numpy(cooc.col.astype(np.int64))
x_arr = torch.from_numpy(cooc.data.astype(np.float32))

fx = torch.where(x_arr < X_MAX, (x_arr / X_MAX) ** ALPHA, torch.ones_like(x_arr))
logx = torch.log(x_arr)


class CooccurrenceDataset(Dataset):
    def __init__(self, i, j, fx, logx):
        self.i, self.j, self.fx, self.logx = i, j, fx, logx

    def __len__(self):
        return len(self.i)

    def __getitem__(self, idx):
        return self.i[idx], self.j[idx], self.fx[idx], self.logx[idx]


i_arr, j_arr = i_arr.to(device), j_arr.to(device)
fx, logx = fx.to(device), logx.to(device)
N_PAIRS = len(i_arr)
n_batches_per_epoch = (N_PAIRS + BATCH_SIZE - 1) // BATCH_SIZE


# ──────────────────────────────────────────────────────────────────────────
# 4) Model — GloVe (sıfırdan)
# ──────────────────────────────────────────────────────────────────────────
class GloVeModel(nn.Module):
    def __init__(self, vocab_size, emb_dim):
        super().__init__()
        self.w = nn.Embedding(vocab_size, emb_dim)
        self.wc = nn.Embedding(vocab_size, emb_dim)
        self.b = nn.Embedding(vocab_size, 1)
        self.bc = nn.Embedding(vocab_size, 1)
        nn.init.uniform_(self.w.weight, -0.5 / emb_dim, 0.5 / emb_dim)
        nn.init.uniform_(self.wc.weight, -0.5 / emb_dim, 0.5 / emb_dim)
        nn.init.zeros_(self.b.weight)
        nn.init.zeros_(self.bc.weight)

    def forward(self, i, j):
        return (self.w(i) * self.wc(j)).sum(dim=1) + self.b(i).squeeze(1) + self.bc(j).squeeze(1)


model = GloVeModel(VOCAB_SIZE, EMB_DIM).to(device)
optimizer = torch.optim.Adagrad(model.parameters(), lr=LR)

print(f"\nModel parametre sayısı: {sum(p.numel() for p in model.parameters()):,}")
print(f"Epoch başına adım: {n_batches_per_epoch:,} | Toplam adım: {EPOCHS * n_batches_per_epoch:,}")


# ──────────────────────────────────────────────────────────────────────────
# 5) Eğitim döngüsü
# ──────────────────────────────────────────────────────────────────────────
print("\n── Eğitim başlıyor ──")
loss_history = []
t_train_start = time.time()

for epoch in range(1, EPOCHS + 1):
    perm = torch.randperm(N_PAIRS)
    epoch_loss, n_seen = 0.0, 0
    for s in range(0, N_PAIRS, BATCH_SIZE):
        idx = perm[s:s + BATCH_SIZE]
        i_b, j_b = i_arr[idx], j_arr[idx]
        fx_b, logx_b = fx[idx], logx[idx]

        optimizer.zero_grad()
        pred = model(i_b, j_b)
        loss = (fx_b * (pred - logx_b) ** 2).mean()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * len(idx)
        n_seen += len(idx)

    avg_loss = epoch_loss / n_seen
    loss_history.append(avg_loss)
    if epoch % 5 == 0 or epoch == 1:
        elapsed = time.time() - t_train_start
        print(f"Epoch {epoch:3d}/{EPOCHS} | ortalama kayıp: {avg_loss:.4f} | geçen süre: {elapsed:.0f}s")

print(f"Eğitim tamamlandı. Toplam süre: {time.time()-t_train_start:.0f}s")

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(range(1, EPOCHS + 1), loss_history, color="#1D9E75")
ax.set_xlabel("Epoch"); ax.set_ylabel("Ağırlıklı MSE Kaybı")
ax.set_title("Eğitim Kaybı — GloVe (Oyun Korpusu, Weighted Least Squares)")
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "01_training_loss.png"), dpi=130)
plt.close(fig)

final_vectors = (model.w.weight.detach().cpu().numpy() + model.wc.weight.detach().cpu().numpy())
norm_vectors = final_vectors / (np.linalg.norm(final_vectors, axis=1, keepdims=True) + 1e-9)


# ──────────────────────────────────────────────────────────────────────────
# 6) En yakın komşu sorguları
# ──────────────────────────────────────────────────────────────────────────
def nearest_neighbors(vecs, w2id, i2w, word, k=8):
    if word not in w2id:
        return None
    idx = w2id[word]
    sims = vecs @ vecs[idx]
    top_idx = np.argsort(-sims)[1:k + 1]
    return [(i2w[i], float(sims[i])) for i in top_idx]


query_words = ["battle", "sword", "dragon", "space", "zombie", "level", "quest", "boss"]
print("\n── En Yakın Komşular ──")
neighbor_results = {}
for w in query_words:
    nn_list = nearest_neighbors(norm_vectors, word2id, id2word, w)
    neighbor_results[w] = nn_list
    if nn_list:
        print(f"  {w:<12} → " + ", ".join(f"{n}({s:.2f})" for n, s in nn_list[:6]))


# ──────────────────────────────────────────────────────────────────────────
# 7) Analoji görevi — resmi Google analoji test seti (3CosAdd)
# ──────────────────────────────────────────────────────────────────────────
def load_analogy_questions(path):
    categories, cur_cat = {}, None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(":"):
                cur_cat = line[1:].strip(); categories[cur_cat] = []; continue
            words = line.lower().split()
            if len(words) == 4:
                categories[cur_cat].append(words)
    return categories


def evaluate_analogies(vectors_norm, w2id, categories, per_category=N_ANALOGY_PER_CATEGORY):
    results = {}
    vt = torch.tensor(vectors_norm)
    for cat, questions in categories.items():
        valid = [q for q in questions if all(w in w2id for w in q)]
        if not valid:
            results[cat] = (0, 0); continue
        sample = random.sample(valid, min(per_category, len(valid)))
        correct = 0
        for a, b, c, d in sample:
            ia, ib, ic, id_ = w2id[a], w2id[b], w2id[c], w2id[d]
            query = vt[ib] - vt[ia] + vt[ic]
            query = query / (query.norm() + 1e-9)
            sims = vt @ query
            sims[ia] = -1; sims[ib] = -1; sims[ic] = -1
            pred = int(torch.argmax(sims))
            if pred == id_:
                correct += 1
        results[cat] = (correct, len(sample))
    return results


categories = load_analogy_questions(QUESTIONS_PATH)
print(f"\n── Analoji Değerlendirmesi ──")
t0 = time.time()
glove_results = evaluate_analogies(norm_vectors, word2id, categories)
total_correct = sum(c for c, n in glove_results.values())
total_n = sum(n for c, n in glove_results.values())
for cat, (c, n) in glove_results.items():
    acc = 100 * c / n if n else 0
    print(f"  {cat:<32} {c:4d}/{n:<4d} (%{acc:5.1f})")
print(f"  {'TOPLAM':<32} {total_correct:4d}/{total_n:<4d} (%{100*total_correct/max(total_n,1):5.1f})  [{time.time()-t0:.1f}s]")


# ──────────────────────────────────────────────────────────────────────────
# 8) Word2Vec projesiyle kıyaslama (count-based vs prediction-based)
# ──────────────────────────────────────────────────────────────────────────
w2v_ckpt_path = "reference_word2vec_model.pt"
w2v_results, w2v_total_correct, w2v_total_n = None, None, None
if os.path.exists(w2v_ckpt_path):
    print("\n── Word2Vec Projesiyle Kıyaslama (aynı korpus) ──")
    ckpt = torch.load(w2v_ckpt_path, weights_only=False)
    w2v_vectors = ckpt["in_embed"].numpy()
    w2v_word2id = ckpt["word2id"]
    w2v_norm = w2v_vectors / (np.linalg.norm(w2v_vectors, axis=1, keepdims=True) + 1e-9)
    w2v_results = evaluate_analogies(w2v_norm, w2v_word2id, categories)
    w2v_total_correct = sum(c for c, n in w2v_results.values())
    w2v_total_n = sum(n for c, n in w2v_results.values())
    print(f"Word2Vec analoji doğruluğu: {w2v_total_correct}/{w2v_total_n} (%{100*w2v_total_correct/max(w2v_total_n,1):.1f})")
    print(f"GloVe analoji doğruluğu   : {total_correct}/{total_n} (%{100*total_correct/max(total_n,1):.1f})")
else:
    print("\n(Word2Vec kontrol noktası bulunamadı — kıyas atlanıyor. Kendi başına çalışabilir.")
    print(" ../word2vec/data/word2vec_model.pt dosyasını buraya 'reference_word2vec_model.pt' olarak kopyalayabilirsin.)")


# ──────────────────────────────────────────────────────────────────────────
# 9) Görseller
# ──────────────────────────────────────────────────────────────────────────
cats_sorted = sorted(glove_results.keys())
glove_acc = [100 * glove_results[c][0] / max(glove_results[c][1], 1) for c in cats_sorted]

fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(cats_sorted))
if w2v_results:
    w2v_acc = [100 * w2v_results[c][0] / max(w2v_results[c][1], 1) for c in cats_sorted]
    width = 0.38
    ax.bar(x - width/2, glove_acc, width, label="GloVe (sıfırdan)", color="#1D9E75")
    ax.bar(x + width/2, w2v_acc, width, label="Word2Vec (aynı korpus)", color="#534AB7")
    ax.legend()
    title = "Analoji Görevi — GloVe vs Word2Vec (Kategori Bazlı, Oyun Korpusu)"
else:
    ax.bar(x, glove_acc, color="#1D9E75")
    title = "Analoji Görevi — Kategori Bazlı Doğruluk (GloVe, Oyun Korpusu)"
ax.set_xticks(x); ax.set_xticklabels(cats_sorted, rotation=40, ha="right")
ax.set_ylabel("Doğruluk (%)"); ax.set_title(title)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "02_analogy_accuracy_comparison.png"), dpi=130)
plt.close(fig)

tsne_groups = {
    "Savas/Aksiyon": ["battle", "sword", "war", "combat", "fight", "attack", "enemy", "weapon"],
    "Fantastik": ["dragon", "magic", "spell", "wizard", "castle", "kingdom", "quest", "hero"],
    "Bilim Kurgu": ["space", "alien", "robot", "planet", "future", "laser", "ship", "galaxy"],
    "Oyun Mekanigi": ["level", "boss", "player", "score", "mission", "puzzle", "survival", "multiplayer"],
    "Korku": ["zombie", "horror", "dark", "monster", "survive", "fear", "night", "blood"],
    "Genel Sifat": ["good", "bad", "fast", "slow", "large", "small", "new", "classic"],
}
plot_words, plot_labels = [], []
for group, words in tsne_groups.items():
    for w in words:
        if w in word2id:
            plot_words.append(w); plot_labels.append(group)
plot_vecs = np.array([norm_vectors[word2id[w]] for w in plot_words])
tsne = TSNE(n_components=2, perplexity=15, random_state=SEED, init="pca", max_iter=1000)
coords = tsne.fit_transform(plot_vecs)

fig, ax = plt.subplots(figsize=(11, 8.5))
palette = ["#534AB7", "#E24B4A", "#1D9E75", "#BA7517", "#3178C6", "#C2338B"]
for idx, group in enumerate(tsne_groups.keys()):
    mask = [j for j, l in enumerate(plot_labels) if l == group]
    ax.scatter(coords[mask, 0], coords[mask, 1], color=palette[idx % len(palette)], label=group, s=60)
    for j in mask:
        ax.annotate(plot_words[j], (coords[j, 0], coords[j, 1]), fontsize=9, alpha=0.85,
                     xytext=(4, 4), textcoords="offset points")
ax.set_title("t-SNE ile GloVe Kelime Vektörü Görselleştirmesi (Oyun Korpusu)")
ax.legend(loc="best"); ax.set_xticks([]); ax.set_yticks([])
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "03_tsne_word_clusters.png"), dpi=130)
plt.close(fig)

fig, axes = plt.subplots(2, 4, figsize=(16, 7))
for ax, w in zip(axes.flat, query_words):
    nn_list = neighbor_results.get(w)
    if not nn_list:
        ax.axis("off"); continue
    names = [n for n, s in nn_list[:6]][::-1]
    scores = [s for n, s in nn_list[:6]][::-1]
    ax.barh(names, scores, color="#1D9E75")
    ax.set_title(f'"{w}"', fontsize=12)
    ax.set_xlim(0, 1)
fig.suptitle("En Yakın Komşu Kelimeler (kosinüs benzerliği) — GloVe (Oyun Korpusu)", fontsize=14)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "04_nearest_neighbors.png"), dpi=130)
plt.close(fig)

print(f"\nGörseller '{FIG_DIR}/' klasörüne kaydedildi.")

# ──────────────────────────────────────────────────────────────────────────
# 10) Modeli kaydet
# ──────────────────────────────────────────────────────────────────────────
torch.save({
    "w": model.w.weight.detach().cpu(),
    "wc": model.wc.weight.detach().cpu(),
    "word2id": word2id, "id2word": id2word, "emb_dim": EMB_DIM,
}, os.path.join(DATA_DIR, "glove_model.pt"))

print("\n── Özet ──")
print(f"Vocab: {VOCAB_SIZE:,} | Co-occurrence çifti: {cooc.nnz:,}")
print(f"GloVe analoji doğruluğu: %{100*total_correct/max(total_n,1):.1f}")
print("Tamamlandı.")
