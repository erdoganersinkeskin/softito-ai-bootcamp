"""
==============================================================================
 Word2Vec (Skip-gram + Negative Sampling) — Sıfırdan PyTorch İmplementasyonu
 (OYUN VERSİYONU)
==============================================================================

ÖĞRENME NOTU (yönlendirme):
  Bu dosyanın amacı, Skip-gram + Negative Sampling, subsampling, doğrusal
  LR azaltma ve gensim kıyaslaması içeren bir Word2Vec implementasyonunu
  sıfırdan PyTorch ile kurup bir metin korpusu üzerinde eğitmektir.

  VERI SETI NOTU: Kelime gomme (embedding) egitimi BUYUK bir metin
  korpusu gerektirir; paylasilan 9 Kaggle veri setinin 8'i tamamen tablo
  (sayisal/kategorik) veridir. Ancak `fronkongames/steam-games-dataset`
  her oyun icin bir "About the game" (oyun aciklamasi) serbest metin
  kolonu icerir - butun katalogdaki aciklamalari birlestirerek GERCEK ve
  OYUN DOMAININE ozgu bir metin korpusu olusturmak mumkun. Bu yuzden
  text8 (Wikipedia) yerine bu korpus kullanilir; mimari ve egitim mantigi
  BIREBIR aynidir.

  Degerlendirme icin hala Mikolov'un resmi analoji test setini
  (questions-words.txt: ulke/baskent, meslek, sayi gibi GENEL Ingilizce
  kategoriler) kullaniyoruz - oyun aciklamalari domaine ozgu oldugundan bu
  genel kategorilerde dogruluk muhtemelen text8'e gore daha dusuk cikacak
  (bu, "domain-specific corpus, general benchmark" uyumsuzlugunun dogal
  bir sonucudur ve kendi basina ogretici bir gozlemdir).

Veri seti: Kaggle fronkongames/steam-games-dataset - "About the game"
kolonlarinin birlestirilmesiyle olusturulan ~12M karakterlik alt kume.
==============================================================================
"""

import os
import re
import time
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.manifold import TSNE
from collections import Counter

import kagglehub
import urllib.request

# ──────────────────────────────────────────────────────────────────────────
# 0) Sabitler / Hiperparametreler
# ──────────────────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DATA_DIR = "data"
FIG_DIR = "figures"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

QUESTIONS_URL = "https://raw.githubusercontent.com/tmikolov/word2vec/master/questions-words.txt"
CORPUS_PATH = os.path.join(DATA_DIR, "game_corpus.txt")
QUESTIONS_PATH = os.path.join(DATA_DIR, "questions-words.txt")

SUBSET_CHARS = 12_000_000     # ~2M kelimelik alt küme — hız/kalite dengesi
MIN_COUNT = 5
SUBSAMPLE_T = 1e-5
WINDOW_SIZE = 5
EMB_DIM = 100
K_NEGATIVE = 5
BATCH_SIZE = 65536
EPOCHS = 4
LR_START = 0.01
LR_END = 0.0005

N_ANALOGY_PER_CATEGORY = 150   # kategori başına en fazla bu kadar soru değerlendirilir

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Cihaz: {device} | CPU çekirdek: {torch.get_num_threads()}")


# ──────────────────────────────────────────────────────────────────────────
# 1) Oyun aciklamasi korpusunu olustur (Kaggle) + analoji test setini indir
# ──────────────────────────────────────────────────────────────────────────
def build_game_corpus():
    if os.path.exists(CORPUS_PATH):
        print(f"Oyun aciklamasi korpusu zaten mevcut: {CORPUS_PATH}")
        return
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
    parts = []
    total_chars = 0
    for chunk in pd.read_csv(catalog_path, usecols=[about_col], chunksize=5000):
        for text in chunk[about_col].dropna().astype(str):
            parts.append(text)
            total_chars += len(text)
        if total_chars > SUBSET_CHARS * 2:   # temizlik oncesi yeterince biriktiginde dur
            break

    raw_text = " ".join(parts)
    raw_text = re.sub(r"<[^>]+>", " ", raw_text)            # HTML etiketlerini temizle
    raw_text = re.sub(r"[^a-zA-Z\s]", " ", raw_text).lower()
    raw_text = re.sub(r"\s+", " ", raw_text)

    with open(CORPUS_PATH, "w") as f:
        f.write(raw_text)
    print(f"Korpus olusturuldu: {len(raw_text):,} karakter -> {CORPUS_PATH}")


def download_questions():
    if os.path.exists(QUESTIONS_PATH):
        print(f"Analoji test seti zaten mevcut: {QUESTIONS_PATH}")
        return
    print("Google analoji test seti indiriliyor...")
    urllib.request.urlretrieve(QUESTIONS_URL, QUESTIONS_PATH)


build_game_corpus()
download_questions()

with open(CORPUS_PATH) as f:
    raw_text = f.read(SUBSET_CHARS)
tokens = raw_text.split()
print(f"\nAlt küme boyutu: {len(raw_text):,} karakter → {len(tokens):,} kelime (token)")


# ──────────────────────────────────────────────────────────────────────────
# 2) Kelime dağarcığı (vocab) + alt örnekleme (subsampling)
# ──────────────────────────────────────────────────────────────────────────
counts = Counter(tokens)
vocab_words = [w for w, c in counts.items() if c >= MIN_COUNT]
vocab_words.sort(key=lambda w: -counts[w])   # en sık kelime id=0

word2id = {w: i for i, w in enumerate(vocab_words)}
id2word = {i: w for w, i in word2id.items()}
VOCAB_SIZE = len(vocab_words)
print(f"Vocab boyutu (min_count={MIN_COUNT}): {VOCAB_SIZE:,}")

token_ids = np.array([word2id[w] for w in tokens if w in word2id], dtype=np.int64)
print(f"Kelime dağarcığındaki toplam token: {len(token_ids):,}")

freqs = np.zeros(VOCAB_SIZE, dtype=np.float64)
for i, w in id2word.items():
    freqs[i] = counts[w]
freq_ratio = freqs / freqs.sum()
keep_prob = (np.sqrt(freq_ratio / SUBSAMPLE_T) + 1) * (SUBSAMPLE_T / freq_ratio)
keep_prob = np.clip(keep_prob, 0, 1)

rand_vals = np.random.rand(len(token_ids))
train_ids = token_ids[rand_vals < keep_prob[token_ids]]
print(f"Alt örnekleme sonrası token sayısı: {len(train_ids):,}")

noise_dist = torch.tensor(freqs ** 0.75)
noise_dist = noise_dist / noise_dist.sum()


# ──────────────────────────────────────────────────────────────────────────
# 3) Skip-gram çiftlerini üret (vektörize, kayan pencere)
# ──────────────────────────────────────────────────────────────────────────
def build_skipgram_pairs(ids, window):
    centers, contexts = [], []
    for offset in range(1, window + 1):
        centers.append(ids[:-offset]);  contexts.append(ids[offset:])
        centers.append(ids[offset:]);   contexts.append(ids[:-offset])
    return np.concatenate(centers), np.concatenate(contexts)


t0 = time.time()
center_ids, context_ids = build_skipgram_pairs(train_ids, WINDOW_SIZE)
print(f"Skip-gram çift sayısı: {len(center_ids):,} ({time.time()-t0:.1f}s)")


class SkipGramDataset(Dataset):
    def __init__(self, centers, contexts):
        self.centers = torch.from_numpy(centers)
        self.contexts = torch.from_numpy(contexts)

    def __len__(self):
        return len(self.centers)

    def __getitem__(self, idx):
        return self.centers[idx], self.contexts[idx]


train_loader = DataLoader(
    SkipGramDataset(center_ids, context_ids),
    batch_size=BATCH_SIZE, shuffle=True, drop_last=True,
)


# ──────────────────────────────────────────────────────────────────────────
# 4) Model — Skip-gram + Negative Sampling (sıfırdan)
# ──────────────────────────────────────────────────────────────────────────
class SkipGramNegativeSampling(nn.Module):
    def __init__(self, vocab_size, emb_dim):
        super().__init__()
        self.in_embed = nn.Embedding(vocab_size, emb_dim)
        self.out_embed = nn.Embedding(vocab_size, emb_dim)
        nn.init.uniform_(self.in_embed.weight, -0.5 / emb_dim, 0.5 / emb_dim)
        nn.init.zeros_(self.out_embed.weight)

    def forward(self, center, context, negatives):
        v_c = self.in_embed(center)
        v_o = self.out_embed(context)
        v_neg = self.out_embed(negatives)

        pos_score = torch.sum(v_c * v_o, dim=1)
        pos_loss = torch.nn.functional.logsigmoid(pos_score)

        neg_score = torch.bmm(v_neg, v_c.unsqueeze(2)).squeeze(2)
        neg_loss = torch.nn.functional.logsigmoid(-neg_score).sum(dim=1)

        return -(pos_loss + neg_loss).mean()


model = SkipGramNegativeSampling(VOCAB_SIZE, EMB_DIM).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR_START)
noise_dist = noise_dist.to(device)

NEG_POOL_SIZE = 20_000_000
neg_pool = torch.multinomial(noise_dist, NEG_POOL_SIZE, replacement=True)
neg_pool_ptr = 0

n_total_steps = EPOCHS * len(train_loader)
print(f"\nModel parametre sayısı: {sum(p.numel() for p in model.parameters()):,}")
print(f"Epoch başına adım: {len(train_loader):,} | Toplam adım: {n_total_steps:,}")


# ──────────────────────────────────────────────────────────────────────────
# 5) Eğitim döngüsü (doğrusal öğrenme oranı azaltma — orijinal word2vec gibi)
# ──────────────────────────────────────────────────────────────────────────
print("\n── Eğitim başlıyor ──")
step = 0
loss_history = []
t_train_start = time.time()

for epoch in range(1, EPOCHS + 1):
    epoch_loss, n_batches = 0.0, 0
    for center, context in train_loader:
        center, context = center.to(device), context.to(device)
        n_needed = len(center) * K_NEGATIVE
        if neg_pool_ptr + n_needed > NEG_POOL_SIZE:
            neg_pool_ptr = 0
        negatives = neg_pool[neg_pool_ptr: neg_pool_ptr + n_needed].view(len(center), K_NEGATIVE)
        neg_pool_ptr += n_needed

        lr = LR_START - (LR_START - LR_END) * (step / n_total_steps)
        for g in optimizer.param_groups:
            g["lr"] = max(lr, LR_END)

        optimizer.zero_grad()
        loss = model(center, context, negatives)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        n_batches += 1
        step += 1

    avg_loss = epoch_loss / n_batches
    loss_history.append(avg_loss)
    elapsed = time.time() - t_train_start
    print(f"Epoch {epoch}/{EPOCHS} | ortalama kayıp: {avg_loss:.4f} | geçen süre: {elapsed:.0f}s")

print(f"Eğitim tamamlandı. Toplam süre: {time.time()-t_train_start:.0f}s")

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(range(1, EPOCHS + 1), loss_history, marker="o", color="#534AB7")
ax.set_xlabel("Epoch")
ax.set_ylabel("Ortalama Kayıp (Negative Sampling)")
ax.set_title("Eğitim Kaybı — Skip-gram + Negative Sampling (Oyun Korpusu)")
ax.set_xticks(range(1, EPOCHS + 1))
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "01_training_loss.png"), dpi=130)
plt.close(fig)

word_vectors = model.in_embed.weight.detach().cpu().numpy()
norm_vectors = word_vectors / (np.linalg.norm(word_vectors, axis=1, keepdims=True) + 1e-9)


# ──────────────────────────────────────────────────────────────────────────
# 6) En yakın komşu sorguları
# ──────────────────────────────────────────────────────────────────────────
def nearest_neighbors(word, k=8):
    if word not in word2id:
        return None
    idx = word2id[word]
    sims = norm_vectors @ norm_vectors[idx]
    top_idx = np.argsort(-sims)[1:k + 1]
    return [(id2word[i], float(sims[i])) for i in top_idx]


query_words = ["battle", "sword", "dragon", "space", "zombie", "level", "quest", "boss"]
print("\n── En Yakın Komşular ──")
neighbor_results = {}
for w in query_words:
    nn_list = nearest_neighbors(w)
    neighbor_results[w] = nn_list
    if nn_list:
        formatted = ", ".join(f"{n}({s:.2f})" for n, s in nn_list[:6])
        print(f"  {w:<12} → {formatted}")
    else:
        print(f"  {w:<12} → [vocab dışında]")


# ──────────────────────────────────────────────────────────────────────────
# 7) Analoji görevi — resmi Google analoji test seti (3CosAdd yöntemi)
# ──────────────────────────────────────────────────────────────────────────
def load_analogy_questions(path):
    categories = {}
    cur_cat = None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(":"):
                cur_cat = line[1:].strip()
                categories[cur_cat] = []
                continue
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
            results[cat] = (0, 0)
            continue
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
print(f"\n── Analoji Değerlendirmesi (genel Ingilizce benchmark, kategori başına en fazla {N_ANALOGY_PER_CATEGORY} soru) ──")
t0 = time.time()
custom_results = evaluate_analogies(norm_vectors, word2id, categories)

total_correct = sum(c for c, n in custom_results.values())
total_n = sum(n for c, n in custom_results.values())
for cat, (c, n) in custom_results.items():
    acc = 100 * c / n if n else 0
    print(f"  {cat:<32} {c:4d}/{n:<4d} (%{acc:5.1f})")
print(f"  {'TOPLAM':<32} {total_correct:4d}/{total_n:<4d} (%{100*total_correct/max(total_n,1):5.1f})  [{time.time()-t0:.1f}s]")


# ──────────────────────────────────────────────────────────────────────────
# 8) gensim ile kıyaslama — endüstri standardı kütüphaneye karşı
# ──────────────────────────────────────────────────────────────────────────
print("\n── Gensim Word2Vec ile Kıyaslama ──")
from gensim.models import Word2Vec as GensimWord2Vec

t0 = time.time()
GENSIM_CHUNK = 1000
sentences = [tokens[i:i + GENSIM_CHUNK] for i in range(0, len(tokens), GENSIM_CHUNK)]
gensim_model = GensimWord2Vec(
    sentences=sentences,
    vector_size=EMB_DIM,
    window=WINDOW_SIZE,
    min_count=MIN_COUNT,
    sg=1,
    negative=K_NEGATIVE,
    sample=SUBSAMPLE_T,
    epochs=EPOCHS,
    workers=1,
    seed=SEED,
)
print(f"Gensim eğitim süresi: {time.time()-t0:.0f}s | vocab: {len(gensim_model.wv):,}")

gensim_vectors = np.zeros((VOCAB_SIZE, EMB_DIM), dtype=np.float32)
gensim_word2id = {}
for i, w in enumerate(vocab_words):
    if w in gensim_model.wv:
        gensim_vectors[i] = gensim_model.wv[w]
        gensim_word2id[w] = i
gensim_norm = gensim_vectors / (np.linalg.norm(gensim_vectors, axis=1, keepdims=True) + 1e-9)

gensim_results = evaluate_analogies(gensim_norm, gensim_word2id, categories)
gensim_correct = sum(c for c, n in gensim_results.values())
gensim_n = sum(n for c, n in gensim_results.values())
print(f"Gensim analoji doğruluğu: {gensim_correct}/{gensim_n} (%{100*gensim_correct/max(gensim_n,1):.1f})")
print(f"Bizim modelin analoji doğruluğu: {total_correct}/{total_n} (%{100*total_correct/max(total_n,1):.1f})")


# ──────────────────────────────────────────────────────────────────────────
# 9) Görseller
# ──────────────────────────────────────────────────────────────────────────
cats_sorted = sorted(custom_results.keys())
our_acc = [100 * custom_results[c][0] / max(custom_results[c][1], 1) for c in cats_sorted]
gs_acc = [100 * gensim_results[c][0] / max(gensim_results[c][1], 1) for c in cats_sorted]

fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(cats_sorted))
width = 0.38
ax.bar(x - width / 2, our_acc, width, label="Bizim Model (sıfırdan)", color="#534AB7")
ax.bar(x + width / 2, gs_acc, width, label="Gensim Word2Vec", color="#1D9E75")
ax.set_xticks(x)
ax.set_xticklabels(cats_sorted, rotation=40, ha="right")
ax.set_ylabel("Doğruluk (%)")
ax.set_title("Analoji Görevi — Kategori Bazlı Doğruluk Karşılaştırması (Oyun Korpusu)")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "02_analogy_accuracy_comparison.png"), dpi=130)
plt.close(fig)

# t-SNE görselleştirmesi — oyun dunyasina ozgu kelime gruplari
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
            plot_words.append(w)
            plot_labels.append(group)

plot_vecs = np.array([norm_vectors[word2id[w]] for w in plot_words])
tsne = TSNE(n_components=2, perplexity=15, random_state=SEED, init="pca", max_iter=1500)
coords = tsne.fit_transform(plot_vecs)

fig, ax = plt.subplots(figsize=(11, 8.5))
palette = ["#534AB7", "#E24B4A", "#1D9E75", "#BA7517", "#3178C6", "#C2338B"]
for i, group in enumerate(tsne_groups.keys()):
    mask = [j for j, l in enumerate(plot_labels) if l == group]
    ax.scatter(coords[mask, 0], coords[mask, 1], color=palette[i % len(palette)], label=group, s=60)
    for j in mask:
        ax.annotate(plot_words[j], (coords[j, 0], coords[j, 1]), fontsize=9, alpha=0.85,
                     xytext=(4, 4), textcoords="offset points")
ax.set_title("t-SNE ile Kelime Vektörü Görselleştirmesi (Oyun Domaini Kategorilerine Göre Renklendirilmiş)")
ax.legend(loc="best")
ax.set_xticks([]); ax.set_yticks([])
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "03_tsne_word_clusters.png"), dpi=130)
plt.close(fig)

fig, axes = plt.subplots(2, 4, figsize=(16, 7))
for ax, w in zip(axes.flat, query_words):
    nn_list = neighbor_results.get(w)
    if not nn_list:
        ax.axis("off")
        continue
    names = [n for n, s in nn_list[:6]][::-1]
    scores = [s for n, s in nn_list[:6]][::-1]
    ax.barh(names, scores, color="#534AB7")
    ax.set_title(f'"{w}"', fontsize=12)
    ax.set_xlim(0, 1)
fig.suptitle("En Yakın Komşu Kelimeler (kosinüs benzerliği) — Oyun Korpusu", fontsize=14)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "04_nearest_neighbors.png"), dpi=130)
plt.close(fig)

print(f"\nGörseller '{FIG_DIR}/' klasörüne kaydedildi.")


# ──────────────────────────────────────────────────────────────────────────
# 10) Modeli ve kelime dağarcığını kaydet
# ──────────────────────────────────────────────────────────────────────────
torch.save({
    "in_embed": model.in_embed.weight.detach().cpu(),
    "out_embed": model.out_embed.weight.detach().cpu(),
    "word2id": word2id,
    "id2word": id2word,
    "emb_dim": EMB_DIM,
}, os.path.join(DATA_DIR, "word2vec_model.pt"))

print("\n── Özet ──")
print(f"Vocab: {VOCAB_SIZE:,} | Eğitim çifti: {len(center_ids):,}")
print(f"Bizim model analoji doğruluğu : %{100*total_correct/max(total_n,1):.1f}")
print(f"Gensim analoji doğruluğu       : %{100*gensim_correct/max(gensim_n,1):.1f}")
print("Tamamlandı.")
