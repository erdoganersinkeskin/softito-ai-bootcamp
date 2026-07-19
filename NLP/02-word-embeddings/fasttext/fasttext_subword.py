"""
==============================================================================
 FastText — Subword (Karakter n-gram) Tabanlı Kelime Vektörleri (OYUN VERSİYONU)
==============================================================================

ÖĞRENME NOTU (yönlendirme):
  Bu dosyanın amacı, karakter n-gram (subword) tabanlı FastText yöntemini
  (word2vec/glove'un aksine, bu dosyada sıfırdan degil) egitip, sözlük
  dışı (OOV) kelimeler için de vektör üretebilen bir model kurmaktır.

  KUTUPHANE NOTU: Facebook'un resmi `fasttext` C++/pybind11 paketi,
  2023'ten beri guncellenmedigi icin Python 3.13 + guncel MSVC ile
  DERLENEMIYOR (ssize_t/pybind11 uyumsuzlugu). Bu yuzden ayni skip-gram +
  subword-ngram FastText algoritmasini uygulayan, aktif olarak
  gelistirilen `gensim.models.FastText` kullaniliyor - matematiksel
  yontem ve OOV vektor uretme yetenegi birebir aynidir, sadece C++
  bagimliligi Python-native bir implementasyonla degistirilmistir.

  VERI SETI NOTU: word2vec/glove projeleriyle AYNI gerekce -
  `fronkongames/steam-games-dataset` katalogundaki "About the game"
  aciklamalari birlestirilerek olusturulan GERCEK oyun-domaini metin
  korpusu kullaniliyor (uc proje ile AYNI korpus, adil uclu kiyas icin).

Veri seti: Kaggle fronkongames/steam-games-dataset — diger iki embedding
projesiyle AYNI ~12M karakterlik "About the game" alt kumesi.
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
from sklearn.manifold import TSNE
from collections import Counter

import kagglehub
from gensim.models import FastText

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
CORPUS_PATH = os.path.join(DATA_DIR, "game_corpus_full.txt")
SUBSET_PATH = os.path.join(DATA_DIR, "game_corpus_subset.txt")
QUESTIONS_PATH = os.path.join(DATA_DIR, "questions-words.txt")

SUBSET_CHARS = 12_000_000     # diğer iki embedding projesiyle AYNI alt küme
MIN_COUNT = 5
WINDOW_SIZE = 5
EMB_DIM = 100
MIN_N = 3                     # en kısa karakter n-gram
MAX_N = 5                     # en uzun karakter n-gram
K_NEGATIVE = 5
EPOCHS = 5

N_ANALOGY_PER_CATEGORY = 150

print(f"CPU çekirdek: {torch.get_num_threads()}")


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
    import urllib.request
    urllib.request.urlretrieve(QUESTIONS_URL, QUESTIONS_PATH)


build_game_corpus()
download_questions()

with open(CORPUS_PATH) as f:
    raw_text = f.read(SUBSET_CHARS)
with open(SUBSET_PATH, "w") as f:
    f.write(raw_text)
tokens = raw_text.split()
print(f"\nAlt küme boyutu: {len(raw_text):,} karakter → {len(tokens):,} kelime (token)")


# ──────────────────────────────────────────────────────────────────────────
# 2) FastText'in ÇEKİRDEK FİKRİ — subword (karakter n-gram) çıkarımı (sıfırdan)
# ──────────────────────────────────────────────────────────────────────────
def get_subwords(word, nmin=MIN_N, nmax=MAX_N):
    w = "<" + word + ">"
    ngrams = []
    for n in range(nmin, nmax + 1):
        for i in range(len(w) - n + 1):
            ngrams.append(w[i:i + n])
    return ngrams


print("\n── FastText Çekirdek Fikri: Subword Çıkarımı (örnek) ──")
for demo_word in ["multiplayer", "roguelike", "speedrunning"]:
    sw = get_subwords(demo_word)
    print(f"  '{demo_word}' → {sw[:8]}{' ...' if len(sw) > 8 else ''}  (toplam {len(sw)} subword)")


# ──────────────────────────────────────────────────────────────────────────
# 3) gensim FastText ile eğitim (skip-gram + subword) — bkz. dosya başındaki
#    KUTUPHANE NOTU: resmi fasttext C++ paketi bu ortamda derlenemiyor.
# ──────────────────────────────────────────────────────────────────────────
print("\n── FastText Modeli Eğitiliyor (gensim, skip-gram + subword) ──")
# gensim cümle listesi bekler; token akışını sabit uzunlukta pasajlara bölüyoruz
# (word2vec/glove projelerindeki gibi cümle sınırı olmayan sürekli bir korpus).
SENTENCE_LEN = 20
sentences = [tokens[i:i + SENTENCE_LEN] for i in range(0, len(tokens), SENTENCE_LEN)]

t0 = time.time()
model = FastText(
    sentences=sentences,
    vector_size=EMB_DIM,
    window=WINDOW_SIZE,
    min_count=MIN_COUNT,
    min_n=MIN_N,
    max_n=MAX_N,
    negative=K_NEGATIVE,
    epochs=EPOCHS,
    sg=1,             # skip-gram (CBOW degil)
    workers=1,
    seed=SEED,
)
vocab_words = model.wv.index_to_key
print(f"Eğitim süresi: {time.time()-t0:.0f}s | vocab: {len(vocab_words):,}")

word2id = {w: i for i, w in enumerate(vocab_words)}
id2word = {i: w for w, i in word2id.items()}
VOCAB_SIZE = len(vocab_words)

word_vectors = np.array([model.wv[w] for w in vocab_words])
norm_vectors = word_vectors / (np.linalg.norm(word_vectors, axis=1, keepdims=True) + 1e-9)


def get_word_vector(w):
    """fasttext.get_word_vector(w) yerine: gensim de subword'lerden OOV vektoru uretir."""
    return model.wv[w]


def get_nearest_neighbors(w, k=6):
    """fasttext.get_nearest_neighbors(w) ile ayni imza: (skor, kelime) ciftleri dondurur.
    gensim OOV kelimeler icin de calisir (subword bucket'lari sayesinde)."""
    return [(score, word) for word, score in model.wv.most_similar(w, topn=k)]


# ──────────────────────────────────────────────────────────────────────────
# 4) En yakın komşu sorguları
# ──────────────────────────────────────────────────────────────────────────
query_words = ["battle", "sword", "dragon", "space", "zombie", "level", "quest", "boss"]
print("\n── En Yakın Komşular ──")
neighbor_results = {}
for w in query_words:
    neighbors = get_nearest_neighbors(w, k=6)
    neighbor_results[w] = [(word, score) for score, word in neighbors]
    print(f"  {w:<12} → " + ", ".join(f"{word}({score:.2f})" for score, word in neighbors))


# ──────────────────────────────────────────────────────────────────────────
# 5) FastText'in İMZA ÖZELLİĞİ — OOV (görülmemiş kelime) vektörü üretebilme
# ──────────────────────────────────────────────────────────────────────────
# Oyun endustrisi jargonunda siklikla uydurulan/turetilmis kelimeler
# (bazilari vocab'da olmayabilir) - FastText subword'lerden anlamli
# komsular bulabiliyor mu?
print("\n── OOV (Görülmemiş Kelime) Testi — FastText'in İmza Özelliği ──")
oov_words = ["roguelikeish", "unplayable", "resurvivalize", "overpowered"]
oov_results = {}
for w in oov_words:
    in_vocab = w in word2id
    neighbors = get_nearest_neighbors(w, k=5)
    oov_results[w] = [(word, score) for score, word in neighbors]
    tag = "(vocab'da)" if in_vocab else "(vocab DIŞI - sadece subword'lerden!)"
    print(f"  '{w}' {tag}")
    print(f"      → " + ", ".join(f"{word}({score:.2f})" for score, word in neighbors))


# ──────────────────────────────────────────────────────────────────────────
# 6) Analoji görevi
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
ft_results = evaluate_analogies(norm_vectors, word2id, categories)
total_correct = sum(c for c, n in ft_results.values())
total_n = sum(n for c, n in ft_results.values())
for cat, (c, n) in ft_results.items():
    acc = 100 * c / n if n else 0
    print(f"  {cat:<32} {c:4d}/{n:<4d} (%{acc:5.1f})")
print(f"  {'TOPLAM':<32} {total_correct:4d}/{total_n:<4d} (%{100*total_correct/max(total_n,1):5.1f})  [{time.time()-t0:.1f}s]")


# ──────────────────────────────────────────────────────────────────────────
# 7) Word2Vec ve GloVe projeleriyle kıyaslama (üçlü karşılaştırma)
# ──────────────────────────────────────────────────────────────────────────
def eval_checkpoint(path, key_in):
    if not os.path.exists(path):
        return None
    ckpt = torch.load(path, weights_only=False)
    if key_in == "sum":
        vecs = (ckpt["w"] + ckpt["wc"]).numpy()
    else:
        vecs = ckpt["in_embed"].numpy()
    w2id = ckpt["word2id"]
    norm = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9)
    res = evaluate_analogies(norm, w2id, categories)
    return res


print("\n── Üçlü Karşılaştırma (aynı korpus) ──")
w2v_res = eval_checkpoint("reference_word2vec_model.pt", "in_embed")
glove_res = eval_checkpoint("reference_glove_model.pt", "sum")

comparison = {"FastText": ft_results}
if w2v_res:
    comparison["Word2Vec"] = w2v_res
if glove_res:
    comparison["GloVe"] = glove_res

for name, res in comparison.items():
    tc = sum(c for c, n in res.values()); tn = sum(n for c, n in res.values())
    print(f"  {name:<10} analoji doğruluğu: {tc}/{tn} (%{100*tc/max(tn,1):.1f})")


# ──────────────────────────────────────────────────────────────────────────
# 8) Görseller
# ──────────────────────────────────────────────────────────────────────────
cats_sorted = sorted(ft_results.keys())
fig, ax = plt.subplots(figsize=(13, 6))
x = np.arange(len(cats_sorted))
colors = {"FastText": "#E24B4A", "Word2Vec": "#534AB7", "GloVe": "#1D9E75"}
n_models = len(comparison)
width = 0.8 / n_models
for mi, (name, res) in enumerate(comparison.items()):
    acc = [100 * res[c][0] / max(res[c][1], 1) for c in cats_sorted]
    ax.bar(x + (mi - (n_models - 1) / 2) * width, acc, width, label=name, color=colors.get(name, "#888"))
ax.set_xticks(x); ax.set_xticklabels(cats_sorted, rotation=40, ha="right")
ax.set_ylabel("Doğruluk (%)")
ax.set_title("Analoji Görevi — FastText vs Word2Vec vs GloVe (Oyun Korpusu)")
ax.legend()
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
ax.set_title("t-SNE ile FastText Kelime Vektörü Görselleştirmesi (Oyun Korpusu)")
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
    ax.barh(names, scores, color="#E24B4A")
    ax.set_title(f'"{w}"', fontsize=12)
    ax.set_xlim(0, 1)
fig.suptitle("En Yakın Komşu Kelimeler (kosinüs benzerliği) — FastText (Oyun Korpusu)", fontsize=14)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "04_nearest_neighbors.png"), dpi=130)
plt.close(fig)

fig, axes = plt.subplots(1, len(oov_words), figsize=(16, 4.5))
for ax, w in zip(axes, oov_words):
    nn_list = oov_results.get(w, [])
    names = [n for n, s in nn_list[:5]][::-1]
    scores = [s for n, s in nn_list[:5]][::-1]
    ax.barh(names, scores, color="#BA7517")
    ax.set_title(f'OOV: "{w}"', fontsize=11)
    ax.set_xlim(0, 1)
fig.suptitle("Görülmemiş (OOV) Oyun Jargonu İçin Bile Anlamlı Komşular — FastText'in Farkı", fontsize=13)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "01_oov_capability.png"), dpi=130)
plt.close(fig)

print(f"\nGörseller '{FIG_DIR}/' klasörüne kaydedildi.")

# ──────────────────────────────────────────────────────────────────────────
# 9) Modeli kaydet
# ──────────────────────────────────────────────────────────────────────────
model.save(os.path.join(DATA_DIR, "fasttext_model.bin"))

print("\n── Özet ──")
print(f"Vocab: {VOCAB_SIZE:,}")
print(f"FastText analoji doğruluğu: %{100*total_correct/max(total_n,1):.1f}")
print("FastText, OOV kelimelere bile subword'lerden vektör üretebilir (imza özellik).")
print("Tamamlandı.")
