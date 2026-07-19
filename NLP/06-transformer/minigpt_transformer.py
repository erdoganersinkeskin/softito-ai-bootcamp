"""
==============================================================================
 Self-Attention / Transformer — Karakter Düzeyinde Mini-GPT (OYUN VERSİYONU)
==============================================================================

ÖĞRENME NOTU (yönlendirme):
  Bu dosyanın amacı, karakter düzeyinde decoder-only bir Transformer/mini-GPT
  mimarisini (multi-head self-attention, positional embedding, causal mask,
  feed-forward, residual, layer norm) sıfırdan PyTorch ile inşa edip bir
  oyun metni korpusu üzerinde eğitmektir.

  VERI SETI NOTU: word-embeddings projeleriyle AYNI gerekce -
  `fronkongames/steam-games-dataset` katalogundaki "About the game"
  aciklamalari, Tiny Shakespeare'in yerini alan karakter-duzeyinde
  egitim korpusu olarak kullaniliyor. ÖNEMLİ FARK: word2vec/glove/
  fasttext projelerindeki korpus kelime-duzeyinde oldugu icin agresif
  temizlenip (kucuk harf, noktalama yok) tek kelimelere ayrilmisti; bu
  projede ise KARAKTER duzeyinde ogrenim yapildigindan buyuk/kucuk harf
  ve noktalama BILEREK KORUNUYOR (Shakespeare korpusunun da orijinal
  noktalama/karakter cesitliligini korudugu gibi) - aksi halde modelin
  ogrenecegi karakter kumesi anlamsizlasirdi.

  Egitilen model, YENI "Steam oyun aciklamasi" tarzi metin uretir -
  Shakespeare yerine "AI tarafindan yazilmis oyun tanitim metni".

Veri seti: Kaggle fronkongames/steam-games-dataset — "About the game"
metinlerinin birlestirilmesiyle olusturulan karakter-duzeyinde korpus.
==============================================================================
"""

import os
import re
import time

import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

import kagglehub

# ──────────────────────────────────────────────────────────────────────────
# 0) Sabitler / Hiperparametreler
# ──────────────────────────────────────────────────────────────────────────
SEED = 42
torch.manual_seed(SEED)

DATA_DIR = "data"; FIG_DIR = "figures"
os.makedirs(DATA_DIR, exist_ok=True); os.makedirs(FIG_DIR, exist_ok=True)

CORPUS_PATH = os.path.join(DATA_DIR, "game_blurbs.txt")
CORPUS_CHAR_LIMIT = 4_000_000   # Shakespeare'e (~1.1M) kiyasla biraz daha genis, ceside karsi

BLOCK_SIZE = 64        # modelin bir seferde göreceği maksimum bağlam uzunluğu
N_EMBD = 64            # gömme (embedding) boyutu
N_HEAD = 4             # multi-head attention'daki kafa sayısı
N_LAYER = 3            # üst üste dizilen Transformer blok sayısı
DROPOUT = 0.1
BATCH_SIZE = 64
MAX_ITERS = 2500
EVAL_INTERVAL = 250
LR = 3e-4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Cihaz: {device} | CPU çekirdek: {torch.get_num_threads()}")


# ──────────────────────────────────────────────────────────────────────────
# 1) Oyun aciklamasi korpusunu olustur ve karakter duzeyinde tokenize et
# ──────────────────────────────────────────────────────────────────────────
def build_game_blurb_corpus():
    if os.path.exists(CORPUS_PATH):
        print(f"Korpus zaten mevcut: {CORPUS_PATH}"); return
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
    # NOT: Karakter duzeyinde ogrenim icin buyuk/kucuk harf ve noktalama
    # BILEREK korunuyor - sadece HTML etiketleri ve fazla bosluklar temizlenir.
    parts, total_chars = [], 0
    for chunk in pd.read_csv(catalog_path, usecols=[about_col], chunksize=5000):
        for text in chunk[about_col].dropna().astype(str):
            text = re.sub(r"<[^>]+>", " ", text)
            text = re.sub(r"[ \t]+", " ", text).strip()
            if text:
                parts.append(text)
                total_chars += len(text)
        if total_chars > CORPUS_CHAR_LIMIT * 1.5:
            break

    raw_text = "\n".join(parts)
    with open(CORPUS_PATH, "w", encoding="utf-8") as f:
        f.write(raw_text)
    print(f"Korpus oluşturuldu: {len(raw_text):,} karakter -> {CORPUS_PATH}")


build_game_blurb_corpus()
with open(CORPUS_PATH, encoding="utf-8") as f:
    text = f.read(CORPUS_CHAR_LIMIT)
print(f"\nToplam karakter: {len(text):,}")

chars = sorted(set(text))
VOCAB_SIZE = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
print(f"Benzersiz karakter (vocab): {VOCAB_SIZE}")


def encode(s):
    return [stoi[c] for c in s]


def decode(ids):
    return "".join(itos[i] for i in ids)


data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]
print(f"Eğitim: {len(train_data):,} karakter | Doğrulama: {len(val_data):,} karakter")


def get_batch(split):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([d[i:i + BLOCK_SIZE] for i in ix])
    y = torch.stack([d[i + 1:i + BLOCK_SIZE + 1] for i in ix])
    return x.to(device), y.to(device)


# ──────────────────────────────────────────────────────────────────────────
# 2) Model — Decoder-only Transformer (sıfırdan)
# ──────────────────────────────────────────────────────────────────────────
class SelfAttentionHead(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(N_EMBD, head_size, bias=False)
        self.query = nn.Linear(N_EMBD, head_size, bias=False)
        self.value = nn.Linear(N_EMBD, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x); q = self.query(x); v = self.value(x)
        weights = q @ k.transpose(-2, -1) * (C ** -0.5)
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        weights = F.softmax(weights, dim=-1)
        weights = self.dropout(weights)
        return weights @ v


class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, head_size):
        super().__init__()
        self.heads = nn.ModuleList([SelfAttentionHead(head_size) for _ in range(n_head)])
        self.proj = nn.Linear(N_EMBD, N_EMBD)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd), nn.Dropout(DROPOUT),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.sa = MultiHeadAttention(n_head, n_embd // n_head)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class MiniGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(VOCAB_SIZE, N_EMBD)
        self.position_embedding = nn.Embedding(BLOCK_SIZE, N_EMBD)
        self.blocks = nn.Sequential(*[TransformerBlock(N_EMBD, N_HEAD) for _ in range(N_LAYER)])
        self.ln_f = nn.LayerNorm(N_EMBD)
        self.lm_head = nn.Linear(N_EMBD, VOCAB_SIZE)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -BLOCK_SIZE:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx


model = MiniGPT().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
print(f"\nModel parametre sayısı: {sum(p.numel() for p in model.parameters()):,}")


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(20)
        for k in range(20):
            xb, yb = get_batch(split)
            _, loss = model(xb, yb)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


# ──────────────────────────────────────────────────────────────────────────
# 3) Eğitim döngüsü
# ──────────────────────────────────────────────────────────────────────────
print("\n── Eğitim başlıyor ──")
history = {"iter": [], "train_loss": [], "val_loss": []}
t_start = time.time()

for it in range(MAX_ITERS + 1):
    if it % EVAL_INTERVAL == 0 or it == MAX_ITERS:
        losses = estimate_loss()
        history["iter"].append(it)
        history["train_loss"].append(losses["train"])
        history["val_loss"].append(losses["val"])
        elapsed = time.time() - t_start
        print(f"Adım {it:5d}/{MAX_ITERS} | train_loss={losses['train']:.4f} | "
              f"val_loss={losses['val']:.4f} | {elapsed:.0f}s")

    xb, yb = get_batch("train")
    _, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print(f"Eğitim tamamlandı: {time.time()-t_start:.0f}s")

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(history["iter"], history["train_loss"], label="Eğitim", color="#534AB7")
ax.plot(history["iter"], history["val_loss"], label="Doğrulama", color="#E24B4A")
ax.set_xlabel("Adım"); ax.set_ylabel("Cross-Entropy Kaybı")
ax.set_title("Mini-GPT Eğitim Kaybı (Karakter Düzeyinde Tahmin, Oyun Açıklamaları)")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "01_training_loss.png"), dpi=130)
plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────
# 4) Metin Üretimi — Eğitilen Modelle Yeni "Oyun Açıklaması" Yaz
# ──────────────────────────────────────────────────────────────────────────
print("\n── Üretilen Metin Örnekleri ──")
model.eval()

context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated_ids = model.generate(context, max_new_tokens=500)[0].tolist()
generated_text = decode(generated_ids)
print(generated_text)

prompt = "This game"
prompt_ids = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
continued_ids = model.generate(prompt_ids, max_new_tokens=300)[0].tolist()
continued_text = decode(continued_ids)
print("\n--- Prompt tamamlama ---")
print(continued_text)

with open(os.path.join(DATA_DIR, "generated_samples.txt"), "w", encoding="utf-8") as f:
    f.write("=== Boş bağlamdan üretim ===\n")
    f.write(generated_text)
    f.write("\n\n=== 'This game' prompt'undan devam ===\n")
    f.write(continued_text)


# ──────────────────────────────────────────────────────────────────────────
# 5) Attention Görselleştirmesi — Bir Kafa Neye Bakıyor?
# ──────────────────────────────────────────────────────────────────────────
sample_text = "This game is a"
sample_ids = torch.tensor([encode(sample_text)], dtype=torch.long, device=device)

with torch.no_grad():
    x = model.token_embedding(sample_ids) + model.position_embedding(
        torch.arange(sample_ids.shape[1], device=device))
    x_norm = model.blocks[0].ln1(x)
    head = model.blocks[0].sa.heads[0]
    k = head.key(x_norm); q = head.query(x_norm)
    T = sample_ids.shape[1]
    weights = (q @ k.transpose(-2, -1)) * (k.shape[-1] ** -0.5)
    weights = weights.masked_fill(head.tril[:T, :T] == 0, float("-inf"))
    weights = F.softmax(weights, dim=-1)[0].numpy()

fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(weights, cmap="Purples")
ax.set_xticks(range(len(sample_text))); ax.set_xticklabels(list(sample_text), fontsize=9)
ax.set_yticks(range(len(sample_text))); ax.set_yticklabels(list(sample_text), fontsize=9)
ax.set_xlabel("Dikkat Edilen Karakter (Key)"); ax.set_ylabel("Sorgulayan Karakter (Query)")
ax.set_title("Self-Attention Matrisi (1. Blok, 1. Kafa)\nAlt üçgen: nedensel (causal) maske")
plt.colorbar(im, ax=ax, fraction=0.046)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "02_self_attention_matrix.png"), dpi=130)
plt.close(fig)

print(f"\nGörseller '{FIG_DIR}/' klasörüne kaydedildi.")

# ──────────────────────────────────────────────────────────────────────────
# 6) Modeli kaydet
# ──────────────────────────────────────────────────────────────────────────
torch.save({
    "model_state": model.state_dict(),
    "stoi": stoi, "itos": itos,
    "block_size": BLOCK_SIZE, "n_embd": N_EMBD, "n_head": N_HEAD, "n_layer": N_LAYER,
}, os.path.join(DATA_DIR, "minigpt_model.pt"))

print("\n── Özet ──")
print(f"Vocab: {VOCAB_SIZE} | Parametre: {sum(p.numel() for p in model.parameters()):,}")
print(f"Son eğitim kaybı: {history['train_loss'][-1]:.4f} | Son doğrulama kaybı: {history['val_loss'][-1]:.4f}")
print("Tamamlandı.")
