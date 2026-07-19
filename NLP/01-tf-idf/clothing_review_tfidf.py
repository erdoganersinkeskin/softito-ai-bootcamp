"""
clothing_review_tfidf.py  (Oyun Versiyonu)
===========================================
Oyun yorumlarından "oyuncu bu oyunu tavsiye eder mi?" (Recommended IND: 1/0)
tahmini yapan, uctan uca tek dosyalik bir metin siniflandirma projesi.

ÖĞRENME NOTU (yönlendirme):
  Bu dosyanın amacı, metin temizliği -> EDA -> TF-IDF+LogReg ->
  değerlendirme -> SVD/LSA boyut indirgeme takas analizi işlem hattını
  uçtan uca kurmaktır.

  VERI SETI NOTU: Paylasilan 9 Kaggle veri setinden hicbiri HAM OYUN YORUMU
  METNI + "tavsiye eder mi" etiketini BIRLIKTE icermiyor (bkz.
  MachineLearning/Supervised/08-naive-bayes README'sindeki ayni tespit).
  Bu yuzden gorev tanimindaki istisna uygulanarak, Title, Review Text,
  Recommended IND sütun yapısını taklit eden, kasıtlı olarak GERÇEKÇİ
  "kirlilik" iceren (eksik baslik, mukerrer yorum, degisken uzunluk) daha
  buyuk olcekli SENTETIK bir oyun yorumu veri seti uretiliyor - boylece
  temizlik/EDA adimlari anlamini koruyor.

Akis:
  1) Kütüphaneler
  2) Sentetik oyun yorumu veri seti üretimi
  3) Veri temizleme (eksik değerler, mükerrerler)
  4) Keşifsel analiz ve görselleştirme
  5) TF-IDF vektörleştirme + Logistic Regression
  6) Değerlendirme (metrikler, confusion matrix, en etkili kelimeler)
  7) SVD (LSA) ile boyut indirgeme ve performans takası analizi
  8) Örnek tahmin
"""

# ======================================================================
# 1) KÜTÜPHANELER
# ======================================================================
import os
import re
import random

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix,
)
import joblib

# --- Sabitler -----------------------------------------------------------
DATA_DIR = "data"
CSV_PATH = os.path.join(DATA_DIR, "game_reviews.csv")
FIG_DIR = "figures"
MODEL_PATH = os.path.join("models", "model.joblib")

TITLE_COL = "Title"
BODY_COL = "Review Text"
TARGET_COL = "Recommended IND"

RANDOM_STATE = 42
random.seed(RANDOM_STATE)
plt.rcParams["figure.dpi"] = 120


# ======================================================================
# 2) SENTETIK OYUN YORUMU VERI SETI URETIMI
# ======================================================================
GAMES = ["Nova Strike", "Shadow Realm", "Pixel Kingdom", "Iron Frontier",
         "Mystic Trails", "Velocity Rush", "Ember Legacy", "Storm Tactics",
         "Lunar Outpost", "Crimson Order"]

POS_SENTENCES = [
    "Grafikleri gercekten cok basarili ve akici calisiyor.",
    "Oynanis mekanikleri son derece tatmin edici.",
    "Hikaye anlatimi surukleyici, elimden birakamadim.",
    "Cok saat oynadim ve hic sikilmadim.",
    "Fiyatina gore inanilmaz derecede kaliteli bir yapim.",
    "Multiplayer deneyimi cok dengeli ve eglenceli.",
    "Muzikleri ve ses tasarimi harika atmosfer yaratiyor.",
    "Guncellemelerle surekli gelistiriliyor, gelistiriciler ilgili.",
    "Kontroller cok akici, hicbir sorun yasamadim.",
    "Bu fiyata bu kalitede bir oyun bulmak zor, kesinlikle tavsiye ederim.",
    "Boss savaslari cok yaratici tasarlanmis.",
    "Acik dunya kesif duygusu gercekten tatmin edici.",
]

NEG_SENTENCES = [
    "Optimizasyon berbat, surekli donuyor ve dusuyor.",
    "Bug'lar oyunu oynanamaz hale getiriyor.",
    "Hikaye cok siradan ve tahmin edilebilir.",
    "Pay-to-win mekanikleri dengeyi tamamen bozuyor.",
    "Sunucular surekli cevrimdisi, coklu oyuncu neredeyse imkansiz.",
    "Fiyatina gore icerik cok az, hayal kirikligina ugradim.",
    "Kontroller cok sert ve tepkisiz hissettiriyor.",
    "Destek ekibi hicbir sikayete yanit vermiyor.",
    "Reklamlarda vaat edilenle oyun ici deneyim cok farkli.",
    "Ilk saatlerden sonra tekrara dusuyor, sikici oluyor.",
    "Cok fazla cokme (crash) yasadim, kayitlarim bile silindi.",
    "Bu paraya degmez, pisman oldum.",
]

NEUTRAL_FILLERS = [
    "Oyunu birkac hafta once satin aldim.",
    "Arkadaslarimin tavsiyesiyle denedim.",
    "Steam indiriminde uygun fiyata aldim.",
    "PC'de orta ayarlarda oynuyorum.",
    "Yaklasik 20 saattir oynuyorum.",
    "Bu turden oyunlari severim.",
]


def make_review_text(is_recommended):
    pool = POS_SENTENCES if is_recommended else NEG_SENTENCES
    n_main = random.randint(2, 4)
    sentences = random.sample(pool, min(n_main, len(pool)))
    # gercekci "karma" gorus icin dusuk olasilikla karsi kutuptan bir cumle ekle
    if random.random() < 0.12:
        other_pool = NEG_SENTENCES if is_recommended else POS_SENTENCES
        sentences.append(random.choice(other_pool))
    if random.random() < 0.4:
        sentences.insert(0, random.choice(NEUTRAL_FILLERS))
    random.shuffle(sentences)
    return " ".join(sentences)


def make_title(is_recommended):
    if is_recommended:
        options = ["Harika bir deneyim", "Kesinlikle tavsiye ederim",
                    "Paranin karsiligini fazlasiyla veriyor", "Cok begendim",
                    "Suruklendim gitti"]
    else:
        options = ["Hayal kirikligi", "Tavsiye etmiyorum", "Pisman oldum",
                    "Beklentimi karsilamadi", "Sorunlu bir yapim"]
    return random.choice(options)


def generate_dataset(n_rows=23000, positive_rate=0.78):
    os.makedirs(DATA_DIR, exist_ok=True)
    records = []
    for _ in range(n_rows):
        is_rec = 1 if random.random() < positive_rate else 0
        game = random.choice(GAMES)
        title = make_title(is_rec) if random.random() > 0.08 else None  # ~%8 eksik baslik
        body = make_review_text(is_rec)
        records.append({
            "Clothing ID": game,   # sema tutarliligi icin kolon adi boyle birakildi
            TITLE_COL: title,
            BODY_COL: body,
            TARGET_COL: is_rec,
        })

    df = pd.DataFrame(records)

    # gercekci mukerrerlik: bazi yorumlari bilerek kopyala ("mukerrer/bos
    # atilan" temizlik adimini anlamli kilmak icin)
    dup_sample = df.sample(frac=0.03, random_state=RANDOM_STATE)
    df = pd.concat([df, dup_sample], ignore_index=True)

    # gercekci bos yorum: birkac satirin govde metnini bosalt
    empty_idx = df.sample(frac=0.01, random_state=RANDOM_STATE).index
    df.loc[empty_idx, BODY_COL] = np.nan

    df = df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    df.to_csv(CSV_PATH, index=False)
    print(f"[veri] Sentetik oyun yorumu veri seti uretildi -> {CSV_PATH} ({len(df)} satir)")


def ensure_dataset():
    if os.path.exists(CSV_PATH):
        print(f"[veri] Mevcut: {CSV_PATH}")
        return
    generate_dataset()


# ======================================================================
# 3) VERİ TEMİZLEME
# ======================================================================
_ws = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Tek metni normalize eder: küçük harf + fazla boşlukları sadeleştirme."""
    if not isinstance(text, str):
        return ""
    return _ws.sub(" ", text.lower().strip())


def load_and_clean():
    """Ham CSV'yi yükler, temizler; (df, raw_df, rapor) döndürür."""
    raw = pd.read_csv(CSV_PATH)
    report = {"ham_satir": len(raw), "ham_sutun": raw.shape[1],
              "eksik": raw.isnull().sum().to_dict()}

    df = raw.drop(columns=[c for c in raw.columns if str(c).startswith("Unnamed")],
                  errors="ignore").copy()

    before = len(df)
    df = df.dropna(subset=[BODY_COL])                 # yorum metni yoksa sınıflandıramayız
    report["bos_yorum_atilan"] = before - len(df)

    df[TITLE_COL] = df.get(TITLE_COL, "").fillna("")
    df["text"] = (df[TITLE_COL].astype(str) + " " + df[BODY_COL].astype(str)).map(clean_text)

    before = len(df)
    df = df[df["text"].str.len() > 0]
    df = df.drop_duplicates(subset=["text"])
    report["mukerrer_bos_atilan"] = before - len(df)

    df = df.dropna(subset=[TARGET_COL])
    df[TARGET_COL] = df[TARGET_COL].astype(int)
    df["review_len"] = df[BODY_COL].astype(str).str.split().map(len)

    report["temiz_satir"] = len(df)
    report["pozitif_oran"] = float(df[TARGET_COL].mean())
    return df.reset_index(drop=True), raw, report


def print_report(r):
    print("=" * 56)
    print("VERİ TEMİZLEME RAPORU")
    print("=" * 56)
    print(f"Ham veri            : {r['ham_satir']} satır, {r['ham_sutun']} sütun")
    print("Eksik değerler (ham):")
    for c, n in r["eksik"].items():
        if n > 0:
            print(f"   - {c:<25}: {n}")
    print(f"Atılan (boş yorum)  : {r['bos_yorum_atilan']}")
    print(f"Atılan (mükerrer/boş): {r['mukerrer_bos_atilan']}")
    print(f"Temiz veri          : {r['temiz_satir']} satır")
    print(f"Pozitif sınıf oranı : %{r['pozitif_oran']*100:.1f} (dengesiz)")
    print("=" * 56)


# ======================================================================
# 4) KEŞİFSEL ANALİZ / GÖRSELLEŞTİRME
# ======================================================================
def plot_eda(df, raw, report):
    """Sınıf dağılımı, eksik değerler ve yorum uzunluğu görselleri."""
    os.makedirs(FIG_DIR, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    counts = df[TARGET_COL].value_counts().sort_index()
    axes[0].bar(["Tavsiye etmez (0)", "Tavsiye eder (1)"], counts.values,
                color=["#d9534f", "#5cb85c"])
    for i, v in enumerate(counts.values):
        axes[0].text(i, v, f"{v}\n%{100*v/counts.sum():.1f}", ha="center", va="bottom")
    axes[0].set_title("Sınıf Dağılımı (dengesiz)")
    axes[0].set_ylabel("Yorum sayısı")
    axes[0].set_ylim(0, counts.max() * 1.18)

    miss = {k: v for k, v in report["eksik"].items() if v > 0}
    axes[1].barh(list(miss.keys()), list(miss.values()), color="#f0ad4e")
    for i, v in enumerate(miss.values()):
        axes[1].text(v, i, f" {v}", va="center")
    axes[1].set_title("Ham Veride Eksik Değerler")
    axes[1].invert_yaxis()

    for cls, color, lab in [(1, "#5cb85c", "Tavsiye eder"), (0, "#d9534f", "Tavsiye etmez")]:
        axes[2].hist(df.loc[df[TARGET_COL] == cls, "review_len"], bins=40,
                     alpha=0.6, color=color, label=lab)
    axes[2].set_title("Yorum Uzunluğu (kelime)")
    axes[2].set_xlabel("Kelime sayısı")
    axes[2].set_xlim(0, 60)
    axes[2].legend()

    fig.tight_layout()
    out = os.path.join(FIG_DIR, "01_eda.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[görsel] EDA kaydedildi -> {out}")


# ======================================================================
# 5) TF-IDF + MODEL  (yardımcı)
# ======================================================================
def build_tfidf(max_features):
    return TfidfVectorizer(ngram_range=(1, 2),
                           min_df=5, max_features=max_features, sublinear_tf=True)


# ======================================================================
# 6) DEĞERLENDİRME GÖRSELLERİ
# ======================================================================
def plot_confusion(y_true, y_pred):
    os.makedirs(FIG_DIR, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4.6, 4))
    im = ax.imshow(cm, cmap="Blues")
    labs = ["Tavsiye etmez", "Tavsiye eder"]
    ax.set_xticks([0, 1], labels=labs); ax.set_yticks([0, 1], labels=labs)
    ax.set_xlabel("Tahmin"); ax.set_ylabel("Gerçek")
    ax.set_title("Confusion Matrix (TF-IDF + LogReg)")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "02_confusion_matrix.png")
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    print(f"[görsel] Confusion matrix kaydedildi -> {out}")


def plot_top_words(vectorizer, clf, n=15):
    """LogReg katsayılarından en etkili pozitif/negatif kelimeler (yorumlanabilirlik)."""
    os.makedirs(FIG_DIR, exist_ok=True)
    names = np.array(vectorizer.get_feature_names_out())
    coefs = clf.coef_[0]
    top_pos = np.argsort(coefs)[-n:]
    top_neg = np.argsort(coefs)[:n]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].barh(names[top_pos], coefs[top_pos], color="#5cb85c")
    axes[0].set_title("Tavsiyeye en çok iten kelimeler")
    axes[1].barh(names[top_neg], coefs[top_neg], color="#d9534f")
    axes[1].set_title("Tavsiyeden en çok caydıran kelimeler")
    for ax in axes:
        ax.axvline(0, color="gray", lw=0.8)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "03_top_words.png")
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    print(f"[görsel] En etkili kelimeler kaydedildi -> {out}")


# ======================================================================
# 7) SVD ANALİZİ + GÖRSEL
# ======================================================================
def svd_analysis(X_tr, X_te, y_tr, y_te, max_features, components):
    """Farklı SVD bileşen sayıları için doğruluk/F1 ve açıklanan varyansı ölçer."""
    os.makedirs(FIG_DIR, exist_ok=True)
    accs, f1s, evrs = [], [], []
    for k in components:
        pipe = make_pipeline(
            build_tfidf(max_features),
            TruncatedSVD(n_components=k, random_state=RANDOM_STATE),
            LogisticRegression(max_iter=1000, class_weight="balanced"),
        )
        pipe.fit(X_tr, y_tr)
        pred = pipe.predict(X_te)
        accs.append(accuracy_score(y_te, pred))
        f1s.append(f1_score(y_te, pred, average="macro"))
        evrs.append(pipe.named_steps["truncatedsvd"].explained_variance_ratio_.sum())
        print(f"   SVD k={k:<4} -> acc {accs[-1]:.4f} | F1 {f1s[-1]:.4f} | varyans %{100*evrs[-1]:.1f}")

    fig, ax1 = plt.subplots(figsize=(7.5, 4.6))
    ax1.plot(components, accs, "o-", color="#0275d8", label="Accuracy")
    ax1.plot(components, f1s, "s-", color="#5cb85c", label="F1 (macro)")
    ax1.set_xlabel("SVD bileşen sayısı (boyut)")
    ax1.set_ylabel("Skor")
    ax1.legend(loc="lower right")
    ax2 = ax1.twinx()
    ax2.plot(components, [100*e for e in evrs], "^--", color="#f0ad4e", label="Açıklanan varyans %")
    ax2.set_ylabel("Açıklanan varyans (%)", color="#f0ad4e")
    ax1.set_title("SVD: Boyut ↓ ↔ Performans Takası")
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "04_svd_tradeoff.png")
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    print(f"[görsel] SVD takas grafiği kaydedildi -> {out}")
    return accs, f1s, evrs


# ======================================================================
# ANA AKIŞ
# ======================================================================
def main():
    MAX_FEATURES = 20000
    SVD_COMPONENTS = [50, 100, 200, 300, 500]

    ensure_dataset()

    df, raw, report = load_and_clean()
    print_report(report)

    plot_eda(df, raw, report)

    # NOT: pandas >= 3.0'da metin kolonlari varsayilan olarak Arrow destekli
    # string dtype kullaniyor; bu, sklearn'in fancy-indexing yardimcisiyla
    # (train_test_split icinde) uyumsuz oldugundan duz numpy object array'e
    # ceviriyoruz.
    X = df["text"].astype(object).to_numpy()
    y = df[TARGET_COL].to_numpy()
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
    print(f"\nEğitim: {len(X_tr)} | Test: {len(X_te)}")

    model = make_pipeline(
        build_tfidf(MAX_FEATURES),
        LogisticRegression(max_iter=1000, class_weight="balanced"),
    )
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    vec = model.named_steps["tfidfvectorizer"]
    clf = model.named_steps["logisticregression"]
    n_feat = len(vec.get_feature_names_out())

    acc = accuracy_score(y_te, pred)
    f1m = f1_score(y_te, pred, average="macro")
    print("\n" + "=" * 56)
    print("TF-IDF + LOGISTIC REGRESSION")
    print("=" * 56)
    print(f"Öznitelik (kelime) sayısı: {n_feat}")
    print(f"Accuracy : {acc:.4f}   F1 (macro): {f1m:.4f}")
    print(classification_report(y_te, pred,
          target_names=["Tavsiye etmez (0)", "Tavsiye eder (1)"]))
    plot_confusion(y_te, pred)
    plot_top_words(vec, clf)

    print("\n" + "=" * 56)
    print("SVD (LSA) İLE BOYUT İNDİRGEME ANALİZİ")
    print("=" * 56)
    accs, f1s, evrs = svd_analysis(X_tr, X_te, y_tr, y_te, MAX_FEATURES, SVD_COMPONENTS)
    k = SVD_COMPONENTS[len(SVD_COMPONENTS)//2]
    idx = SVD_COMPONENTS.index(k)
    print(f"\nÖrnek: {n_feat} -> {k} boyut (%{100*(1-k/n_feat):.1f} sıkışma) ile "
          f"accuracy {acc:.4f} -> {accs[idx]:.4f}.")
    print("Yorum: Boyut çok büyük oranda düşerken doğrulukta yalnızca küçük bir kayıp var.")

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"\n[model] Kaydedildi -> {MODEL_PATH}")

    print("\nÖRNEK TAHMİNLER:")
    for s in ["bu oyun gercekten harika, grafikleri ve oynanisi cok akici",
              "cok kotu bir deneyimdi, surekli cokuyor ve bug dolu"]:
        p = model.predict([clean_text(s)])[0]
        prob = model.predict_proba([clean_text(s)])[0][1]
        print(f"   '{s[:45]}...' -> {'TAVSIYE EDER' if p else 'TAVSIYE ETMEZ'} (%{prob*100:.1f})")


if __name__ == "__main__":
    main()
