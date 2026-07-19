"""
OYUN YORUMU DUYGU ANALIZI - Multinomial Naive Bayes (3 Sinif)

ÖĞRENME NOTU (yönlendirme):
  Bu dosyanın amacı, TF-IDF + Multinomial Naive Bayes ile 3-sinifli duygu
  analizi yapmak, sinir bulaniligini simule eden etiket gurultusu eklemek
  ve sinif basina en ayirt edici kelimeleri incelemektir.

  VERI SETI NOTU: Paylasilan 9 Kaggle veri setinden hicbiri HAM YORUM
  METNI icermiyor (antonkozyriev/game-recommendations-on-steam bile sadece
  oy/tavsiye META VERISI tutuyor, gercek yorum cumlelerini degil). Bu
  yuzden gorev tanimindaki "uygun veri seti yoksa oyun dunyasindan uygun
  icerik" istisnasi uygulanarak, SENTETIK sablon teknigiyle OYUN YORUMU
  metinleri uretilmistir.
"""
import os
import random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

RANDOM_STATE = 42
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
os.makedirs('figures', exist_ok=True)

print("Oyun Yorumu Duygu Analizi (3 Sinif) - Sentetik Yorum Uretimi")

games = ["bu oyun", "bu strateji oyunu", "bu RPG", "bu FPS", "bu platform oyunu",
         "bu bulmaca oyunu", "bu spor oyunu", "bu simulasyon", "bu battle royale",
         "bu indie oyun"]

positive_templates = [
    "{p} gerçekten harika, saatlerce oynadım ve hiç sıkılmadım.",
    "{p} beklediğimden çok daha akıcı çıktı, kesinlikle tavsiye ederim.",
    "Fiyatına göre mükemmel bir oyun, {p} herkese öneririm.",
    "{p} bug'sız ve optimizasyonu çok iyi, teşekkürler geliştiricilere.",
    "Uzun zamandır böyle iyi bir oyun oynamamıştım, {p} çok başarılı.",
    "{p} tam beklediğim gibiydi, hiç pişman olmadım.",
    "Kaliteli grafikler, sürükleyici hikaye, {p}'a bayıldım.",
    "{p} kullanıcı dostu ve dengeli, paramın karşılığını aldım.",
    "Harika bir deneyimdi, {p} çok kaliteli.",
    "Bu fiyata bu kalitede bir oyun bulmak zor, {p} çok beğendim.",
]

negative_templates = [
    "{p} tam bir hayal kırıklığı, hiç memnun kalmadım.",
    "{p} beklediğimin çok altında, param boşa gitti.",
    "Optimizasyon berbat, {p} sürekli donuyor.",
    "{p} bug dolu ve destek ekibi hiç yanıt vermiyor, çok kötü deneyim.",
    "Hiç tavsiye etmiyorum, {p} paranıza değmez.",
    "{p} mağaza açıklamasından çok farklı çıktı, kandırıldım.",
    "Berbat bir denge, {p} pay-to-win mekanikleriyle dolu.",
    "{p} çok pahalı ve düşük kaliteli, pişman oldum.",
    "Sunucular da kötüydü, {p} de kalitesizdi.",
    "Bu oyunu asla tekrar almam, {p} tam bir zaman kaybı.",
]

# Gercek bir NOTR sinif: ne belirgin pozitif ne belirgin negatif duygu iceren,
# tanimlayici/olgusal veya "ortalama" ifadeler - gercek Steam yorumlarinda
# "karisik" etiketli yorumlarin tipik dili.
neutral_templates = [
    "{p} mağaza sayfasında yazıldığı gibi geldi, standart bir oyun.",
    "Fiyatı ortalama, {p} performansı da ortalama seviyede.",
    "{p} fena değil ama beklediğim kadar iyi de değildi.",
    "Ne çok iyi ne çok kötü, sıradan bir oyun, {p} idare eder.",
    "{p} güzel ama güncellemeler çok geç geliyor, karışık duygular içindeyim.",
    "Bazı yönleri iyi ama {p} genel olarak vasat kaldı.",
    "{p} idare eder, fiyatına göre normal.",
    "İlk saatlerinde {p} beklediğim gibiydi, ne fazla ne eksik.",
    "{p} hakkında henüz kesin bir fikrim yok, oynadıkça göreceğiz.",
    "Ortalama bir oyun, {p}'ı ne öve öve bitiremem ne de şikayetçiyim.",
]

records = []
n_per_class = 700

for _ in range(n_per_class):
    p = random.choice(games)
    text = random.choice(positive_templates).format(p=p)
    records.append((text, "Pozitif"))

for _ in range(n_per_class):
    p = random.choice(games)
    text = random.choice(negative_templates).format(p=p)
    records.append((text, "Negatif"))

for _ in range(n_per_class):
    p = random.choice(games)
    text = random.choice(neutral_templates).format(p=p)
    records.append((text, "Nötr"))

df = pd.DataFrame(records, columns=["review", "sentiment"])

# Gercek yorum etiketleme surecinde siniflar arasi (ozellikle komsu
# sinifllar arasi) belirsizlik/anlasmazlik olur - bunu simule etmek icin
# bilincli bir etiket gurultusu eklenir.
def add_boundary_noise(label, p=0.15):
    if random.random() > p:
        return label
    if label == "Pozitif":
        return random.choice(["Pozitif", "Nötr"])
    if label == "Negatif":
        return random.choice(["Negatif", "Nötr"])
    return random.choice(["Nötr", "Pozitif", "Negatif"])

df["sentiment"] = df["sentiment"].apply(add_boundary_noise)
df = df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
print(f"Toplam yorum sayisi: {len(df)}")
print(f"Sinif dagilimi:\n{df['sentiment'].value_counts().to_string()}")
print(f"\nOrnek yorumlar:")
print(df.sample(6, random_state=RANDOM_STATE).to_string(index=False))

X_train, X_test, y_train, y_test = train_test_split(
    df["review"], df["sentiment"], test_size=0.2, random_state=RANDOM_STATE, stratify=df["sentiment"]
)
print(f"\nEgitim: {len(X_train)} | Test: {len(X_test)}")

print("\nTF-IDF vektorlestirme uygulaniyor...")
vectorizer = TfidfVectorizer(max_features=2000, ngram_range=(1, 2))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
print(f"Ozellik (kelime/n-gram) sayisi: {X_train_tfidf.shape[1]}")

print("\nMultinomial Naive Bayes egitiliyor (3 sinif)...")
model = MultinomialNB(alpha=0.5)
model.fit(X_train_tfidf, y_train)

y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.4f}")
print("\nSiniflandirma Raporu:")
class_labels = ["Negatif", "Nötr", "Pozitif"]
print(classification_report(y_test, y_pred, labels=class_labels))

print("\nConfusion matrix kaydediliyor...")
cm = confusion_matrix(y_test, y_pred, labels=class_labels)
plt.figure(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel("Tahmin")
plt.ylabel("Gercek")
plt.title("Confusion Matrix - Oyun Yorumu Duygu Analizi (3 Sinif)")
plt.tight_layout()
plt.savefig("figures/confusion_matrix.png", dpi=150)
plt.close()

print("En ayirt edici kelimeler hesaplaniyor (her sinif icin)...")
feature_names = np.array(vectorizer.get_feature_names_out())
class_order = list(model.classes_)

top_words_per_class = {}
for i, cls in enumerate(class_order):
    other_idx = [j for j in range(len(class_order)) if j != i]
    other_mean = model.feature_log_prob_[other_idx].mean(axis=0)
    diff = model.feature_log_prob_[i] - other_mean
    top_idx = np.argsort(diff)[-12:][::-1]
    top_words_per_class[cls] = pd.DataFrame({
        "word": feature_names[top_idx],
        "score": diff[top_idx]
    })
    print(f"\nEn ayirt edici 8 kelime/ifade - {cls}:")
    print(top_words_per_class[cls].head(8).to_string(index=False))
    top_words_per_class[cls].to_csv(f"figures/top_words_{cls.lower().replace('ö','o')}.csv", index=False)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
colors = {"Negatif": "#dc2626", "Nötr": "#f59e0b", "Pozitif": "#059669"}
for ax, cls in zip(axes, ["Negatif", "Nötr", "Pozitif"]):
    data = top_words_per_class[cls].sort_values("score")
    ax.barh(data["word"], data["score"], color=colors[cls])
    ax.set_title(f"{cls} Sinifi icin En Ayirt Edici Ifadeler")
    ax.set_xlabel("Ayirt Edicilik Skoru")
plt.tight_layout()
plt.savefig("figures/top_words_by_sentiment.png", dpi=150)
plt.close()

print("\nGorseller kaydedildi: figures/")
print("Tamamlandi.")
