"""
OYUN ICERIK UYGUNLUK RISKI SINIFLANDIRMASI - Decision Tree

ÖĞRENME NOTU (yönlendirme):
  Bu dosyanın amacı, Dusuk/Orta/Yuksek riskini siniflandiran, ACIK VE
  TAKIP EDILEBILIR kurallar ureten bir Decision Tree kurmaktir. Amac
  "neden bu oyun yuksek icerik riskli (yas siniri yuksek)" sorusuna
  aciklanabilir bir cevap vermek - ebeveynler/magaza moderasyon ekipleri
  bu tur aciklanabilir kurallari kullanir.

  Hedef degisken (icerik_riski) veri setindeki gercek "Required Age"
  kolonundan turetilir (0 -> Dusuk, 1-15 -> Orta, 16+ -> Yuksek); bu kolon
  SIZINTI olmasin diye ozelliklerden CIKARILIR, model sadece tur/fiyat/
  begeni/platform gibi dolayli ozelliklerden tahmin yapar.

Kullanilan veri seti (Kaggle): fronkongames/steam-games-dataset
  -> Gercek "Required Age" (yas siniri) kolonu + tur/fiyat/begeni gibi
     dolayli ozellikler icerdigi icin, aciklanabilir bir siniflandirma
     kurali cikarmak isteyen bu egzersize uygun tek veri seti bu.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import kagglehub

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
os.makedirs('figures', exist_ok=True)

print("Oyun Icerik Uygunluk Riski Siniflandirmasi - Veri Hazirligi")

dataset_path = kagglehub.dataset_download("fronkongames/steam-games-dataset")
csv_files = [f for f in os.listdir(dataset_path) if f.lower().endswith(".csv")]
data_path = os.path.join(dataset_path, csv_files[0])

needed_cols = ["Price", "Genres", "Categories", "Achievements", "Required age",
               "Positive", "Negative", "Windows", "Mac", "Linux"]
raw = pd.read_csv(data_path, usecols=lambda c: c in needed_cols)
raw.columns = [c.strip() for c in raw.columns]
raw = raw.rename(columns={"Required age": "Required_Age"})
raw = raw.dropna(subset=["Required_Age", "Genres", "Price"])

raw["Positive"] = pd.to_numeric(raw.get("Positive", 0), errors="coerce").fillna(0)
raw["Negative"] = pd.to_numeric(raw.get("Negative", 0), errors="coerce").fillna(0)
review_count = raw["Positive"] + raw["Negative"]
raw["positive_ratio"] = np.where(review_count > 0, raw["Positive"] / review_count, 0.5)
raw["platform_count"] = raw[["Windows", "Mac", "Linux"]].astype(bool).sum(axis=1)
raw["primary_genre"] = raw["Genres"].astype(str).str.split(",").str[0].str.strip()
raw["achievements"] = pd.to_numeric(raw.get("Achievements", 0), errors="coerce").fillna(0)

def to_risk_level(age):
    if age <= 0:
        return "Dusuk"
    if age < 16:
        return "Orta"
    return "Yuksek"

raw["risk_level"] = raw["Required_Age"].apply(to_risk_level)

df = raw.sample(n=min(6000, len(raw)), random_state=RANDOM_STATE).reset_index(drop=True)
print(f"Kullanilan oyun sayisi: {len(df)}")
print(df["risk_level"].value_counts().to_string())

le_genre = LabelEncoder()
df["genre_enc"] = le_genre.fit_transform(df["primary_genre"])

feature_cols = ["Price", "achievements", "positive_ratio", "platform_count", "genre_enc"]
X = df[feature_cols]
y = df["risk_level"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

print("\nDecision Tree egitiliyor (max_depth=5, yorumlanabilirlik icin sinirli)...")
# class_weight="balanced": "Orta" sinifi %80 pay aldigi icin agirliksiz
# agac her zaman "Orta" tahmin edip minority siniflari (Dusuk/Yuksek) hic
# yakalayamiyordu; dengeleme bu sorunu duzeltir.
model = DecisionTreeClassifier(
    max_depth=5, min_samples_leaf=25, random_state=RANDOM_STATE, class_weight="balanced"
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc:.4f}")
print("\n" + classification_report(y_test, y_pred))

print("\nCikarilan karar kurallari (ilk birkac dal):")
rules_text = export_text(model, feature_names=feature_cols)
print(rules_text[:1200])
with open("figures/decision_rules.txt", "w") as f:
    f.write(rules_text)

print("\nGorseller kaydediliyor...")

labels_order = ["Dusuk", "Orta", "Yuksek"]
cm = confusion_matrix(y_test, y_pred, labels=labels_order)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples',
            xticklabels=labels_order, yticklabels=labels_order)
plt.xlabel("Tahmin")
plt.ylabel("Gercek")
plt.title("Confusion Matrix - Icerik Uygunluk Riski Siniflandirmasi")
plt.tight_layout()
plt.savefig("figures/confusion_matrix.png", dpi=150)
plt.close()

plt.figure(figsize=(20, 10))
plot_tree(model, feature_names=feature_cols, class_names=model.classes_,
          filled=True, rounded=True, fontsize=9, max_depth=3)
plt.title("Karar Agaci - Icerik Uygunluk Riski (ilk 3 seviye)")
plt.tight_layout()
plt.savefig("figures/decision_tree.png", dpi=150)
plt.close()

importance_df = pd.DataFrame({
    "Feature": feature_cols,
    "Importance": model.feature_importances_
}).sort_values("Importance", ascending=False)

plt.figure(figsize=(8, 5))
sns.barplot(data=importance_df, x="Importance", y="Feature",
            hue="Feature", palette="Purples_r", legend=False)
plt.title("Ozellik Onem Duzeyleri (Feature Importance)")
plt.tight_layout()
plt.savefig("figures/feature_importance.png", dpi=150)
plt.close()
importance_df.to_csv("figures/feature_importance.csv", index=False)

print("Kaydedildi: figures/confusion_matrix.png")
print("Kaydedildi: figures/decision_tree.png")
print("Kaydedildi: figures/feature_importance.png")
print("Kaydedildi: figures/decision_rules.txt")

print("\nTamamlandi.")
