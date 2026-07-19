"""
OYUN KRITIK BEGENI TAHMINI (Credit Scoring Analogu) - Logistic Regression

ÖĞRENME NOTU (yönlendirme):
  Bu dosyanın amacı, "kara kutu olmayan" aciklanabilir bir Logistic
  Regression modeli (+ katsayi yorumlama + ROC-AUC) kurmaktır.

  Hedef: "oyun kritik acidan iyi karsilanacak mi" (Positive review orani
  > %70) ikili karari - ozelliklerden bir karari aciklanabilir sekilde
  tahmin etme mantigi izlenir.

Kullanilan veri seti (Kaggle): fronkongames/steam-games-dataset
  -> Gercek Steam katalogu; fiyat, tur, basarim sayisi, gerekli yas,
     platform destegi gibi "basvuru ozelligi" niteliginde kolonlar +
     Positive/Negative oy sayisindan turetilebilen bir "basari" hedefi
     icerir. Bu proje, hem sayisal hem kategorik acikanabilir ozelliklere
     sahip gercek bir katalog gerektirdigi icin secildi.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, roc_curve
)

import kagglehub

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
os.makedirs('figures', exist_ok=True)


# 1) VERI YUKLEME

print("=" * 60)
print("OYUN KRITIK BEGENI TAHMINI - VERI HAZIRLIGI")
print("=" * 60)

dataset_path = kagglehub.dataset_download("fronkongames/steam-games-dataset")
csv_files = [f for f in os.listdir(dataset_path) if f.lower().endswith(".csv")]
data_path = os.path.join(dataset_path, csv_files[0])

needed_cols = ["Price", "Genres", "Categories", "Achievements", "Required age",
               "Windows", "Mac", "Linux", "Positive", "Negative"]
raw = pd.read_csv(data_path, usecols=lambda c: c in needed_cols)
raw.columns = [c.strip() for c in raw.columns]
rename_map = {}
for col in raw.columns:
    lc = col.lower()
    if lc == "required age":
        rename_map[col] = "Required_Age"
    elif lc == "achievements":
        rename_map[col] = "Achievements"
    elif lc == "genres":
        rename_map[col] = "Genres"
raw = raw.rename(columns=rename_map)

raw["Positive"] = pd.to_numeric(raw.get("Positive", 0), errors="coerce").fillna(0)
raw["Negative"] = pd.to_numeric(raw.get("Negative", 0), errors="coerce").fillna(0)
raw["review_count"] = raw["Positive"] + raw["Negative"]
raw = raw[raw["review_count"] >= 10]   # anlamli bir yorum kitlesi olan oyunlar

raw["positive_ratio"] = raw["Positive"] / raw["review_count"]
raw["well_received"] = (raw["positive_ratio"] > 0.70).astype(int)
raw["primary_genre"] = raw["Genres"].astype(str).str.split(",").str[0].str.strip()
raw["platform_count"] = raw[["Windows", "Mac", "Linux"]].astype(bool).sum(axis=1)

df = raw.sample(n=min(6000, len(raw)), random_state=RANDOM_STATE).reset_index(drop=True)
print(f"Kullanilan oyun sayisi: {len(df)}")
print(f"Kritik acidan iyi karsilanma orani: %{100*df['well_received'].mean():.1f}")


# 2) ON ISLEME

print("\n[2] Kategorik degiskenler encode ediliyor...")
le_genre = LabelEncoder()
df["genre_enc"] = le_genre.fit_transform(df["primary_genre"])

feature_cols = ["Price", "Achievements", "Required_Age", "platform_count", "genre_enc"]
X = df[feature_cols]
y = df["well_received"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)


# 3) MODEL EGITIMI

print("\n[3] Logistic Regression egitiliyor...")
# class_weight="balanced": %71 "Iyi Karsilandi" / %29 "Zayif Karsilandi"
# sinif dengesizliginde, dengesiz agirliksiz model azinlik sinifi neredeyse
# hic yakalayamiyordu (recall ~0.01); dengeleme bu sorunu duzeltir.
model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, class_weight="balanced")
model.fit(X_train_s, y_train)
y_pred = model.predict(X_test_s)
y_proba = model.predict_proba(X_test_s)[:, 1]

acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)
print(f"    Accuracy: {acc:.4f}")
print(f"    ROC-AUC : {auc:.4f}")
print("\n" + classification_report(y_test, y_pred, target_names=["Zayif Karsilandi", "Iyi Karsilandi"]))


# 4) GORSELLESTIRME

print("\n[4] Gorseller kaydediliyor...")

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=["Zayif Karsilandi", "Iyi Karsilandi"],
            yticklabels=["Zayif Karsilandi", "Iyi Karsilandi"])
plt.xlabel("Tahmin")
plt.ylabel("Gercek")
plt.title("Confusion Matrix - Kritik Begeni Tahmini")
plt.tight_layout()
plt.savefig("figures/confusion_matrix.png", dpi=150)
plt.close()

fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"ROC (AUC = {auc:.3f})", color="#2563eb", linewidth=2)
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Egrisi")
plt.legend()
plt.tight_layout()
plt.savefig("figures/roc_curve.png", dpi=150)
plt.close()

coef_df = pd.DataFrame({
    "Feature": feature_cols,
    "Coefficient": model.coef_[0]
}).sort_values("Coefficient", key=abs, ascending=False)

plt.figure(figsize=(8, 5))
colors = ["#059669" if c > 0 else "#dc2626" for c in coef_df["Coefficient"]]
sns.barplot(data=coef_df, x="Coefficient", y="Feature", hue="Feature", palette=colors, legend=False)
plt.axvline(0, color="black", linewidth=0.8)
plt.title("Model Katsayilari (Iyi Karsilanma Olasiligina Etki)")
plt.tight_layout()
plt.savefig("figures/coefficients.png", dpi=150)
plt.close()

coef_df.to_csv("figures/coefficients.csv", index=False)

print("    Kaydedildi: figures/confusion_matrix.png")
print("    Kaydedildi: figures/roc_curve.png")
print("    Kaydedildi: figures/coefficients.png")

print("\n" + "=" * 60)
print("TAMAMLANDI")
print("=" * 60)
