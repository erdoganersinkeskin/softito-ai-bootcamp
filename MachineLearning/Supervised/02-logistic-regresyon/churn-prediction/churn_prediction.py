"""
OYUN TAVSIYE ETMEME (CHURN ANALOGU) TAHMINI - Logistic Regression

ÖĞRENME NOTU (yönlendirme):
  Bu dosyanın amacı, standardizasyon + ROC-AUC + katsayı yorumlama içeren
  bir Logistic Regression churn (kayıp) tahmini kurmaktır.

  Hedef gercek bir Steam davranis sinyaline dayanir: "is_recommended" -
  yani bir oyuncu, oynadigi oyunu BASKALARINA TAVSIYE ETMIYOR MU (bu bir
  nevi 'memnuniyetsizlik/churn' sinyalidir - churn=1 -> tavsiye etmedi).
  GERCEK bir veri seti ve GERCEK bir hedef kolon kullaniliyor (sentetik
  formul degil).

Kullanilan veri seti (Kaggle): antonkozyriev/game-recommendations-on-steam
  -> recommendations.csv dosyasi, her satirda bir kullanicinin bir oyun
     icin biraktigi gercek degerlendirme (oynama suresi, faydali/komik oy
     sayisi, tarih, tavsiye edip etmedigi) bulunur. Bu proje bir
     SINIFLANDIRMA (churn benzeri) egzersizi ve zaten hazir bir "is
     recommended" hedef kolonuna sahip tek veri seti bu oldugu icin
     secildi (formulle etiket TURETMEYE gerek kalmadi).
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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
print("OYUN TAVSIYE ETMEME (CHURN ANALOGU) TAHMINI - VERI HAZIRLIGI")
print("=" * 60)

dataset_path = kagglehub.dataset_download("antonkozyriev/game-recommendations-on-steam")
csv_files = [f for f in os.listdir(dataset_path) if f.lower().endswith(".csv")]
rec_file = next((f for f in csv_files if "recommendation" in f.lower()), csv_files[0])
data_path = os.path.join(dataset_path, rec_file)

df = pd.read_csv(data_path)
df.columns = [c.strip().lower() for c in df.columns]

# Buyuk veri setini makul boyuta ornekle
df = df.sample(n=min(60000, len(df)), random_state=RANDOM_STATE).reset_index(drop=True)
df = df.dropna(subset=["hours", "is_recommended"])

df["date"] = pd.to_datetime(df["date"], errors="coerce")
reference_date = df["date"].max()
df["days_since_review"] = (reference_date - df["date"]).dt.days
df["days_since_review"] = df["days_since_review"].fillna(df["days_since_review"].median())

# churn analogu: tavsiye ETMEMEK (memnuniyetsizlik/ayrilma sinyali)
df["churn"] = (~df["is_recommended"].astype(bool)).astype(int)

print(f"Kullanilan degerlendirme sayisi: {len(df)}")
print(f"Churn (tavsiye etmeme) orani: %{100*df['churn'].mean():.1f}")


# 2) ON ISLEME

feature_cols = ["hours", "helpful", "funny", "days_since_review"]
for col in feature_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

X = df[feature_cols]
y = df["churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)


# 3) MODEL EGITIMI

print("\n[3] Logistic Regression egitiliyor...")
# class_weight="balanced": churn orani sadece %14.2 oldugundan, agirliksiz
# model azinlik sinifini (tavsiye etmeyenler) neredeyse hic yakalayamiyordu
# (recall ~0.00); dengeleme bu sorunu duzeltir.
model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, class_weight="balanced")
model.fit(X_train_s, y_train)
y_pred = model.predict(X_test_s)
y_proba = model.predict_proba(X_test_s)[:, 1]

acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)
print(f"    Accuracy: {acc:.4f}")
print(f"    ROC-AUC : {auc:.4f}")
print("\n" + classification_report(y_test, y_pred, target_names=["Tavsiye Etti", "Tavsiye Etmedi"]))


# 4) GORSELLESTIRME

print("\n[4] Gorseller kaydediliyor...")

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges',
            xticklabels=["Tavsiye Etti", "Tavsiye Etmedi"],
            yticklabels=["Tavsiye Etti", "Tavsiye Etmedi"])
plt.xlabel("Tahmin")
plt.ylabel("Gercek")
plt.title("Confusion Matrix - Tavsiye Etmeme Tahmini")
plt.tight_layout()
plt.savefig("figures/confusion_matrix.png", dpi=150)
plt.close()

fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"ROC (AUC = {auc:.3f})", color="#ea580c", linewidth=2)
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
colors = ["#dc2626" if c > 0 else "#059669" for c in coef_df["Coefficient"]]
sns.barplot(data=coef_df, x="Coefficient", y="Feature", hue="Feature", palette=colors, legend=False)
plt.axvline(0, color="black", linewidth=0.8)
plt.title("Model Katsayilari (Tavsiye Etmeme Riskine Etki)")
plt.tight_layout()
plt.savefig("figures/coefficients.png", dpi=150)
plt.close()
coef_df.to_csv("figures/coefficients.csv", index=False)

plt.figure(figsize=(7, 5))
hours_bin = pd.qcut(df["hours"], q=5, duplicates="drop")
churn_by_hours = df.groupby(hours_bin)["churn"].mean() * 100
sns.barplot(x=[str(i) for i in churn_by_hours.index], y=churn_by_hours.values, color="#f97316")
plt.ylabel("Tavsiye Etmeme Orani (%)")
plt.xlabel("Oynama Suresi Dilimi (saat)")
plt.xticks(rotation=20)
plt.title("Oynama Suresine Gore Tavsiye Etmeme Orani")
plt.tight_layout()
plt.savefig("figures/churn_by_contract.png", dpi=150)
plt.close()

print("    Kaydedildi: figures/confusion_matrix.png")
print("    Kaydedildi: figures/roc_curve.png")
print("    Kaydedildi: figures/coefficients.png")
print("    Kaydedildi: figures/churn_by_contract.png")

print("\n" + "=" * 60)
print("TAMAMLANDI")
print("=" * 60)
