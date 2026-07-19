"""
SUPHELI (SAHTE OLASILIKLI) OYUN YORUMU TESPITI - Random Forest

ÖĞRENME NOTU (yönlendirme):
  Bu dosyanın amacı, dengesiz sınıflarla çalışan bir Random Forest
  siniflandirmasi (class_weight="balanced", ROC-AUC, Precision-Recall
  egrisi, feature importance) kurmaktır.

  Burada gercekci bir Steam yorum-kalitesi sinyali hedefleniyor: bir
  kullanicinin OYUNU NEREDEYSE HIC OYNAMADAN (0.5 saatten az) yorum/tavsiye
  birakmasi - bu, gercek platformlarda "supheli/sahte olabilecek yorum"
  olarak isaretlenen tipik bir orintudur (bot hesaplar, odullu sahte
  yorumlar vb.). Bu bir AZINLIK sinif problemidir (supheli yorumlar
  toplam yorumlarin kucuk bir yuzdesidir).

Kullanilan veri seti (Kaggle): antonkozyriev/game-recommendations-on-steam
  -> recommendations.csv, gercek kullanici yorumu meta verisi (oynama
     suresi, faydali/komik oy, tarih, tavsiye durumu) icerir; supheli
     yorum orintusunu (dusuk oynama suresi + yorum davranisi) gercek
     veriden turetebilen tek veri seti bu oldugu icin secildi.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, roc_curve, precision_recall_curve, average_precision_score
)

import kagglehub

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
os.makedirs('figures', exist_ok=True)

print("Supheli Oyun Yorumu Tespiti (Random Forest) - Veri Hazirligi")

dataset_path = kagglehub.dataset_download("antonkozyriev/game-recommendations-on-steam")
csv_files = [f for f in os.listdir(dataset_path) if f.lower().endswith(".csv")]
rec_file = next((f for f in csv_files if "recommendation" in f.lower()), csv_files[0])
data_path = os.path.join(dataset_path, rec_file)

df = pd.read_csv(data_path)
df.columns = [c.strip().lower() for c in df.columns]
df = df.sample(n=min(60000, len(df)), random_state=RANDOM_STATE).reset_index(drop=True)
df = df.dropna(subset=["hours", "is_recommended", "date"])

df["date"] = pd.to_datetime(df["date"], errors="coerce")
df["day_of_week"] = df["date"].dt.dayofweek
df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
df["is_recommended"] = df["is_recommended"].astype(int)
for col in ["helpful", "funny", "hours"]:
    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

# "supheli yorum": oyunu neredeyse hic oynamadan (0.5 saatten az) yorum birakmak
df["is_suspicious"] = (df["hours"] < 0.5).astype(int)
print(f"Uretilen yorum sayisi: {len(df)}")
print(f"Supheli yorum orani: %{100*df['is_suspicious'].mean():.2f}")

feature_cols = ["helpful", "funny", "is_recommended", "day_of_week", "is_weekend"]
X = df[feature_cols]
y = df["is_suspicious"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
)

print("\nRandom Forest egitiliyor (class_weight=balanced, dengesiz siniflar icin)...")
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    min_samples_leaf=5,
    class_weight="balanced",
    random_state=RANDOM_STATE,
    n_jobs=-1
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)
avg_precision = average_precision_score(y_test, y_proba)
print(f"Accuracy: {acc:.4f}")
print(f"ROC-AUC : {auc:.4f}")
print(f"Average Precision (PR-AUC): {avg_precision:.4f}")
print("\n" + classification_report(y_test, y_pred, target_names=["Normal", "Supheli"]))

print("\nGorseller kaydediliyor...")

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds',
            xticklabels=["Normal", "Supheli"], yticklabels=["Normal", "Supheli"])
plt.xlabel("Tahmin")
plt.ylabel("Gercek")
plt.title("Confusion Matrix - Supheli Yorum Tespiti")
plt.tight_layout()
plt.savefig("figures/confusion_matrix.png", dpi=150)
plt.close()

fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"ROC (AUC = {auc:.3f})", color="#dc2626", linewidth=2)
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Egrisi")
plt.legend()
plt.tight_layout()
plt.savefig("figures/roc_curve.png", dpi=150)
plt.close()

precision, recall, _ = precision_recall_curve(y_test, y_proba)
plt.figure(figsize=(6, 5))
plt.plot(recall, precision, color="#991b1b", linewidth=2,
         label=f"PR (Avg Precision = {avg_precision:.3f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Egrisi (dengesiz siniflar icin)")
plt.legend()
plt.tight_layout()
plt.savefig("figures/precision_recall_curve.png", dpi=150)
plt.close()

importance_df = pd.DataFrame({
    "Feature": feature_cols,
    "Importance": model.feature_importances_
}).sort_values("Importance", ascending=False)

plt.figure(figsize=(8, 5))
sns.barplot(data=importance_df, x="Importance", y="Feature",
            hue="Feature", palette="Reds_r", legend=False)
plt.title("Ozellik Onem Duzeyleri (Feature Importance)")
plt.tight_layout()
plt.savefig("figures/feature_importance.png", dpi=150)
plt.close()
importance_df.to_csv("figures/feature_importance.csv", index=False)

print("Kaydedildi: figures/confusion_matrix.png")
print("Kaydedildi: figures/roc_curve.png")
print("Kaydedildi: figures/precision_recall_curve.png")
print("Kaydedildi: figures/feature_importance.png")

print("\nTamamlandi.")
