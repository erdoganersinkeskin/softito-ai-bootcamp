"""
SUPHELI OYUN YORUMU ANOMALI TESPITI (ETIKETSIZ) - Isolation Forest

ÖĞRENME NOTU (yönlendirme):
  Bu dosyanın amacı, ETIKET KULLANMADAN (unsupervised) anomali tespiti
  yapan bir Isolation Forest kurmaktır.

  Bu proje, Supervised/04-random-forest/fraud_detection_rf.py projesiyle
  (supheli/sahte oyun yorumu tespiti) TEMATIK OLARAK BAGLANTILIDIR - ikisi
  de "dusuk oynama suresiyle birakilan yorum" orintusunu inceler, ama
  Random Forest projesi ETIKETLI (supervised) calisirken, bu proje HICBIR
  ETIKET GORMEDEN (unsupervised) calisir. "is_suspicious" etiketi SADECE
  MODELIN BASARISINI DEGERLENDIRMEK icin kullanilir, egitimde hic
  gorulmez.

Kullanilan veri seti (Kaggle): antonkozyriev/game-recommendations-on-steam
  -> recommendations.csv, gercek kullanici yorumu meta verisi.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve,
    average_precision_score, classification_report, confusion_matrix
)

import kagglehub

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
os.makedirs('figures', exist_ok=True)

print("Supheli Oyun Yorumu Anomali Tespiti (Isolation Forest) - Veri Hazirligi")

dataset_path = kagglehub.dataset_download("antonkozyriev/game-recommendations-on-steam")
csv_files = [f for f in os.listdir(dataset_path) if f.lower().endswith(".csv")]
rec_file = next((f for f in csv_files if "recommendation" in f.lower()), csv_files[0])
data_path = os.path.join(dataset_path, rec_file)

df = pd.read_csv(data_path)
df.columns = [c.strip().lower() for c in df.columns]
df = df.sample(n=min(20000, len(df)), random_state=RANDOM_STATE).reset_index(drop=True)
df = df.dropna(subset=["hours", "is_recommended", "date"])

df["date"] = pd.to_datetime(df["date"], errors="coerce")
df["day_of_week"] = df["date"].dt.dayofweek
df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
df["is_recommended"] = df["is_recommended"].astype(int)
for col in ["helpful", "funny", "hours"]:
    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

# "gercek" supheli etiket - SADECE DEGERLENDIRME icin, model bunu hic gormeyecek
df["is_suspicious_true"] = (df["hours"] < 0.5).astype(int)
print(f"Uretilen yorum sayisi: {len(df)}")
print(f"Gercek supheli orani (SADECE degerlendirme icin, model bunu gormeyecek): %{100*df['is_suspicious_true'].mean():.2f}")

feature_cols = ["helpful", "funny", "is_recommended", "day_of_week", "is_weekend"]
X = df[feature_cols]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

contamination = max(min(df["is_suspicious_true"].mean(), 0.5), 0.01)
print("\nIsolation Forest egitiliyor (ETIKETSIZ - is_suspicious_true sutunu hic kullanilmiyor)...")
iso_forest = IsolationForest(
    n_estimators=200,
    contamination=contamination,   # beklenen anomali orani (is bilgisinden - etiketten degil)
    random_state=RANDOM_STATE,
    n_jobs=-1
)
iso_forest.fit(X_scaled)

anomaly_pred = iso_forest.predict(X_scaled)   # -1: anomali, 1: normal
anomaly_score = -iso_forest.score_samples(X_scaled)  # yuksek skor = daha supheli

df["predicted_anomaly"] = (anomaly_pred == -1).astype(int)
df["anomaly_score"] = anomaly_score

print("\nModel performansi DEGERLENDIRILIYOR (gercek etiketlerle kiyaslanarak)...")
auc = roc_auc_score(df["is_suspicious_true"], df["anomaly_score"])
ap = average_precision_score(df["is_suspicious_true"], df["anomaly_score"])
print(f"ROC-AUC: {auc:.4f}")
print(f"PR-AUC (Average Precision): {ap:.4f}")
print("\nSiniflandirma Raporu (isaretlenen anomaliler vs gercek supheli yorumlar):")
print(classification_report(df["is_suspicious_true"], df["predicted_anomaly"],
                             target_names=["Normal", "Anomali/Supheli"]))

print("\nConfusion matrix kaydediliyor...")
cm = confusion_matrix(df["is_suspicious_true"], df["predicted_anomaly"])
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds',
            xticklabels=["Normal", "Anomali"], yticklabels=["Normal", "Supheli"])
plt.xlabel("Model Tahmini")
plt.ylabel("Gercek Durum")
plt.title("Isolation Forest - Anomali Tespiti Sonucu")
plt.tight_layout()
plt.savefig("figures/confusion_matrix.png", dpi=150)
plt.close()

print("ROC ve Precision-Recall egrileri kaydediliyor...")
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
fpr, tpr, _ = roc_curve(df["is_suspicious_true"], df["anomaly_score"])
axes[0].plot(fpr, tpr, color="#7c3aed", linewidth=2, label=f"AUC={auc:.3f}")
axes[0].plot([0, 1], [0, 1], "--", color="gray")
axes[0].set_xlabel("False Positive Rate")
axes[0].set_ylabel("True Positive Rate")
axes[0].set_title("ROC Egrisi")
axes[0].legend()

prec, rec, _ = precision_recall_curve(df["is_suspicious_true"], df["anomaly_score"])
axes[1].plot(rec, prec, color="#7c3aed", linewidth=2, label=f"AP={ap:.3f}")
axes[1].set_xlabel("Recall")
axes[1].set_ylabel("Precision")
axes[1].set_title("Precision-Recall Egrisi")
axes[1].legend()
plt.tight_layout()
plt.savefig("figures/roc_pr_curves.png", dpi=150)
plt.close()

print("Anomali skoru dagilimi kaydediliyor...")
plt.figure(figsize=(9, 6))
sns.histplot(data=df, x="anomaly_score", hue="is_suspicious_true", bins=60,
             palette={0: "#3b82f6", 1: "#dc2626"}, alpha=0.6,
             element="step", stat="density", common_norm=False)
plt.xlabel("Anomali Skoru (yuksek = daha supheli)")
plt.title("Anomali Skoru Dagilimi (Gercek Supheli vs Normal)")
plt.legend(labels=["Supheli", "Normal"])
plt.tight_layout()
plt.savefig("figures/anomaly_score_distribution.png", dpi=150)
plt.close()

print("PCA ile 2 boyutlu gorsellestirme...")
pca = PCA(n_components=2, random_state=RANDOM_STATE)
X_pca = pca.fit_transform(X_scaled)
plt.figure(figsize=(9, 7))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=df["predicted_anomaly"],
                       cmap="coolwarm", alpha=0.5, s=15)
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
plt.title("Isolation Forest Anomali Tespiti (PCA ile 2 Boyut)")
plt.colorbar(scatter, label="0=Normal, 1=Anomali")
plt.tight_layout()
plt.savefig("figures/anomaly_pca.png", dpi=150)
plt.close()

print("\nGorseller kaydedildi: figures/")
print("Tamamlandi.")
