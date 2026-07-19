"""
OYUN SUNUCUSU GUVENLIGI - BOT/DDOS TRAFIGI TESPITI - One-Class SVM

ÖĞRENME NOTU (yönlendirme):
  Bu dosyanın amacı, SADECE NORMAL trafik örnekleriyle eğitilen bir
  One-Class SVM ile anomali/saldırı tespiti yapmaktır.

  VERI SETI NOTU: Paylasilan 9 Kaggle veri setinden hicbiri AG/SUNUCU
  BAGLANTI TELEMETRISI (paket boyutu, oturum suresi, basarisiz giris
  sayisi vb.) icermiyor - hepsi oyun katalog/satis/degerlendirme verisi.
  Gercek bir ag trafigi veri seti (NSL-KDD, CICIDS) de bu 9 veri setinin
  disinda kaldigi icin, gorev tanimindaki istisna uygulanarak SENTETIK
  veri uretiliyor; baglam "OYUN SUNUCUSU guvenligi / bot-cheat-DDoS
  trafigi tespiti"ne cevrilmistir - ag guvenligi kavramlari (paket,
  oturum, baglanti) zaten dogrudan oyun sunucusu altyapisina
  uygulanabilir.

  Bu proje ayni zamanda oyun-versiyonlari/BigData/big-data-log-analytics
  projesindeki oyun magazasi log analitigiyle TEMATIK OLARAK
  BAGLANTILIDIR - ikisi de oyun platformu sunucu/ag trafigi verisiyle
  ilgilenir, ama bu proje "anomali tespiti" katmanini ekler.

Not: Gercek bir oyun sunucusu ag trafigi veri seti bu ortamda bulunmadigi
     icin, gercekci baglanti oruntularini yansitan SENTETIK bir veri seti
     uretilir. Bot/saldiri etiketleri SADECE DEGERLENDIRME icin kullanilir,
     model egitiminde hic gorulmez.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve,
    average_precision_score, classification_report, confusion_matrix
)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
os.makedirs('figures', exist_ok=True)

print("Oyun Sunucusu Guvenligi - Bot/DDoS Trafigi Tespiti (One-Class SVM) - Veri Uretimi")

n = 12000
attack_rate = 0.08

is_attack_true = np.random.choice([0, 1], n, p=[1 - attack_rate, attack_rate])

duration_sec = np.where(
    is_attack_true == 1,
    np.random.exponential(18, n),
    np.random.exponential(8, n)
)
packet_size_avg = np.where(
    is_attack_true == 1,
    np.clip(np.random.normal(320, 140, n), 20, 1500),
    np.clip(np.random.normal(480, 160, n), 20, 1500)
)
num_failed_logins = np.where(
    is_attack_true == 1,
    np.random.poisson(1.8, n),
    np.random.poisson(0.3, n)
)
connection_count_per_min = np.where(
    is_attack_true == 1,
    np.random.poisson(12, n),
    np.random.poisson(5, n)
)
bytes_sent = np.where(
    is_attack_true == 1,
    np.random.exponential(1900, n),
    np.random.exponential(1300, n)
)
unique_game_servers_accessed = np.where(
    is_attack_true == 1,
    np.random.poisson(5, n),
    np.random.poisson(2.2, n)
)

df = pd.DataFrame({
    "duration_sec": duration_sec.round(2),
    "packet_size_avg": packet_size_avg.round(1),
    "num_failed_logins": num_failed_logins,
    "connection_count_per_min": connection_count_per_min,
    "bytes_sent": bytes_sent.round(0),
    "unique_game_servers_accessed": unique_game_servers_accessed,
    "is_attack_true": is_attack_true
})
print(f"Uretilen oturum sayisi: {len(df)}")
print(f"Gercek saldiri/bot orani (SADECE degerlendirme icin): %{100*df['is_attack_true'].mean():.2f}")

feature_cols = ["duration_sec", "packet_size_avg", "num_failed_logins",
                 "connection_count_per_min", "bytes_sent", "unique_game_servers_accessed"]

# One-Class SVM'in klasik kullanim bicimi: SADECE NORMAL ornekler ile egitilir.
normal_data = df[df["is_attack_true"] == 0][feature_cols]
print(f"\nEgitim SADECE normal oyuncu trafigiyle yapiliyor: {normal_data.shape[0]} oturum")
print("(One-Class SVM boylece 'normalin siniri' ogrenir, bot/saldiri hic gormez)")

scaler = StandardScaler()
normal_scaled = scaler.fit_transform(normal_data)
X_all_scaled = scaler.transform(df[feature_cols])

print("\nOne-Class SVM egitiliyor (RBF kernel, nu=beklenen anomali orani)...")
ocsvm = OneClassSVM(kernel="rbf", gamma="scale", nu=0.05)
ocsvm.fit(normal_scaled)

decision_scores = -ocsvm.decision_function(X_all_scaled)  # yuksek = daha supheli
predictions = ocsvm.predict(X_all_scaled)  # -1: anomali, 1: normal
df["predicted_attack"] = (predictions == -1).astype(int)
df["anomaly_score"] = decision_scores

print("\nModel performansi DEGERLENDIRILIYOR (gercek etiketlerle kiyaslanarak)...")
auc = roc_auc_score(df["is_attack_true"], df["anomaly_score"])
ap = average_precision_score(df["is_attack_true"], df["anomaly_score"])
print(f"ROC-AUC: {auc:.4f}")
print(f"PR-AUC (Average Precision): {ap:.4f}")
print("\nSiniflandirma Raporu:")
print(classification_report(df["is_attack_true"], df["predicted_attack"],
                             target_names=["Normal", "Bot/Saldiri"]))

print("\nConfusion matrix kaydediliyor...")
cm = confusion_matrix(df["is_attack_true"], df["predicted_attack"])
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges',
            xticklabels=["Normal", "Bot/Saldiri"], yticklabels=["Normal", "Bot/Saldiri"])
plt.xlabel("Model Tahmini")
plt.ylabel("Gercek Durum")
plt.title("One-Class SVM - Oyun Sunucusu Bot/DDoS Tespiti Sonucu")
plt.tight_layout()
plt.savefig("figures/confusion_matrix.png", dpi=150)
plt.close()

print("ROC ve Precision-Recall egrileri kaydediliyor...")
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
fpr, tpr, _ = roc_curve(df["is_attack_true"], df["anomaly_score"])
axes[0].plot(fpr, tpr, color="#ea580c", linewidth=2, label=f"AUC={auc:.3f}")
axes[0].plot([0, 1], [0, 1], "--", color="gray")
axes[0].set_xlabel("False Positive Rate")
axes[0].set_ylabel("True Positive Rate")
axes[0].set_title("ROC Egrisi")
axes[0].legend()

prec, rec, _ = precision_recall_curve(df["is_attack_true"], df["anomaly_score"])
axes[1].plot(rec, prec, color="#ea580c", linewidth=2, label=f"AP={ap:.3f}")
axes[1].set_xlabel("Recall")
axes[1].set_ylabel("Precision")
axes[1].set_title("Precision-Recall Egrisi")
axes[1].legend()
plt.tight_layout()
plt.savefig("figures/roc_pr_curves.png", dpi=150)
plt.close()

print("Anomali skoru dagilimi kaydediliyor...")
plt.figure(figsize=(9, 6))
sns.histplot(data=df, x="anomaly_score", hue="is_attack_true", bins=60,
             palette={0: "#3b82f6", 1: "#ea580c"}, alpha=0.6,
             element="step", stat="density", common_norm=False)
plt.xlabel("Anomali Skoru (yuksek = daha supheli)")
plt.title("Anomali Skoru Dagilimi (Gercek Bot/Saldiri vs Normal)")
plt.legend(labels=["Bot/Saldiri", "Normal"])
plt.tight_layout()
plt.savefig("figures/anomaly_score_distribution.png", dpi=150)
plt.close()

print("PCA ile 2 boyutlu gorsellestirme...")
pca = PCA(n_components=2, random_state=RANDOM_STATE)
X_pca = pca.fit_transform(X_all_scaled)
plt.figure(figsize=(9, 7))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=df["predicted_attack"],
                       cmap="coolwarm", alpha=0.5, s=15)
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
plt.title("One-Class SVM Bot/DDoS Tespiti (PCA ile 2 Boyut)")
plt.colorbar(scatter, label="0=Normal, 1=Bot/Saldiri")
plt.tight_layout()
plt.savefig("figures/attack_pca.png", dpi=150)
plt.close()

print("\nGorseller kaydedildi: figures/")
print("Tamamlandi.")
