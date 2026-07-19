"""
OYUN PAZARI SEGMENTASYONU - K-Means, Hiyerarsik, DBSCAN, GMM Kiyaslamasi

ÖĞRENME NOTU (yönlendirme):
  Bu dosyanın amacı, 4 farklı kümeleme algoritmasını (K-Means, Hiyerarşik,
  DBSCAN, GMM) aynı veri seti üzerinde uygulayıp karşılaştırmaktır.
  OYUNLAR fiyat/basarim sayisi/ortalama oynanma suresi/begeni oranina gore
  kumeleniyor.

Kullanilan veri seti (Kaggle): fronkongames/steam-games-dataset
  -> Gercek Steam katalogu; fiyat, basarim, ortalama oynanma suresi ve
     begeni orani gibi sayisal ozellikler icerdigi ve GERCEK eksik/
     bilinmeyen degerler (orn. hic izlenmemis oynanma suresi) barindirdigi
     icin, eksik veri isleme (imputation) ogretisini de dogal olarak
     iceren tek veri seti bu oldugundan secildi.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score

import kagglehub

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
os.makedirs('figures', exist_ok=True)

print("Oyun Pazari Segmentasyonu - Veri Hazirligi")

dataset_path = kagglehub.dataset_download("fronkongames/steam-games-dataset")
csv_files = [f for f in os.listdir(dataset_path) if f.lower().endswith(".csv")]
data_path = os.path.join(dataset_path, csv_files[0])

col_candidates = {
    "Price": ["Price"],
    "Achievements": ["Achievements"],
    "Average_Playtime": ["Average playtime forever", "Average playtime two weeks"],
    "Positive": ["Positive"],
    "Negative": ["Negative"],
}
header = pd.read_csv(data_path, nrows=0).columns
usecols = []
resolved = {}
for target, candidates in col_candidates.items():
    for cand in candidates:
        if cand in header:
            usecols.append(cand)
            resolved[cand] = target
            break

raw = pd.read_csv(data_path, usecols=usecols).rename(columns=resolved)
raw = raw.sample(n=min(1200, len(raw)), random_state=RANDOM_STATE).reset_index(drop=True)
raw = raw.reset_index().rename(columns={"index": "AppIndex"})
raw["CustomerID"] = "GAME-" + raw["AppIndex"].astype(str).str.zfill(5)

# Bu veri setinde "0" cogunlukla "hic izlenmemis/olculmemis" anlamina gelir,
# bu yuzden mantiken 0 olan Average_Playtime'i eksik deger gibi ele aliyoruz
# (gercek dunya eksik veri simulasyonu).
if "Average_Playtime" in raw.columns:
    raw["Average_Playtime"] = raw["Average_Playtime"].replace(0, np.nan)

raw["Positive"] = pd.to_numeric(raw.get("Positive", 0), errors="coerce").fillna(0)
raw["Negative"] = pd.to_numeric(raw.get("Negative", 0), errors="coerce").fillna(0)
review_count = raw["Positive"] + raw["Negative"]
raw["Positive_Ratio"] = np.where(review_count > 0, raw["Positive"] / review_count, np.nan)

df = raw[["CustomerID", "Price", "Achievements", "Average_Playtime", "Positive_Ratio"]].copy()
print(f"Uretilen oyun sayisi: {len(df)}")
print(f"\nVeri seti bilgisi:")
print(f"Sutun sayisi: {len(df.columns)}")
print(f"Satir sayisi: {len(df)}")

print("\nIstatistiksel ozet:")
print(df.describe().T.round(2).to_string())

print("\nEksik deger raporu:")
print(df.isnull().sum()[df.isnull().sum() > 0].to_string())
print(f"Toplam eksik deger: {df.isnull().sum().sum()}")

feature_cols = ["Price", "Achievements", "Average_Playtime", "Positive_Ratio"]
imputer = SimpleImputer(strategy="median")
X_imputed = imputer.fit_transform(df[feature_cols])
df_imputed = pd.DataFrame(X_imputed, columns=feature_cols)

scaler = StandardScaler()
x_scaled = scaler.fit_transform(df_imputed)

print("\n--- K-MEANS ---")
inertia, silhouette_scores = [], []
K_range = range(2, 11)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    labels = kmeans.fit_predict(x_scaled)
    inertia.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(x_scaled, labels))

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].plot(list(K_range), inertia, "bo-")
axes[0].set_title("Elbow Yontemi")
axes[0].set_xlabel("K"); axes[0].set_ylabel("Inertia")
axes[1].plot(list(K_range), silhouette_scores, "ro-")
axes[1].set_title("Silhouette Skoru")
axes[1].set_xlabel("K"); axes[1].set_ylabel("Skor")
plt.tight_layout()
plt.savefig("figures/optimal_k.png", dpi=150)
plt.close()

optimal_k = 4   # is baglaminda anlamli, karsilastirmayi 4 algoritmada tutarli tutmak icin
kmeans = KMeans(n_clusters=optimal_k, init="k-means++", random_state=RANDOM_STATE, n_init=10)
kmeans_labels = kmeans.fit_predict(x_scaled)
kmeans_sil = silhouette_score(x_scaled, kmeans_labels)
print(f"K-Means (K={optimal_k}) Silhouette Skoru: {kmeans_sil:.4f}")

print("\n--- HIYERARSIK (AGGLOMERATIVE) ---")
agg = AgglomerativeClustering(n_clusters=optimal_k, linkage="ward")
agg_labels = agg.fit_predict(x_scaled)
agg_sil = silhouette_score(x_scaled, agg_labels)
print(f"Agglomerative (K={optimal_k}) Silhouette Skoru: {agg_sil:.4f}")

print("\n--- DBSCAN ---")
neighbors = NearestNeighbors(n_neighbors=5)
neighbors_fit = neighbors.fit(x_scaled)
distances, _ = neighbors_fit.kneighbors(x_scaled)
distances = np.sort(distances[:, 4], axis=0)

plt.figure(figsize=(9, 5))
plt.plot(distances, "b-")
plt.xlabel("Nokta Sirasi")
plt.ylabel("5. En Yakin Komsu Mesafesi")
plt.title("DBSCAN - Eps Secimi icin K-Distance Grafigi")
plt.tight_layout()
plt.savefig("figures/dbscan_kdistance.png", dpi=150)
plt.close()

dbscan = DBSCAN(eps=0.8, min_samples=5)
dbscan_labels = dbscan.fit_predict(x_scaled)
n_clusters_db = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
n_noise = list(dbscan_labels).count(-1)
print(f"DBSCAN kume sayisi: {n_clusters_db}, gurultu noktasi: {n_noise}")
if n_clusters_db > 1:
    mask = dbscan_labels != -1
    dbscan_sil = silhouette_score(x_scaled[mask], dbscan_labels[mask])
    print(f"DBSCAN Silhouette Skoru (gurultu haric): {dbscan_sil:.4f}")
else:
    dbscan_sil = np.nan
    print("DBSCAN tek kume/gurultu uretti, silhouette hesaplanamadi.")

print("\n--- GAUSSIAN MIXTURE MODEL ---")
gmm = GaussianMixture(n_components=optimal_k, covariance_type="full",
                       max_iter=200, random_state=RANDOM_STATE)
gmm_labels = gmm.fit_predict(x_scaled)
gmm_sil = silhouette_score(x_scaled, gmm_labels)
print(f"GMM (K={optimal_k}) Silhouette Skoru: {gmm_sil:.4f}")

print("\n--- ALGORITMA KARSILASTIRMASI ---")
comparison = pd.DataFrame({
    "Algoritma": ["K-Means", "Hiyerarsik (Agglomerative)", "DBSCAN", "Gaussian Mixture Model"],
    "Silhouette_Skoru": [kmeans_sil, agg_sil, dbscan_sil, gmm_sil],
    "Kume_Sayisi": [optimal_k, optimal_k, n_clusters_db, optimal_k]
})
print(comparison.to_string(index=False))
comparison.to_csv("figures/algorithm_comparison.csv", index=False)

plt.figure(figsize=(8, 5))
sns.barplot(data=comparison, x="Silhouette_Skoru", y="Algoritma",
            hue="Algoritma", palette="viridis", legend=False)
plt.title("Algoritma Karsilastirmasi (Silhouette Skoru)")
plt.tight_layout()
plt.savefig("figures/algorithm_comparison.png", dpi=150)
plt.close()

print("\nPCA ile gorsellestirme (tum algoritmalar)...")
pca = PCA(n_components=2, random_state=RANDOM_STATE)
X_pca = pca.fit_transform(x_scaled)

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
configs = [
    (kmeans_labels, "K-Means", "viridis", axes[0, 0]),
    (agg_labels, "Hiyerarsik (Agglomerative)", "plasma", axes[0, 1]),
    (dbscan_labels, "DBSCAN (-1 = Gurultu)", "tab10", axes[1, 0]),
    (gmm_labels, "Gaussian Mixture Model", "coolwarm", axes[1, 1]),
]
for labels, title, cmap, ax in configs:
    sc = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap=cmap,
                     alpha=0.7, s=35, edgecolors="k", linewidth=0.3)
    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    plt.colorbar(sc, ax=ax, label="Kume")
plt.tight_layout()
plt.savefig("figures/all_algorithms_pca.png", dpi=150)
plt.close()

print("\nGorseller kaydedildi: figures/")
print("Tamamlandi.")
