"""
OYUNCU SEGMENTASYONU - K-Means + PCA

ÖĞRENME NOTU (yönlendirme):
  Bu dosyanın amacı, K-Means + PCA ile davranışsal müşteri/kullanıcı
  segmentasyonu yapmak, Elbow + Silhouette ile optimal K'yı bulmak ve iş
  bağlamında K seçimini yorumlamaktır. OYUNCULAR kutuphanelerindeki oyun
  sayisina, toplam oynama suresine ve ortalama oyun basina saatine gore
  segmentleniyor (casual / duzenli / hardcore oyuncu).

Kullanilan veri seti (Kaggle): tamber/steam-video-games
  -> Gercek kullanici-oyun etkilesim gunlugu; kullanici basina agregatlar
     (kutuphane buyuklugu, toplam saat, ortalama saat) cikarilabildigi
     icin, GERCEK oyuncu davranis verisiyle segmentasyon yapmaya uygun
     tek veri seti bu oldugundan secildi.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

import kagglehub

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
os.makedirs('figures', exist_ok=True)

print("Oyuncu Segmentasyonu - Veri Hazirligi")

dataset_path = kagglehub.dataset_download("tamber/steam-video-games")
csv_files = [f for f in os.listdir(dataset_path) if f.lower().endswith(".csv")]
data_path = os.path.join(dataset_path, csv_files[0])

raw = pd.read_csv(data_path, header=None,
                   names=["user_id", "game", "behavior", "value", "_unused"])
purchases = raw[raw["behavior"] == "purchase"]
plays = raw[raw["behavior"] == "play"]

game_count = purchases.groupby("user_id").size().rename("Game_Count")
total_hours = plays.groupby("user_id")["value"].sum().rename("Total_Hours")
played_game_count = plays.groupby("user_id").size().rename("Played_Game_Count")

df = pd.concat([game_count, total_hours, played_game_count], axis=1).fillna(0).reset_index()
df = df.rename(columns={"user_id": "UserID"})
df = df[df["Game_Count"] > 0]
df["Avg_Hours_Per_Played_Game"] = np.where(
    df["Played_Game_Count"] > 0, df["Total_Hours"] / df["Played_Game_Count"], 0
)

print(f"Uretilen oyuncu sayisi: {len(df)}")
print(df.describe().round(2).to_string())

feature_cols = ["Game_Count", "Total_Hours", "Avg_Hours_Per_Played_Game"]
profile_cols = feature_cols

# Cok carpik dagilimlari normallestirmek icin log-donusum
X = df[feature_cols].apply(lambda col: np.log1p(col))

print("\nEksik deger kontrolu...")
print(f"Eksik deger: {X.isnull().sum().sum()}")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\nOptimal kume sayisi araniyor (Elbow + Silhouette)...")
inertias, silhouettes = [], []
K_range = range(2, 11)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    sil = silhouette_score(X_scaled, labels)
    silhouettes.append(sil)
    print(f"    K={k}: inertia={km.inertia_:.1f}, silhouette={sil:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].plot(list(K_range), inertias, "bo-")
axes[0].set_title("Elbow Yontemi")
axes[0].set_xlabel("Kume Sayisi (K)")
axes[0].set_ylabel("Inertia")
axes[1].plot(list(K_range), silhouettes, "ro-")
axes[1].set_title("Silhouette Skoru")
axes[1].set_xlabel("Kume Sayisi (K)")
axes[1].set_ylabel("Skor")
plt.tight_layout()
plt.savefig("figures/optimal_k.png", dpi=150)
plt.close()

optimal_k_statistical = list(K_range)[int(np.argmax(silhouettes))]
print(f"\nIstatistiksel optimal K (Silhouette'e gore): {optimal_k_statistical}")

# Silhouette skoru istatistiksel olarak dusuk K'yi one cikarabilir, ancak is
# baglaminda (casual / duzenli / hardcore oyuncu) K=3 daha anlamli ve
# pazarlama/oyun-onerisi ekibinin kullanabilecegi bir segmentasyon sunar -
# is ihtiyacinin K secimine yon vermesi prensibi.
optimal_k = 3
print(f"Secilen K (is baglami nedeniyle): {optimal_k}")
print("Gerekce: K=2 sadece kaba bir 'az/cok oynayan' ayrimi verirken, K=3")
print("oyun onerisi ekibinin dogrudan aksiyon alabilecegi (casual/duzenli/")
print("hardcore) katmanlari ortaya cikarir.")

print(f"\nK-Means egitiliyor (K={optimal_k})...")
kmeans = KMeans(n_clusters=optimal_k, init="k-means++", random_state=RANDOM_STATE, n_init=10)
kmeans_labels = kmeans.fit_predict(X_scaled)
final_silhouette = silhouette_score(X_scaled, kmeans_labels)
print(f"Silhouette Skoru: {final_silhouette:.4f}")

df["KMeans_Cluster"] = kmeans_labels
cluster_summary = df.groupby("KMeans_Cluster")[profile_cols].mean().round(2)
cluster_summary["Oyuncu_Sayisi"] = df["KMeans_Cluster"].value_counts().sort_index()
print("\nKume Profilleri:")
print(cluster_summary.to_string())
cluster_summary.to_csv("figures/cluster_profiles.csv")

print("\nPCA ile 2 boyutlu gorsellestirme...")
pca = PCA(n_components=2, random_state=RANDOM_STATE)
X_pca = pca.fit_transform(X_scaled)
print(f"Aciklanan varyans orani: {pca.explained_variance_ratio_.sum():.3f}")

plt.figure(figsize=(9, 7))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans_labels, cmap="viridis",
                       alpha=0.7, s=40, edgecolors="k", linewidth=0.3)
plt.colorbar(scatter, label="Kume")
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
plt.title(f"K-Means Kumeleme Sonuclari (PCA ile, K={optimal_k})")
plt.tight_layout()
plt.savefig("figures/kmeans_pca.png", dpi=150)
plt.close()

plt.figure(figsize=(8, 6))
for c in sorted(df["KMeans_Cluster"].unique()):
    subset = df[df["KMeans_Cluster"] == c]
    plt.scatter(subset["Game_Count"], subset["Total_Hours"], alpha=0.5, s=40, label=f"Kume {c}")
plt.xlabel("Kutuphanedeki Oyun Sayisi")
plt.ylabel("Toplam Oynama Suresi (saat)")
plt.title("Kumelere Gore Oyun Sayisi vs Toplam Oynama Suresi")
plt.legend()
plt.tight_layout()
plt.savefig("figures/order_vs_spending.png", dpi=150)
plt.close()

print("\nGorseller kaydedildi: figures/")
print("Tamamlandi.")
