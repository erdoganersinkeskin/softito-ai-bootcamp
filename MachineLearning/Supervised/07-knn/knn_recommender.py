"""
OYUN ONERI SISTEMI - K-Nearest Neighbors (KNN) ile Item-Based Collaborative Filtering

ÖĞRENME NOTU (yönlendirme):
  Bu dosyanın amacı, cosine benzerligi tabanli item-based + user-based KNN
  ile bir oneri sistemi kurmak ve kategori tutarliligi/Precision@5 ile
  degerlendirmektir. GERCEK Steam kullanici davranis verisi kullanildigi
  icin 2 veri seti birlikte kullanilmistir (biri etkilesim verisi, digeri
  tur/kategori zenginlestirmesi icin).

Kullanilan veri setleri (Kaggle):
  1) tamber/steam-video-games -> Gercek kullanici-oyun etkilesim gunlugu
     (kullanici ID, oyun adi, 'purchase'/'play' davranisi, oynama suresi).
     Bu, GERCEK davranissal veridir.
  2) gregorut/videogamesales -> Oyun adindan TUR (Genre) bilgisine
     ulasmak icin (tamber veri setinde tur kolonu yok); sadece oneri
     kalitesini "ayni turden mi" diye degerlendirmek icin kullanilir,
     tamber #1'deki gibi bir "referans katalog" rolündedir.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

import kagglehub

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
os.makedirs('figures', exist_ok=True)

print("Oyun Oneri Sistemi (KNN) - Veri Hazirligi")

# 1) Kullanici-oyun etkilesim verisi (tamber/steam-video-games)
interactions_path = kagglehub.dataset_download("tamber/steam-video-games")
csv_files = [f for f in os.listdir(interactions_path) if f.lower().endswith(".csv")]
interactions_file = os.path.join(interactions_path, csv_files[0])

raw = pd.read_csv(interactions_file, header=None,
                   names=["user_id", "game", "behavior", "value", "_unused"])
play_rows = raw[raw["behavior"] == "play"][["user_id", "game", "value"]].rename(columns={"value": "hours"})
print(f"Toplam oynama kaydi: {len(play_rows)}")

# En aktif kullanicilari ve en cok oynanan oyunlari alarak matrisi yonetilebilir tut
top_games = play_rows.groupby("game")["hours"].sum().sort_values(ascending=False).head(80).index
play_rows = play_rows[play_rows["game"].isin(top_games)]
top_users = play_rows.groupby("user_id")["hours"].sum().sort_values(ascending=False).head(700).index
play_rows = play_rows[play_rows["user_id"].isin(top_users)]
print(f"Kullanilan kayit sayisi (filtrelenmis): {len(play_rows)}")
print(f"Kullanici sayisi: {play_rows['user_id'].nunique()} | Oyun sayisi: {play_rows['game'].nunique()}")

# Saatleri 1-5 araliginda bir "rating"e cevir (log-olcek, gercek dunyada
# oynama suresi cok carpik dagilir - birkac saat ile yuzlerce saat arasi fark buyuktur)
play_rows["log_hours"] = np.log1p(play_rows["hours"])
play_rows["rating"] = pd.qcut(play_rows["log_hours"], q=5, labels=[1, 2, 3, 4, 5], duplicates="drop").astype(int)
ratings_df = play_rows[["user_id", "game", "rating"]].rename(columns={"game": "product_id"})

print("\nKullanici-oyun puanlama matrisi olusturuluyor...")
user_item_matrix = ratings_df.pivot_table(
    index="user_id", columns="product_id", values="rating", fill_value=0
)
print(f"Matris boyutu: {user_item_matrix.shape}")

item_user_matrix = user_item_matrix.T  # KNN oyunlar arasi benzerlik icin transpoze

print("\nKNN modeli (item-based, cosine benzerligi) egitiliyor...")
knn_model = NearestNeighbors(metric="cosine", algorithm="brute", n_neighbors=6)
knn_model.fit(item_user_matrix.values)


def get_similar_products(product_id, k=5):
    idx = item_user_matrix.index.get_loc(product_id)
    distances, indices = knn_model.kneighbors(
        item_user_matrix.iloc[idx, :].values.reshape(1, -1), n_neighbors=k + 1
    )
    similar = []
    for dist, i in zip(distances.flatten()[1:], indices.flatten()[1:]):
        similar.append((item_user_matrix.index[i], round(1 - dist, 3)))
    return similar


example_game = item_user_matrix.index[0]
print(f"\nOrnek oneri: '{example_game}' oyununa en benzer 5 oyun")
example_recs = get_similar_products(example_game, k=5)
for pid, sim in example_recs:
    print(f"    {pid} - benzerlik: {sim}")

# 2) Tur (Genre) referans katalogu (gregorut/videogamesales) - oneri kalitesi degerlendirmesi icin
print("\nTur referans katalogu indiriliyor (gregorut/videogamesales)...")
vgsales_path = kagglehub.dataset_download("gregorut/videogamesales")
vg_csv = [f for f in os.listdir(vgsales_path) if f.lower().endswith(".csv")][0]
vgsales = pd.read_csv(os.path.join(vgsales_path, vg_csv), encoding="latin-1")
vgsales.columns = [c.strip() for c in vgsales.columns]
genre_lookup = (
    vgsales.dropna(subset=["Name", "Genre"])
    .assign(name_norm=lambda d: d["Name"].str.lower().str.strip())
    .drop_duplicates("name_norm")
    .set_index("name_norm")["Genre"]
)


def lookup_genre(game_name):
    return genre_lookup.get(str(game_name).lower().strip(), None)


print("\nOneri kalitesi degerlendiriliyor: onerilen oyunlar ayni turden mi?")
correct, total = 0, 0
for pid in item_user_matrix.index[:40]:
    true_genre = lookup_genre(pid)
    if true_genre is None:
        continue
    recs = get_similar_products(pid, k=5)
    for rec_pid, _ in recs:
        rec_genre = lookup_genre(rec_pid)
        if rec_genre is None:
            continue
        total += 1
        if rec_genre == true_genre:
            correct += 1
category_precision = correct / total if total > 0 else float("nan")
print(f"Kategori tutarliligi (Precision@5, tur eslesen oyunlar arasinda): %{100*category_precision:.1f}")
print(f"(Degerlendirilebilen oneri-cifti sayisi: {total} - tamber veri setindeki")
print(" oyun isimlerinin bir kismi vgsales katalogunda bulunamayabilir.)")

print("\nKullanici-bazli puan tahmini (user-based KNN) test ediliyor...")
train_ratings, test_ratings = train_test_split(
    ratings_df, test_size=0.2, random_state=RANDOM_STATE
)
train_matrix = train_ratings.pivot_table(
    index="user_id", columns="product_id", values="rating", fill_value=0
)

user_knn = NearestNeighbors(metric="cosine", algorithm="brute", n_neighbors=10)
user_knn.fit(train_matrix.values)

predictions, actuals = [], []
for _, row in test_ratings.iterrows():
    u, p, actual_rating = row["user_id"], row["product_id"], row["rating"]
    if u not in train_matrix.index or p not in train_matrix.columns:
        continue
    idx = train_matrix.index.get_loc(u)
    distances, indices = user_knn.kneighbors(
        train_matrix.iloc[idx, :].values.reshape(1, -1), n_neighbors=11
    )
    neighbor_ratings = []
    for dist, i in zip(distances.flatten()[1:], indices.flatten()[1:]):
        neighbor_id = train_matrix.index[i]
        r = train_matrix.loc[neighbor_id, p]
        if r > 0:
            neighbor_ratings.append(r)
    if neighbor_ratings:
        predictions.append(np.mean(neighbor_ratings))
        actuals.append(actual_rating)

mae = mean_absolute_error(actuals, predictions) if predictions else float("nan")
print(f"Test seti Puan Tahmini MAE: {mae:.3f} (1-5 puan olceginde)")

print("\nGorseller kaydediliyor...")

plt.figure(figsize=(10, 7))
sample_matrix = user_item_matrix.iloc[:40, :30]
sns.heatmap(sample_matrix, cmap="YlOrRd", cbar_kws={"label": "Puan (0=puanlanmadi)"})
plt.title("Kullanici-Oyun Puanlama Matrisi (ornek 40x30 kesit)")
plt.xlabel("Oyun")
plt.ylabel("Kullanici")
plt.tight_layout()
plt.savefig("figures/rating_matrix_heatmap.png", dpi=150)
plt.close()

genre_counts = pd.Series(
    [g for g in (lookup_genre(pid) for pid in item_user_matrix.index) if g is not None]
).value_counts()
plt.figure(figsize=(8, 5))
sns.barplot(x=genre_counts.index, y=genre_counts.values, hue=genre_counts.index,
            palette="viridis", legend=False)
plt.ylabel("Oyun Sayisi")
plt.title("Tur Basina Oyun Sayisi (esleşen oyunlar arasinda)")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig("figures/category_distribution.png", dpi=150)
plt.close()

plt.figure(figsize=(7, 5))
plt.bar(["Kategori Tutarliligi\n(Precision@5)"], [category_precision * 100], color="#0ea5e9")
plt.ylim(0, 100)
plt.ylabel("Yuzde (%)")
plt.title("Oneri Sistemi Kalite Metrigi")
plt.text(0, category_precision * 100 + 2, f"%{100*category_precision:.1f}",
          ha="center", fontweight="bold")
plt.tight_layout()
plt.savefig("figures/recommendation_quality.png", dpi=150)
plt.close()

example_df = pd.DataFrame(example_recs, columns=["product_id", "similarity"])
example_df["genre"] = example_df["product_id"].apply(lookup_genre)
example_df.to_csv("figures/example_recommendations_P000.csv", index=False)

print("Kaydedildi: figures/rating_matrix_heatmap.png")
print("Kaydedildi: figures/category_distribution.png")
print("Kaydedildi: figures/recommendation_quality.png")
print("Kaydedildi: figures/example_recommendations_P000.csv")

print("\nTamamlandi.")
