"""
MOBIL OYUN FIYAT SEGMENTI SINIFLANDIRMASI - Decision Tree

ÖĞRENME NOTU (yönlendirme):
  Bu dosyanın amacı, bir Decision Tree ile fiyat segmenti sınıflandırması
  yapmaktır: MOBIL OYUNUN "ozellik setinden" (kullanici puani, puan
  sayisi, boyut, yas siniri, dil sayisi, icinde satin alma sayisi) fiyat
  segmenti tahmin ediliyor. price_range 4 ceyreklige (qcut) bolunur ve
  derinlik/dogruluk karsilastirma analizleri (overfitting kontrolu,
  max_depth taramasi) uygulanir.

Kullanilan veri seti (Kaggle): tristan581/17k-apple-app-store-strategy-games
  -> Gercek mobil oyun kataloğu; Price hedef degiskeni turetmek icin,
     Average User Rating / User Rating Count / Size / Age Rating / Languages
     / In-app Purchases ise "donanim ozellikleri" karsiligi olarak
     kullanilir. Bu proje bir mobil UYGULAMA fiyat siniflandirmasi oldugu
     icin, listedeki veri setleri arasinda gercek mobil oyun katalog
     ozelliklerine sahip tek veri seti bu oldugundan secildi.
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

print("Mobil Oyun Fiyat Segmenti Siniflandirmasi - Veri Hazirligi")

dataset_path = kagglehub.dataset_download("tristan581/17k-apple-app-store-strategy-games")
csv_files = [f for f in os.listdir(dataset_path) if f.lower().endswith(".csv")]
data_path = os.path.join(dataset_path, csv_files[0])

df = pd.read_csv(data_path)
df.columns = [c.strip() for c in df.columns]
df = df.dropna(subset=["Price", "Average User Rating", "User Rating Count", "Size"])
df = df[df["Price"] > 0]  # ucretsiz oyunlar fiyat siniflandirmasinda anlamsiz

df["age_rating_numeric"] = pd.to_numeric(
    df["Age Rating"].astype(str).str.replace("+", "", regex=False), errors="coerce"
).fillna(4)
df["size_mb"] = df["Size"] / (1024 * 1024)
# NOT: pandas >= 3.0'da .astype(str) eksik degerleri her zaman "nan" string'ine
# cevirmeyebiliyor (Arrow destekli string dtype); bu yuzden dogrudan pd.isna
# ile kontrol edip ardindan str()'e ceviriyoruz.
def _count_comma_items(value):
    if pd.isna(value) or value == "":
        return 0
    return len(str(value).split(","))


df["language_count"] = df["Languages"].apply(_count_comma_items)
df["iap_count"] = df["In-app Purchases"].apply(_count_comma_items)

le_genre = LabelEncoder()
df["primary_genre_enc"] = le_genre.fit_transform(df["Primary Genre"].astype(str))

# fiyati ceyreklige bol (price_range 0-3 mantigi)
df["price_range"], bin_edges = pd.qcut(df["Price"], q=4, labels=False, retbins=True, duplicates="drop")
df = df.dropna(subset=["price_range"])
df["price_range"] = df["price_range"].astype(int)
n_classes = df["price_range"].nunique()
class_names = [f"Segment {i}" for i in sorted(df["price_range"].unique())]

print(f"Veri seti boyutu: {df.shape}")
print(f"Fiyat sinir degerleri: {np.round(bin_edges, 2)}")
print(f"Sinif dagilimi:\n{df['price_range'].value_counts().sort_index()}")

feature_cols = ["Average User Rating", "User Rating Count", "size_mb",
                 "age_rating_numeric", "language_count", "iap_count", "primary_genre_enc"]
X = df[feature_cols]
y = df["price_range"]
print(f"\nOzellik matrisi boyutu: {X.shape}")
print(f"Hedef vektoru boyutu: {y.shape}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=RANDOM_STATE, stratify=y
)
print(f"\nEgitim seti: {X_train.shape}")
print(f"Test seti: {X_test.shape}")

print("\nDecision Tree modeli egitiliyor...")
model = DecisionTreeClassifier(
    criterion="gini",
    max_depth=6,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=RANDOM_STATE,
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Dogruluk (Accuracy): {accuracy:.4f}")
print("\nSiniflandirma Raporu:")
print(classification_report(y_test, y_pred, target_names=class_names))

print("\nConfusion matrix kaydediliyor...")
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.title("Confusion Matrix")
plt.xlabel("Tahmin Edilen")
plt.ylabel("Gercek")
plt.tight_layout()
plt.savefig("figures/confusion_matrix.png", dpi=150)
plt.close()

print("Karar agaci gorsellestiriliyor...")
plt.figure(figsize=(28, 14))
plot_tree(
    model, feature_names=list(X.columns), class_names=class_names,
    filled=True, rounded=True, fontsize=8, max_depth=3
)
plt.title("Decision Tree Gorsellestirmesi (ilk 3 seviye)", fontsize=20)
plt.tight_layout()
plt.savefig("figures/decision_tree.png", dpi=150)
plt.close()

tree_rules = export_text(model, feature_names=list(X.columns), spacing=3)
with open("figures/decision_rules.txt", "w") as f:
    f.write(tree_rules)
print("Karar kurallari kaydedildi: figures/decision_rules.txt")

print("\nOzellik onem siralamasi hesaplaniyor...")
feature_importance = pd.DataFrame({
    "feature": X.columns,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=False)
print("En Onemli Ozellikler:")
print(feature_importance.to_string(index=False))
feature_importance.to_csv("figures/feature_importance.csv", index=False)

plt.figure(figsize=(10, 6))
sns.barplot(data=feature_importance, x="importance", y="feature",
            hue="feature", palette="viridis", legend=False)
plt.title("Feature Importance")
plt.xlabel("Onem Skoru")
plt.ylabel("Ozellik")
plt.tight_layout()
plt.savefig("figures/feature_importance.png", dpi=150)
plt.close()

print("\nOverfitting kontrolu...")
train_accuracy = model.score(X_train, y_train)
test_accuracy = model.score(X_test, y_test)
print(f"Egitim Dogrulugu: {train_accuracy:.4f}")
print(f"Test Dogrulugu:   {test_accuracy:.4f}")
if train_accuracy - test_accuracy > 0.1:
    print("Uyari: Model overfitting yapiyor olabilir (fark > %10).")
else:
    print("Egitim-test farki makul seviyede, belirgin overfitting yok.")

print("\nFarkli max_depth degerleriyle model karsilastirmasi...")
depths = range(1, 16)
train_scores, test_scores = [], []
for depth in depths:
    dt = DecisionTreeClassifier(
        max_depth=depth, criterion="gini",
        min_samples_split=10, min_samples_leaf=5, random_state=RANDOM_STATE
    )
    dt.fit(X_train, y_train)
    train_scores.append(dt.score(X_train, y_train))
    test_scores.append(dt.score(X_test, y_test))

plt.figure(figsize=(10, 6))
plt.plot(list(depths), train_scores, "b-o", label="Egitim Dogrulugu")
plt.plot(list(depths), test_scores, "r-s", label="Test Dogrulugu")
plt.xlabel("Max Depth")
plt.ylabel("Dogruluk")
plt.title("Derinlik - Dogruluk Iliskisi")
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(list(depths))
plt.tight_layout()
plt.savefig("figures/depth_accuracy_curve.png", dpi=150)
plt.close()

best_depth = list(depths)[int(np.argmax(test_scores))]
print(f"En iyi test dogrulugunu veren derinlik: {best_depth} (test acc={max(test_scores):.4f})")

print("\nGorseller kaydedildi: figures/")
print("Tamamlandi.")
