"""
STEAM OYUN KAPAK GORSELLERI ILE EVRISIMLI SINIR AGI (CNN) TUR SINIFLANDIRMA
Amac: Steam magazasindaki oyunlarin kapak/header gorsellerini, oyunun ana
      turune (Action, RPG, Strategy, vb.) gore bir Evrisimli Sinir Agi (CNN)
      ile siniflandirmak.

ÖĞRENME NOTU (yönlendirme):
  Bu dosyanın amacı, bir CNN mimarisini uçtan uca kurmaktır (veri yukleme
  -> normalize -> CNN -> egitim -> confusion matrix -> ornek tahminler).

  Steam kataloğundaki en yaygin 8 oyun turu (genre) siniflandiriliyor;
  Kaggle'dan indirilen gercek Steam katalogundaki oyunlarin header (kapak)
  gorselleri kullaniliyor.

Kullanilan veri seti (Kaggle):
  fronkongames/steam-games-dataset
  -> 85.000+ Steam oyunu; her satirda AppID, isim, tur (genres) ve
     "header_image" (magaza kapak gorseli) URL'si bulunuyor. Bu proje bir
     GORUNTU siniflandirma egzersizi oldugu icin, listedeki veri setleri
     arasinda gercek oyun gorseline (URL) sahip tek veri seti bu oldugundan
     tercih edildi.

  NOT: Bu script'i calistirmadan once Kaggle kimlik dogrulamasi
  (kaggle.json) ve internet baglantisi gereklidir; ayrica header
  gorsellerinin indirilmesi de internet erisimi ister.
    1) https://www.kaggle.com/settings -> "Create New Token" -> kaggle.json
    2) Windows: C:\\Users\\<kullanici_adi>\\.kaggle\\kaggle.json
    3) pip install -r requirements.txt
"""
import io
import os
import time
import numpy as np
import pandas as pd
import requests
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from PIL import Image
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, Dropout
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

import kagglehub

RANDOM_STATE = 42
tf.random.set_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
os.makedirs('figures', exist_ok=True)
os.makedirs('data/images', exist_ok=True)

IMAGE_SIZE = 64          # Her kapak gorseli bu boyuta (kare) yeniden olceklenir
GENRES = [               # Steam kataloğunda siklikla gecen 8 ana tur
    "Action", "Adventure", "RPG", "Strategy",
    "Simulation", "Sports", "Racing", "Puzzle",
]
IMAGES_PER_GENRE = 200    # Egitim suresini makul tutmak icin tur basina ornek sayisi


def download_game_catalog():
    """Kaggle'dan Steam oyun katalogunu indirir ve DataFrame olarak dondurur."""
    print("Steam oyun katalogu indiriliyor (Kaggle: fronkongames/steam-games-dataset)...")
    dataset_path = kagglehub.dataset_download("fronkongames/steam-games-dataset")
    csv_files = [f for f in os.listdir(dataset_path) if f.lower().endswith(".csv")]
    if not csv_files:
        raise FileNotFoundError(f"'{dataset_path}' icinde .csv bulunamadi.")
    catalog_path = os.path.join(dataset_path, csv_files[0])
    print("Katalog dosyasi:", catalog_path)

    # NOT: Kolon adlarini once TUM basliklari okuyup normalize ederek eslestiriyoruz
    # (ornegin "Header image" gibi bosluklu/farkli sermayeli adlar da yakalanir);
    # dogrudan usecols=lambda c: c in needed_cols ile tam esitlik kontrolu, kolon
    # adi surumden surume degisirse (ornegin "header_image" -> "Header image")
    # sessizce hicbir satir donmemesine yol acabiliyordu.
    header_cols = pd.read_csv(catalog_path, nrows=0).columns
    def _norm(c):
        return c.strip().lower().replace(" ", "_")
    wanted = {"appid": "AppID", "name": "Name", "genres": "Genres", "header_image": "header_image"}
    usecols = [c for c in header_cols if _norm(c) in wanted]
    catalog = pd.read_csv(catalog_path, usecols=usecols)
    catalog = catalog.rename(columns={c: wanted[_norm(c)] for c in usecols})
    # NOT: Bazi satirlarda serbest metin kolonlarindaki kacis karakteri
    # sorunlari AppID'nin sayisal olmayan bir degere kaymasina yol acabiliyor;
    # bu yuzden AppID'yi sayisala zorlayip gecersiz satirlari atiyoruz.
    catalog["AppID"] = pd.to_numeric(catalog["AppID"], errors="coerce")
    catalog = catalog[np.isfinite(catalog["AppID"])]
    catalog = catalog.dropna(subset=["AppID", "Genres", "header_image"])
    return catalog


def pick_primary_genre(genres_value):
    """'Genres' kolonundaki virgulle ayrilmis turlerden ilkini secili tur listesiyle eslestirir."""
    for genre in str(genres_value).split(","):
        genre = genre.strip()
        if genre in GENRES:
            return genre
    return None


# NOT: Yuzlerce ardisik indirmede her cagride yeni bir TCP/TLS baglantisi
# acmak (requests.get varsayilani) Windows'ta port tukenmesine yol acip
# indirmelerin sessizce basarisiz olmasina neden olabiliyor; bu yuzden
# tum indirmeler icin TEK bir requests.Session (baglanti havuzu) kullanilir.
_http_session = requests.Session()
_http_session.headers.update({"User-Agent": "Mozilla/5.0"})


def download_image(url, timeout=10):
    """Bir kapak gorseli URL'sini indirir, RGB'ye cevirir, IMAGE_SIZE x IMAGE_SIZE olcekler."""
    response = _http_session.get(url, timeout=timeout)
    response.raise_for_status()
    image = Image.open(io.BytesIO(response.content)).convert("RGB")
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    return np.array(image, dtype=np.uint8)


def build_image_dataset(catalog):
    """Her turden IMAGES_PER_GENRE kadar kapak gorselini indirip X, y dizilerini olusturur."""
    catalog = catalog.copy()
    catalog["primary_genre"] = catalog["Genres"].apply(pick_primary_genre)
    catalog = catalog.dropna(subset=["primary_genre"])

    images = []
    labels = []
    genre_to_idx = {genre: idx for idx, genre in enumerate(GENRES)}

    for genre in GENRES:
        subset = catalog[catalog["primary_genre"] == genre].sample(
            frac=1.0, random_state=RANDOM_STATE
        )
        collected = 0
        first_error = None
        print(f"'{genre}' turu icin kapak gorselleri indiriliyor...")
        for _, row in subset.iterrows():
            if collected >= IMAGES_PER_GENRE:
                break
            cache_path = f"data/images/{int(row['AppID'])}.png"
            try:
                if os.path.exists(cache_path):
                    image_array = np.array(Image.open(cache_path).convert("RGB"))
                else:
                    image_array = download_image(row["header_image"])
                    Image.fromarray(image_array).save(cache_path)
            except Exception as exc:
                if first_error is None:
                    first_error = f"{type(exc).__name__}: {exc}"
                continue  # Bozuk/erisilemeyen gorselleri atla
            images.append(image_array)
            labels.append(genre_to_idx[genre])
            collected += 1
        print(f"   Toplanan gorsel sayisi: {collected}/{IMAGES_PER_GENRE}")
        if first_error and collected < IMAGES_PER_GENRE:
            print(f"   (ornek hata: {first_error})")

    X = np.array(images, dtype=np.float32)
    y = np.array(labels, dtype=np.int64)
    return X, y


print("Steam Kapak Gorseli CNN Tur Siniflandirma")
print(f"TensorFlow Versiyonu: {tf.__version__}")

gpu_devices = tf.config.list_physical_devices("GPU")
if len(gpu_devices) > 0:
    print(f"GPU aktif: {gpu_devices[0]}")
else:
    print("GPU bulunamadi, egitim CPU uzerinden yapilacak.")

print("\nSteam katalogu ve kapak gorselleri hazirlaniyor (ilk calistirmada indirilir)...")
game_catalog = download_game_catalog()
X, y = build_image_dataset(game_catalog)

# Guvenlik: stratify icin her sinifta en az 2 ornek gerekir. Indirme
# sirasinda cok az gorsel toplanan (ag hatasi / gecersiz URL) siniflar
# varsa, egitim/test ayrimini bozmamalari icin veri setinden cikarilir.
class_counts = pd.Series(y).value_counts()
valid_classes = class_counts[class_counts >= 2].index
if len(valid_classes) < len(GENRES):
    dropped = [GENRES[i] for i in range(len(GENRES)) if i not in valid_classes]
    print(f"\nUYARI: Yetersiz ornekli turler egitimden cikarildi: {dropped}")
    keep_mask = np.isin(y, valid_classes)
    X, y = X[keep_mask], y[keep_mask]
    remap = {old: new for new, old in enumerate(sorted(valid_classes))}
    y = np.array([remap[v] for v in y])
    GENRES = [GENRES[i] for i in sorted(valid_classes)]

print(f"\nToplam ornek sayisi: {len(X)}")
print("Normalize ediliyor (0-255 -> 0.0-1.0)...")
X = X / 255.0

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
print(f"Egitim seti: {X_train.shape} | Test seti: {X_test.shape}")

print("\nCNN mimarisi olusturuluyor...")
model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, kernel_size=(3, 3), activation="relu"),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.3),
    Dense(len(GENRES), activation="softmax"),
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
model.summary()

print("\nModel egitiliyor...")
history = model.fit(
    X_train, y_train, epochs=8, batch_size=32, validation_split=0.1, verbose=2
)

print("\nEgitim tamamlandi, grafikler kaydediliyor...")
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], "b-o", label="Egitim Dogrulugu")
plt.plot(history.history["val_accuracy"], "r-o", label="Dogrulama Dogrulugu")
plt.title("Model Dogruluk Degisimi (Accuracy)")
plt.xlabel("Epoch")
plt.ylabel("Skor")
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], "b-o", label="Egitim Kaybi")
plt.plot(history.history["val_loss"], "r-o", label="Dogrulama Kaybi")
plt.title("Model Kayip Degisimi (Loss)")
plt.xlabel("Epoch")
plt.ylabel("Kayip Degeri")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("figures/training_curves.png", dpi=150)
plt.close()

print("\nTest verisiyle final degerlendirme...")
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print("=" * 55)
print(f" TEST DOGRULUGU (ACCURACY): %{test_acc * 100:.2f}")
print(f" TEST KAYBI (LOSS): {test_loss:.4f}")
print("=" * 55)

print("\nConfusion matrix ve siniflandirma raporu hazirlaniyor...")
y_pred_proba = model.predict(X_test, verbose=0)
y_pred = np.argmax(y_pred_proba, axis=1)

print(classification_report(y_test, y_pred, target_names=GENRES))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=GENRES, yticklabels=GENRES)
plt.xlabel("Tahmin")
plt.ylabel("Gercek")
plt.title(f"Confusion Matrix - Steam Kapak Gorselleri ({len(GENRES)} Tur)")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig("figures/confusion_matrix.png", dpi=150)
plt.close()

print("\nOrnek tahminler gorsellestiriliyor...")
rng = np.random.default_rng(RANDOM_STATE)
sample_ids = rng.choice(len(X_test), size=min(8, len(X_test)), replace=False)

fig, axes = plt.subplots(2, 4, figsize=(14, 7))
for ax, idx in zip(axes.flat, sample_ids):
    img = X_test[idx]
    true_label = y_test[idx]
    pred_label = y_pred[idx]
    color = "green" if true_label == pred_label else "red"
    ax.imshow(img)
    ax.set_title(f"Gercek: {GENRES[true_label]}\nTahmin: {GENRES[pred_label]}",
                 color=color, fontsize=9)
    ax.axis("off")
plt.tight_layout()
plt.savefig("figures/sample_predictions.png", dpi=150)
plt.close()

print("\nGorseller kaydedildi: figures/")
print("Tamamlandi.")
