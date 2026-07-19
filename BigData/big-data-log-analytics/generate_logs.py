# ==============================================================================
# generate_logs.py — Oyun Mağazası Sunucu Log Üreticisi
# ------------------------------------------------------------------------------
# ÖĞRENME NOTU (yönlendirme):
#   Bu dosyanın amacı, sentetik veri üretimi + PySpark ile büyük ölçekli
#   analiz tekniğiyle bir OYUN MAĞAZASI'nın (Steam benzeri bir platform)
#   sunucu loglarını üretmektir.
#
#   Amaç: Kaggle'daki gerçek bir Steam oyun kataloğunu (appid, isim, tür,
#   fiyat) temel alıp, kullanıcıların bu oyunlarla ilgili sayfalara/API'lere
#   yaptığı sahte (ama gerçekçi) HTTP isteklerini büyük hacimde üretmek.
#   Böylece analyze_logs.py tarafında "en çok görüntülenen oyun", "en
#   popüler tür" gibi anlamlı sonuçlar elde edilebiliyor.
#
# Kullanılan veri seti (Kaggle):
#   fronkongames/steam-games-dataset
#   -> ~85.000+ Steam oyununu; AppID, isim, tür (genre), fiyat gibi
#      kolonlarla içeren güncel ve kapsamlı bir katalog olduğu için bu
#      egzersize en uygun seçim budur (liste içindeki diğer Steam/oyun satış
#      veri setleri de kullanılabilir, ama bu en zengin/güncel olanı).
#
#   NOT: Bu script'i çalıştırmadan önce Kaggle hesabınızla kimlik doğrulama
#   yapmanız gerekir:
#     1) https://www.kaggle.com/settings -> "Create New Token" -> kaggle.json indirin
#     2) Windows'ta şu klasöre koyun: C:\Users\<kullanici_adi>\.kaggle\kaggle.json
#        (veya KAGGLE_USERNAME / KAGGLE_KEY ortam değişkenlerini ayarlayın)
#     3) pip install kagglehub pandas
# ==============================================================================

import csv
import os
import random
import re
import time

import kagglehub
import pandas as pd

# ------------------------------------------------------------------------------
# 1) ADIM: Gerçek oyun kataloğunu Kaggle'dan indir
# ------------------------------------------------------------------------------
# kagglehub.dataset_download, veri setini yerel bir önbellek klasörüne indirir
# ve o klasörün yolunu döndürür. Aynı veri seti tekrar istenirse indirmez,
# önbellekten okur.
print("Steam oyun kataloğu indiriliyor (Kaggle: fronkongames/steam-games-dataset)...")
dataset_path = kagglehub.dataset_download("fronkongames/steam-games-dataset")
print("Veri seti yolu:", dataset_path)

# Bu veri seti genelde tek bir "games.csv" (bazen "games_march2025_cleaned.csv" gibi
# tarih içeren bir isimle) dosyası içerir. Klasördeki ilk .csv dosyasını otomatik
# buluyoruz ki dataset güncellenip dosya adı değişse de script kırılmasın.
csv_files = [f for f in os.listdir(dataset_path) if f.lower().endswith(".csv")]
if not csv_files:
    raise FileNotFoundError(
        f"'{dataset_path}' içinde .csv bulunamadı. Kaggle indirmesini kontrol edin."
    )
games_csv_path = os.path.join(dataset_path, csv_files[0])
print("Oyun kataloğu dosyası:", games_csv_path)

# ------------------------------------------------------------------------------
# 2) ADIM: Kataloğu okuyup, log üretimi için ihtiyacımız olan kolonları çıkar
# ------------------------------------------------------------------------------
# Not: Bu veri seti oldukça büyük olabileceğinden (85k+ satır, çok sayıda
# kolon), sadece ihtiyacımız olan kolonları okuyarak belleği koruyoruz.
raw = pd.read_csv(games_csv_path, usecols=lambda c: c in ("AppID", "Name", "Genres", "Price"))

# Bazı sürümlerde kolon adları farklı olabilir (ör. "appid" küçük harf).
# ÖĞRENME NOTU: gerçek dünyada veri setleri sürümden sürüme değişebilir,
# bu yüzden kolon adlarını normalize etmek iyi bir alışkanlıktır.
raw.columns = [c.strip() for c in raw.columns]
rename_map = {}
for col in raw.columns:
    lc = col.lower()
    if lc == "appid":
        rename_map[col] = "AppID"
    elif lc == "name":
        rename_map[col] = "Name"
    elif lc == "genres":
        rename_map[col] = "Genres"
    elif lc == "price":
        rename_map[col] = "Price"
raw = raw.rename(columns=rename_map)

# Eksik/boş isim veya AppID içeren satırları at, isim tekrarlarını temizle.
# Not: Bazı satırlarda serbest metin kolonlarındaki (About the game vb.)
# kaçış karakteri sorunları AppID'nin sayısal olmayan bir değere kaymasına
# yol açabiliyor; bu yüzden AppID'yi sayısala zorlayıp geçersiz satırları atıyoruz.
raw["AppID"] = pd.to_numeric(raw["AppID"], errors="coerce")
raw = raw[raw["AppID"].apply(lambda v: pd.notna(v) and v not in (float("inf"), float("-inf")))]
raw = raw.dropna(subset=["AppID", "Name"]).drop_duplicates(subset=["AppID"])

# Log üretimini makul sürede tutmak için (85k oyunun tamamı değil) rastgele
# 3.000 oyunluk bir örneklem seçiyoruz. Bu, gerçek bir mağazanın "aktif
# katalogdaki popüler oyunlar" alt kümesini simüle ediyor.
SAMPLE_SIZE = 3000
if len(raw) > SAMPLE_SIZE:
    raw = raw.sample(n=SAMPLE_SIZE, random_state=42)

def slugify(name: str) -> str:
    """Oyun ismini URL-dostu bir "slug"a çevirir. Ör: 'Half-Life 2' -> 'half-life-2'."""
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", str(name).strip().lower())
    return slug.strip("-")[:60] or "game"

def first_genre(genres_value) -> str:
    """'Genres' kolonu genelde virgülle ayrılmış birden fazla tür içerir
    (ör. 'Action,Indie'). Basitlik için ilk türü alıyoruz."""
    if pd.isna(genres_value):
        return "Unknown"
    return str(genres_value).split(",")[0].strip() or "Unknown"

games = []
for _, row in raw.iterrows():
    games.append({
        "appid": int(row["AppID"]),
        "slug": slugify(row["Name"]),
        "genre": first_genre(row.get("Genres", None)),
    })

print(f"Log üretiminde kullanılacak oyun sayısı: {len(games)}")

# ------------------------------------------------------------------------------
# 3) ADIM: Sentetik "oyun mağazası" HTTP log şeması
# ------------------------------------------------------------------------------
# Bir oyun platformunun gerçekçi uç noktalarını (endpoint) kullanıyoruz.
# Bunların bir kısmı
# genel mağaza sayfaları, bir kısmı belirli bir oyuna özel (appid içeren).
GENERIC_URLS = ["/login", "/logout", "/store", "/cart", "/checkout", "/profile",
                 "/search", "/wishlist", "/library", "/support"]
METHODS = ["GET", "POST", "PUT", "DELETE"]
STATUS_CODES = [200, 200, 200, 200, 201, 304, 400, 401, 403, 404, 500, 502]
USER_AGENTS = [
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 16_0 like Mac OS X)",
    "Mozilla/5.0 (Linux; Android 13; Pixel 7)",
    "SteamClient/1.0 (Windows)",
    "curl/7.79.1",
    "Python-urllib/3.11",
    "Mozilla/5.0 (compatible; Googlebot/2.1)",
]
IPS = [f"192.168.{random.randint(0,255)}.{random.randint(1,254)}" for _ in range(1000)]
COUNTRIES = ["US", "DE", "GB", "FR", "BR", "IN", "JP", "CA", "AU", "TR"]

# Oyuna özel uç nokta şablonları. {appid} ve {slug} gerçek katalogdan gelen
# değerlerle dolduruluyor; {genre} ayrı bir kolon olarak da saklanıyor
# (analiz tarafında "en popüler tür" sorgusu için).
GAME_URL_TEMPLATES = [
    "/store/app/{appid}/{slug}",
    "/api/games/{appid}/reviews",
    "/api/games/{appid}/purchase",
    "/api/games/{appid}/screenshots",
    "/library/install/{appid}",
]

# İsteklerin %70'i belirli bir oyuna özel (mağaza gerçekçiliği için), %30'u
# genel sayfalara (login, arama, sepet vb.) gidiyor.
GAME_URL_RATIO = 0.7

total_rows = 1_000_000
file_count = 5
rows_per_file = total_rows // file_count

os.makedirs("logs", exist_ok=True)

start_ts = int(time.time()) - 86400 * 7  # son 7 gün içinde rastgele zaman damgaları

for f_idx in range(file_count):
    filepath = f"logs/gamelogs_{f_idx:03d}.csv"
    pc = f_idx + 1
    print(f"{pc}/{file_count} Üretiliyor: {filepath} ({rows_per_file} satır)")
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "ip", "timestamp", "method", "url", "status", "response_time_ms",
            "user_agent", "country", "bytes_sent", "appid", "genre",
        ])
        for _ in range(rows_per_file):
            ts = start_ts + random.randint(0, 86400 * 7)

            if random.random() < GAME_URL_RATIO:
                game = random.choice(games)
                template = random.choice(GAME_URL_TEMPLATES)
                url = template.format(appid=game["appid"], slug=game["slug"])
                appid = game["appid"]
                genre = game["genre"]
            else:
                url = random.choice(GENERIC_URLS)
                appid = ""
                genre = ""

            writer.writerow([
                random.choice(IPS),
                ts,
                random.choice(METHODS),
                url,
                random.choice(STATUS_CODES),
                random.randint(1, 5000),
                random.choice(USER_AGENTS),
                random.choice(COUNTRIES),
                random.randint(100, 50000),
                appid,
                genre,
            ])
    size_mb = os.path.getsize(filepath) / 1024 / 1024
    print(f"   Tamam: {size_mb:.1f} MB")

print(f"\nToplam {total_rows} satır, {file_count} dosya olarak logs/ dizinine yazıldı.")
print("Sıradaki adım: python analyze_logs.py")
