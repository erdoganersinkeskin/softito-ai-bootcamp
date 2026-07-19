# ==============================================================================
# prepare_dataset.py — Kaggle Steam Veri Setini Indirip Standart Sema ile Kaydeder
# ------------------------------------------------------------------------------
# ÖĞRENME NOTU (yönlendirme):
#   Bu script'i HOST makinende bir kere calistir, ardindan `docker compose
#   up --build` ile devam et. Boylece 5 mikroservisin her biri ayni,
#   ONCEDEN TEMIZLENMIS "steam_games.csv" dosyasini kullanir; kolon adi
#   normalizasyonu (Kaggle veri setleri surumden surume kolon adi
#   degistirebiliyor) tek bir yerde, burada yapilir.
#
#   Calistirma: python prepare_dataset.py
# ==============================================================================
import os

import kagglehub
import pandas as pd

OUTPUT_CSV = "steam_games.csv"

# Kaggle veri setindeki olasi kolon adlarini, bizim kullanacagimiz sabit
# isimlere esleyen sozluk (kucuk/buyuk harf ve bosluk farklarina karsi
# esnek olmasi icin anahtarlar kucuk harfle karsilastirilir).
COLUMN_ALIASES = {
    "appid": "AppID",
    "steam_appid": "AppID",
    "name": "Name",
    "release_date": "Release_Date",
    "required_age": "Required_Age",
    "genres": "Genres",
    "categories": "Categories",
    "developer": "Developers",
    "developers": "Developers",
    "publisher": "Publishers",
    "publishers": "Publishers",
    "positive_ratings": "Positive",
    "positive": "Positive",
    "negative_ratings": "Negative",
    "negative": "Negative",
    "achievements": "Achievements",
    "windows": "Windows",
    "mac": "Mac",
    "linux": "Linux",
    "price": "Price",
}

FINAL_COLUMNS = [
    "AppID", "Name", "Release_Date", "Required_Age", "Genres", "Categories",
    "Developers", "Publishers", "Positive", "Negative", "Achievements",
    "Windows", "Mac", "Linux", "Price",
]


def main():
    print("Kaggle'dan Steam oyun katalogu indiriliyor...")
    print("(Kaggle: artermiloff/steam-games-dataset)")
    dataset_path = kagglehub.dataset_download("artermiloff/steam-games-dataset")
    print("Veri seti yolu:", dataset_path)

    csv_files = [f for f in os.listdir(dataset_path) if f.lower().endswith(".csv")]
    if not csv_files:
        raise FileNotFoundError(f"'{dataset_path}' icinde .csv bulunamadi.")
    source_csv = os.path.join(dataset_path, csv_files[0])
    print("Kaynak dosya:", source_csv)

    df = pd.read_csv(source_csv)
    df.columns = [c.strip() for c in df.columns]

    rename_map = {}
    for col in df.columns:
        alias = COLUMN_ALIASES.get(col.lower())
        if alias:
            rename_map[col] = alias
    df = df.rename(columns=rename_map)

    # Sadece bildigimiz/standartlastirdigimiz kolonlari tut (mevcut olanlar)
    keep_cols = [c for c in FINAL_COLUMNS if c in df.columns]
    missing_cols = [c for c in FINAL_COLUMNS if c not in df.columns]
    if missing_cols:
        print(f"UYARI: Veri setinde bulunamayan kolonlar (atlaniyor): {missing_cols}")
        print("       Kaggle veri seti sema degistirmis olabilir - bu durumda")
        print("       COLUMN_ALIASES sozlugunu gercek kolon adlarina gore guncelle.")

    df = df[keep_cols].dropna(subset=["Price"] if "Price" in keep_cols else [])

    # Release_Date -> Release_Year (mikroservislerde sayisal ozellik olarak kullanilacak)
    # NOT: Bu Kaggle veri setinin bazi surumlerinde "Release date" kolonu
    # kaymis/bozuk satirlar icerebiliyor (ornegin "Estimated owners" degeri
    # sizabiliyor); parse edilen yili makul bir araliga (1995-2026)
    # sikistirip aralik disi degerleri eksik olarak isaretliyoruz.
    if "Release_Date" in df.columns:
        parsed_year = pd.to_datetime(df["Release_Date"], errors="coerce").dt.year
        df["Release_Year"] = parsed_year.where(parsed_year.between(1995, 2026))
        valid_median = df["Release_Year"].median()
        df["Release_Year"] = df["Release_Year"].fillna(valid_median if pd.notna(valid_median) else 2020)
        df = df.drop(columns=["Release_Date"])

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Kaydedildi: {OUTPUT_CSV}  (shape={df.shape})")
    print(f"Kolonlar: {list(df.columns)}")
    print("\nSimdi 'docker compose up --build' calistirabilirsin.")


if __name__ == "__main__":
    main()
