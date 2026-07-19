# ==============================================================================
# prepare_dataset.py — Kaggle Veri Setini Indirip Docker Icin Hazirlar
# ------------------------------------------------------------------------------
# ÖĞRENME NOTU (yönlendirme):
#   Gercek bir Kaggle veri seti kullaniyoruz; ancak Docker imajinin ICINE
#   gomulecek dosyanin `docker build` ANINDA diskte hazir olmasi gerekir
#   (Kaggle kimlik bilgilerini konteynerin icine tasimamak icin).
#
#   Bu yuzden akis su sekilde: 1) bu script'i HOST makinende (Docker'in
#   disinda) bir kere calistirip CSV'yi indir -> 2) sonra `docker build` ile
#   imaji olustur (Dockerfile bu CSV'yi COPY ile imaja kopyalar).
#
#   Calistirma: python prepare_dataset.py
# ==============================================================================
import os

import kagglehub
import pandas as pd

OUTPUT_CSV = "video_games_sales_with_ratings.csv"


def main():
    print("Kaggle'dan video oyunu satis/puan veri seti indiriliyor...")
    print("(Kaggle: rush4ratio/video-game-sales-with-ratings)")
    dataset_path = kagglehub.dataset_download("rush4ratio/video-game-sales-with-ratings")
    print("Veri seti yolu:", dataset_path)

    csv_files = [f for f in os.listdir(dataset_path) if f.lower().endswith(".csv")]
    if not csv_files:
        raise FileNotFoundError(f"'{dataset_path}' icinde .csv bulunamadi.")
    source_csv = os.path.join(dataset_path, csv_files[0])
    print("Kaynak dosya:", source_csv)

    df = pd.read_csv(source_csv, encoding="latin-1")
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Kaydedildi: {OUTPUT_CSV}  (shape={df.shape})")
    print("\nSimdi 'docker build' calistirabilirsin.")


if __name__ == "__main__":
    main()
