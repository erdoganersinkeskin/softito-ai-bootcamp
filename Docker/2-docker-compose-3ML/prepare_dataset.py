# ==============================================================================
# prepare_dataset.py — Kaggle Veri Setini Indirip Docker Icin Hazirlar
# ------------------------------------------------------------------------------
# ÖĞRENME NOTU (yönlendirme):
#   Bu script'i HOST makinende (Docker'in disinda) bir kere calistirip
#   CSV'yi indir, sonra `docker compose up --build` ile devam et. Boylece
#   Kaggle kimlik bilgilerini konteynerlerin icine tasimana gerek kalmaz.
#
#   Calistirma: python prepare_dataset.py
# ==============================================================================
import os

import kagglehub
import pandas as pd

OUTPUT_CSV = "vgsales.csv"


def main():
    print("Kaggle'dan video oyunu satis veri seti indiriliyor...")
    print("(Kaggle: gregorut/videogamesales)")
    dataset_path = kagglehub.dataset_download("gregorut/videogamesales")
    print("Veri seti yolu:", dataset_path)

    csv_files = [f for f in os.listdir(dataset_path) if f.lower().endswith(".csv")]
    if not csv_files:
        raise FileNotFoundError(f"'{dataset_path}' icinde .csv bulunamadi.")
    source_csv = os.path.join(dataset_path, csv_files[0])
    print("Kaynak dosya:", source_csv)

    df = pd.read_csv(source_csv, encoding="latin-1")
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Kaydedildi: {OUTPUT_CSV}  (shape={df.shape})")
    print("\nSimdi 'docker compose up --build' calistirabilirsin.")


if __name__ == "__main__":
    main()
