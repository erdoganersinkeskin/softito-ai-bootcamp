"""
INDIE FIYAT FARKI ANALIZI (Indie Pricing Gap Analysis) - Linear Regression

ÖĞRENME NOTU (yönlendirme):
  Bu dosyanın amacı, kontrol degiskenli Linear Regression ile "ham fark vs
  kontrol edilmis fark" analizi yapmaktır.

  Soru: "INDIE (bagimsiz) oyunlar ile AAA/buyuk
  yayincili oyunlar arasindaki HAM fiyat farkinin ne kadari mesru
  faktorlerle (tur, oyun kalitesi/begeni, basarim sayisi, cikis yili)
  ACIKLANABILIYOR, ne kadari ACIKLANAMIYOR."

  Bu, ayrimcilik denetimi degil, bir FIYATLANDIRMA ADALETI/SEFFAFLIGI
  ornegidir: amac indie gelistiricilerin sistematik olarak dusuk/yuksek
  fiyatlandirilip fiyatlandirilmadigini ORTAYA CIKARMAKTIR.

Kullanilan veri seti (Kaggle): fronkongames/steam-games-dataset
  -> Gercek Steam kataloğu; Price, Genres, Categories (Indie etiketi icin),
     Achievements, Positive/Negative oy sayisi, Release date kolonlarini
     icerir. Bu proje "ham fark vs kontrol edilmis fark" metodolojisi icin
     hem sayisal hem kategorik kontrol degiskenine sahip gercek bir katalog
     gerektirdiginden bu veri seti secildi.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

import kagglehub

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
os.makedirs('figures', exist_ok=True)


# 1) STEAM KATALOGUNU INDIR VE HAZIRLA

print("INDIE FIYAT FARKI ANALIZI - VERI HAZIRLIGI")
print("=" * 60)

dataset_path = kagglehub.dataset_download("fronkongames/steam-games-dataset")
csv_files = [f for f in os.listdir(dataset_path) if f.lower().endswith(".csv")]
catalog_path = os.path.join(dataset_path, csv_files[0])

needed_cols = ["AppID", "Name", "Genres", "Categories", "Price",
               "Achievements", "Positive", "Negative", "Release date"]
raw = pd.read_csv(catalog_path, usecols=lambda c: c in needed_cols)
raw.columns = [c.strip() for c in raw.columns]

# Kolon adlarini normalize et (Kaggle surumleri arasinda kucuk farklar olabilir)
rename_map = {}
for col in raw.columns:
    lc = col.lower()
    if lc == "appid":
        rename_map[col] = "AppID"
    elif lc == "name":
        rename_map[col] = "Name"
    elif lc == "genres":
        rename_map[col] = "Genres"
    elif lc == "categories":
        rename_map[col] = "Categories"
    elif lc == "price":
        rename_map[col] = "Price"
    elif lc == "achievements":
        rename_map[col] = "Achievements"
    elif lc == "positive":
        rename_map[col] = "Positive"
    elif lc == "negative":
        rename_map[col] = "Negative"
    elif lc in ("release date", "release_date"):
        rename_map[col] = "Release_Date"
raw = raw.rename(columns=rename_map)
raw = raw.dropna(subset=["Price", "Genres", "Categories"])
raw = raw[raw["Price"] > 0]  # ucretsiz oyunlar fiyat farki analizinde anlamsiz

# "Indie" etiketi: Categories veya Genres icinde "Indie" geciyorsa
raw["is_indie"] = (
    raw["Categories"].astype(str).str.contains("Indie", case=False)
    | raw["Genres"].astype(str).str.contains("Indie", case=False)
).astype(int)

# ana tur (ilk deger)
raw["primary_genre"] = raw["Genres"].astype(str).str.split(",").str[0].str.strip()

raw["Achievements"] = pd.to_numeric(raw.get("Achievements", 0), errors="coerce").fillna(0)
raw["Positive"] = pd.to_numeric(raw.get("Positive", 0), errors="coerce").fillna(0)
raw["Negative"] = pd.to_numeric(raw.get("Negative", 0), errors="coerce").fillna(0)
raw["review_count"] = raw["Positive"] + raw["Negative"]
raw["positive_ratio"] = np.where(raw["review_count"] > 0, raw["Positive"] / raw["review_count"], np.nan)
raw["positive_ratio"] = raw["positive_ratio"].fillna(raw["positive_ratio"].median())

if "Release_Date" in raw.columns:
    # NOT: Bu Kaggle veri setinin bazi surumlerinde "Release date" kolonu
    # bozuk/kaymis satirlar icerebiliyor (ornegin "0 - 20000" gibi bir
    # "Estimated owners" degeri sizabiliyor). Bu yuzden parse edilen yili
    # makul bir araliga (1995-2026) sikistirip, aralik disi/parse edilemeyen
    # degerleri eksik olarak isaretliyoruz.
    parsed_year = pd.to_datetime(raw["Release_Date"], errors="coerce").dt.year
    raw["release_year"] = parsed_year.where(parsed_year.between(1995, 2026))
    valid_median = raw["release_year"].median()
    raw["release_year"] = raw["release_year"].fillna(valid_median if pd.notna(valid_median) else 2020)
else:
    raw["release_year"] = 2020

# Makul buyuklukte, temiz bir orneklem al (85k satirin tamamiyla ugrasmaya gerek yok)
df = raw.sample(n=min(6000, len(raw)), random_state=RANDOM_STATE).reset_index(drop=True)
print(f"Kullanilan oyun sayisi: {len(df)}")

raw_gap = df.groupby("is_indie")["Price"].mean()
raw_gap_pct = (1 - raw_gap[1] / raw_gap[0]) * 100
print(f"Ham ortalama fiyat (AAA/non-indie) : {raw_gap[0]:.2f}")
print(f"Ham ortalama fiyat (Indie)         : {raw_gap[1]:.2f}")
print(f"HAM (kontrolsuz) fark               : %{raw_gap_pct:.1f}")


# 2) ON ISLEME + MODEL (kontrol degiskenleriyle)

print("\n[2] Kategorik degiskenler encode ediliyor...")
le_genre = LabelEncoder()
df["genre_enc"] = le_genre.fit_transform(df["primary_genre"])

feature_cols = [
    "Achievements", "positive_ratio", "release_year",
    "genre_enc", "is_indie"
]
# Bazi sutunlarda (ozellikle release_year) medyan doldurma sonrasi bile
# NaN kalabiliyor (ornegin tum tarih degerleri parse edilemediyse); modele
# girmeden once bu satirlari guvenli sekilde eliyoruz.
df = df.dropna(subset=feature_cols + ["Price"]).reset_index(drop=True)
X = df[feature_cols]
y = df["Price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)

print("\n[3] Linear Regression egitiliyor (kontrol degiskenleriyle)...")
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print(f"    R2 : {r2:.4f}")
print(f"    MAE: {mae:.2f}")

indie_coef = model.coef_[feature_cols.index("is_indie")]
print(f"\n>>> KONTROL EDILMIS (aciklanamayan) indie fiyat etkisi: {indie_coef:.2f} $")
print(f"    (Tur, basarim sayisi, begeni orani, cikis yili SABIT tutulduğunda,")
print(f"     indie oyunlarin fiyati AAA/non-indie oyunlara gore ortalama {indie_coef:.2f} $ farkli.)")

adjusted_gap_pct = abs(indie_coef) / raw_gap[0] * 100


# 3) GORSELLESTIRME

print("\n[4] Gorseller kaydediliyor...")

plt.figure(figsize=(7, 5))
bars = plt.bar(
    ["Ham Fark\n(kontrolsuz)", "Kontrol Edilmis Fark\n(aciklanamayan)"],
    [raw_gap_pct, adjusted_gap_pct],
    color=["#f59e0b", "#dc2626"]
)
for b in bars:
    plt.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.3,
              f"%{b.get_height():.1f}", ha="center", fontweight="bold")
plt.ylabel("Fiyat Farki (%)")
plt.title("Ham Fark vs Kontrol Edilmis (Aciklanamayan) Fark - Indie vs AAA")
plt.tight_layout()
plt.savefig("figures/raw_vs_adjusted_gap.png", dpi=150)
plt.close()

plt.figure(figsize=(9, 5))
top_genres = df["primary_genre"].value_counts().head(6).index
genre_indie = df[df["primary_genre"].isin(top_genres)].groupby(
    ["primary_genre", "is_indie"])["Price"].mean().unstack()
genre_indie.columns = ["AAA/Non-Indie", "Indie"]
genre_indie.plot(kind="bar", ax=plt.gca(), color=["#f59e0b", "#2563eb"])
plt.ylabel("Ortalama Fiyat ($)")
plt.xlabel("Ana Tur")
plt.title("Ture ve Indie Durumuna Gore Ortalama Fiyat")
plt.xticks(rotation=20)
plt.legend(title="Yayinci Tipi")
plt.tight_layout()
plt.savefig("figures/salary_by_department_gender.png", dpi=150)
plt.close()

plt.figure(figsize=(8, 6))
for flag, label, c in [(0, "AAA/Non-Indie", "#2563eb"), (1, "Indie", "#f59e0b")]:
    subset = df[df["is_indie"] == flag]
    plt.scatter(subset["Achievements"], subset["Price"],
                alpha=0.35, label=label, color=c, s=18)
plt.xlabel("Basarim (Achievement) Sayisi")
plt.ylabel("Fiyat ($)")
plt.title("Basarim Sayisina Gore Fiyat Dagilimi (Indie Durumuna Gore)")
plt.xlim(0, df["Achievements"].quantile(0.95))
plt.legend()
plt.tight_layout()
plt.savefig("figures/experience_vs_salary.png", dpi=150)
plt.close()

coef_df = pd.DataFrame({
    "Feature": feature_cols,
    "Coefficient": model.coef_
}).sort_values("Coefficient", key=abs, ascending=False)
coef_df["Feature"] = coef_df["Feature"].replace({"is_indie": "is_indie (Indie=1)"})

plt.figure(figsize=(8, 5))
colors = ["#dc2626" if f.startswith("is_indie") else "#6366f1" for f in coef_df["Feature"]]
sns.barplot(data=coef_df, x="Coefficient", y="Feature", hue="Feature", palette=colors, legend=False)
plt.axvline(0, color="black", linewidth=0.8)
plt.title("Regresyon Katsayilari (Fiyata Etki, $)")
plt.tight_layout()
plt.savefig("figures/coefficients.png", dpi=150)
plt.close()
coef_df.to_csv("figures/coefficients.csv", index=False)

print("    Kaydedildi: figures/raw_vs_adjusted_gap.png")
print("    Kaydedildi: figures/salary_by_department_gender.png")
print("    Kaydedildi: figures/experience_vs_salary.png")
print("    Kaydedildi: figures/coefficients.png")

print("\n" + "=" * 60)
print("TAMAMLANDI")
print("=" * 60)
