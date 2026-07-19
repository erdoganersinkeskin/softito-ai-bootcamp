"""
OYUN ELESTIRMEN PUANI TAHMINI - Basit ve Coklu Dogrusal Regresyon

ÖĞRENME NOTU (yönlendirme):
  Bu dosyanın amacı, basit ve çoklu doğrusal regresyonu aynı problem
  üzerinde karşılaştırmaktır: "User_Score -> Critic_Score" (basit) ve
  "tum oyun istatistikleri -> Critic_Score" (coklu) olarak uygulanir.

Kullanilan veri seti (Kaggle): rush4ratio/video-game-sales-with-ratings
  -> Gercek elestirmen puani (Critic_Score) ve kullanici puani (User_Score)
     ile birlikte satis/platform/tur bilgisi icerir; bu proje ozunde bir
     "bir sayisal degiskenden digerini tahmin etme" egzersizi oldugundan,
     listedeki veri setleri arasinda hem hedef (Critic_Score) hem de
     aciklayici degiskenler (User_Score, satislar, elestirmen/kullanici
     sayisi) icin zengin sayisal kolonlara sahip tek veri seti bu oldugu
     icin secildi.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import kagglehub

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
os.makedirs('figures', exist_ok=True)

print("Oyun Elestirmen Puani Tahmini - Veri Hazirligi")

dataset_path = kagglehub.dataset_download("rush4ratio/video-game-sales-with-ratings")
csv_files = [f for f in os.listdir(dataset_path) if f.lower().endswith(".csv")]
data_path = os.path.join(dataset_path, csv_files[0])

df = pd.read_csv(data_path, encoding="latin-1")
df.columns = [c.strip() for c in df.columns]

needed = ["User_Score", "Critic_Score", "Critic_Count", "User_Count",
          "Global_Sales", "Year_of_Release"]
df["User_Score"] = pd.to_numeric(df["User_Score"], errors="coerce")
df = df.dropna(subset=needed)
df = df[(df["Critic_Score"] > 0) & (df["User_Score"] > 0)]

print(f"Kullanilan kayit sayisi: {len(df)} (elestirmen + kullanici puani olan oyunlar)")

print("\nKesifci veri analizi - korelasyon matrisi hesaplaniyor...")
numeric_df = df[needed]
corr = numeric_df.corr()

plt.figure(figsize=(9, 7))
sns.heatmap(corr, cmap="coolwarm", annot=True, fmt=".2f")
plt.title("Korelasyon Matrisi")
plt.tight_layout()
plt.savefig("figures/correlation_matrix.png", dpi=150)
plt.close()

print("\nBasit Dogrusal Regresyon (tek degisken: User_Score)")
x = df[["User_Score"]].values
y = df["Critic_Score"].values
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=RANDOM_STATE
)

model_simple = LinearRegression()
model_simple.fit(x_train, y_train)
y_pred_test = model_simple.predict(x_test)

r2_simple = r2_score(y_test, y_pred_test)
rmse_simple = np.sqrt(mean_squared_error(y_test, y_pred_test))
mae_simple = mean_absolute_error(y_test, y_pred_test)
print(f"    Kesisim noktasi: {model_simple.intercept_:.3f}")
print(f"    Egim (User_Score): {model_simple.coef_[0]:.3f}")
print(f"    R2: {r2_simple:.4f} | RMSE: {rmse_simple:.2f} | MAE: {mae_simple:.2f}")

plt.figure(figsize=(9, 6))
plt.scatter(x, y, color="#2563eb", alpha=0.3, label="Gercek Veri", s=30)
x_line = np.linspace(x.min(), x.max(), 100).reshape(-1, 1)
y_line = model_simple.predict(x_line)
plt.plot(x_line, y_line, color="#dc2626", linewidth=2, label="Regresyon Dogrusu")
plt.xlabel("Kullanici Puani (User_Score, 0-10)")
plt.ylabel("Elestirmen Puani (Critic_Score, 0-100)")
plt.title("Basit Dogrusal Regresyon: Kullanici Puani -> Elestirmen Puani")
plt.legend()
plt.tight_layout()
plt.savefig("figures/simple_regression.png", dpi=150)
plt.close()

print("\nCoklu Dogrusal Regresyon (tum degiskenler)")
feature_cols = ["User_Score", "Critic_Count", "User_Count", "Global_Sales", "Year_of_Release"]
X_multi = df[feature_cols].values
y_multi = df["Critic_Score"].values

X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(
    X_multi, y_multi, test_size=0.2, random_state=RANDOM_STATE
)

model_multi = LinearRegression()
model_multi.fit(X_train_m, y_train_m)
y_pred_test_m = model_multi.predict(X_test_m)

r2_multi = r2_score(y_test_m, y_pred_test_m)
rmse_multi = np.sqrt(mean_squared_error(y_test_m, y_pred_test_m))
mae_multi = mean_absolute_error(y_test_m, y_pred_test_m)
print(f"    R2: {r2_multi:.4f} | RMSE: {rmse_multi:.2f} | MAE: {mae_multi:.2f}")

coef_df = pd.DataFrame({
    "Feature": feature_cols,
    "Coefficient": model_multi.coef_
}).sort_values("Coefficient", key=abs, ascending=False)
print("\nCoklu regresyon katsayilari:")
print(coef_df.to_string(index=False))
coef_df.to_csv("figures/coefficients.csv", index=False)

plt.figure(figsize=(8, 5))
sns.barplot(data=coef_df, x="Coefficient", y="Feature",
            hue="Feature", palette="Blues_r", legend=False)
plt.axvline(0, color="black", linewidth=0.8)
plt.title("Coklu Regresyon Katsayilari (Elestirmen Puanina Etki)")
plt.tight_layout()
plt.savefig("figures/coefficients.png", dpi=150)
plt.close()

plt.figure(figsize=(7, 6))
plt.scatter(y_test_m, y_pred_test_m, alpha=0.3, color="#059669", s=30)
lims = [min(y_test_m.min(), y_pred_test_m.min()), max(y_test_m.max(), y_pred_test_m.max())]
plt.plot(lims, lims, "r--")
plt.xlabel("Gercek Elestirmen Puani")
plt.ylabel("Tahmin Edilen Elestirmen Puani")
plt.title("Coklu Regresyon: Gercek vs Tahmin")
plt.tight_layout()
plt.savefig("figures/actual_vs_predicted.png", dpi=150)
plt.close()

print("\nKarsilastirma:")
print(f"    Basit regresyon  R2: {r2_simple:.4f}")
print(f"    Coklu regresyon  R2: {r2_multi:.4f}")

print("\nGorseller kaydedildi: figures/")
print("Tamamlandi.")
