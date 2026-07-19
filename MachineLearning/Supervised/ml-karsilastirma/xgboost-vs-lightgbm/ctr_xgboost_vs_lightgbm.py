"""
SATIN ALMA SONRASI OYNANMAMA (CTR Analogu) TAHMINI: XGBoost vs LightGBM

ÖĞRENME NOTU (yönlendirme):
  Bu dosyanın amacı, XGBoost ve LightGBM'i AYNI VERIDE hiz ve dogruluk
  acisindan kiyaslamaktır.

  Hedef gercek bir Steam davranis sinyaline dayanir: "kullanici satin
  aldigi oyunu HIC OYNAMADI MI" (raftaki oyun / backlog fenomeni). Bu,
  "bir eylemden (tiklama / satin alma) sonra beklenen ikinci eylem
  (donusum / oynama) gerceklesti mi?" sorusuyla ayni yapidadir. Iki
  kutuphane AYNI VERIDE
  egitilip HIZ ve DOGRULUK acisindan kiyaslanir.

Kullanilan veri seti (Kaggle): tamber/steam-video-games
  -> Gercek kullanici-oyun satin alma ve oynama gunlugu; "satin alindi ama
     hic oynanmadi mi" sorusuna gercek veriyle cevap verebilen tek veri
     seti bu oldugu icin secildi.
"""
import os
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, roc_curve, average_precision_score,
    precision_recall_curve, accuracy_score, classification_report
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

import kagglehub

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
os.makedirs('figures', exist_ok=True)

print("Satin Alma Sonrasi Oynanmama Tahmini - Veri Hazirligi")

dataset_path = kagglehub.dataset_download("tamber/steam-video-games")
csv_files = [f for f in os.listdir(dataset_path) if f.lower().endswith(".csv")]
data_path = os.path.join(dataset_path, csv_files[0])

raw = pd.read_csv(data_path, header=None,
                   names=["user_id", "game", "behavior", "value", "_unused"])

purchases = raw[raw["behavior"] == "purchase"][["user_id", "game"]].drop_duplicates()
plays = raw[raw["behavior"] == "play"][["user_id", "game", "value"]].rename(columns={"value": "hours"})

df = purchases.merge(plays, on=["user_id", "game"], how="left")
df["hours"] = df["hours"].fillna(0)
df["abandoned"] = (df["hours"] == 0).astype(int)  # satin alindi, hic oynanmadi

print(f"Uretilen satin alma kaydi: {len(df)}")
print(f"Oynanmama (abandoned) orani: %{100*df['abandoned'].mean():.2f}")

print("\nKullanici ve oyun bazli ozellikler turetiliyor...")
user_library_size = purchases.groupby("user_id").size().rename("user_library_size")
game_popularity = purchases.groupby("game").size().rename("game_popularity")
user_avg_hours = df.groupby("user_id")["hours"].mean().rename("user_avg_hours")
game_avg_hours = df.groupby("game")["hours"].mean().rename("game_avg_hours")

df = df.join(user_library_size, on="user_id")
df = df.join(game_popularity, on="game")
df = df.join(user_avg_hours, on="user_id")
df = df.join(game_avg_hours, on="game")

df["popularity_tier"] = pd.qcut(df["game_popularity"], q=4, labels=[0, 1, 2, 3], duplicates="drop").astype(int)

feature_cols = ["user_library_size", "game_popularity", "user_avg_hours",
                 "game_avg_hours", "popularity_tier"]
X = df[feature_cols]
y = df["abandoned"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
print(f"Egitim seti: {X_train.shape} | Test seti: {X_test.shape}")

print("\n=== XGBoost egitiliyor ===")
t0 = time.time()
xgb_model = XGBClassifier(
    n_estimators=300, learning_rate=0.08, max_depth=5,
    subsample=0.9, colsample_bytree=0.9, eval_metric="logloss",
    random_state=RANDOM_STATE, n_jobs=-1
)
xgb_model.fit(X_train, y_train)
xgb_train_time = time.time() - t0
print(f"Egitim suresi: {xgb_train_time:.2f} saniye")

xgb_pred = xgb_model.predict(X_test)
xgb_proba = xgb_model.predict_proba(X_test)[:, 1]
xgb_auc = roc_auc_score(y_test, xgb_proba)
xgb_ap = average_precision_score(y_test, xgb_proba)
xgb_acc = accuracy_score(y_test, xgb_pred)
print(f"Accuracy: {xgb_acc:.4f} | ROC-AUC: {xgb_auc:.4f} | PR-AUC: {xgb_ap:.4f}")

print("\n=== LightGBM egitiliyor ===")
t0 = time.time()
lgbm_model = LGBMClassifier(
    n_estimators=300, learning_rate=0.08, max_depth=5,
    subsample=0.9, colsample_bytree=0.9,
    random_state=RANDOM_STATE, n_jobs=-1, verbose=-1
)
lgbm_model.fit(X_train, y_train)
lgbm_train_time = time.time() - t0
print(f"Egitim suresi: {lgbm_train_time:.2f} saniye")

lgbm_pred = lgbm_model.predict(X_test)
lgbm_proba = lgbm_model.predict_proba(X_test)[:, 1]
lgbm_auc = roc_auc_score(y_test, lgbm_proba)
lgbm_ap = average_precision_score(y_test, lgbm_proba)
lgbm_acc = accuracy_score(y_test, lgbm_pred)
print(f"Accuracy: {lgbm_acc:.4f} | ROC-AUC: {lgbm_auc:.4f} | PR-AUC: {lgbm_ap:.4f}")

print("\nROC egrisi kiyaslamasi kaydediliyor...")
fpr_x, tpr_x, _ = roc_curve(y_test, xgb_proba)
fpr_l, tpr_l, _ = roc_curve(y_test, lgbm_proba)
plt.figure(figsize=(7, 6))
plt.plot(fpr_x, tpr_x, color="#dc2626", label=f"XGBoost (AUC={xgb_auc:.3f})", linewidth=2)
plt.plot(fpr_l, tpr_l, color="#059669", label=f"LightGBM (AUC={lgbm_auc:.3f})", linewidth=2)
plt.plot([0, 1], [0, 1], "--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Egrisi: XGBoost vs LightGBM")
plt.legend()
plt.tight_layout()
plt.savefig("figures/roc_comparison.png", dpi=150)
plt.close()

print("Precision-Recall egrisi kaydediliyor (dengesiz siniflar icin)...")
prec_x, rec_x, _ = precision_recall_curve(y_test, xgb_proba)
prec_l, rec_l, _ = precision_recall_curve(y_test, lgbm_proba)
plt.figure(figsize=(7, 6))
plt.plot(rec_x, prec_x, color="#dc2626", label=f"XGBoost (AP={xgb_ap:.3f})", linewidth=2)
plt.plot(rec_l, prec_l, color="#059669", label=f"LightGBM (AP={lgbm_ap:.3f})", linewidth=2)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Egrisi: XGBoost vs LightGBM")
plt.legend()
plt.tight_layout()
plt.savefig("figures/pr_comparison.png", dpi=150)
plt.close()

print("Egitim suresi kiyaslamasi kaydediliyor...")
plt.figure(figsize=(7, 5))
bars = plt.bar(["XGBoost", "LightGBM"], [xgb_train_time, lgbm_train_time],
                color=["#dc2626", "#059669"])
for b in bars:
    plt.text(b.get_x() + b.get_width()/2, b.get_height() + 0.02,
              f"{b.get_height():.2f}s", ha="center", fontweight="bold")
plt.ylabel("Egitim Suresi (saniye)")
plt.title(f"Egitim Suresi Kiyaslamasi ({len(X_train):,} satir)")
plt.tight_layout()
plt.savefig("figures/training_time_comparison.png", dpi=150)
plt.close()

print("Feature importance kiyaslamasi kaydediliyor...")
xgb_imp = pd.DataFrame({
    "Feature": feature_cols, "Importance": xgb_model.feature_importances_, "Model": "XGBoost"
})
lgbm_imp_raw = lgbm_model.feature_importances_ / lgbm_model.feature_importances_.sum()
lgbm_imp = pd.DataFrame({
    "Feature": feature_cols, "Importance": lgbm_imp_raw, "Model": "LightGBM"
})
imp_combined = pd.concat([xgb_imp, lgbm_imp])
imp_combined.to_csv("figures/feature_importance_comparison.csv", index=False)

plt.figure(figsize=(10, 6))
sns.barplot(data=imp_combined, x="Importance", y="Feature", hue="Model",
            palette={"XGBoost": "#dc2626", "LightGBM": "#059669"})
plt.title("Feature Importance Kiyaslamasi")
plt.tight_layout()
plt.savefig("figures/feature_importance_comparison.png", dpi=150)
plt.close()

print("\n=== SONUC OZETI ===")
summary = pd.DataFrame({
    "Model": ["XGBoost", "LightGBM"],
    "Accuracy": [xgb_acc, lgbm_acc],
    "ROC_AUC": [xgb_auc, lgbm_auc],
    "PR_AUC": [xgb_ap, lgbm_ap],
    "Egitim_Suresi_sn": [xgb_train_time, lgbm_train_time],
})
print(summary.to_string(index=False))
summary.to_csv("figures/model_comparison_summary.csv", index=False)

speed_ratio = xgb_train_time / lgbm_train_time
print(f"\nLightGBM, XGBoost'tan {speed_ratio:.1f}x daha hizli egitildi." if speed_ratio > 1
      else f"\nXGBoost, LightGBM'den {1/speed_ratio:.1f}x daha hizli egitildi.")

print("\nGorseller kaydedildi: figures/")
print("Tamamlandi.")
