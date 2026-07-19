"""
OYUN COK SATANLAR (BESTSELLER) TAHMINI - LightGBM

ÖĞRENME NOTU (yönlendirme):
  Bu dosyanın amacı, LightGBM + early stopping ile düşük oranlı/dengesiz
  bir sınıflandırma probleminde doğruluğu maksimize etmektir.

  Hedef: "bu oyun COK SATANLAR (bestseller) listesine girecek mi"
  (kuresel satisin en ust %11'i) - dusuk oranli/dengesiz siniflandirma
  yapisi. LightGBM, early stopping ve ogrenme egrisi takibiyle TEK BASINA
  en iyi sonucu cikarmaya odaklaniyor.

Kullanilan veri seti (Kaggle): gregorut/videogamesales
  -> Gercek satis rakamlari (Global_Sales) + platform/tur/yayinci/yil
     bilgisi icerir; buyuklugu (16.500+ satir) LightGBM'in early stopping
     mekanizmasini gostermeye uygun oldugu icin secildi.
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
from sklearn.metrics import (
    roc_auc_score, roc_curve, average_precision_score,
    precision_recall_curve, accuracy_score, classification_report, confusion_matrix
)
from lightgbm import LGBMClassifier, early_stopping, log_evaluation

import kagglehub

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
os.makedirs('figures', exist_ok=True)

print("Oyun Cok Satanlar (Bestseller) Tahmini - Veri Hazirligi")

dataset_path = kagglehub.dataset_download("gregorut/videogamesales")
csv_files = [f for f in os.listdir(dataset_path) if f.lower().endswith(".csv")]
data_path = os.path.join(dataset_path, csv_files[0])

df = pd.read_csv(data_path, encoding="latin-1")
df.columns = [c.strip() for c in df.columns]
df = df.dropna(subset=["Platform", "Genre", "Publisher", "Year", "Global_Sales"])

bestseller_threshold = df["Global_Sales"].quantile(0.89)  # ~%11 dengesiz oran
df["bestseller"] = (df["Global_Sales"] > bestseller_threshold).astype(int)
print(f"Uretilen oyun sayisi: {len(df)}")
print(f"Cok satan (bestseller) orani: %{100*df['bestseller'].mean():.2f}")

print("\nKategorik degiskenler encode ediliyor...")
cat_cols = ["Platform", "Genre", "Publisher"]
for col in cat_cols:
    le = LabelEncoder()
    df[col + "_enc"] = le.fit_transform(df[col].astype(str))

feature_cols = ["Platform_enc", "Genre_enc", "Publisher_enc", "Year"]
X = df[feature_cols]
y = df["bestseller"]

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=RANDOM_STATE, stratify=y_temp
)
print(f"Egitim: {X_train.shape} | Dogrulama: {X_val.shape} | Test: {X_test.shape}")

print("\nLightGBM egitiliyor (early stopping ile, en iyi iterasyon otomatik secilir)...")
model = LGBMClassifier(
    n_estimators=1000,
    learning_rate=0.03,
    num_leaves=31,
    max_depth=-1,
    subsample=0.85,
    colsample_bytree=0.85,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbose=-1,
)
model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_val, y_val)],
    eval_metric="auc",
    callbacks=[early_stopping(stopping_rounds=50, verbose=False), log_evaluation(period=0)],
)
print(f"Erken durdurmayla secilen en iyi iterasyon: {model.best_iteration_}")

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

test_acc = accuracy_score(y_test, y_pred)
test_auc = roc_auc_score(y_test, y_proba)
test_ap = average_precision_score(y_test, y_proba)
print(f"\nTest Accuracy: {test_acc:.4f}")
print(f"Test ROC-AUC : {test_auc:.4f}")
print(f"Test PR-AUC  : {test_ap:.4f}")
print("\nSiniflandirma Raporu:")
print(classification_report(y_test, y_pred, target_names=["Normal", "Bestseller"]))

print("\nOgrenme egrisi (train vs validation AUC) kaydediliyor...")
results = model.evals_result_
plt.figure(figsize=(9, 6))
plt.plot(results["training"]["auc"], label="Egitim AUC", color="#2563eb")
plt.plot(results["valid_1"]["auc"], label="Dogrulama AUC", color="#dc2626")
plt.axvline(model.best_iteration_, color="gray", linestyle="--",
            label=f"Erken durdurma noktasi ({model.best_iteration_})")
plt.xlabel("Iterasyon (Agac Sayisi)")
plt.ylabel("AUC")
plt.title("LightGBM Ogrenme Egrisi (Early Stopping ile)")
plt.legend()
plt.tight_layout()
plt.savefig("figures/learning_curve.png", dpi=150)
plt.close()

print("Confusion matrix kaydediliyor...")
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
            xticklabels=["Normal", "Bestseller"], yticklabels=["Normal", "Bestseller"])
plt.xlabel("Tahmin")
plt.ylabel("Gercek")
plt.title("Confusion Matrix - Bestseller Tahmini")
plt.tight_layout()
plt.savefig("figures/confusion_matrix.png", dpi=150)
plt.close()

print("ROC ve Precision-Recall egrileri kaydediliyor...")
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
fpr, tpr, _ = roc_curve(y_test, y_proba)
axes[0].plot(fpr, tpr, color="#059669", linewidth=2, label=f"AUC={test_auc:.3f}")
axes[0].plot([0, 1], [0, 1], "--", color="gray")
axes[0].set_xlabel("False Positive Rate")
axes[0].set_ylabel("True Positive Rate")
axes[0].set_title("ROC Egrisi")
axes[0].legend()

prec, rec, _ = precision_recall_curve(y_test, y_proba)
axes[1].plot(rec, prec, color="#059669", linewidth=2, label=f"AP={test_ap:.3f}")
axes[1].set_xlabel("Recall")
axes[1].set_ylabel("Precision")
axes[1].set_title("Precision-Recall Egrisi")
axes[1].legend()

plt.tight_layout()
plt.savefig("figures/roc_pr_curves.png", dpi=150)
plt.close()

print("Feature importance kaydediliyor...")
importance_df = pd.DataFrame({
    "Feature": feature_cols,
    "Importance": model.feature_importances_
}).sort_values("Importance", ascending=False)
print(importance_df.to_string(index=False))
importance_df.to_csv("figures/feature_importance.csv", index=False)

plt.figure(figsize=(9, 6))
sns.barplot(data=importance_df, x="Importance", y="Feature",
            hue="Feature", palette="Greens_r", legend=False)
plt.title("LightGBM Feature Importance")
plt.tight_layout()
plt.savefig("figures/feature_importance.png", dpi=150)
plt.close()

print("\nGorseller kaydedildi: figures/")
print("Tamamlandi.")
