"""
INDIE OYUN TESHISI (Tumor Diagnosis Analogu) - Support Vector Machine (SVM)

ÖĞRENME NOTU (yönlendirme):
  Bu dosyanın amacı, Linear ve RBF kernel'i karşılaştıran bir SVM ikili
  sınıflandırma kurmaktır.

  Burada "bagimsiz (Indie) mi, yoksa AAA/buyuk yayincili mi" ikili
  siniflandirmasi ele aliniyor: sadece SAYISAL "ayak izi" ozelliklerinden
  (fiyat, basarim sayisi, olumlu/olumsuz oy sayisi, gerekli yas, begeni
  orani) bir oyunun indie mi AAA mi oldugu tahmin edilebiliyor mu? Linear
  ve RBF kernel karsilastirilarak "karar siniri dogrusal mi yoksa daha
  karmasik mi olmali" sorusuna veriyle cevap araniyor.

Kullanilan veri seti (Kaggle): fronkongames/steam-games-dataset
  -> Gercek Steam katalogu; hem hedefi (Indie etiketi) hem de siniflandirma
     icin gereken COK SAYIDA surekli sayisal ozelligi (fiyat, basarim,
     oy sayilari) icerdigi icin secildi - SVM kernel karsilastirmasi
     yuksek boyutlu surekli ozellik uzayi gerektirir.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
)

import kagglehub

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
os.makedirs('figures', exist_ok=True)

print("Indie Oyun Teshisi (SVM) - Veri Hazirligi")

dataset_path = kagglehub.dataset_download("fronkongames/steam-games-dataset")
csv_files = [f for f in os.listdir(dataset_path) if f.lower().endswith(".csv")]
data_path = os.path.join(dataset_path, csv_files[0])

needed_cols = ["Price", "Genres", "Categories", "Achievements", "Required age",
               "Positive", "Negative", "DiscountDLC count"]
raw = pd.read_csv(data_path, usecols=lambda c: c in needed_cols)
raw.columns = [c.strip() for c in raw.columns]
raw = raw.rename(columns={"Required age": "Required_Age", "DiscountDLC count": "DLC_Count"})
raw = raw.dropna(subset=["Price", "Genres", "Categories"])

raw["Achievements"] = pd.to_numeric(raw.get("Achievements", 0), errors="coerce").fillna(0)
raw["Positive"] = pd.to_numeric(raw.get("Positive", 0), errors="coerce").fillna(0)
raw["Negative"] = pd.to_numeric(raw.get("Negative", 0), errors="coerce").fillna(0)
raw["DLC_Count"] = pd.to_numeric(raw.get("DLC_Count", 0), errors="coerce").fillna(0)
raw["Required_Age"] = pd.to_numeric(raw.get("Required_Age", 0), errors="coerce").fillna(0)
review_count = raw["Positive"] + raw["Negative"]
raw["positive_ratio"] = np.where(review_count > 0, raw["Positive"] / review_count, 0.5)

raw["is_indie"] = (
    raw["Categories"].astype(str).str.contains("Indie", case=False)
    | raw["Genres"].astype(str).str.contains("Indie", case=False)
).astype(int)

df = raw.sample(n=min(4000, len(raw)), random_state=RANDOM_STATE).reset_index(drop=True)
print(f"Veri seti boyutu: {df.shape}")
print(f"Tani dagilimi (Indie):\n{df['is_indie'].value_counts().to_string()}")

feature_cols = ["Price", "Achievements", "Required_Age", "DLC_Count",
                 "positive_ratio", "Positive", "Negative"]
X = df[feature_cols]
y = df["is_indie"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n=== Linear Kernel SVM ===")
# class_weight="balanced": Indie/AAA dagilimi ~%71/%29 dengesiz oldugundan,
# agirliksiz SVM azinlik sinifini (AAA) hic yakalayamiyordu (recall 0.00).
svm_linear = SVC(kernel="linear", probability=True, random_state=RANDOM_STATE, class_weight="balanced")
svm_linear.fit(X_train_scaled, y_train)
pred_linear = svm_linear.predict(X_test_scaled)
acc_linear = accuracy_score(y_test, pred_linear)
proba_linear = svm_linear.predict_proba(X_test_scaled)[:, 1]
auc_linear = roc_auc_score(y_test, proba_linear)
print(f"Accuracy: {acc_linear:.4f} | ROC-AUC: {auc_linear:.4f}")

print("\n=== RBF Kernel SVM (dogrusal olmayan) ===")
svm_rbf = SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE, class_weight="balanced")
svm_rbf.fit(X_train_scaled, y_train)
pred_rbf = svm_rbf.predict(X_test_scaled)
acc_rbf = accuracy_score(y_test, pred_rbf)
proba_rbf = svm_rbf.predict_proba(X_test_scaled)[:, 1]
auc_rbf = roc_auc_score(y_test, proba_rbf)
print(f"Accuracy: {acc_rbf:.4f} | ROC-AUC: {auc_rbf:.4f}")

# NOT: Dengesiz siniflarda ham accuracy yaniltici olabilir (cogunluk
# sinifini hep tahmin eden bir model yuksek accuracy alabilir); esikten
# bagimsiz oldugu icin ROC-AUC'a gore secim yapiyoruz.
best_name = "Linear" if auc_linear >= auc_rbf else "RBF"
print(f"\nEn iyi performansi veren kernel (ROC-AUC'a gore): {best_name}")
best_pred = pred_linear if best_name == "Linear" else pred_rbf
print("\nSiniflandirma Raporu (en iyi kernel):")
print(classification_report(y_test, best_pred, target_names=["AAA/Non-Indie", "Indie"]))

print("\nConfusion matrix kaydediliyor...")
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
cm_linear = confusion_matrix(y_test, pred_linear)
sns.heatmap(cm_linear, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=["AAA/Non-Indie", "Indie"], yticklabels=["AAA/Non-Indie", "Indie"])
axes[0].set_title(f"Linear Kernel (Acc={acc_linear:.3f})")
axes[0].set_xlabel("Tahmin"); axes[0].set_ylabel("Gercek")

cm_rbf = confusion_matrix(y_test, pred_rbf)
sns.heatmap(cm_rbf, annot=True, fmt='d', cmap='Purples', ax=axes[1],
            xticklabels=["AAA/Non-Indie", "Indie"], yticklabels=["AAA/Non-Indie", "Indie"])
axes[1].set_title(f"RBF Kernel (Acc={acc_rbf:.3f})")
axes[1].set_xlabel("Tahmin"); axes[1].set_ylabel("Gercek")
plt.tight_layout()
plt.savefig("figures/confusion_matrices.png", dpi=150)
plt.close()

print("ROC egrisi kiyaslamasi kaydediliyor...")
fpr_l, tpr_l, _ = roc_curve(y_test, proba_linear)
fpr_r, tpr_r, _ = roc_curve(y_test, proba_rbf)
plt.figure(figsize=(7, 6))
plt.plot(fpr_l, tpr_l, color="#2563eb", label=f"Linear (AUC={auc_linear:.3f})", linewidth=2)
plt.plot(fpr_r, tpr_r, color="#7c3aed", label=f"RBF (AUC={auc_rbf:.3f})", linewidth=2)
plt.plot([0, 1], [0, 1], "--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Egrisi: Linear vs RBF Kernel")
plt.legend()
plt.tight_layout()
plt.savefig("figures/roc_comparison.png", dpi=150)
plt.close()

print("Karar siniri (decision boundary) gorsellestiriliyor (PCA ile 2 boyut)...")
pca = PCA(n_components=2, random_state=RANDOM_STATE)
X_train_pca = pca.fit_transform(X_train_scaled)

svm_2d = SVC(kernel="rbf", random_state=RANDOM_STATE)
svm_2d.fit(X_train_pca, y_train)

x_min, x_max = X_train_pca[:, 0].min() - 1, X_train_pca[:, 0].max() + 1
y_min, y_max = X_train_pca[:, 1].min() - 1, X_train_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
Z = svm_2d.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

plt.figure(figsize=(8, 7))
plt.contourf(xx, yy, Z, alpha=0.25, cmap="coolwarm")
plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train,
            cmap="coolwarm", edgecolors="k", s=35, alpha=0.8)
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
plt.title("SVM Karar Siniri (RBF Kernel, PCA ile 2 Boyut)")
plt.tight_layout()
plt.savefig("figures/decision_boundary.png", dpi=150)
plt.close()

print("\nGorseller kaydedildi: figures/")
print("Tamamlandi.")
