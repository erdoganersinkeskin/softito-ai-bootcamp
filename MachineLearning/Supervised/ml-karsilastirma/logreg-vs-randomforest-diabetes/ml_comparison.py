"""
MODEL KARSILASTIRMA: Logistic Regression vs Random Forest (Yuksek Puanli Oyun Tahmini)

ÖĞRENME NOTU (yönlendirme):
  Bu dosyanın amacı, Logistic Regression vs Random Forest kiyaslamasini
  5-Fold Cross-Validation ile "hangi model gercekten guvenilir" sorusuna
  odaklanarak yapmaktir.

  Hedef: "bu oyun yuksek elestirmen puani aldi mi" (Critic_Score, medyan
  uzeri) ikili siniflandirmasi. Karsilastirma cercevesi (tek train/test
  split'in sansa bagli olabilecegi, CV'nin daha guvenilir oldugu) esas
  alinir.

Kullanilan veri seti (Kaggle): rush4ratio/video-game-sales-with-ratings
  -> Gercek Critic_Score hedefi + User_Score/Platform/Genre/satis/yil gibi
     karisik sayisal+kategorik ozellikler icerdigi icin, iki farkli model
     ailesini (dogrusal vs agac tabanli) anlamli sekilde kiyaslamaya
     uygun tek veri seti bu oldugundan secildi.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, roc_curve, auc
)

import kagglehub

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
os.makedirs('figures', exist_ok=True)

print("Model Karsilastirma - Veri Hazirligi (Yuksek Puanli Oyun Tahmini)")

dataset_path = kagglehub.dataset_download("rush4ratio/video-game-sales-with-ratings")
csv_files = [f for f in os.listdir(dataset_path) if f.lower().endswith(".csv")]
data_path = os.path.join(dataset_path, csv_files[0])

df = pd.read_csv(data_path, encoding="latin-1")
df.columns = [c.strip() for c in df.columns]
df["User_Score"] = pd.to_numeric(df["User_Score"], errors="coerce")

needed = ["Platform", "Genre", "Year_of_Release", "Global_Sales",
          "Critic_Score", "Critic_Count", "User_Score", "User_Count"]
df = df.dropna(subset=needed)
df = df[df["Critic_Score"] > 0]

df["High_Rated"] = (df["Critic_Score"] > df["Critic_Score"].median()).astype(int)
print(f"Veri seti boyutu: {df.shape}")
print(f"Sinif dagilimi:\n{df['High_Rated'].value_counts().to_string()}")

le_platform = LabelEncoder()
df["Platform_enc"] = le_platform.fit_transform(df["Platform"])
le_genre = LabelEncoder()
df["Genre_enc"] = le_genre.fit_transform(df["Genre"])

feature_cols = ["Platform_enc", "Genre_enc", "Year_of_Release", "Global_Sales",
                 "Critic_Count", "User_Score", "User_Count"]
X = df[feature_cols]
y = df["High_Rated"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
)

print("\nLogistic Regression icin veriler standartlastiriliyor (scaling)...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n=== MODEL 1: Logistic Regression (Basit/Dogrusal Model) ===")
log_model = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)
log_model.fit(X_train_scaled, y_train)
log_pred = log_model.predict(X_test_scaled)
log_acc = accuracy_score(y_test, log_pred)
print(f"Accuracy: {log_acc:.4f}")
print(classification_report(y_test, log_pred))

print("\n=== MODEL 2: Random Forest (Karmasik/Agac Tabanli Model) ===")
rf_model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
rf_model.fit(X_train, y_train)   # agac tabanli modeller scaling gerektirmez
rf_pred = rf_model.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)
print(f"Accuracy: {rf_acc:.4f}")
print(classification_report(y_test, rf_pred))

print("\nConfusion matrix'ler kaydediliyor...")
fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=100)
log_cm = confusion_matrix(y_test, log_pred)
sns.heatmap(log_cm, annot=True, fmt='d', cmap='Blues', ax=axes[0])
axes[0].set_title('Logistic Regression Confusion Matrix')
axes[0].set_xlabel('Tahmin Edilen')
axes[0].set_ylabel('Gercek Durum')

rf_cm = confusion_matrix(y_test, rf_pred)
sns.heatmap(rf_cm, annot=True, fmt='d', cmap='Greens', ax=axes[1])
axes[1].set_title('Random Forest Confusion Matrix')
axes[1].set_xlabel('Tahmin Edilen')
axes[1].set_ylabel('Gercek Durum')
plt.tight_layout()
plt.savefig("figures/confusion_matrices.png", dpi=150)
plt.close()

print("ROC-AUC egrisi kaydediliyor...")
plt.figure(figsize=(8, 6), dpi=100)
log_pos_probs = log_model.predict_proba(X_test_scaled)[:, 1]
fpr_log, tpr_log, _ = roc_curve(y_test, log_pos_probs)
roc_auc_log = auc(fpr_log, tpr_log)
plt.plot(fpr_log, tpr_log, color='blue', label=f'Logistic Regression (AUC = {roc_auc_log:.2f})')

rf_pos_probs = rf_model.predict_proba(X_test)[:, 1]
fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_pos_probs)
roc_auc_rf = auc(fpr_rf, tpr_rf)
plt.plot(fpr_rf, tpr_rf, color='green', label=f'Random Forest (AUC = {roc_auc_rf:.2f})')

plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Egrisi Karsilastirmasi')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("figures/roc_comparison.png", dpi=150)
plt.close()

print("\n5-Fold Cross-Validation hesaplaniyor...")
log_cv_scores = cross_val_score(log_model, X_train_scaled, y_train, cv=5, scoring='accuracy')
rf_cv_scores = cross_val_score(rf_model, X_train, y_train, cv=5, scoring='accuracy')

print(f"Logistic Regression CV skorlari: {np.round(log_cv_scores, 3)}")
print(f"Logistic Regression CV ortalama: {log_cv_scores.mean():.4f} (std: {log_cv_scores.std():.4f})")
print(f"Random Forest CV skorlari: {np.round(rf_cv_scores, 3)}")
print(f"Random Forest CV ortalama: {rf_cv_scores.mean():.4f} (std: {rf_cv_scores.std():.4f})")

folds = ["1. Katlama", "2. Katlama", "3. Katlama", "4. Katlama", "5. Katlama"]
plt.figure(figsize=(10, 6))
plt.plot(folds, log_cv_scores, marker='o', linewidth=2, color='blue',
         label=f'Logistic Regression (Ort: {log_cv_scores.mean():.3f})')
plt.plot(folds, rf_cv_scores, marker='s', linewidth=2, color='green',
         label=f'Random Forest (Ort: {rf_cv_scores.mean():.3f})')
plt.title('Modellerin 5-Fold Cross-Validation Katlama Performanslari', fontsize=14, fontweight='bold')
plt.xlabel('Cross-Validation Katlamalari', fontsize=12)
plt.ylabel('Dogruluk Skoru (Accuracy)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc='lower left')
plt.tight_layout()
plt.savefig("figures/cross_validation_comparison.png", dpi=150)
plt.close()

print("\n=== SONUC OZETI ===")
summary = pd.DataFrame({
    "Model": ["Logistic Regression", "Random Forest"],
    "Tek_Test_Accuracy": [log_acc, rf_acc],
    "CV_Ortalama_Accuracy": [log_cv_scores.mean(), rf_cv_scores.mean()],
    "CV_Std": [log_cv_scores.std(), rf_cv_scores.std()],
    "ROC_AUC": [roc_auc_log, roc_auc_rf],
})
print(summary.to_string(index=False))
summary.to_csv("figures/model_comparison_summary.csv", index=False)

if (rf_acc > log_acc) and (rf_cv_scores.mean() < log_cv_scores.mean()):
    print("\nCELISKI TESPIT EDILDI: Random Forest tek test setinde daha iyi,")
    print("ama CV ortalamasinda Logistic Regression daha iyi/yakin cikti.")
    print("Bu, tek train/test split'in sansa bagli olabilecegini, CV'nin")
    print("daha guvenilir bir olcut oldugunu gosteren tipik bir ornektir.")

print("\nGorseller kaydedildi: figures/")
print("Tamamlandi.")
