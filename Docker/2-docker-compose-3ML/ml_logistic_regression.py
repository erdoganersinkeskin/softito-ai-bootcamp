# ==============================================================================
# ml_logistic_regression.py — Oyun Satis Sinifi (High/Low) Siniflandirmasi
# ------------------------------------------------------------------------------
# ÖĞRENME NOTU (yönlendirme):
#   Bu dosyanın amacı, bir oyunun kuresel satisi (Global_Sales) medyanina
#   gore High/Low sinifi turetip Logistic Regression ile siniflandirmaktir.
#   Bolgesel satis kolonlari (NA/EU/JP/Other_Sales) veri sizintisini
#   onlemek icin ozelliklerden cikarilir (ml_xgboost.py ile ayni gerekce).
# ==============================================================================
import os
import glob
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

warnings.filterwarnings('ignore')

# config kismi
DATA_PATH = 'vgsales.csv'
SOURCE_COLUMN = 'Global_Sales'
LEAKAGE_COLUMNS = ['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Rank', 'Name']
TEST_SIZE = 0.2
RANDOM_STATE = 42


os.makedirs('output', exist_ok=True)

print("=" * 60)
print("VIDEO GAMES - CLASSIFICATION (LOGISTIC REGRESSION)")
print("=" * 60)

# ----------------------------------------------------------------
print("\n[1] Loading data...")
if DATA_PATH is None:
    cands = [c for c in glob.glob('*.csv') + glob.glob('**/*.csv', recursive=True)
             if not c.startswith('output')]
    if not cands:
        raise FileNotFoundError("Klasorde .csv bulunamadi.")
    DATA_PATH = cands[0]
    print(f"    Otomatik bulunan dosya: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)
print(f"    Shape: {df.shape}")
print(f"    Columns: {list(df.columns)}")

# ----------------------------------------------------------------
print("\n[2] Cleaning data...")
df = df.dropna(how='all').dropna(axis=1, how='all')

before = df.shape[0]
df = df.dropna(subset=[SOURCE_COLUMN])
print(f"    Hedef kaynagi bos olan {before - df.shape[0]} satir atildi.")

missing = df.isnull().sum()
missing = missing[missing > 0]
if len(missing) > 0:
    print("    Eksik deger iceren kolonlar:")
    for col, n in missing.items():
        print(f"        {col}: {n}")
    for col in df.columns:
        if df[col].isnull().sum() == 0:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(df[col].mode().iloc[0])
else:
    print("    Eksik deger yok.")

dups = df.duplicated().sum()
if dups > 0:
    df = df.drop_duplicates()
    print(f"    {dups} yinelenen satir atildi.")
print(f"    Temizlik sonrasi shape: {df.shape}")

# ----------------------------------------------------------------
print("\n[3] Creating binary target (High / Low)...")
median_val = df[SOURCE_COLUMN].median()
df['SalesClass'] = np.where(df[SOURCE_COLUMN] > median_val, 'High', 'Low')
print(f"    Medyan {SOURCE_COLUMN}: {median_val:.2f}")
print(f"    Sinif dagilimi:\n{df['SalesClass'].value_counts().to_string()}")

y_raw = df['SalesClass']
drop_now = [c for c in LEAKAGE_COLUMNS if c in df.columns] + ['SalesClass', SOURCE_COLUMN]
X = df.drop(columns=drop_now)
print(f"    Ozelliklerden cikarilan kolonlar: {drop_now}")

# ----------------------------------------------------------------
print("\n[4] Encoding & scaling...")
cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
for col in cat_cols:
    X[col] = LabelEncoder().fit_transform(X[col].astype(str))
print(f"    Encode edilen kolonlar: {cat_cols}")

target_le = LabelEncoder()
y = target_le.fit_transform(y_raw)   # High/Low -> 0/1
print(f"    Hedef siniflar: {list(target_le.classes_)}")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)   # Logistic Regression olcekleme ister

# ----------------------------------------------------------------
print("\n[5] Train/test split + training...")
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)
print(f"    Train: {X_train.shape}, Test: {X_test.shape}")

model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("    Model egitildi.")

# ----------------------------------------------------------------
print("\n[6] Evaluation:")
acc = accuracy_score(y_test, y_pred)
print(f"    Accuracy: {acc:.4f}\n")
print(classification_report(y_test, y_pred, target_names=list(target_le.classes_)))

# ----------------------------------------------------------------
print("\n[7] Feature coefficients (etki yonu)...")
coef = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_[0]
}).sort_values('Coefficient', key=abs, ascending=False)
print(coef.to_string(index=False))
coef.to_csv('output/clf_coefficients.csv', index=False)

plt.figure(figsize=(9, 5))
sns.barplot(data=coef, x='Coefficient', y='Feature', color='indianred')
plt.title('Logistic Regression Coefficients')
plt.axvline(0, color='black', linewidth=0.8)
plt.tight_layout()
plt.savefig('output/clf_coefficients.png', dpi=150)
plt.close()
print("    Saved: output/clf_coefficients.png")

# ----------------------------------------------------------------
print("\n[8] Confusion matrix...")
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=target_le.classes_, yticklabels=target_le.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig('output/clf_confusion_matrix.png', dpi=150)
plt.close()
print("    Saved: output/clf_confusion_matrix.png")

# ----------------------------------------------------------------
print("\n[9] Saving predictions...")
out = df.copy()
out['Predicted'] = target_le.inverse_transform(model.predict(X_scaled))
out[['SalesClass', 'Predicted']].to_csv('output/clf_predictions.csv', index=False)
print("    Saved: output/clf_predictions.csv")

print("\n" + "=" * 60)
print("CLASSIFICATION COMPLETED SUCCESSFULLY")
print("=" * 60)
