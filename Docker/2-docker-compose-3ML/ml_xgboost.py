# ==============================================================================
# ml_xgboost.py — Kuresel Oyun Satisi (Global_Sales) Regresyonu
# ------------------------------------------------------------------------------
# ÖĞRENME NOTU (yönlendirme):
#   Bu dosyanın amacı, XGBoost regresyon hattıyla bir oyunun kuresel satis
#   rakamini (Global_Sales, milyon adet) tahmin etmektir.
#
#   ONEMLI: NA_Sales/EU_Sales/JP_Sales/Other_Sales kolonlari TOPLANDIGINDA
#   Global_Sales'e esit oldugu icin (veri sizintisi olmasin diye) bu
#   kolonlar OZELLIKLERDEN CIKARILDI - sadece oyunun "meta verisi"
#   (Platform, Year, Genre, Publisher) ile tahmin yapiliyor. Bu, "hedefle
#   dogrudan iliskili kolonu ozelliklerden cikar" prensibidir (bkz.
#   ml_logistic_regression.py'deki charges/CostClass ayrimi) prensibiyle
#   aynidir.
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

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

warnings.filterwarnings('ignore')

# ============================ CONFIG ============================
DATA_PATH = 'vgsales.csv'
TARGET_COLUMN = 'Global_Sales'
# Hedefle dogrudan iliskili / sizinti yaratan kolonlar + yuksek kardinaliteli kimlikler
LEAKAGE_COLUMNS = ['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Rank', 'Name']
TEST_SIZE = 0.2
RANDOM_STATE = 42
# ===============================================================

os.makedirs('output', exist_ok=True)

print("=" * 60)
print("VIDEO GAMES - REGRESSION (XGBOOST)")
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
df = df.dropna(subset=[TARGET_COLUMN])
print(f"    Hedef bos olan {before - df.shape[0]} satir atildi.")

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
print("\n[3] Encoding categorical features...")
y = df[TARGET_COLUMN]
drop_now = [c for c in LEAKAGE_COLUMNS if c in df.columns] + [TARGET_COLUMN]
X = df.drop(columns=drop_now)
print(f"    Ozelliklerden cikarilan kolonlar: {drop_now}")

cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
for col in cat_cols:
    X[col] = LabelEncoder().fit_transform(X[col].astype(str))
print(f"    Encode edilen kolonlar: {cat_cols}")

# ----------------------------------------------------------------
print("\n[4] Train/test split...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)
print(f"    Train: {X_train.shape}, Test: {X_test.shape}")

# ----------------------------------------------------------------
print("\n[5] Training XGBoost Regressor...")
model = XGBRegressor(
    n_estimators=300, learning_rate=0.1, max_depth=4,
    subsample=0.9, colsample_bytree=0.9, random_state=RANDOM_STATE
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("    Model egitildi.")

# ----------------------------------------------------------------
print("\n[6] Evaluation:")
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"    R2  : {r2:.4f}")
print(f"    MAE : {mae:.2f}")
print(f"    RMSE: {rmse:.2f}")

# ----------------------------------------------------------------
print("\n[7] Feature importance...")
importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)
print(importance.to_string(index=False))
importance.to_csv('output/reg_feature_importance.csv', index=False)

plt.figure(figsize=(9, 5))
sns.barplot(data=importance, x='Importance', y='Feature', color='steelblue')
plt.title('XGBoost Feature Importance (Global_Sales)')
plt.tight_layout()
plt.savefig('output/reg_feature_importance.png', dpi=150)
plt.close()
print("    Saved: output/reg_feature_importance.png")

# ----------------------------------------------------------------
print("\n[8] Actual vs Predicted plot...")
plt.figure(figsize=(7, 6))
plt.scatter(y_test, y_pred, alpha=0.4, color='steelblue')
lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
plt.plot(lims, lims, 'r--')
plt.xlabel('Actual Global_Sales (million units)')
plt.ylabel('Predicted Global_Sales (million units)')
plt.title('Actual vs Predicted')
plt.tight_layout()
plt.savefig('output/reg_actual_vs_predicted.png', dpi=150)
plt.close()
print("    Saved: output/reg_actual_vs_predicted.png")

# ----------------------------------------------------------------
print("\n[9] Saving predictions...")
results = X_test.copy()
results['Actual'] = np.array(y_test)
results['Predicted'] = np.round(np.array(y_pred), 2)
results.to_csv('output/reg_predictions.csv', index=False)
print("    Saved: output/reg_predictions.csv")

print("\n" + "=" * 60)
print("REGRESSION COMPLETED SUCCESSFULLY")
print("=" * 60)
