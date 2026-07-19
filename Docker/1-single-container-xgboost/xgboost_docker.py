# ==============================================================================
# xgboost_docker.py — Video Oyunu ESRB Puan Siniflandirmasi (XGBoost)
# ------------------------------------------------------------------------------
# ÖĞRENME NOTU (yönlendirme):
#   Bu dosyanın amacı, genel bir XGBoost islem hattini (veri yukle ->
#   temizle -> encode et -> gorev tipini otomatik belirle -> XGBoost egit
#   -> degerlendir -> feature importance -> confusion matrix /
#   actual-predicted -> tahminleri kaydet) kurmaktır.
#
#   Hedef, bir oyunun ESRB icerik puani (Rating: E, T, M, E10+ vb.)
#   -- script'in "gorev tipini otomatik belirle" (siniflandirma/
#   regresyon) mantigi, sadece CONFIG degistirilerek yeniden
#   kullaniliyor.
#
# Kullanilan veri seti (Kaggle):
#   rush4ratio/video-game-sales-with-ratings
#   -> Video oyunlarinin satis rakamlari YANI SIRA Critic_Score, User_Score,
#      Platform, Genre, Publisher, Year_of_Release ve ESRB "Rating" kolonunu
#      da icerir. Bu proje bir SINIFLANDIRMA egzersizi oldugu ve hedef
#      kolonun (Rating) veri setinde ZATEN hazir bulunmasi gerektigi icin,
#      listedeki veri setleri
#      arasinda dogrudan kategorik/hazir bir hedef kolona sahip tek veri
#      seti bu oldugundan tercih edildi.
#
#   NOT: Bu script'i calistirmadan once "prepare_dataset.py"yi HOST
#   makinende calistirip CSV'yi indirmen gerekir (bkz. README).
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
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    r2_score, mean_absolute_error, mean_squared_error
)
from xgboost import XGBClassifier, XGBRegressor

warnings.filterwarnings('ignore')

# ============================ CONFIG ============================
# DATA_PATH None ise script, klasorde otomatik csv/xlsx arar.
DATA_PATH = 'video_games_sales_with_ratings.csv'

# Tahmin etmek istedigin hedef kolon. Bu veri setinde dogal secim:
#   'Rating'       -> siniflandirma (ESRB: E, T, M, E10+, ...)
#   'Global_Sales'  -> regresyon (istersen deneyebilirsin)
TARGET_COLUMN = 'Rating'

# Modele girmeyecek kimlik / gereksiz kolonlar
DROP_COLUMNS = ['Name']

TEST_SIZE = 0.2
RANDOM_STATE = 42
# ===============================================================

os.makedirs('output', exist_ok=True)

print("=" * 60)
print("VIDEO GAME ESRB RATING - XGBOOST")
print("=" * 60)

# ----------------------------------------------------------------
print("\n[1] Loading data...")

if DATA_PATH is None:
    candidates = (glob.glob('*.csv') + glob.glob('*.xlsx')
                  + glob.glob('**/*.csv', recursive=True)
                  + glob.glob('**/*.xlsx', recursive=True))
    candidates = [c for c in candidates if not c.startswith('output')]
    if not candidates:
        raise FileNotFoundError("Klasorde .csv veya .xlsx bulunamadi. Veri dosyasini bu klasore koy.")
    DATA_PATH = candidates[0]
    print(f"    Otomatik bulunan dosya: {DATA_PATH}")

if DATA_PATH.lower().endswith('.xlsx'):
    df = pd.read_excel(DATA_PATH)
else:
    df = pd.read_csv(DATA_PATH)

print(f"    Shape: {df.shape}")
print(f"    Columns: {list(df.columns)}")

if TARGET_COLUMN not in df.columns:
    raise KeyError(
        f"'{TARGET_COLUMN}' kolonu bulunamadi. "
        f"Mevcut kolonlar: {list(df.columns)}. "
        f"CONFIG bolumunden TARGET_COLUMN'i guncelle."
    )

# ----------------------------------------------------------------
print("\n[2] Cleaning data...")

df = df.dropna(how='all').dropna(axis=1, how='all')

before = df.shape[0]
df = df.dropna(subset=[TARGET_COLUMN])
print(f"    Hedef '{TARGET_COLUMN}' bos olan {before - df.shape[0]} satir atildi.")

drop_now = [c for c in DROP_COLUMNS if c in df.columns]
if drop_now:
    df = df.drop(columns=drop_now)
    print(f"    Dusurulen kolonlar: {drop_now}")

missing = df.isnull().sum()
missing = missing[missing > 0]
if len(missing) > 0:
    print("    Eksik deger iceren kolonlar:")
    for col, n in missing.items():
        print(f"        {col}: {n}")
else:
    print("    Eksik deger yok.")

for col in df.columns:
    if df[col].isnull().sum() == 0:
        continue
    if pd.api.types.is_numeric_dtype(df[col]):
        df[col] = df[col].fillna(df[col].median())
    else:
        df[col] = df[col].fillna(df[col].mode().iloc[0])

dups = df.duplicated().sum()
if dups > 0:
    df = df.drop_duplicates()
    print(f"    {dups} yinelenen satir atildi.")

print(f"    Temizlik sonrasi shape: {df.shape}")

# ----------------------------------------------------------------
print("\n[3] Preparing features & target...")

y = df[TARGET_COLUMN]
X = df.drop(columns=[TARGET_COLUMN])

is_classification = (not pd.api.types.is_numeric_dtype(y)) or (y.nunique() <= 10)
task = "CLASSIFICATION" if is_classification else "REGRESSION"
print(f"    Task type: {task}  (hedef benzersiz deger sayisi: {y.nunique()})")

if is_classification:
    # Gercek dunya veri setlerinde bazi siniflar (ornegin ESRB "AO" - Adults
    # Only) cok nadir gorulur; stratified train/test split her sinifta en az
    # 2 ornek gerektirdigi icin, tek orneklik siniflari once eliyoruz.
    class_counts = y.value_counts()
    rare_classes = class_counts[class_counts < 2].index
    if len(rare_classes) > 0:
        print(f"    UYARI: Cok nadir siniflar (<2 ornek) cikarildi: {list(rare_classes)}")
        keep_mask = ~y.isin(rare_classes)
        X, y = X[keep_mask], y[keep_mask]

encoders = {}
cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
for col in cat_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    encoders[col] = le
print(f"    Encode edilen kategorik ozellikler: {cat_cols if cat_cols else 'yok'}")

target_encoder = None
if is_classification and not pd.api.types.is_numeric_dtype(y):
    target_encoder = LabelEncoder()
    y = target_encoder.fit_transform(y.astype(str))
    print(f"    Hedef siniflar: {list(target_encoder.classes_)}")

# ----------------------------------------------------------------
print("\n[4] Train/test split...")
stratify = y if is_classification else None
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=stratify
)
print(f"    Train: {X_train.shape}, Test: {X_test.shape}")

# ----------------------------------------------------------------
print("\n[5] Training XGBoost...")
if is_classification:
    model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=RANDOM_STATE,
        eval_metric='mlogloss'
    )
else:
    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=RANDOM_STATE
    )

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("    Model egitildi.")

# ----------------------------------------------------------------
print("\n[6] Evaluation:")
if is_classification:
    acc = accuracy_score(y_test, y_pred)
    print(f"    Accuracy: {acc:.4f}")
    if target_encoder is not None:
        names = [str(c) for c in target_encoder.classes_]
        # Nadir siniflardan bazilari test setinde hic gorunmeyebilir; bu
        # durumda target_names uzunlugu ile gercek sinif sayisi uyusmaz.
        # labels=range(len(names)) ile tum sinif uzayini sabitliyoruz.
        print("\n" + classification_report(
            y_test, y_pred, labels=range(len(names)), target_names=names, zero_division=0
        ))
    else:
        print("\n" + classification_report(y_test, y_pred))
else:
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"    R2 : {r2:.4f}")
    print(f"    MAE: {mae:.4f}")
    print(f"    RMSE: {rmse:.4f}")

# ----------------------------------------------------------------
print("\n[7] Feature importance...")
importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

print(importance.to_string(index=False))
importance.to_csv('output/feature_importance.csv', index=False)
print("    Saved: output/feature_importance.csv")

plt.figure(figsize=(10, 6))
sns.barplot(data=importance.head(15), x='Importance', y='Feature', color='steelblue')
plt.title('XGBoost Feature Importance (Top 15)')
plt.tight_layout()
plt.savefig('output/feature_importance.png', dpi=150)
plt.close()
print("    Saved: output/feature_importance.png")

# ----------------------------------------------------------------
print("\n[8] Diagnostic plot...")
if is_classification:
    cm = confusion_matrix(y_test, y_pred)
    labels = ([str(c) for c in target_encoder.classes_]
              if target_encoder is not None else sorted(np.unique(y_test)))
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('output/confusion_matrix.png', dpi=150)
    plt.close()
    print("    Saved: output/confusion_matrix.png")
else:
    plt.figure(figsize=(7, 6))
    plt.scatter(y_test, y_pred, alpha=0.4, color='steelblue')
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    plt.plot(lims, lims, 'r--')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Actual vs Predicted')
    plt.tight_layout()
    plt.savefig('output/actual_vs_predicted.png', dpi=150)
    plt.close()
    print("    Saved: output/actual_vs_predicted.png")

# ----------------------------------------------------------------
print("\n[9] Saving predictions...")
results = X_test.copy()
if is_classification and target_encoder is not None:
    results['Actual'] = target_encoder.inverse_transform(np.array(y_test))
    results['Predicted'] = target_encoder.inverse_transform(np.array(y_pred))
else:
    results['Actual'] = np.array(y_test)
    results['Predicted'] = np.array(y_pred)
results.to_csv('output/predictions.csv', index=False)
print("    Saved: output/predictions.csv")

print("\n" + "=" * 60)
print("PROJECT COMPLETED SUCCESSFULLY")
print("=" * 60)
