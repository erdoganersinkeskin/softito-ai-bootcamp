# video-game-sales-ml-docker-compose (Oyun Versiyonu)

> Video oyunu satış verisiyle **3 farklı ML görevini** (regresyon, sınıflandırma, kümeleme) tek bir Docker imajından, **Docker Compose** ile paralel çalıştıran bir proje.

## 🎓 Bu Proje Hakkında

Bu çalışmanın amacı, aynı Docker Compose mimarisi altında 3 bağımsız ML
görevini (regresyon, sınıflandırma, kümeleme) paralel çalıştırmaktır;
video oyunu satış verisi kullanılıyor.

## 📌 Proje Ne Yapıyor?

Video oyunu satış verisi (`vgsales.csv`: isim, platform, çıkış yılı, tür,
yayıncı, bölgesel + küresel satış rakamları) üzerinde **3 bağımsız ML
görevi** çalıştırılıyor. Her görev kendi konteynerinde, aynı Docker
imajından, farklı bir komutla ayağa kalkıyor.

| Servis | Script | Görev Türü | Ne Yapıyor |
|---|---|---|---|
| `ml_xgboost` | `ml_xgboost.py` | Regresyon | Küresel satışı (`Global_Sales`) oyun meta verisinden (platform/tür/yıl/yayıncı) tahmin eder |
| `ml_logistic_regression` | `ml_logistic_regression.py` | Sınıflandırma | Satışı medyana göre `High`/`Low` sınıflarına ayırıp tahmin eder |
| `ml_kmeans` | `ml_kmeans.py` | Kümeleme | Oyunları çıkış yılı / bölgesel satış / küresel satışa göre segmentlere ayırır |

## 📊 Veri Seti

**Kaggle:** [`gregorut/videogamesales`](https://www.kaggle.com/datasets/gregorut/videogamesales)
— 16.500+ oyun; `Rank`, `Name`, `Platform`, `Year`, `Genre`, `Publisher`,
`NA_Sales`, `EU_Sales`, `JP_Sales`, `Other_Sales`, `Global_Sales` kolonları.

**Neden bu veri seti seçildi?** Az sayıda, temiz, sayısal + kategorik
karışık kolona sahip küçük-orta ölçekli bir tablo veri seti gerekiyordu —
3 farklı ML görevine (regresyon/sınıflandırma/kümeleme) aynı anda uygun
olan bu veri seti tercih edildi.

**Veri sızıntısı notu:** `NA_Sales + EU_Sales + JP_Sales + Other_Sales =
Global_Sales` olduğundan, regresyon ve sınıflandırma scriptlerinde bu
bölgesel kolonlar **özelliklerden çıkarılmıştır** — sadece oyunun meta
verisiyle (platform, tür, yıl, yayıncı) tahmin yapılır. Kümeleme scripti
bu sınırlamaya tabi değildir (hedef tahmini yok), bu yüzden bölgesel satış
kolonlarından biri (`NA_Sales`) kümeleme özelliği olarak kullanılabilir.

## 📂 Dizin Yapısı

```
2-docker-compose-3ML/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── prepare_dataset.py     # Kaggle'dan veriyi indirip CSV'ye kaydeder (host'ta çalıştırılır)
├── vgsales.csv            # prepare_dataset.py çalıştırılınca oluşur (repoya dahil değil)
├── ml_xgboost.py
├── ml_logistic_regression.py
├── ml_kmeans.py
├── .gitignore
└── output/
```

## 🐳 Neden Docker Compose (Dockerfile değil)?

3 ayrı görev aynı anda / birbirinden bağımsız çalışması gerektiği için
Compose kullanıldı. Tek bir
`Dockerfile` ortak imajı tanımlar, `docker-compose.yml` bu aynı imajı 3 kez
farklı `command` ile çalıştırır; üç servis de paylaşımlı `ml_output`
volume'una yazar.

## 🚀 Nasıl Çalıştırılır?

### 1) Veri setini indir (host makinende)

```bash
pip install kagglehub pandas
python prepare_dataset.py
```

### 2) Servisleri çalıştır

```bash
docker compose up --build

# Sadece birini çalıştırmak istersen:
docker compose up --build ml_xgboost

# Bitince temizle
docker compose down
```

Sonuçlar `output/` klasöründe (Docker volume üzerinden) birikir:
- `reg_*` → regresyon çıktıları
- `clf_*` → sınıflandırma çıktıları
- `clu_*` → kümeleme çıktıları

## 📊 Sonuçlar (gerçek çalıştırma — ML mantığı Docker dışında doğrudan test edildi)

| Görev | Metrik | Değer |
|---|---|---|
| Regresyon (XGBoost) | R² / MAE / RMSE | 0.095 / 0.54 / 1.95 |
| Sınıflandırma (Logistic Regression) | Accuracy | %57.5 |
| Kümeleme (K-Means) | En iyi K (Silhouette) | K=2 (Silhouette=0.41) |

Sadece platform/tür/yıl/yayıncı gibi meta veriyle küresel satışı tahmin
etmek zor bir problem (R²=0.095) — beklenen bir sonuç, çünkü asıl satış
sinyalini taşıyan bölgesel satış kolonları veri sızıntısını önlemek için
kasıtlı olarak dışlandı. Kümeleme K=2'de en net ayrımı buluyor: **%99.5
"Yeni Nesil - Düşük Satış"** ve **%0.5 "Klasik - Yüksek Satış (efsane)"**
(77 oyun, ortalama 16.5M küresel satış).

| | |
|---|---|
| ![Regresyon: gerçek vs tahmin](output/reg_actual_vs_predicted.png) | ![Sınıflandırma: confusion matrix](output/clf_confusion_matrix.png) |
| ![Optimal K](output/clu_optimal_k.png) | ![Küme profilleri](output/clu_profiles.png) |

## 🛠️ Kullanılan Teknolojiler

`Python 3.13` · `Docker` · `Docker Compose` · `pandas` · `scikit-learn` · `XGBoost` · `matplotlib` · `seaborn` · `kagglehub`

<p align="center"><i>Docker Compose & ML pratiği amaçlı, öğrenme sürecinde egzersiz olarak hazırlanmış bir versiyondur.</i></p>
