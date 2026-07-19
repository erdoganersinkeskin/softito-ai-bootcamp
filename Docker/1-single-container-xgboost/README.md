# 🐳 video-game-rating-docker (Oyun Versiyonu)

> Bir video oyununun **ESRB içerik puanını** (E, T, M, E10+ vb.) tahmin eden, **tek bir Dockerfile** ile konteynerize edilmiş bir XGBoost sınıflandırma modeli.

## 🎓 Bu Proje Hakkında

Bu çalışmanın amacı, genel bir XGBoost işlem hattı (veri yükle → temizle →
encode et → görev tipini otomatik belirle → XGBoost eğit → değerlendir →
feature importance → confusion matrix → tahminleri kaydet) kurup oyunların
**ESRB puanını** tahmin etmektir.

## 📌 Proje Ne Yapıyor?

Video oyunu satış/puan verisiyle (platform, tür, yayıncı, çıkış yılı,
eleştirmen/kullanıcı puanı vb.) her oyunun **ESRB içerik puanını**
(`E`, `T`, `M`, `E10+`, ...) tahmin eden bir **XGBoost** modeli eğitir.

## 📊 Veri Seti

**Kaggle:** [`rush4ratio/video-game-sales-with-ratings`](https://www.kaggle.com/datasets/rush4ratio/video-game-sales-with-ratings)

**Neden bu veri seti seçildi?** Hedefin doğrudan bir formülle türetilmesi
yerine, gerçekten hazır kategorik bir hedef kolona (ESRB `Rating`) sahip
olması isteniyordu — paylaşılan 9 veri setinden bu koşulu sağlayan tek
veri seti bu olduğundan tercih edildi.

## 📂 Dizin Yapısı

```
1-single-container-xgboost/
├── Dockerfile
├── xgboost_docker.py          # Ana script (veri → model → sonuç)
├── prepare_dataset.py         # Kaggle'dan veriyi indirip CSV'ye kaydeder (host'ta çalıştırılır)
├── requirements.txt
├── .gitignore
├── README.md
└── output/                    # Script çalışınca üretilen çıktılar
    ├── confusion_matrix.png
    ├── feature_importance.png
    ├── feature_importance.csv
    └── predictions.csv
```

## 🚀 Nasıl Çalıştırılır?

### 1) Veri setini indir (host makinende, Docker'ın dışında)

Kaggle kimlik doğrulaması gerekir (`kaggle.json` → `C:\Users\<kullanici_adi>\.kaggle\kaggle.json`).

```bash
pip install kagglehub pandas
python prepare_dataset.py
```

Bu adım `video_games_sales_with_ratings.csv` dosyasını bu klasöre indirir —
Dockerfile bu dosyayı imaja `COPY` ile gömer.

### 2) İmajı oluştur ve çalıştır

```bash
docker build -t video-game-rating-model .
docker run --rm -v "$(pwd)/output:/app/output" video-game-rating-model
```

Çalışma bitince `output/` klasöründe grafikleri ve tahmin sonuçlarını
bulursun.

## 📊 Sonuçlar (gerçek çalıştırma — ML mantığı Docker dışında doğrudan test edildi)

**Test doğruluğu: %72.35** (7 ESRB sınıfı üzerinde: E, E10+, EC, K-A, M, RP, T
— veri setinde tek örnekli `AO` sınıfı otomatik olarak elendi).

| Sınıf | Precision | Recall | F1 | Destek |
|---|---|---|---|---|
| E | 0.78 | 0.84 | 0.81 | 797 |
| E10+ | 0.64 | 0.49 | 0.55 | 284 |
| M | 0.75 | 0.65 | 0.70 | 313 |
| T | 0.66 | 0.71 | 0.69 | 592 |

Model, en çok örneği olan sınıflarda (E, T) güçlü; nadir sınıflarda
(EC, K-A, RP) yeterli örnek olmadığı için tahmin yapamıyor — gerçek
dünya sınıf dengesizliğinin tipik bir sonucu.

| | |
|---|---|
| ![Confusion Matrix](output/confusion_matrix.png) | ![Feature Importance](output/feature_importance.png) |

## 🛠️ Kullanılan Teknolojiler

`Python 3.10` · `Docker` · `pandas` · `scikit-learn` · `XGBoost` · `matplotlib` · `seaborn` · `kagglehub`

<p align="center"><i>Docker & ML pratiği amaçlı, öğrenme sürecinde egzersiz olarak hazırlanmış bir versiyondur.</i></p>
