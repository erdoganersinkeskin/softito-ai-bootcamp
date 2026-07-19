# 3-microservices-ml-gateway (Oyun Versiyonu)

> Steam oyun fiyat verisi üzerinde çalışan **5 bağımsız ML servisi** ve bunları tek noktadan yöneten bir **API Gateway** — Docker Compose ile ayağa kaldırılan bir mikroservis mimarisi.

## 🎓 Bu Proje Hakkında

Bu çalışmanın amacı, 5 bağımsız Flask servisi + API Gateway deseniyle
Steam oyun fiyatı tahmin eden bir mikroservis mimarisi kurmaktır.

Bu proje mikroservis mimarisinin temel bileşenlerini (bağımsız servisler,
API Gateway, servisler arası HTTP iletişimi) küçük ölçekte uygular. Service
discovery, authentication, veritabanı-per-servis gibi production-seviyesi
bileşenler kapsam dışıdır.

## 📌 Servisler

| Servis | Port | Görev | Tür |
|---|---|---|---|
| `ml-linear` | 5001 | Fiyat tahmini | Regresyon |
| `ml-knn` | 5002 | Fiyat tahmini | Regresyon |
| `ml-rf` | 5003 | Fiyat tahmini | Regresyon |
| `ml-dtree` | 5004 | Pahalı / Ucuz | Sınıflandırma |
| `ml-svm` | 5005 | Pahalı / Ucuz | Sınıflandırma |
| `ml-gateway` | 8080 | Yönlendirme (tek giriş noktası) | — |

Hepsi aynı veri setini (`steam_games.csv`) kullanır, ancak her servis kendi
modelini bağımsız olarak eğitir ve ayrı bir konteynerde çalışır.

## 📊 Veri Seti

**Kaggle:** [`artermiloff/steam-games-dataset`](https://www.kaggle.com/datasets/artermiloff/steam-games-dataset)
— oyun adı, tür (genres), kategori, geliştirici/yayıncı, olumlu/olumsuz
yorum sayısı, başarım (achievement) sayısı, platform desteği (Windows/Mac/
Linux), çıkış tarihi ve **fiyat** kolonlarını içerir.

**Neden bu veri seti seçildi?** Hem kategorik (genre, developer, publisher)
hem sayısal (review sayısı, achievement sayısı, yıl) özelliklerle
zenginleştirilmiş, doğrudan **fiyat** hedefine sahip bir veri seti
gerekiyordu — listedeki 9 veri seti arasında bu ihtiyacı en iyi karşılayan,
geniş kapsamlı Steam kataloğu bu.

**Şema notu:** `prepare_dataset.py`, Kaggle'dan gelen ham kolon adlarını
(`COLUMN_ALIASES` sözlüğü ile) standart bir şemaya (`AppID`, `Name`,
`Genres`, `Categories`, `Developers`, `Publishers`, `Positive`, `Negative`,
`Achievements`, `Windows`, `Mac`, `Linux`, `Price`, `Release_Year`) çevirir.
Kaggle veri seti şema değiştirirse (kolon adları farklıysa), script konsola
hangi kolonların bulunamadığını yazar — `COLUMN_ALIASES` sözlüğünü gerçek
kolon adlarına göre güncellemen yeterli.

## 📂 Dizin Yapısı

```
3-microservices-ml-gateway/
├── Dockerfile
├── docker-compose.yml         # ml-gateway dahil, inline Flask ile yazılmıştır
├── requirements.txt
├── prepare_dataset.py         # Kaggle'dan veriyi indirip standart şemayla CSV'ye kaydeder (host'ta çalıştırılır)
├── steam_games.csv            # prepare_dataset.py çalıştırılınca oluşur (repoya dahil değil)
├── 1_linear_regression.py
├── 2_knn_regressor.py
├── 3_random_forest_regressor.py
├── 4_decision_tree_classifier.py
├── 5_svm_classifier.py
├── .gitignore
└── output/
```

## 🚀 Kurulum ve Çalıştırma

### 1) Veri setini indir (host makinende)

```bash
pip install kagglehub pandas
python prepare_dataset.py
```

### 2) Servisleri ayağa kaldır

```bash
docker compose up --build
```

İlk açılış biraz sürer — her servis ayağa kalkarken kendi modelini eğitir.

## 🧪 Test

```bash
# Tüm servislerin durumu
curl http://localhost:8080/health

# Fiyat tahmini (regresyon: linear / knn / rf)
curl -X POST http://localhost:8080/linear/predict \
  -H "Content-Type: application/json" \
  -d '{"Required_Age":0,"Genres":"Action","Categories":"Single-player","Developers":"Valve","Publishers":"Valve","Positive":50000,"Negative":2000,"Achievements":20,"Windows":true,"Mac":true,"Linux":true,"Release_Year":2012}'

# Pahalı/Ucuz sınıflandırma (dtree / svm)
curl -X POST http://localhost:8080/dtree/predict \
  -H "Content-Type: application/json" \
  -d '{"Required_Age":0,"Genres":"RPG","Categories":"Multi-player","Developers":"CD Projekt Red","Publishers":"CD Projekt","Positive":300000,"Negative":15000,"Achievements":78,"Windows":true,"Mac":false,"Linux":false,"Release_Year":2015}'
```

## 📊 Sonuçlar (gerçek çalıştırma — her servisin `train()` mantığı Docker dışında doğrudan test edildi)

| Servis | Metrik | Değer |
|---|---|---|
| Linear Regression | R² | -0.174 |
| KNN Regressor | R² | -0.043 |
| Random Forest Regressor | R² | **0.163** (en iyi regresyon) |
| Decision Tree Classifier | Accuracy | %62.1 |
| SVM Classifier | Accuracy | %61.5 |

**Gözlem:** Doğrusal ve KNN regresyon, oyun fiyatını meta verilerden
(tür, geliştirici, inceleme sayısı vb.) tahmin etmekte başarısız kalıyor
(negatif R² = ortalamayı tahmin etmekten bile kötü) — fiyat, doğrusal
olmayan ve bu özelliklerle zayıf ilişkili bir hedef. Random Forest, doğrusal
olmayan etkileşimleri yakalayabildiği için belirgin şekilde daha iyi
performans gösteriyor. Bu, mikroservis mimarisinin "farklı algoritmaları
aynı veri/arayüzle kolayca karşılaştırma" avantajını gerçek verilerle
gösteren iyi bir örnek.

## ⚠️ Önemli Notlar

- `AppID` ve `Name` kolonları (yüksek kardinaliteli kimlikler) modele dahil edilmez.
- SVM ve KNN büyük veride yavaş çalıştığı için eğitim sırasında en fazla 10.000 satır örneklenir.
- `/predict` isteğindeki JSON alan adları `steam_games.csv`'deki kolon adlarıyla birebir aynı olmalı. Bilinmeyen bir kategorik değer gelirse hata vermez, `0` değerine düşer.

## 🛠️ Kullanılan Teknolojiler

`Python 3.13` · `Flask` · `Docker` · `Docker Compose` · `scikit-learn` · `pandas` · `kagglehub`

<p align="center"><i>Docker Compose ile mikroservis mimarisi pratiği amaçlı, öğrenme sürecinde egzersiz olarak hazırlanmış bir versiyondur.</i></p>
