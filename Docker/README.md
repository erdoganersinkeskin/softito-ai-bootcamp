# 🐳 Docker (Oyun Dünyası Versiyonu)

Egzersizlerin oyun dünyası
veri setleriyle hazırlanmış kişisel alıştırma versiyonlarını içeren klasör.
Tek konteynerden çok servisli mikroservis mimarisine kadar artan bir
karmaşıklıkla ilerliyor.

## 📁 Projeler

### 1️⃣ [1-single-container-xgboost](./1-single-container-xgboost)
Tek bir `Dockerfile` ile paketlenmiş, bir video oyununun **ESRB içerik
puanını** (E/T/M/E10+...) tahmin eden bir XGBoost sınıflandırma modeli.

**Veri seti:** Kaggle `rush4ratio/video-game-sales-with-ratings`

`Python` · `Docker` · `XGBoost` · `scikit-learn`

---

### 2️⃣ [2-docker-compose-3ML](./2-docker-compose-3ML)
Video oyunu satış verisiyle **3 farklı ML görevini** (regresyon,
sınıflandırma, kümeleme) tek bir Docker imajından, **Docker Compose** ile
paralel çalıştıran bir proje.

**Veri seti:** Kaggle `gregorut/videogamesales`

`Python` · `Docker Compose` · `XGBoost` · `scikit-learn`

---

### 3️⃣ [3-microservices-ml-gateway](./3-microservices-ml-gateway)
Steam oyun fiyat verisi üzerinde çalışan **5 bağımsız ML servisi** (Flask
API) ve bunları tek noktadan yöneten bir **API Gateway**.

**Veri seti:** Kaggle `artermiloff/steam-games-dataset`

`Python` · `Flask` · `Docker Compose` · `Microservices`

---

## 🛠️ Ortak Teknolojiler

`Docker` · `Docker Compose` · `Python` · `scikit-learn` · `XGBoost` · `Flask` · `pandas` · `kagglehub`

> Her alt projenin kendi README'sinde veri seti seçim gerekçesi ve
> çalıştırma adımları (Kaggle veri setini indirme dahil) ayrıntılı olarak
> açıklanmıştır.

<p align="center"><i>Docker öğrenme/pratik amaçlı, öğrenme sürecinde egzersiz olarak hazırlanmış bir versiyondur.</i></p>
