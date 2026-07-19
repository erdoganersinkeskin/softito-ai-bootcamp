# Yapay Zeka Projeleri — Oyun Dünyası Versiyonu

softITo Machine Learning Bootcamp öğrenme sürecinde egzersiz amaçlı derslerde görülen konuların tekrarları için yapılan tüm egzersizlerin, aşağıdaki Kaggle oyun veri
setlerinden biriyle (veya hiçbiri uymuyorsa uygun başka bir oyun
içeriğiyle) yeniden üretilmiş **alıştırma** versiyonlarını içeren
ana repo:

- `artermiloff/steam-games-dataset`
- `gregorut/videogamesales`
- `anandshaw2001/video-game-sales`
- `rush4ratio/video-game-sales-with-ratings`
- `tristan581/17k-apple-app-store-strategy-games`
- `tamber/steam-video-games`
- `nikdavis/steam-store-games`
- `fronkongames/steam-games-dataset`
- `antonkozyriev/game-recommendations-on-steam`

Hiçbir script bu oturumda çalıştırılmamıştır — hepsi syntax kontrolünden
geçmiştir, sonuçlar/görseller kullanıcı çalıştırınca oluşur. Her alt
projenin kendi README'sinde veri seti seçim gerekçesi ve çalıştırma
adımları ayrıntılı olarak açıklanmıştır.

## Yapı

```
oyun-versiyonlari/
├── BigData/               # PySpark ile oyun mağazası log analitiği
├── DeepLearning/           # CNN — Steam kapak görseli sınıflandırma, görüntü işleme
├── Docker/                 # Tek konteynerden mikroservis mimarisine, oyun veri setleriyle
├── EDA/                    # Keşifsel veri analizi — mobil oyun & Steam katalog notebook'ları
├── LLM/                    # Prompt engineering — oyun endüstrisi senaryolarıyla
├── MachineLearning/         # Supervised + Unsupervised — 17 klasik ML egzersizi
├── NLP/                     # TF-IDF → Kelime Vektörleri → RNN → LSTM → Attention → Transformer
├── Python/                  # Temel Python & OOP alıştırmaları (veri seti gerektirmez)
└── SLM/                     # Türkçe küçük dil modeli + RAG — oyun dünyası Wikipedia korpusu
```

## BigData

| Proje | Bu Projede Ne Yapıldı | Veri Seti |
|---|---|---|
| [big-data-log-analytics](BigData/big-data-log-analytics) | Gerçek Steam kataloğundan (appid/isim/tür) referans alınarak 5 milyon satırlık sentetik oyun mağazası log verisi üretilip PySpark ile analiz edildi (durum kodları, en çok görüntülenen oyunlar, tür popülerliği vb.) | `fronkongames/steam-games-dataset` |

## DeepLearning

| Proje | Bu Projede Ne Yapıldı | Veri Seti |
|---|---|---|
| [CNN/01-fashion-mnist-cnn](DeepLearning/CNN/01-fashion-mnist-cnn) | Steam kapak görsellerinden 8 oyun türünü sınıflandıran CNN | `fronkongames/steam-games-dataset` |
| [CNN/02-goruntu-on-isleme](DeepLearning/CNN/02-goruntu-on-isleme) | Sentetik oyun ikonlarıyla (Coin/Shield/Potion/PowerUp) augmentation, PCA, t-SNE | *(sentetik — hiçbir veri seti ham piksel içermiyor)* |
| [CNN/03-opencv](DeepLearning/CNN/03-opencv) | Gerçek bir Steam kapak görseli üzerinde 10 bölümlük klasik görüntü işleme (renk uzayları, filtreleme, kenar tespiti, kontur) | `fronkongames/steam-games-dataset` |

## Docker

| Proje | Bu Projede Ne Yapıldı | Veri Seti |
|---|---|---|
| [1-single-container-xgboost](Docker/1-single-container-xgboost) | Tek Dockerfile ile paketlenmiş, oyunun ESRB içerik puanını tahmin eden XGBoost modeli | `rush4ratio/video-game-sales-with-ratings` |
| [2-docker-compose-3ML](Docker/2-docker-compose-3ML) | Video oyunu satış verisiyle 3 ML görevi (regresyon/sınıflandırma/kümeleme) Docker Compose ile paralel | `gregorut/videogamesales` |
| [3-microservices-ml-gateway](Docker/3-microservices-ml-gateway) | Steam oyun fiyatı tahmin eden 5 bağımsız Flask mikroservisi + API Gateway | `artermiloff/steam-games-dataset` |

## EDA

| Notebook | Bu Projede Ne Yapıldı | Veri Seti |
|---|---|---|
| [MobileStrategyGamesEDA.ipynb](EDA/MobileStrategyGamesEDA.ipynb) | Mobil strateji oyunlarının puan/fiyat/boyut/yaş sınırı üzerinden tek ve çift değişkenli analizi | `tristan581/17k-apple-app-store-strategy-games` |
| [SteamStoreGamesEDA.ipynb](EDA/SteamStoreGamesEDA.ipynb) | Steam oyun kataloğunda eksik veri temizliği + tek değişkenli analiz | `nikdavis/steam-store-games` |

## LLM

| Proje | Bu Projede Ne Yapıldı | Veri Seti |
|---|---|---|
| [05-prompt-engineering](LLM/05-prompt-engineering) | Zero-shot / few-shot / CoT / system prompt / temperature tekniklerinin oyun endüstrisi senaryolarıyla (churn, oyun yorumu, battle pass) karşılaştırılması | *(veri seti yok — gerçek zamanlı OpenAI API çağrıları)* |

## MachineLearning — Supervised

| # | Proje | Yöntem | Veri Seti |
|---|---|---|---|
| 01 | [pay-equity-analysis](MachineLearning/Supervised/01-linear-regresyon/pay-equity-analysis) | Linear Regression | `fronkongames/steam-games-dataset` |
| 01 | [superlig-goal-prediction](MachineLearning/Supervised/01-linear-regresyon/superlig-goal-prediction) | Basit + Çoklu Linear Regression | `rush4ratio/video-game-sales-with-ratings` |
| 02 | [churn-prediction](MachineLearning/Supervised/02-logistic-regresyon/churn-prediction) | Logistic Regression | `antonkozyriev/game-recommendations-on-steam` |
| 02 | [credit-scoring](MachineLearning/Supervised/02-logistic-regresyon/credit-scoring) | Logistic Regression | `fronkongames/steam-games-dataset` |
| 03 | [decision_tree_clinical](MachineLearning/Supervised/03-decision-tree/decision_tree_clinical) | Decision Tree | `fronkongames/steam-games-dataset` |
| 03 | [mobile-price-decision-tree](MachineLearning/Supervised/03-decision-tree/mobile-price-decision-tree) | Decision Tree | `tristan581/17k-apple-app-store-strategy-games` |
| 04 | [random-forest](MachineLearning/Supervised/04-random-forest) | Random Forest | `antonkozyriev/game-recommendations-on-steam` |
| 05 | [lightgbm](MachineLearning/Supervised/05-lightgbm) | LightGBM | `gregorut/videogamesales` |
| 06 | [svm](MachineLearning/Supervised/06-svm) | SVM (Linear + RBF) | `fronkongames/steam-games-dataset` |
| 07 | [knn](MachineLearning/Supervised/07-knn) | KNN | `tamber/steam-video-games` + `gregorut/videogamesales` |
| 08 | [naive-bayes](MachineLearning/Supervised/08-naive-bayes) | Naive Bayes + TF-IDF | *(veri seti yok — sentetik)* |
| — | [logreg-vs-randomforest-diabetes](MachineLearning/Supervised/ml-karsilastirma/logreg-vs-randomforest-diabetes) | LogReg vs Random Forest | `rush4ratio/video-game-sales-with-ratings` |
| — | [xgboost-vs-lightgbm](MachineLearning/Supervised/ml-karsilastirma/xgboost-vs-lightgbm) | XGBoost vs LightGBM | `tamber/steam-video-games` |

## MachineLearning — Unsupervised

| # | Proje | Yöntem | Veri Seti |
|---|---|---|---|
| 01 | [kmeans](MachineLearning/Unsupervised/01-kmeans) | K-Means + PCA | `tamber/steam-video-games` |
| 02 | [clustering-comparison](MachineLearning/Unsupervised/02-clustering-comparison) | K-Means, Hierarchical, DBSCAN, GMM | `fronkongames/steam-games-dataset` |
| 03 | [isolation-forest](MachineLearning/Unsupervised/03-isolation-forest) | Isolation Forest | `antonkozyriev/game-recommendations-on-steam` |
| 04 | [one-class-svm](MachineLearning/Unsupervised/04-one-class-svm) | One-Class SVM | *(veri seti yok — sentetik)* |

## NLP

| # | Proje | Yöntem | Veri Seti |
|---|---|---|---|
| 01 | [tf-idf](NLP/01-tf-idf) | TF-IDF + Logistic Regression + SVD | *(veri seti yok — sentetik)* |
| 02 | [word-embeddings/word2vec](NLP/02-word-embeddings/word2vec) | Skip-gram + Negative Sampling | `fronkongames/steam-games-dataset` |
| 02 | [word-embeddings/glove](NLP/02-word-embeddings/glove) | GloVe | `fronkongames/steam-games-dataset` |
| 02 | [word-embeddings/fasttext](NLP/02-word-embeddings/fasttext) | FastText (subword) | `fronkongames/steam-games-dataset` |
| 03 | [rnn](NLP/03-rnn) | Vanilla RNN | *(veri seti yok — sentetik)* |
| 04 | [lstm](NLP/04-lstm) | LSTM Zaman Serisi + Anomali | `antonkozyriev/game-recommendations-on-steam` |
| 05 | [attention](NLP/05-attention) | BiLSTM + Bahdanau Attention | *(veri seti yok — sentetik)* |
| 06 | [transformer](NLP/06-transformer) | Decoder-only Mini-GPT | `fronkongames/steam-games-dataset` |

## Python

Veri seti gerektirmeyen, saf Python/OOP alıştırmaları — aynı soru
sırası/sayısı, oyun dünyası örnekleriyle (Oyuncu, Araç, Karakter/Savaşçı/
Büyücü, OyuncuCüzdanı, OyunKütüphanesi vb.):
[python_baslangic.py](Python/python_baslangic.py) ·
[python_1_ders.py](Python/python_1_ders.py) ·
[temel_python.py](Python/temel_python.py) ·
[temel_python_2.py](Python/temel_python_2.py) ·
[python_class_sorular.py](Python/python_class_sorular.py)

## SLM

| Proje | Bu Projede Ne Yapıldı | Veri Kaynağı |
|---|---|---|
| [turkce-slm-rag](SLM/turkce-slm-rag) | Türkçe, karakter seviyeli, decoder-only Transformer tabanlı küçük dil modeli + TF-IDF tabanlı RAG — bilim/teknoloji yerine video oyunu konularında | Wikipedia API (video oyunu kategorileri) |

## Teknolojiler

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat&logo=docker&logoColor=white)
![Apache Spark](https://img.shields.io/badge/Apache%20Spark-E25A1C?style=flat&logo=apachespark&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-000000?style=flat&logo=flask&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-black?style=flat)
![LightGBM](https://img.shields.io/badge/LightGBM-black?style=flat)
![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=flat&logo=kaggle&logoColor=white)

<p align="center"><i>Öğrenme sürecinde egzersiz olarak hazırlanmış bir versiyondur.</i></p>
