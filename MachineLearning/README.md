# Machine Learning (Oyun Dünyası Versiyonu)

Klasik ML tekniklerinin oyun dünyası veri setleriyle hazırlanmış kişisel
alıştırma versiyonlarını içeren klasör. Mümkün olan her yerde **gerçek
Kaggle oyun veri setleri** kullanıldı — sadece 2 istisnada (08-naive-bayes,
Unsupervised/04-one-class-svm) hiçbir Kaggle veri seti uygun olmadığından
sentetik/simüle veri üretilip temayla uyumlu hale getirildi (her ikisinde
de gerekçe ilgili README'de açıklanmıştır).

## Proje Haritası — Supervised (Denetimli Öğrenme)

| # | Klasör | Yöntem | Veri Seti (Kaggle) |
|---|---|---|---|
| 01 | [linear-regresyon/pay-equity-analysis](./Supervised/01-linear-regresyon/pay-equity-analysis) | Linear Regression | `fronkongames/steam-games-dataset` — Indie vs AAA fiyat farkı analizi |
| 01 | [linear-regresyon/superlig-goal-prediction](./Supervised/01-linear-regresyon/superlig-goal-prediction) | Basit + Çoklu Linear Regression | `rush4ratio/video-game-sales-with-ratings` — User_Score → Critic_Score tahmini |
| 02 | [logistic-regresyon/churn-prediction](./Supervised/02-logistic-regresyon/churn-prediction) | Logistic Regression | `antonkozyriev/game-recommendations-on-steam` — tavsiye etmeme (churn analoğu) tahmini |
| 02 | [logistic-regresyon/credit-scoring](./Supervised/02-logistic-regresyon/credit-scoring) | Logistic Regression | `fronkongames/steam-games-dataset` — kritik beğeni tahmini |
| 03 | [decision-tree/decision_tree_clinical](./Supervised/03-decision-tree/decision_tree_clinical) | Decision Tree | `fronkongames/steam-games-dataset` — içerik uygunluk riski sınıflandırması |
| 03 | [decision-tree/mobile-price-decision-tree](./Supervised/03-decision-tree/mobile-price-decision-tree) | Decision Tree | `tristan581/17k-apple-app-store-strategy-games` — mobil oyun fiyat segmenti |
| 04 | [random-forest](./Supervised/04-random-forest) | Random Forest | `antonkozyriev/game-recommendations-on-steam` — şüpheli yorum tespiti (dengesiz sınıflar) |
| 05 | [lightgbm](./Supervised/05-lightgbm) | LightGBM | `gregorut/videogamesales` — bestseller tahmini (early stopping) |
| 06 | [svm](./Supervised/06-svm) | SVM (Linear + RBF) | `fronkongames/steam-games-dataset` — Indie/AAA teşhisi |
| 07 | [knn](./Supervised/07-knn) | KNN | `tamber/steam-video-games` + `gregorut/videogamesales` — item-based oyun öneri sistemi |
| 08 | [naive-bayes](./Supervised/08-naive-bayes) | Naive Bayes + TF-IDF | *(veri seti yok — sentetik oyun yorumu metni)* |
| — | [ml-karsilastirma/logreg-vs-randomforest-diabetes](./Supervised/ml-karsilastirma/logreg-vs-randomforest-diabetes) | LogReg vs Random Forest | `rush4ratio/video-game-sales-with-ratings` — yüksek puanlı oyun tahmini |
| — | [ml-karsilastirma/xgboost-vs-lightgbm](./Supervised/ml-karsilastirma/xgboost-vs-lightgbm) | XGBoost vs LightGBM | `tamber/steam-video-games` — satın alma sonrası oynanmama (CTR analoğu) |

## Proje Haritası — Unsupervised (Denetimsiz Öğrenme)

| # | Klasör | Yöntem | Veri Seti (Kaggle) |
|---|---|---|---|
| 01 | [kmeans](./Unsupervised/01-kmeans) | K-Means + PCA | `tamber/steam-video-games` — oyuncu segmentasyonu |
| 02 | [clustering-comparison](./Unsupervised/02-clustering-comparison) | K-Means, Hierarchical, DBSCAN, GMM | `fronkongames/steam-games-dataset` — oyun pazarı segmentasyonu |
| 03 | [isolation-forest](./Unsupervised/03-isolation-forest) | Isolation Forest | `antonkozyriev/game-recommendations-on-steam` — etiketsiz şüpheli yorum tespiti |
| 04 | [one-class-svm](./Unsupervised/04-one-class-svm) | One-Class SVM | *(veri seti yok — sentetik oyun sunucusu trafiği)* |

---

Her proje README'sinde veri seti seçim gerekçesi ve çalıştırma adımları
(Kaggle kimlik doğrulaması dahil) ayrıntılı olarak açıklanmıştır. Hiçbir
script bu oturumda çalıştırılmamıştır
— syntax kontrolünden geçmiştir, sonuçlar/görseller kullanıcı çalıştırınca
oluşacaktır.

## 🛠️ Kullanılan Teknolojiler

`Python` · `scikit-learn` · `XGBoost` · `LightGBM` · `pandas` · `numpy` · `matplotlib` · `seaborn` · `kagglehub`

<p align="center"><i>Öğrenme sürecinde egzersiz olarak hazırlanmış bir versiyondur.</i></p>
