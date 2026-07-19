# NLP (Oyun Dünyası Versiyonu)

Kümülatif NLP serisinin
(TF-IDF → Word Embeddings → RNN → LSTM → Attention → Transformer) oyun
dünyası veri setleri/senaryolarıyla hazırlanmış kişisel alıştırma
versiyonlarını içeren klasör.

## Proje Haritası

| # | Klasör | Yöntem | Veri Kaynağı |
|---|---|---|---|
| 01 | [tf-idf](01-tf-idf) | TF-IDF + Logistic Regression + SVD | *(veri seti yok — sentetik oyun yorumu; hiçbir Kaggle veri seti ham metin içermiyor)* |
| 02 | [word-embeddings/word2vec](02-word-embeddings/word2vec) | Skip-gram + Negative Sampling | `fronkongames/steam-games-dataset` — "About the game" açıklamaları korpusu |
| 02 | [word-embeddings/glove](02-word-embeddings/glove) | GloVe (count-based) | Aynı oyun açıklaması korpusu (word2vec ile adil kıyas) |
| 02 | [word-embeddings/fasttext](02-word-embeddings/fasttext) | FastText (subword) | Aynı oyun açıklaması korpusu |
| 03 | [rnn](03-rnn) | Vanilla RNN | *(veri seti yok — sentetik oyun sunucusu oturum logu; Kaggle-dışı bir veri türü)* |
| 04 | [lstm](04-lstm) | LSTM Zaman Serisi + Anomali | `antonkozyriev/game-recommendations-on-steam` — günlük yorum hacmi zaman serisi (gerçek veri) |
| 05 | [attention](05-attention) | BiLSTM + Bahdanau Attention | *(veri seti yok — sentetik uzun formlu oyun yorumu; Kaggle-dışı bir veri türü)* |
| 06 | [transformer](06-transformer) | Decoder-only Mini-GPT (karakter düzeyi) | `fronkongames/steam-games-dataset` — aynı oyun açıklaması korpusu (karakter düzeyinde, noktalama korunmuş) |

## Veri Seti Stratejisi Özeti

Bu kategori diğerlerinden farklı olarak çoğunlukla **büyük metin/zaman
serisi korpusu** gerektiren tekniklerden oluşuyor. 9 Kaggle veri setinin
8'i tamamen tablo verisi olduğundan:

- **Gerçek veri kullanılabilen yerler:** `fronkongames/steam-games-dataset`
  içindeki "About the game" serbest metin kolonu (word embeddings +
  transformer projelerinde) ve `antonkozyriev/game-recommendations-on-steam`
  içindeki gerçek yorum tarihleri (LSTM zaman serisi projesinde).
- **Hiçbir veri setinin uymadığı yerler** (01-tf-idf, 03-rnn, 05-attention):
  gerekli veri türü (gerçek metin/log verisi) paylaşılan 9 Kaggle veri
  setinde bulunmadığından, bu 3 projede aynı yapıyı taklit eden
  **sentetik, oyun temalı** veri üretiliyor.

## 🛠️ Kullanılan Teknolojiler

`Python` · `PyTorch` · `scikit-learn` · `gensim` · `fasttext` · `pandas` · `matplotlib` · `seaborn` · `kagglehub`

<p align="center"><i>Öğrenme sürecinde egzersiz olarak hazırlanmış bir versiyondur.</i></p>
