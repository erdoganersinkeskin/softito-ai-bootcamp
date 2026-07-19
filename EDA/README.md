# EDA (Oyun Dünyası Versiyonu)

Keşifsel veri analizi (EDA)
egzersizlerinin oyun dünyası veri setleriyle hazırlanmış kişisel alıştırma
versiyonlarını içeren klasör.

## Notebook'lar

| Notebook | Veri Seti | Ne Yapıyor |
|---|---|---|
| [MobileStrategyGamesEDA.ipynb](MobileStrategyGamesEDA.ipynb) | Kaggle `tristan581/17k-apple-app-store-strategy-games` | Mobil strateji oyunlarının puan/fiyat/boyut/yaş sınırı üzerinden tek ve çift değişkenli analizi |
| [SteamStoreGamesEDA.ipynb](SteamStoreGamesEDA.ipynb) | Kaggle `nikdavis/steam-store-games` | Steam oyun kataloğunda eksik veri temizliği + tek değişkenli analiz (çift değişkenli analiz başlığında yarım bırakılmıştır) |

## Çalıştırma

Her iki notebook da hücre içinde `kagglehub` ile veri setini otomatik indirir
(Kaggle kimlik doğrulaması gerekir — `kaggle.json` → `~/.kaggle/kaggle.json`).
Colab yerine yerel Jupyter/VS Code'da çalıştırmak için:

```bash
pip install pandas numpy matplotlib seaborn scipy kagglehub jupyter
jupyter notebook
```

## 📊 Sonuçlar (gerçek çalıştırma — hücre çıktıları notebook'ların içinde saklı)

Her iki notebook uçtan uca çalıştırıldı; grafikler ve tablolar artık
notebook dosyalarının kendi hücre çıktılarında görülebilir (dosyaları açıp
inceleyin).

**SteamStoreGamesEDA** — `nikdavis/steam-store-games` (27.075 oyun, `steam.csv`):
- `release_year` dağılımı: ortalama 2016.5, medyan 2017, çarpıklık -2.04
  (dağılım sola çarpık — kataloğun büyük kısmı 2015 sonrasına ait).
- Eksik değer temizliği sonrası `publisher` kolonundaki 14 boş satır giderildi.

**MobileStrategyGamesEDA** — `tristan581/17k-apple-app-store-strategy-games`
(17.007 oyun, `appstore_games.csv`):
- `Average User Rating`: ortalama 4.06, medyan 4.50 — kullanıcı puanları
  belirgin şekilde yüksek uçta yoğunlaşıyor (çarpıklık -1.16).
- `Subtitle` kolonunda %69.07 eksik değer (çoğu oyunun alt başlığı yok).

## 🛠️ Kullanılan Teknolojiler

`Python` · `pandas` · `numpy` · `matplotlib` · `seaborn` · `scipy` · `kagglehub`

<p align="center"><i>Keşifsel veri analizi pratiği amaçlı, öğrenme sürecinde egzersiz olarak hazırlanmış bir versiyondur.</i></p>
