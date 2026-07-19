# Turkce Kucuk Dil Modeli (SLM) + RAG — Oyun Dunyasi Versiyonu

## 🎓 Bu Proje Hakkında

Bu çalışmanın amacı, Türkçe karakter-seviyeli, decoder-only Transformer
tabanlı küçük bir dil modeli (SLM) ve TF-IDF tabanlı bir RAG sistemini
uçtan uca kurmaktır. Proje mimarisi/kodu **tamamen config-driven** olduğu
için (tüm script'ler `configs/config.yaml`'daki Wikipedia kategori/makale
listesini okur, hiçbir konu-özel mantık kodda gömülü değildir), konu
seçimi **`configs/config.yaml`'daki kategori listesinin oyun dünyası
konularına ayarlanmasıyla** yapılır — `src/` altındaki Python dosyalarında
mantık değişikliği bulunmaz.

Wikipedia'dan toplanan Turkce video oyunu makaleleri uzerinde egitilen,
karakter seviyeli, Transformer tabanli (decoder-only) bir kucuk dil modeli
(SLM) projesi. Iki fazdan olusur:

- **Faz 1**: Transformer mimarisiyle metin uretimi (bir sonraki karakteri tahmin etme)
- **Faz 2**: Retrieval-Augmented Generation (RAG) ile kaynak gosteren soru-cevap sistemi

## Proje Yapisi

```
turkce-slm-rag/
├── configs/config.yaml       Tum hiperparametreler ve ayarlar
├── data/                     Ham ve islenmis veri (git'e dahil degil)
├── src/                      Kaynak kod
│   ├── data_collection.py    Wikipedia'dan veri cekme
│   ├── preprocessing.py      Temizleme + tokenizer
│   ├── dataset.py            PyTorch Dataset/DataLoader
│   ├── models/transformer.py  Decoder-only Transformer mimarisi
│   ├── train.py               Egitim script'i
│   ├── generate.py            Metin uretim script'i
│   ├── chunking.py            RAG icin pasajlama (Faz 2)
│   ├── embedding.py           TF-IDF embedding (Faz 2)
│   ├── retrieval.py           Pasaj arama (Faz 2)
│   └── rag_pipeline.py        Bul + uret akisi (Faz 2)
├── demo/app.py                Gradio arayuzu
├── tests/                     Unit testler
├── figures/                   Egitim grafikleri (git'e dahil degil)
└── checkpoints/                Egitilmis model dosyalari (git'e dahil degil)
```

## Kurulum

```bash
git clone <repo-url>
cd turkce-slm-rag

# (opsiyonel ama onerilir) sanal ortam
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS/Linux

pip install -r requirements.txt
```

## Kullanim

Not: `generate.py`, `chunking.py`, `embedding.py`, `rag_pipeline.py` VE
`train.py` dosyalari `src` paketinin icini (`from src.dataset import ...`
gibi) kullandigi icin `python -m src.<dosya_adi>` seklinde, yani **modul
olarak** calistirilmalidir. Sadece `data_collection.py` ve
`preprocessing.py` bagimsiz oldugundan direkt `python src/<dosya_adi>.py`
ile calistirilabilir.

### Faz 1: Veri Toplama ve Egitim

```bash
# 1) Wikipedia'dan veri cek
python src/data_collection.py --config configs/config.yaml

# 2) Veriyi temizle, tokenizer olustur
python src/preprocessing.py --config configs/config.yaml

# 3) Modeli egit
python -m src.train --config configs/config.yaml

# 4) Metin uret
python -m src.generate --prompt "Video oyunu" --uzunluk 300
```

### Faz 2: RAG Kurulumu

```bash
# 5) Makaleleri pasajlara bol
python -m src.chunking --config configs/config.yaml

# 6) Pasajlari vektorlestir
python -m src.embedding --config configs/config.yaml
```

RAG pipeline'ini Python icinden kullanmak icin:

```python
from src.rag_pipeline import RagPipeline

pipeline = RagPipeline("configs/config.yaml", "checkpoints/slm_model.pt")
sonuc = pipeline.soru_sor("Video oyunu nedir?")

print(sonuc["cevap"])
print(sonuc["kaynaklar"])
```

### Demo Arayuzu

```bash
python demo/app.py
```

Tarayicida acilan link (orn. `http://127.0.0.1:7860`) uzerinden iki sekmeli
arayuze erisilir: metin tamamlama (Faz 1) ve RAG soru-cevap (Faz 2).

## Model Mimarisi

Decoder-only, GPT tarzi bir Transformer kullanilir: multi-head self-attention
+ feed-forward bloklari, karakter seviyeli embedding ve pozisyon embedding'i
ile. Katman sayisi, embedding boyutu ve attention head sayisi `config.yaml`
icinde `model` bolumunden ayarlanir.

## Veri Kaynagi

Veri, Wikipedia'nin resmi API'si (`wikipedia-api` kutuphanesi) ile toplanir,
HTML kazima (scraping) yapilmaz. `configs/config.yaml` icindeki kategori/makale
listesi degistirilerek farkli konu alanlarina genisletilebilir. Su an 8
kategori altinda ~35 makale (video oyunu temelleri, oyun turleri, konsol/
donanim, efsanevi oyunlar, oyun sirketleri, elektronik spor, oyun tarihi,
oyun platformlari) tanimlidir. Bazi basliklar Turkce Wikipedia'da farkli
bir adla ya da hic bulunmayabilir - `data_collection.py` bulunamayan
basliklari otomatik atlar, bu beklenen bir davranistir.

## 📊 Sonuçlar (gerçek çalıştırma — uçtan uca Faz 1 + Faz 2)

**Veri toplama:** 34/38 Wikipedia makalesi başarıyla çekildi (221.165
karakter temiz metin, 161 karakterlik vocab).

**Eğitim (Faz 1):** 20 epoch'ta eğitim kaybı **3.57 → 2.04**, doğrulama
kaybı **2.90 → 2.02**'ye düştü (overfitting yok, doğrulama kaybı eğitim
kaybını yakından takip ediyor).

**Üretilen örnek metin** ("Video oyunu" prompt'undan):
> *"Video oyunu için yaplan konyamarı için yanından ve göreliştir. Nintendo Ans verdribulanma oyuncualar, bir azarirler moduştur..."*

Gerçek Türkçe ekler ve hece kalıpları öğrenilmiş (küçük model boyutu ve
kısa eğitim nedeniyle tam anlamlı cümleler henüz oluşmuyor — beklenen bir
sonuç).

**RAG (Faz 2):** 187 pasaj oluşturuldu, TF-IDF embedding matrisi (187×5000).
"Minecraft nedir?" sorgusu test edildi — **retrieval kısmı mükemmel çalıştı**
(en alakalı 3 pasaj, skor 0.52/0.30/0.29 ile doğru şekilde bulundu), ama
**üretim kısmı** (küçük SLM ile bulunan bağlamdan cevap yazma) model
boyutunun küçüklüğü nedeniyle tutarlı bir cevap üretemedi — bu, RAG
mimarisinin "arama" ve "üretim" bileşenlerinin bağımsız olarak
değerlendirilebileceğini gösteren öğretici bir gözlem.

> **Not (gerçek bir performans hatası bulundu ve düzeltildi):**
> `src/dataset.py`'deki `MetinDataset`, başlangıçta HER karakter
> pozisyonunu ayrı bir eğitim örneği olarak kullanıyordu (stride=1) —
> bu, 221K karakterlik bir korpuste epoch başına ~199.000 örnek
> üretip eğitimi pratik olmayacak kadar yavaşlatıyordu (~100+ dakika
> tahmini). Standart karakter-seviyeli dil modeli pratiğine uygun
> şekilde örtüşmeyen bloklara (stride=blok_uzunlugu) geçilerek eğitim
> süresi ~15 dakikaya indirildi.

## Testler

```bash
pytest tests/
```

`test_tokenizer.py`: karakter tokenizer'in encode/decode dogrulugunu ve
Turkce karakter destegini test eder.
`test_retrieval.py`: RAG icin metin parcalama (chunking) mantigini test eder.

## Gelistirme Fikirleri

- Karakter seviyesi yerine BPE/subword tokenizer
- TF-IDF yerine sentence-transformers ile daha guclu embedding
- Daha genis Wikipedia korpusu (40 makale -> 500+)
- Instruction-tuning ile daha dogal soru-cevap davranisi

## Lisans

MIT
