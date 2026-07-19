# SLM (Small Language Model) Çalışmaları — Oyun Dünyası Versiyonu

Türkçe, karakter seviyeli, decoder-only Transformer tabanlı küçük bir dil
modeli (SLM) ve TF-IDF tabanlı RAG uygulamasını içerir.

## İçerik

### turkce-slm-rag/

Wikipedia'dan toplanan Türkçe **video oyunu** makaleleri üzerinde eğitilen,
karakter seviyeli, Transformer tabanlı (decoder-only) bir küçük dil modeli.
İki bölümden oluşur:

- Transformer mimarisiyle metin üretimi (bir sonraki karakteri tahmin etme)
- Retrieval-Augmented Generation (RAG) ile kaynak gösteren soru-cevap sistemi

**Bu projenin oyun dünyasına uyarlanışı diğer kategorilerden farklıdır:**
proje mimarisi tamamen `configs/config.yaml` dosyasındaki Wikipedia
kategori/makale listesine göre çalışır (config-driven) — hiçbir Python
kaynak dosyasında konuya özel mantık yok. Bu yüzden tek gerekli değişiklik
config'teki kategori listesinin bilim/teknoloji (yapay zeka, fizik,
matematik, biyoloji, uzay, kimya, mühendislik) yerine **oyun dünyası**
konularıyla (video oyunu temelleri, oyun türleri, konsol/donanım, efsanevi
oyunlar, oyun şirketleri, elektronik spor, oyun tarihi, oyun platformları)
değiştirilmesiydi — `src/` altındaki hiçbir dosyada mantık değişikliği
yapılmadı, sadece birkaç örnek prompt metni güncellendi.

Veri, Wikipedia'nin resmi API'si ile 8 kategori altında ~35 makaleden
toplanacak şekilde yapılandırılmıştır.

Detaylı kurulum ve kullanım için: [`turkce-slm-rag/README.md`](turkce-slm-rag/README.md)

## Kullanılan Teknolojiler

- PyTorch (model mimarisi ve eğitim)
- Wikipedia API (veri toplama)
- scikit-learn (TF-IDF tabanlı embedding)
- Gradio (demo arayüzü)
- pytest (testler)

<p align="center"><i>Öğrenme sürecinde egzersiz olarak hazırlanmış bir versiyondur.</i></p>
