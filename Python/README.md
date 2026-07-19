# Python Programlama (Oyun Dünyası Versiyonu)

Temel Python/OOP alıştırma
defterlerinin oyun dünyası örnekleriyle hazırlanmış kişisel alıştırma
versiyonlarını içeren klasör. Bu klasördeki dosyalar **veri seti
gerektirmez** — saf Python sözdizimi ve OOP pratiğidir. Sadece örnek/soru
bağlamları (kişi/araba/öğrenci yerine oyun karakteri/araç/oyuncu, kitap
yerine oyun, banka hesabı yerine oyuncu
cüzdanı vb.) oyun temasına uyarlanmıştır — soru sırası, zorluk seviyesi ve
kapsanan konular birebir korunmuştur.

## İçerik

| Dosya | Konu | Seviye |
|-------|------|--------|
| [python_baslangic.py](python_baslangic.py) | print, değişkenler, veri tipleri, koşullar, döngüler, listeler, fonksiyonlar | 🟢 Başlangıç |
| [python_1_ders.py](python_1_ders.py) | String işlemleri, liste metodları, math kütüphanesi, sözlükler, class'a giriş, kalıtım, kapsülleme, dunder metodlar | 🟡 Temel |
| [temel_python.py](temel_python.py) | Operatörler, f-string, list comprehension, dictionary, hata yönetimi, kapsamlı uygulama | 🟡 Temel |
| [temel_python_2.py](temel_python_2.py) | Tuple, set, enumerate, zip, lambda, recursive fonksiyon, dosya işlemleri | 🟠 Orta |
| [python_class_sorular.py](python_class_sorular.py) | Class değişkenleri, @property, @classmethod, @staticmethod, iterator class, Mixin pattern, dunder metodlar (11 soru, her biri kendi `assert` testleriyle) | 🔴 İleri |

## Oyun Dünyası Uyarlaması — Örnek Eşleştirmeler

| Genel Bağlam | Oyun Versiyonu |
|---|---|
| Kişi (ad, soyad, yaş) | Oyuncu (ad, unvan, seviye) |
| Araba (marka, model, yıl) | Araç (üretici, model, yıl) — yarış oyunu aracı |
| Dikdörtgen (en, boy) | OyunHaritası (en, boy) — harita/oda boyutu |
| Hayvan → Köpek/Kedi (kalıtım) | Karakter → Savaşçı/Büyücü (kalıtım) |
| BankaHesabı (bakiye) | OyuncuCüzdanı (altın) |
| Kitap (ad, yazar) | Oyun (ad, geliştirici) |
| Kütüphane (ödünç/iade) | OyunKütüphanesi (kirala/iade et) |
| Çalışan (maaş) | Yarışmacı (puan) |
| Para (miktar, TL) | Ödül (miktar, Altın/Elmas) |

## Kurulum

```bash
python python_baslangic.py
python python_class_sorular.py   # tüm sorular kendi assert testleriyle doğrulanır
```

Python 3.8+ yeterli, harici kütüphane gerekmez.

<p align="center"><i>Öğrenme sürecinde egzersiz olarak hazırlanmış bir versiyondur.</i></p>
