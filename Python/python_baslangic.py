# -*- coding: utf-8 -*-
"""python_baslangic.py (Oyun Versiyonu)

ÖĞRENME NOTU (yönlendirme):
  Bu dosyanın amacı, Python temellerini (print, degiskenler, veri tipleri,
  aritmetik, string, kosullar, dongu, liste, fonksiyon) kademeli bir soru
  dizisiyle pratik etmektir. Bu dosya bir veri seti KULLANMIYOR, saf
  Python sozdizimi alistirmasidir; ornek/soru baglamlari (karakter/sunucu,
  envanter esyasi vb.) oyun temasina uyarlanmistir.

# 🐍 Python Başlangıç Defteri (Oyun Versiyonu)
### Adım adım, oyun dünyası örnekleriyle Python öğrenin!
---

## 1️⃣ print() Fonksiyonu
"""

# Örnek
print("Merhaba, oyuncu!")
print(42)
print("Sonuç:", 10 + 5)

"""### ✏️ Soru 1: Ekrana kendi karakter adınızı yazdırın."""

# Buraya yazın:

"""### ✏️ Soru 2: `print()` kullanarak 3 farklı satırda şunları yazdırın:
- Karakter adınız
- Seviyeniz
- Sunucu bölgeniz
"""

# Buraya yazın:

"""---
## 2️⃣ Değişkenler
"""

# Örnek
karakter_adi = "Kaan"
can_puani = 100
print(karakter_adi)
print(can_puani)

"""### ✏️ Soru 3: `sunucu` adında bir değişken oluşturun ve değerini yazdırın."""

# Buraya yazın:

"""### ✏️ Soru 4: `hasar1 = 8` ve `hasar2 = 3` değişkenlerini tanımlayın. Toplamlarını `toplam_hasar` değişkenine atayın ve yazdırın."""

# Buraya yazın:
hasar1 = 8
hasar2 = 3

"""---
## 3️⃣ Veri Tipleri
"""

# Örnek
print(type("oyun"))      # str
print(type(10))          # int
print(type(3.14))        # float
print(type(True))        # bool

"""### ✏️ Soru 5: Aşağıdaki 4 değişkenin tipini `type()` ile yazdırın."""

oyun_adi = "efsane"
skor = 100
hiz_carpani = 9.99
gece_modu = False

# Buraya yazın:

"""### ✏️ Soru 6: `"25"` metnini sayıya (`int`, seviye), `7` sayısını metne (`str`, skor_metni) çevirin ve yazdırın."""

# Buraya yazın:
metin = "25"
sayi = 7

"""---
## 4️⃣ Aritmetik İşlemler
"""

# Örnek
print(10 + 3)   # 13
print(10 - 3)   # 7
print(10 * 3)   # 30
print(10 / 3)   # 3.333...
print(10 // 3)  # 3  (tam bölme)
print(10 % 3)   # 1  (kalan)
print(2 ** 4)   # 16 (üs)

"""### ✏️ Soru 7: `altin = 15`, `oyuncu_sayisi = 4` için toplama, çıkarma, çarpma ve bölme işlemlerini yapın."""

altin = 15
oyuncu_sayisi = 4
# Buraya yazın:

"""### ✏️ Soru 8: 17 canavarı 5 kişilik gruplara böldüğünüzde kaç canavar dışarıda kalır? `%` operatörü ile bulun."""

# Buraya yazın:

"""---
## 5️⃣ String (Metin) İşlemleri
"""

# Örnek
metin = "zafer"
print(metin.upper())       # ZAFER
print(metin.lower())       # zafer
print(len(metin))          # 5
print(metin[0])            # z  (ilk harf)
print(metin[-1])           # r  (son harf)

"""### ✏️ Soru 9: `takim_adi = "efsane"` değişkeni için:
- Büyük harfle yazdırın
- Kaç harf olduğunu yazdırın
- İlk harfini yazdırın
"""

takim_adi = "efsane"
# Buraya yazın:

"""### ✏️ Soru 10: İki metni birleştirin. `unvan = "Buyucu"` ve `isim = "Kaan"` → `"Buyucu Kaan"` yazdırın."""

unvan = "Buyucu"
isim = "Kaan"
# Buraya yazın:

"""### ✏️ Soru 11: f-string kullanarak şu çıktıyı üretin: `"Merhaba, ben Kaan. Seviyem 25."`"""

isim = "Kaan"
seviye = 25
# Buraya yazın:

"""---
## 6️⃣ Koşullar (if / else)
"""

# Örnek
can = 10
if can > 0:
    print("Hayatta")
else:
    print("Elendi")

"""### ✏️ Soru 12: `seviye = 16` değişkeni için; 18'den büyükse `"Yetiskin sunucusuna girebilir"`, değilse `"Giremez"` yazdırın."""

seviye = 16
# Buraya yazın:

"""### ✏️ Soru 13: `skor = 7` değişkeni için; çift mi tek mi olduğunu yazdırın. (`%` operatörü kullanın)"""

skor = 7
# Buraya yazın:

"""### ✏️ Soru 14: `puan = 65` için rütbe adını yazdırın:
- 90 ve üstü → `"Elmas"`
- 75 ve üstü → `"Altin"`
- 60 ve üstü → `"Gumus"`
- 60 altı → `"Bronz"`
"""

puan = 65
# Buraya yazın:

"""---
## 7️⃣ for Döngüsü
"""

# Örnek
for i in range(5):
    print(i)   # 0, 1, 2, 3, 4

for i in range(1, 6):
    print(i)   # 1, 2, 3, 4, 5

"""### ✏️ Soru 15: 1'den 10'a kadar olan seviyeleri yazdırın."""

# Buraya yazın:

"""### ✏️ Soru 16: 1'den 10'a kadar olan seviyelerin toplam deneyim puanını hesaplayın."""

# Buraya yazın:
toplam_xp = 0

"""### ✏️ Soru 17: Aşağıdaki envanterdeki her eşyayı ayrı satırda yazdırın."""

envanter = ["kilic", "kalkan", "iksir", "anahtar"]
# Buraya yazın:

"""---
## 8️⃣ Listeler
"""

# Örnek
skorlar = [10, 20, 30, 40, 50]
print(skorlar[0])       # 10 (ilk eleman)
print(skorlar[-1])      # 50 (son eleman)
print(len(skorlar))     # 5
skorlar.append(60)      # sona ekle
print(skorlar)

"""### ✏️ Soru 18: Aşağıdaki listede:
- İlk elemanı yazdırın
- Son elemanı yazdırın
- Listenin uzunluğunu yazdırın
"""

karakter_siniflari = ["savasci", "buyucu", "okcu", "suikastci", "sifaci"]
# Buraya yazın:

"""### ✏️ Soru 19: Envantere `"zirh"` ekleyin, sonra `"kalkan"` ı silin ve listeyi yazdırın."""

envanter = ["kilic", "kalkan", "iksir", "anahtar"]
# Buraya yazın:

"""### ✏️ Soru 20: Skorlar listesindeki en büyük, en küçük değeri ve toplamı yazdırın."""

skorlar = [5, 12, 3, 8, 19, 1, 7]
# İpucu: max(), min(), sum() fonksiyonlarını kullanın
# Buraya yazın:

"""---
## 9️⃣ Fonksiyonlar
"""

# Örnek
def topla(a, b):
    return a + b

sonuc = topla(3, 5)
print(sonuc)  # 8

"""### ✏️ Soru 21: İki sayıyı çarpan `altin_carp(a, b)` fonksiyonu yazın ve `altin_carp(4, 6)` ile test edin."""

# Buraya yazın:

"""### ✏️ Soru 22: Bir oyuncu adı alıp `"Hoş geldin, [oyuncu_adi]!"` şeklinde yazdıran `karsila(oyuncu_adi)` fonksiyonu yazın."""

# Buraya yazın:

"""### ✏️ Soru 23: Bir seviyenin çift mi tek mi olduğunu döndüren `seviye_cift_mi(seviye)` fonksiyonu yazın. `True` veya `False` dönsün."""

# Buraya yazın:
def seviye_cift_mi(seviye):
    pass

print(seviye_cift_mi(4))   # True
print(seviye_cift_mi(7))   # False

"""---
## 🔟 Hepsini Birleştir!

### ✏️ Soru 24:
Bir oyun mağazası sepeti uygulaması yapın:
- `sepet = []` listesi oluşturun
- 3 oyun ekleyin (`.append()`)
- `for` döngüsü ile oyunları numaralı yazdırın
- Kaç oyun olduğunu yazdırın
"""

# Buraya yazın:
sepet = []

"""### ✏️ Soru 25:
Bir listede verilen oyuncu puanlarının ortalamasını hesaplayan `ortalama(puanlar)` fonksiyonu yazın.
- `sum()` ve `len()` kullanabilirsiniz
- `[70, 85, 90, 60, 75]` listesiyle test edin
"""

# Buraya yazın:
def ortalama(puanlar):
    pass

test_puanlari = [70, 85, 90, 60, 75]
# print(ortalama(test_puanlari))  # 76.0 olmalı

"""---
## 📊 İlerleme Tablosu

| Konu | Sorular | Bitti mi? |
|------|---------|----------|
| print() | 1, 2 | ☐ |
| Değişkenler | 3, 4 | ☐ |
| Veri Tipleri | 5, 6 | ☐ |
| Aritmetik | 7, 8 | ☐ |
| String | 9, 10, 11 | ☐ |
| if / else | 12, 13, 14 | ☐ |
| for Döngüsü | 15, 16, 17 | ☐ |
| Listeler | 18, 19, 20 | ☐ |
| Fonksiyonlar | 21, 22, 23 | ☐ |
| Birleştirici | 24, 25 | ☐ |

> 💡 Hata almaktan korkmayın — hatalar öğrenmenin en iyi yoludur!
"""
