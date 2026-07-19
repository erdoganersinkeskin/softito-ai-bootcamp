# -*- coding: utf-8 -*-
"""temel_python.py (Oyun Versiyonu)

ÖĞRENME NOTU (yönlendirme):
  Bu dosyanın amacı, 18 soruluk bir alistirma dizisiyle (degiskenler,
  operatorler, string, kosullar, donguler, listeler, sozlukler,
  fonksiyonlar, hata yonetimi, butunlestirici gorev defteri projesi)
  Python pratiği yapmaktır. Veri seti gerektirmez.

# 🐍 Temel Python Komutları - Alıştırma Defteri (Oyun Versiyonu)

Bu defterde Python'un temel konularını oyun dünyası örnekleriyle
öğrenecek ve pratik sorularla kendinizi geliştireceksiniz.

---

## 1️⃣ Değişkenler ve Veri Tipleri
"""



"""### 🔸 Soru 1:
Aşağıdaki değişkenleri tanımlayın ve her birinin tipini `type()` fonksiyonu ile yazdırın:
- `sunucu`: "Efsane-1" (string)
- `aktif_oyuncu`: 57000 (integer)
- `gecikme_ms`: 23.5 (float)
- `bakim_modu`: False (boolean)
"""



# Cevabınızı buraya yazın:

sunucu = "Efsane-1"
aktif_oyuncu = 57000
gecikme_ms = 23.5
bakim_modu = False

print(type(sunucu))
print(type(aktif_oyuncu))
print(type(gecikme_ms))
print(type(bakim_modu))

"""### 🔸 Soru 2:
Bir değişkeni `int`'ten `float`'a, `float`'tan `str`'e dönüştürün ve sonuçları yazdırın.

Örnek: `seviye = 42` → `float(seviye)` → `str(float(seviye))`
"""

# Cevabınızı buraya yazın:

seviye = 8

# int'ten float'a
seviye_float = float(seviye)
print(seviye_float)

# float'tan str'e
seviye_str = str(seviye_float)
print(seviye_str)

"""---
## 2️⃣ Operatörler
"""

# Örnek: Aritmetik operatörler
a = 10
b = 3

"""### 🔸 Soru 3:
Bir dalgada beliren canavar sayısı: `x = 17`, savaşan oyuncu sayısı: `y = 5`
- Toplamlarını, farklarını ve çarpımlarını yazdırın.
- 17'nin 5'e bölümünden kalanı yazdırın.
- 2'nin 8. kuvvetini (büyü gücü çarpanı) hesaplayın.
"""

# Cevabınızı buraya yazın:
x = 17
y = 5

# 1. Toplam, fark ve çarpım
print(f"Toplam: {x + y}")    # 22
print(f"Fark: {x - y}")      # 12
print(f"Çarpım: {x * y}")    # 85

# 2. Bölümden kalan (Modülüs)
print(f"Kalan: {x % y}")     # 2

# 3. Kuvvet hesaplama (Üs alma)
print(f"2'nin 8. kuvveti: {2 ** 8}") # 256

"""### 🔸 Soru 4:
Karşılaştırma operatörlerini kullanarak aşağıdaki ifadelerin sonuçlarını (True/False) yazdırın:
- 10 > 5
- 3 == 3.0
- "kilic" != "kalkan"
- 7 <= 7
"""

# Cevabınızı buraya yazın:
print(10 > 5)               # True (10, 5'ten büyüktür)
print(3 == 3.0)             # True (Değerler eşittir, float/int fark etmez)
print("kilic" != "kalkan")  # True ("kilic", "kalkan"a eşit değildir)
print(7 <= 7)                # True (7, 7'den küçük değildir ama eşittir)

"""---
## 3️⃣ String (Metin) İşlemleri
"""

# Örnek: String metodları
metin = "Zafer Oyunu!"

print(metin.upper())       # Büyük harf
print(metin.lower())       # Küçük harf
print(metin.replace("Oyunu", "Efsanesi"))  # Değiştirme
print(len(metin))          # Uzunluk
print(metin[0:5])          # Dilimleme

"""### 🔸 Soru 5:
`cumle = "  efsane oyun dunyasi cok genis  "` değişkeni için:
- Baştaki ve sondaki boşlukları temizleyin (`.strip()`)
- İlk harfini büyük yapın (`.capitalize()`)
- Kaç kelimeden oluştuğunu bulun (`.split()` ve `len()` kullanın)
- "genis" kelimesinin cümlede olup olmadığını kontrol edin (`in` operatörü)
"""

# Cevabınızı buraya yazın:

cumle = " efsane oyun dunyasi cok genis "

# 1. Boşlukları temizle
temiz_cumle = cumle.strip()

# 2. İlk harfi büyük yap
print(temiz_cumle.capitalize())

# 3. Kelime sayısını bul
kelime_sayisi = len(temiz_cumle.split())
print(f"Kelime sayısı: {kelime_sayisi}")

# 4. "genis" kelimesini kontrol et
print("genis" in temiz_cumle)

"""### 🔸 Soru 6 (f-string):
Karakter adı, lakap ve seviye bilgilerini f-string kullanarak şu formatta yazdırın:

`"Merhaba, ben [KARAKTER_ADI] [LAKAP]. [SEVIYE]. seviyedeyim."`
"""

karakter_adi = "Ahmet"
lakap = "Yıldırım"
seviye = 25

cumle = f"Merhaba, ben {karakter_adi} {lakap}. {seviye}. seviyedeyim."
print(cumle)

"""---
## 4️⃣ Koşullu İfadeler (if / elif / else)
"""

# Örnek
turnuva_puani = 75

if turnuva_puani >= 90:
    print("AA")
elif turnuva_puani >= 80:
    print("BA")
elif turnuva_puani >= 70:
    print("BB")
elif turnuva_puani >= 60:
    print("CB")
else:
    print("Başarısız")

"""### 🔸 Soru 7:
Bir bakiye değişiminin kazanç, kayıp ya da nötr olduğunu kontrol eden if-elif-else bloğu yazın.
"""

# Cevabınızı buraya yazın:
bakiye_degisimi = -7
if bakiye_degisimi > 0:
    print("Kazanç")
elif bakiye_degisimi < 0:
    print("Kayıp")
else:
    print("Nötr")

"""### 🔸 Soru 8:
Bir dalgadaki canavar sayısının 2'ye ve 3'e tam bölünüp bölünmediğini kontrol edin. 4 farklı durum vardır:
- Hem 2'ye hem 3'e bölünebilir
- Sadece 2'ye bölünebilir
- Sadece 3'e bölünebilir
- Hiçbirine bölünemiyor
"""

# Cevabınızı buraya yazın:
dalga_sayisi = 12

dalga_sayisi = 12

if dalga_sayisi % 2 == 0 and dalga_sayisi % 3 == 0:
    print("Hem 2'ye hem 3'e bölünebilir")
elif dalga_sayisi % 2 == 0:
    print("Sadece 2'ye bölünebilir")
elif dalga_sayisi % 3 == 0:
    print("Sadece 3'e bölünebilir")
else:
    print("Hiçbirine bölünemiyor")

"""---
## 5️⃣ Döngüler (for / while)
"""

# Örnek: for döngüsü
for i in range(1, 6):
    print(f"{i}. adım")

print("---")

# Örnek: while döngüsü
sayac = 0
while sayac < 3:
    print(f"Sayaç: {sayac}")
    sayac += 1

"""### 🔸 Soru 9:
`for` döngüsü ile 1'den 10'a kadar olan seviyelerin toplam deneyim puanını hesaplayın.
"""

# Cevabınızı buraya yazın:
# Başlangıçta toplamı sıfır olarak tanımlıyoruz
toplam = 0

# range(1, 11) komutu 1'den başlar, 11'e kadar (11 hariç) sayılar üretir
for seviye in range(1, 11):
    toplam += seviye  # Her adımda seviyeyi toplam değişkenine ekler

print(f"1'den 10'a kadar olan seviyelerin toplam deneyimi: {toplam}")

"""### 🔸 Soru 10:
`while` döngüsü ile bir savaşçının kombo çarpanı dizisini (n=5 terim, Fibonacci mantığıyla) yazdırın.

Örnek çıktı: `0, 1, 1, 2, 3`
"""

# Cevabınızı buraya yazın:
# Kombo dizisinin kaç terim üretileceği
n = 5

# Kombo çarpanı dizisinin ilk iki terimi
a, b = 0, 1
sayac = 0

print(f"{n} terimli kombo çarpanı dizisi:")

# Sayaç n değerine ulaşana kadar devam eder
while sayac < n:
    # Sayıyı yazdır (virgül ile yan yana gelmesi için end parametresini kullanıyoruz)
    if sayac == n - 1:
        print(a)
    else:
        print(a, end=", ")

    # Yeni sayıları hesapla (a bir sonraki sayı olur, b ise toplamları)
    yeni_sayi = a + b
    a = b
    b = yeni_sayi

    # Sayacı bir artır
    sayac += 1

"""### 🔸 Soru 11 (İç İçe Döngü):
5x5 hasar çarpım tablosunu yazdırın. Her satır şu formatta olsun:

`1x1=1  1x2=2  1x3=3 ...`
"""

# Cevabınızı buraya yazın:

# Dış döngü: 1'den 5'e kadar olan satırlar için
for i in range(1, 6):
    # İç döngü: Her satırdaki 1'den 5'e kadar olan hasar çarpımları için
    for j in range(1, 6):
        # Hasarı hesapla ve yan yana yazdır
        # end=" " kullanarak bir sonraki print'in aynı satıra devam etmesini sağlarız
        print(f"{i}x{j}={i*j}", end=" ")

    # Bir satır bittiğinde bir alt satıra geçmek için boş bir print atıyoruz
    print()

"""---
## 6️⃣ Listeler
"""

# Örnek
envanter = ["kilic", "kalkan", "zirh", "iksir"]

print(envanter[0])          # İlk eleman
print(envanter[-1])         # Son eleman
print(envanter[1:3])        # Dilimleme
envanter.append("yay")      # Ekleme
envanter.remove("kalkan")   # Silme
print(envanter)

"""### 🔸 Soru 12:
Aşağıdaki listede:
- En büyük ve en küçük elemanı bulun
- Listeyi sıralayın
- 7 sayısının listede kaç kez geçtiğini bulun
- Listeyi tersine çevirin
"""

# Cevabınızı buraya yazın:
hasar_degerleri = [3, 7, 1, 9, 7, 4, 7, 2, 8]

# 1. En büyük ve en küçük elemanı bulun
en_buyuk = max(hasar_degerleri)
en_kucuk = min(hasar_degerleri)
print(f"En büyük hasar: {en_buyuk}")
print(f"En küçük hasar: {en_kucuk}")

# 2. 7 sayısının listede kaç kez geçtiğini bulun
kac_tane_yedi = hasar_degerleri.count(7)
print(f"7 değeri listede {kac_tane_yedi} kez geçiyor.")

# 3. Listeyi sıralayın (Küçükten büyüğe)
hasar_degerleri.sort()
print(f"Sıralanmış liste: {hasar_degerleri}")

# 4. Listeyi tersine çevirin (Büyükten küçüğe veya mevcut sırayı ters çevirme)
hasar_degerleri.reverse()
print(f"Ters çevrilmiş liste: {hasar_degerleri}")

"""### 🔸 Soru 13 (List Comprehension):
List comprehension kullanarak:
- 1-20 arasındaki çift sayıların (seviyelerin) listesini oluşturun
- Bir canavar isimleri listesindeki 4'ten uzun isimleri filtreleyin
"""

# Cevabınızı buraya yazın:
# 1-20 arası çift seviyeler
ciftler = [x for x in range(1, 21) if x % 2 == 0]
print(ciftler)

# 4'ten uzun canavar isimleri
canavarlar = ["ejder", "kurt", "goblin", "orc", "cin", "troll", "peri"]
uzunlar = [c for c in canavarlar if len(c) > 4]
print(uzunlar)

"""---
## 7️⃣ Sözlükler (Dictionary)
"""

# Örnek
oyuncu = {
    "ad": "Ali",
    "lakap": "Kaya",
    "seviye": 20,
    "skorlar": [85, 90, 78]
}

print(oyuncu["ad"])
print(oyuncu.get("seviye"))
oyuncu["sinif"] = "Savaşçı"
print(oyuncu.keys())
print(oyuncu.values())

"""### 🔸 Soru 14:
Bir oyun sözlüğü oluşturun (`isim`, `gelistirici`, `yil`, `seviye_sayisi`) ve:
- Tüm anahtar-değer çiftlerini döngüyle yazdırın
- `fiyat` anahtarı ekleyin
- `gelistirici` anahtarını silin
"""

# Cevabınızı buraya yazın:
# 1. Oyun sözlüğünü oluşturma
oyun = {
    "isim": "Efsane Diyarlar",
    "gelistirici": "Kaan Studio",
    "yil": 2023,
    "seviye_sayisi": 40
}

# 2. fiyat anahtarı ekleme
oyun["fiyat"] = 150

# 3. gelistirici anahtarını silme
del oyun["gelistirici"]

# 4. Tüm anahtar-değer çiftlerini döngüyle yazdırma
for anahtar, deger in oyun.items():
    print(f"{anahtar}: {deger}")

"""---
## 8️⃣ Fonksiyonlar
"""

# Örnek
def karsila(oyuncu, nezaket="Hoş geldin"):
    return f"{nezaket}, {oyuncu}!"

print(karsila("Ayşe"))
print(karsila("Mehmet", "İyi günler"))

"""### 🔸 Soru 15:
Bir seviyenin "şanslı seviye" (asal sayı) olup olmadığını kontrol eden `asal_mi(n)` fonksiyonu yazın.
- 1'den 20'ye kadar olan şanslı (asal) seviyeleri bu fonksiyonu kullanarak yazdırın.
"""

# Cevabınızı buraya yazın:
def asal_mi(n):
    if n < 2:
      return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0: return False
    return True

# 1'den 20'ye kadar olan şanslı seviyeleri yazdır
for seviye in range(1, 21):
    if asal_mi(seviye):
        print(seviye, end=" ")

"""### 🔸 Soru 16:
Bir hasar listesinin ortalamasını, en büyük ve en küçük elemanını döndüren `liste_istatistik(liste)` fonksiyonu yazın.
Fonksiyon 3 değer döndürsün: `(ortalama, minimum, maksimum)`
"""

# Cevabınızı buraya yazın:
def liste_istatistik(liste):
    ortalama = sum(liste) / len(liste)
    minimum = min(liste)
    maksimum = max(liste)
    return ortalama, minimum, maksimum

test = [4, 8, 15, 16, 23, 42]
ort, mini, maksi = liste_istatistik(test)

print(f"Ortalama: {ort}, Min: {mini}, Max: {maksi}")

"""---
## 9️⃣ Hata Yönetimi (try / except)
"""

# Örnek
try:
    sonuc = 10 / 0
except ZeroDivisionError:
    print("Sıfıra bölme hatası!")
except Exception as e:
    print(f"Beklenmeyen hata: {e}")
finally:
    print("İşlem tamamlandı.")

"""### 🔸 Soru 17:
Kullanıcıdan bir seviye alındığını varsayın (`girdi = "abc"`).
Bu değeri int'e çevirmeye çalışırken `ValueError` hatası oluşabilir.
try-except ile bunu yakalayan ve anlamlı bir mesaj veren kod yazın.
"""

# Cevabınızı buraya yazın:
girdi = "abc"

try:
    # Sayıya çevirme işlemini deniyoruz
    seviye = int(girdi)
    print(f"Girdiğiniz seviye: {seviye}")
except ValueError:
    # Eğer çevirme sırasında hata (ValueError) oluşursa burası çalışır
    print("Hata: Girdiğiniz değer bir sayı değil!")

"""---
## 🔟 Bonus: Kapsamlı Uygulama

### 🔸 Soru 18 (Bütünleştirici):
Bir **görev defteri uygulaması** yazın:
- `gorevler` adında boş bir liste oluşturun
- `gorev_ekle(metin)` fonksiyonu: Görevi listeye ekler
- `gorev_listele()` fonksiyonu: Tüm görevleri numaralandırarak yazdırır
- `gorev_sil(index)` fonksiyonu: Verilen indexteki görevi siler, hatalı index için hata mesajı verir

Fonksiyonları test edin: 3 görev ekleyin, listeleyin, ortadakini silin, tekrar listeleyin.
"""

# Cevabınızı buraya yazın:

# 1. Boş görevler listesini oluşturuyoruz
gorevler = []

# 2. Görev ekleme fonksiyonu
def gorev_ekle(metin):
    gorevler.append(metin)
    print(f"Eklendi: {metin}")

# 3. Görevleri listeleme fonksiyonu
def gorev_listele():
    print("\n--- Güncel Görevler ---")
    if not gorevler:
        print("Görev defteri şu an boş.")
    else:
        for sira, gorev_metni in enumerate(gorevler, start=1):
            print(f"{sira}. {gorev_metni}")
    print("---------------------\n")

# 4. Görev silme fonksiyonu
def gorev_sil(index):
    try:
        # Kullanıcı 1-tabanlı sayı girdiği için index'ten 1 çıkarıyoruz
        silinen = gorevler.pop(index - 1)
        print(f"Silindi: {silinen}")
    except IndexError:
        print(f"Hata: {index} numaralı bir görev bulunamadı!")

# --- TEST ADIMLARI ---

# 3 görev ekleyelim
gorev_ekle("Ejderha inini keşfet")
gorev_ekle("Efsanevi kılıcı bul")
gorev_ekle("Akşam 20:00'de klan savaşı var")

# Listeleme yapalım
gorev_listele()

# Ortadaki görevi (2. görev) silelim
gorev_sil(2)

# Tekrar listeleyelim
gorev_listele()
"""gorevler = []

def gorev_ekle(metin):
    pass

def gorev_listele():
    pass

def gorev_sil(index):
    pass
"""
# Test kodu:
# gorev_ekle("Python öğren")
# gorev_ekle("Alıştırmaları tamamla")
# gorev_ekle("Proje yap")
# gorev_listele()
# gorev_sil(1)
# gorev_listele()

"""---
## 📊 Özet Çizelgesi

| Konu | Sorular | Tamamlandı mı? |
|------|---------|----------------|
| Değişkenler & Tipler | 1, 2 | evet |
| Operatörler | 3, 4 | evet |
| String İşlemleri | 5, 6 | evet |
| Koşullar | 7, 8 | evet|
| Döngüler | 9, 10, 11 | evet|
| Listeler | 12, 13 | evet |
| Sözlükler | 14 |evet |
| Fonksiyonlar | 15, 16 | evet|
| Hata Yönetimi | 17 | evet |
| Bütünleştirici | 18 |evet
 |

> 💡 **İpucu:** Takıldığınız yerde `help()` fonksiyonunu veya Python dokümantasyonunu kullanın!
"""
