# -*- coding: utf-8 -*-
"""temel_python_2.py (Oyun Versiyonu)

ÖĞRENME NOTU (yönlendirme):
  Bu dosyanın amacı, 18 soruluk bir alistirma dizisiyle (coklu atama,
  operatorler, string formatlama, kosullar, donguler, liste/tuple/set,
  sozluk comprehension, *args/**kwargs, dosya islemleri, butunlestirici
  oyuncu skor sistemi projesi) Python pratiği yapmaktır. Veri seti
  gerektirmez.

# 🐍 Temel Python Komutları - Alıştırma Defteri 2 (Oyun Versiyonu)

Bu defterde Python'un temel konularını oyun dünyası örnekleriyle farklı
sorularla pekiştireceksiniz.

---

## 1️⃣ Değişkenler ve Veri Tipleri
"""

# Örnek: Çoklu atama ve değer değiştirme
x = y = z = 0
print(x, y, z)

a, b, c = 1, 2.5, "Oyun"
print(a, b, c)

# İki değişkeni yer değiştirme
a, b = b, a
print("Yer değişti:", a, b)

"""### 🔸 Soru 1:
Tek satırda `oyuncu_adi`, `seviye`, `altin_bakiyesi` değişkenlerine sırasıyla `"Zeynep"`, `32`, `15000.50` değerlerini atayın ve her birini ayrı `print` ile yazdırın.
"""

# Cevabınızı buraya yazın:

"""### 🔸 Soru 2:
Aşağıdaki ifadelerin Python'da geçerli olup olmadığını `try-except` kullanmadan, sadece `type()` ve mantık yürüterek tahmin edin, sonra çalıştırarak doğrulayın:
- `"5" + 5`
- `"5" * 3`
- `True + 1`
- `None == False`
"""

# Tahminlerinizi yorum satırı olarak yazın, sonra çalıştırın:

# "5" + 5    -> Tahminim:
# print("5" + 5)

# "5" * 3    -> Tahminim:
print("5" * 3)

# True + 1   -> Tahminim:
print(True + 1)

# None == False -> Tahminim:
print(None == False)

"""---
## 2️⃣ Operatörler
"""

# Örnek: Mantıksal operatörler
seviye = 20
uye_mi = True

turnuvaya_katilabilir = seviye >= 18 and uye_mi
print("Turnuvaya katılabilir:", turnuvaya_katilabilir)

ozel_hediye_hakki = seviye < 10 or seviye > 90
print("Özel hediye hakkı:", ozel_hediye_hakki)

"""### 🔸 Soru 3:
Bir haritadaki üçgen bölgenin kenar uzunlukları veriliyor. Üçgen eşitsizliği kuralına göre (`a + b > c` vb.) geçerli bir üçgen oluşturup oluşturmadığını tek bir `and` ifadesiyle kontrol edin.
"""

# Cevabınızı buraya yazın:
a, b, c = 3, 4, 5

"""### 🔸 Soru 4:
Atama operatörlerini (`+=`, `-=`, `*=`, `//=`) kullanarak `altin = 100` değişkenini sırasıyla:
- 25 artırın
- 3 ile çarpın
- 5'e tam bölün
- 10 azaltın

Her adımdan sonra değeri yazdırın.
"""

# Cevabınızı buraya yazın:
altin = 100

"""---
## 3️⃣ String (Metin) İşlemleri
"""

# Örnek: join ve split
envanter_esyalari = ["kilic", "kalkan", "zirh"]
metin = ", ".join(envanter_esyalari)
print(metin)

geri = metin.split(", ")
print(geri)

# Sayı formatlama
pi = 3.14159265
print(f"Pi sayısı: {pi:.2f}")
print(f"Ödül: {15000:,} Altın")

"""### 🔸 Soru 5:
Aşağıdaki oyuncu etiketinden `#` işaretini kullanarak oyuncu adını ve numarasını ayırın. Sonuçları ayrı değişkenlere atayıp yazdırın.
"""

# Cevabınızı buraya yazın:
etiket = "KaanTheBrave#4821"

"""### 🔸 Soru 6:
Bir palindrom kontrolü yapın: verilen kelimeyi tersine çevirip orijinaliyle karşılaştırın.
- `"kaçak"` → palindrom mu?
- `"oyun"` → palindrom mu?
"""

# Cevabınızı buraya yazın:
# İpucu: kelime[::-1] ifadesi kelimeyi tersine çevirir
kelime1 = "kaçak"
kelime2 = "oyun"

"""---
## 4️⃣ Koşullu İfadeler (if / elif / else)
"""

# Örnek: Ternary (tek satır if)
sayi = 7
durum = "tek" if sayi % 2 != 0 else "çift"
print(f"{sayi} sayısı {durum}")

"""### 🔸 Soru 7:
Bir sezonun (yılın) artık yıl olup olmadığını kontrol edin (yeni sezon güncellemesinin çıkış tarihini hesaplamak için).
Artık yıl kuralı:
- 4'e tam bölünebiliyorsa artık yıldır
- Ama 100'e bölünebiliyorsa artık yıl **değildir**
- Ama 400'e bölünebiliyorsa yine artık yıldır
"""

# Cevabınızı buraya yazın:
# Test: 2000 (artık), 1900 (değil), 2024 (artık), 2023 (değil)
sezon_yili = 2024

"""### 🔸 Soru 8:
Oyun mağazasında indirim hesabı yapın:
- 500 Altın altı → indirim yok
- 500-1000 Altın arası → %10 indirim
- 1000-2000 Altın arası → %20 indirim
- 2000 Altın üstü → %30 indirim

Ödenmesi gereken tutarı yazdırın.
"""

# Cevabınızı buraya yazın:
sepet_tutari = 1500

"""---
## 5️⃣ Döngüler (for / while)
"""

# Örnek: enumerate ve zip
karakter_siniflari = ["savasci", "buyucu", "okcu"]

for i, sinif in enumerate(karakter_siniflari, start=1):
    print(f"{i}. {sinif}")

print("---")

oyuncular = ["Ali", "Ayşe", "Can"]
skorlar = [85, 92, 78]

for oyuncu, skor in zip(oyuncular, skorlar):
    print(f"{oyuncu}: {skor}")

"""### 🔸 Soru 9:
`break` ve `continue` kullanımını gösterin:
- 1-20 arasında bir dalga taraması döngüsü kurun
- 3'ün katı olan dalgaları `continue` ile atlayın
- 15. dalgaya ulaşınca `break` ile çıkın
- Taranan dalga numaralarını gösterin
"""

# Cevabınızı buraya yazın:

"""### 🔸 Soru 10:
`while` döngüsü ile basamak sayısı bulma: Verilen bir toplam hasarın kaç basamaklı olduğunu, sayıyı 10'a bölerek bulun (string dönüşümü kullanmadan).
"""

# Cevabınızı buraya yazın:
# Örnek: 4752 -> 4 basamaklı
toplam_hasar = 4752

"""### 🔸 Soru 11:
`enumerate` kullanarak bir oyun mağazası sepetini numaralandırıp yazdırın, ardından toplam ürün sayısını gösterin.
"""

# Cevabınızı buraya yazın:
magaza_sepeti = ["kilic", "zirh", "iksir", "anahtar", "harita"]

"""---
## 6️⃣ Listeler ve Demetler (Tuple)
"""

# Örnek: Tuple
oyuncu_konumu = (120.5, 340.2)  # oyun dünyası koordinatları
x_konum, y_konum = oyuncu_konumu       # Unpacking
print(f"X: {x_konum}, Y: {y_konum}")

# Tuple değiştirilemez!
# oyuncu_konumu[0] = 40  # -> TypeError!

"""### 🔸 Soru 12:
Bir oyuncu listesi var. Her oyuncu `(isim, skor)` tuple'ı olarak saklanıyor.
- Skoru 70'in üzerinde olanları filtreleyin
- En yüksek skoru alan oyuncuyu bulun
- Skorların ortalamasını hesaplayın
"""

# Cevabınızı buraya yazın:
oyuncular = [
    ("Ahmet", 85),
    ("Büşra", 62),
    ("Can", 91),
    ("Deniz", 74),
    ("Elif", 58),
    ("Furkan", 88)
]

"""### 🔸 Soru 13 (Set):
İki takımın oyuncu listesi var. `set` kullanarak:
- Her iki takımda da bulunan oyuncuları bulun (kesişim)
- Toplam kaç farklı oyuncu olduğunu bulun (birleşim)
- Sadece A takımında olup B'de olmayan oyuncuları bulun (fark)
"""

# Cevabınızı buraya yazın:
takim_a = {"Ali", "Ayşe", "Can", "Dilan", "Ece"}
takim_b = {"Can", "Dilan", "Fatih", "Gökçe", "Hakan"}

"""---
## 7️⃣ Sözlükler (Dictionary)
"""

# Örnek: Dict comprehension
kareler = {x: x**2 for x in range(1, 6)}
print(kareler)

# Filtrelemeli
cift_kareler = {x: x**2 for x in range(1, 11) if x % 2 == 0}
print(cift_kareler)

"""### 🔸 Soru 14:
Bir görev açıklamasındaki her harfin kaç kez geçtiğini sayan bir sözlük oluşturun (boşlukları saymayin, büyük/küçük harf farkı olmasın).
"""

# Cevabınızı buraya yazın:
gorev_aciklamasi = "efsane oyun dunyasi"

"""---
## 8️⃣ Fonksiyonlar
"""

# Örnek: *args ve **kwargs
def toplam(*sayilar):
    return sum(sayilar)

print(toplam(1, 2, 3))
print(toplam(10, 20, 30, 40, 50))

def oyuncu_bilgi(**bilgiler):
    for anahtar, deger in bilgiler.items():
        print(f"{anahtar}: {deger}")

oyuncu_bilgi(ad="Selin", seviye=25, sunucu="Izmir-1")

"""### 🔸 Soru 15:
Özyinelemeli (recursive) bir fonksiyon yazın: `guc_katsayisi(n)` - n! (n faktöriyel) hesaplasın, bu bir savaşçının güç yükseltme katsayısı olsun.
- 0! = 1
- n! = n × (n-1)!

5!, 10! değerlerini hesaplayın.
"""

# Cevabınızı buraya yazın:
def guc_katsayisi(n):
    pass

"""### 🔸 Soru 16:
`lambda` fonksiyonu kullanarak:
- Bir hasar değerinin karesini hesaplayan lambda
- İki hasar değerinin büyüğünü döndüren lambda
- `sorted()` ile bir karakter isim listesini son harfe göre sıralayan lambda
"""

# Cevabınızı buraya yazın:
karakter_isimleri = ["Zeynep", "Ali", "Mahmut", "Su", "Berk"]

"""---
## 9️⃣ Dosya İşlemleri
"""

# Örnek: Dosya yazma ve okuma
with open("ornek.txt", "w", encoding="utf-8") as dosya:
    dosya.write("Efsane Diyarlar'a hoş geldin!\n")
    dosya.write("Bu bir oyun günlüğü dosyasıdır.\n")

with open("ornek.txt", "r", encoding="utf-8") as dosya:
    icerik = dosya.read()
    print(icerik)

"""### 🔸 Soru 17:
Bir görev listesini `gorevler.txt` dosyasına yazın (her görev ayrı satırda). Sonra dosyayı okuyarak görevleri numaralı şekilde ekrana yazdırın.
"""

# Cevabınızı buraya yazın:
gorevler = ["ejder avla", "kılıç yükselt", "zindan temizle", "ittifak kur", "harita topla"]

"""---
## 🔟 Bonus: Kapsamlı Uygulama

### 🔸 Soru 18 (Bütünleştirici):
Basit bir **oyuncu skor sistemi** yazın:

- `oyuncular` adında boş bir sözlük oluşturun (anahtar: isim, değer: skor listesi)
- `oyuncu_ekle(isim)`: Yeni oyuncu ekler
- `skor_ekle(isim, skor)`: Oyuncuya skor ekler. Oyuncu yoksa hata mesajı verir.
- `ortalama_hesapla(isim)`: Oyuncunun skor ortalamasını döndürür
- `takim_raporu()`: Tüm oyuncuları ve ortalamalarını yazdırır, takım ortalamasını gösterir

En az 3 oyuncu ve her birine 3 skor girerek test edin.
"""

# Cevabınızı buraya yazın:
oyuncular = {}

def oyuncu_ekle(isim):
    pass

def skor_ekle(isim, skor):
    pass

def ortalama_hesapla(isim):
    pass

def takim_raporu():
    pass

# Test kodu:
# oyuncu_ekle("Ahmet")
# oyuncu_ekle("Büşra")
# oyuncu_ekle("Can")
# skor_ekle("Ahmet", 85)
# skor_ekle("Ahmet", 90)
# skor_ekle("Ahmet", 78)
# skor_ekle("Büşra", 92)
# skor_ekle("Büşra", 88)
# skor_ekle("Büşra", 95)
# skor_ekle("Can", 70)
# skor_ekle("Can", 65)
# skor_ekle("Can", 73)
# takim_raporu()

"""---
## 📊 Özet Çizelgesi

| Konu | Sorular | Tamamlandı mı? |
|------|---------|----------------|
| Değişkenler & Tipler | 1, 2 | ☐ |
| Operatörler | 3, 4 | ☐ |
| String İşlemleri | 5, 6 | ☐ |
| Koşullar | 7, 8 | ☐ |
| Döngüler | 9, 10, 11 | ☐ |
| Liste, Tuple, Set | 12, 13 | ☐ |
| Sözlükler | 14 | ☐ |
| Fonksiyonlar | 15, 16 | ☐ |
| Dosya İşlemleri | 17 | ☐ |
| Bütünleştirici | 18 | ☐ |

> 💡 **İpucu:** Takıldığınız yerde `help()` fonksiyonunu veya Python dokümantasyonunu kullanın!
"""
