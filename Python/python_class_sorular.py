# -*- coding: utf-8 -*-
"""python_class_sorular.py (Oyun Versiyonu)

ÖĞRENME NOTU (yönlendirme):
  Bu dosyanın amacı, kolaydan zora 11 class/OOP sorusuyla nesne yönelimli
  programlama pratiği yapmaktır; her soru kendi `assert` testleriyle
  dogrulanir. Veri seti gerektirmez.

# Python Class — Kolaydan Zora Sorular (Oyun Versiyonu)
Her soruyu çözdükten sonra alttaki test hücresini çalıştırarak cevabını kontrol edebilirsin.

---
## 🟢 KOLAY

### Soru 1 — Temel class
Aşağıdaki özelliklere sahip bir `Oyuncu` class'ı yaz:
- `ad`, `unvan`, `seviye` instance değişkenleri olsun
- `tanitim()` metodu `"Merhaba, ben Ali Veli, 25 seviyesindeyim."` formatında yazdırsın
"""

# Cevabını buraya yaz
class Oyuncu:
    def __init__(self, ad, unvan, seviye):
      self.ad = ad
      self.unvan = unvan
      self.seviye = seviye

    def tanitim(self):
      print(f"Merhaba, ben  {self.ad} {self.unvan}, {self.seviye} seviyesindeyim.")

# TEST
o = Oyuncu("Ali", "Veli", 25)
assert o.ad == "Ali"
assert o.unvan == "Veli"
assert o.seviye == 25
o.tanitim()  # Merhaba, ben Ali Veli, 25 seviyesindeyim.
print("✅ Soru 1 doğru!")

"""---
### Soru 2 — Metotlar
Bir `OyunHaritasi` class'ı yaz:
- `en` ve `boy` alsın
- `alan()` → en × boy döndürsün
- `cevre()` → 2 × (en + boy) döndürsün
- `kare_mi()` → en == boy ise `True` döndürsün
"""

# Cevabını buraya yaz
class OyunHaritasi:
    def __init__(self, en, boy):
      self.en = en
      self.boy = boy
    def alan(self):
      return self.en * self.boy

    def cevre(self):
      return 2 * (self.en + self.boy)

    def kare_mi(self):
      return self.en == self.boy

# TEST
h1 = OyunHaritasi(5, 3)
h2 = OyunHaritasi(4, 4)
assert h1.alan() == 15
assert h1.cevre() == 16
assert h1.kare_mi() == False
assert h2.kare_mi() == True
print("✅ Soru 2 doğru!")

"""---
### Soru 3 — `__str__` ve `__repr__`
Bir `Oyun` class'ı yaz:
- `ad` ve `gelistirici` alsın
- `__str__`  → `"The Witcher 3 — CD Projekt Red"` formatında döndürsün
- `__repr__` → `"Oyun('The Witcher 3', 'CD Projekt Red')"` formatında döndürsün
"""

# Cevabını buraya yaz
class Oyun:
    def __init__(self, ad, gelistirici):
      self.ad = ad
      self.gelistirici = gelistirici
    def __str__(self):
      return f"{self.ad} — {self.gelistirici}"

    def __repr__(self):
      return f"Oyun('{self.ad}', '{self.gelistirici}')"

# TEST
oy = Oyun("The Witcher 3", "CD Projekt Red")
assert str(oy) == "The Witcher 3 — CD Projekt Red"
assert repr(oy) == "Oyun('The Witcher 3', 'CD Projekt Red')"
print("✅ Soru 3 doğru!")

"""---
### Soru 4 — Class değişkeni
Bir `Arac` class'ı yaz:
- `uretici` instance değişkeni olsun
- `tekerlek_sayisi = 4` class değişkeni olsun (tüm oyun içi araçlar için sabit)
- `toplam_arac` class değişkeni her yeni nesne oluşturulduğunda 1 artsın
"""

# Cevabını buraya yaz
class Arac:
    toplam_arac = 0
    tekerlek_sayisi = 4

    def __init__(self, uretici):
        self.uretici = uretici
        Arac.toplam_arac += 1


a1 = Arac("HizliMotors")
print(f"1.Araç: {a1.uretici}, Tekerlek: {a1.tekerlek_sayisi}")
print(f"Toplam Araç Sayısı: {Arac.toplam_arac}")

a2 = Arac("VoltTeknik")
print(f"2.Araç: {a2.uretici}, Tekerlek: {a2.tekerlek_sayisi}")
print(f"Toplam Araç Sayısı: {Arac.toplam_arac}")

# TEST
Arac.toplam_arac = 0  # sıfırla
a1 = Arac("VoltTeknik")
a2 = Arac("HizliMotors")
a3 = Arac("RuzgarKanat")
assert Arac.tekerlek_sayisi == 4
assert Arac.toplam_arac == 3
assert a1.uretici == "VoltTeknik"
print("✅ Soru 4 doğru!")

"""---
## 🟡 ORTA

### Soru 5 — Kalıtım
`Karakter` class'ından türeyen `Savasci` ve `Buyucu` class'ları yaz:
- `Karakter`: `isim` alsın, `saldiri()` → `"..."` döndürsün
- `Savasci`: `saldiri()` → `"Kılıç darbesi!"` döndürsün, `kalkanla_blokla()` → `"[isim] kalkanla bloke etti!"` yazdırsın
- `Buyucu`: `saldiri()` → `"Ateş topu!"` döndürsün
"""

# Cevabını buraya yaz
class Karakter:
    def __init__(self, isim):
      self.isim = isim

    def saldiri(self):
      return "..."

class Savasci(Karakter):
    def saldiri(self):
       return "Kılıç darbesi!"

    def kalkanla_blokla(self):
        print(f"{self.isim} kalkanla bloke etti!")

class Buyucu(Karakter):
    def saldiri(self):
        return "Ateş topu!"

# TEST
s = Savasci("Karabaş")
b = Buyucu("Pamuk")
assert s.isim == "Karabaş"
assert s.saldiri() == "Kılıç darbesi!"
assert b.saldiri() == "Ateş topu!"
assert isinstance(s, Karakter)
assert isinstance(b, Karakter)
s.kalkanla_blokla()  # Karabaş kalkanla bloke etti!
print("✅ Soru 5 doğru!")

"""---
### Soru 6 — Kapsülleme ve @property
Bir oyun sunucusunun donanım sıcaklığını temsil eden `SunucuSicakligi` class'ı yaz:
- `_derece` ile saklansın
- `@property` ile `derece` okunabilsin
- `@derece.setter` ile atanabilsin ama `-273`'ün altına düşürülemesin (mutlak sıfır, donanım sensörü hata verir), düşürülmeye çalışılınca `ValueError` fırlatsın
- `@property` ile `fahrenheit` hesaplansın → `(derece × 9/5) + 32`
"""

# Cevabını buraya yaz
class SunucuSicakligi:
    def __init__(self, derece):
        self._derece = derece

    @property
    def derece(self):
        return self._derece

    @derece.setter
    def derece(self, yeni_deger):
        if yeni_deger < -273:
            raise ValueError("Sıcaklık mutlak sıfırın altına düşemez!")
        self._derece = yeni_deger

    @property
    def fahrenheit(self):
        return (self._derece * 9/5) + 32

# TEST
sc = SunucuSicakligi(100)
assert sc.derece == 100
assert sc.fahrenheit == 212.0
sc.derece = 0
assert sc.fahrenheit == 32.0
try:
    sc.derece = -300
    assert False, "ValueError bekleniyor"
except ValueError:
    pass
print("✅ Soru 6 doğru!")

"""---
### Soru 7 — Dunder metotlar
Bir `Odul` class'ı yaz:
- `miktar` ve `odul_turu` alsın (varsayılan: `"Altin"`)
- `__str__`  → `"100 Altin"` formatında
- `__add__`  → iki Odul toplanabilsin (aynı ödül türü kontrolü yap, farklıysa `ValueError`)
- `__eq__`   → miktar ve ödül türü eşitse `True`
- `__lt__`   → miktar karşılaştırması
"""

# Cevabını buraya yaz
class Odul:
    def __init__(self, miktar, odul_turu="Altin"):
        self.miktar = miktar
        self.odul_turu = odul_turu

    def __str__(self):
        return f"{self.miktar} {self.odul_turu}"

    def __add__(self, diger):
        if self.odul_turu != diger.odul_turu:
            raise ValueError("Farklı ödül türleri toplanamaz!")
        return Odul(self.miktar + diger.miktar, self.odul_turu)

    def __eq__(self, diger):
        return self.miktar == diger.miktar and self.odul_turu == diger.odul_turu

    def __lt__(self, diger):
        if self.odul_turu != diger.odul_turu:
             raise ValueError("Farklı türler karşılaştırılamaz!")
        return self.miktar < diger.miktar

# TEST
o1 = Odul(100, "Altin")
o2 = Odul(50, "Altin")
o3 = Odul(100, "Altin")
o4 = Odul(50, "Elmas")
assert str(o1) == "100 Altin"
assert (o1 + o2).miktar == 150
assert o1 == o3
assert o2 < o1
try:
    o1 + o4
    assert False, "ValueError bekleniyor"
except ValueError:
    pass
print("✅ Soru 7 doğru!")

"""---
### Soru 8 — @classmethod ve @staticmethod
Bir `OyunHaritasi` class'ı yaz:
- `en` ve `boy` alsın
- `@classmethod kare_oda_olustur(cls, kenar)` → kare oda döndürsün (en == boy == kenar)
- `@staticmethod gecerli_mi(en, boy)` → ikisi de 0'dan büyükse `True` döndürsün
"""

# Cevabını buraya yaz
class OyunHaritasi:
    def __init__(self, en, boy):
        self.en = en
        self.boy = boy

    @classmethod
    def kare_oda_olustur(cls, kenar):
        return cls(kenar, kenar)

    @staticmethod
    def gecerli_mi(en, boy):
        return en > 0 and boy > 0

# TEST
harita = OyunHaritasi(5, 3)
kare_oda = OyunHaritasi.kare_oda_olustur(4)
assert kare_oda.en == 4
assert kare_oda.boy == 4
assert OyunHaritasi.gecerli_mi(5, 3) == True
assert OyunHaritasi.gecerli_mi(-1, 3) == False
assert OyunHaritasi.gecerli_mi(0, 3) == False
print("✅ Soru 8 doğru!")

"""---
## 🔴 ZOR

### Soru 9 — Iterator class
Bir `SeviyeIterator` class'ı yaz:
- `baslangic`, `bitis`, `adim` alsın
- `for` döngüsünde kullanılabilsin
- Python'un `range()` gibi çalışsın ama class olarak (açılan seviyeler arasında gezinmek için)
- `__len__` da ekle — kaç seviye üreteceğini döndürsün
"""

# Cevabını buraya yaz
class SeviyeIterator:
    def __init__(self, baslangic, bitis, adim):
        self.baslangic = baslangic
        self.bitis = bitis
        self.adim = adim

    def __iter__(self):
        self.mevcut = self.baslangic
        return self

    def __next__(self):
        if (self.adim > 0 and self.mevcut >= self.bitis) or \
           (self.adim < 0 and self.mevcut <= self.bitis):
            raise StopIteration
        deger = self.mevcut
        self.mevcut += self.adim
        return deger

    def __len__(self):
        return max(0, (self.bitis - self.baslangic + self.adim - (1 if self.adim > 0 else -1)) // self.adim)

# TEST
sv = SeviyeIterator(0, 10, 2)
assert list(sv) == [0, 2, 4, 6, 8]
assert len(sv) == 5
sv2 = SeviyeIterator(1, 6, 1)
assert list(sv2) == [1, 2, 3, 4, 5]
# iki kez kullanılabilmeli
assert list(sv) == [0, 2, 4, 6, 8]
print("✅ Soru 9 doğru!")

"""---
### Soru 10 — Mixin
Aşağıdaki iki Mixin'i yaz ve `Yarismaci` class'ına ekle:
- `KarsilastirmaMixin`: `__eq__` ve `__lt__` eklesin — karşılaştırma `puan` değerine göre yapılsın
- `YazdirMixin`: `yazdir()` metodu tüm instance değişkenlerini `anahtar: değer` formatında yazdırsın
- `Yarismaci`: `isim`, `takim`, `puan` alsın
"""

# Cevabını buraya yaz
class KarsilastirmaMixin:
    def __eq__(self, diger):
        return self.puan == diger.puan
    def __lt__(self, diger):
        return self.puan < diger.puan
    def __gt__(self, diger):
        return self.puan > diger.puan

class YazdirMixin:
    def yazdir(self):
        for anahtar, deger in self.__dict__.items():
            print(f"{anahtar}: {deger}")

class Yarismaci(KarsilastirmaMixin, YazdirMixin):
    def __init__(self, isim, takim, puan):
        self.isim = isim
        self.takim = takim
        self.puan = puan

# TEST
y1 = Yarismaci("Ali", "Kirmizi Takim", 50000)
y2 = Yarismaci("Ayşe", "Mavi Takim", 60000)
y3 = Yarismaci("Ali", "Kirmizi Takim", 50000)
assert y1 < y2
assert y2 > y1
assert y1 == y3
y1.yazdir()
# isim: Ali
# takim: Kirmizi Takim
# puan: 50000
print("✅ Soru 10 doğru!")

"""---
### Soru 11 — Hepsini Birleştir (Proje)
Bir `OyunKutuphanesi` sistemi yaz:

**`Oyun` class'ı:**
- `oyun_id`, `ad`, `gelistirici`, `stokta_var` (varsayılan `True`) alsın
- `__str__` → `"[oyun_id] Ad — Gelistirici"`
- `__repr__` → `"Oyun(oyun_id, ad, gelistirici)"`

**`OyunKutuphanesi` class'ı:**
- `oyunlar` listesi tut
- `oyun_ekle(oyun)` → listeye ekle
- `__len__` → toplam oyun sayısı
- `__contains__` → oyun_id ile oyun var mı kontrolü (`"steam-1" in kutuphane`)
- `oyun_bul(oyun_id)` → oyun_id ile oyun döndür, bulamazsa `None`
- `kirala(oyun_id)` → oyunu `stokta_var=False` yap, zaten kiralanmışsa hata ver
- `iade_et(oyun_id)` → oyunu `stokta_var=True` yap
- `stoktaki_oyunlar()` → sadece stokta olan oyunları listele
"""

# Cevabını buraya yaz
class Oyun:
    def __init__(self, oyun_id, ad, gelistirici, stokta_var=True):
      self.oyun_id = oyun_id
      self.ad = ad
      self.gelistirici = gelistirici
      self.stokta_var = stokta_var

    def __str__(self):
        return f"[{self.oyun_id}] {self.ad} - {self.gelistirici}"

    def __repr__(self):
        return f"Oyun('{self.oyun_id}', '{self.ad}', '{self.gelistirici}')"

class OyunKutuphanesi:
    def __init__(self):
        self.oyunlar = []

    def oyun_ekle(self, oyun):
        self.oyunlar.append(oyun)

    def __len__(self):
        return len(self.oyunlar)

    def __contains__(self, oyun_id):
        return any(o.oyun_id == oyun_id for o in self.oyunlar)

    def oyun_bul(self, oyun_id):
        for o in self.oyunlar:
            if o.oyun_id == oyun_id: return o
        return None

    def kirala(self, oyun_id):
        oyun = self.oyun_bul(oyun_id)
        if oyun and oyun.stokta_var:
            oyun.stokta_var = False
        else:
            raise Exception("Oyun stokta değil veya zaten kiralanmış.")

    def iade_et(self, oyun_id):
        oyun = self.oyun_bul(oyun_id)
        if oyun:
            oyun.stokta_var = True

    def stoktaki_oyunlar(self):
        return [o for o in self.oyunlar if o.stokta_var]

# TEST
oy1 = Oyun("steam-1", "Suç ve Ceza: Diriliş", "Dostoyevski Games")
oy2 = Oyun("steam-2", "1984: Isyan", "Orwell Studio")
oy3 = Oyun("steam-3", "Dune: Cöl Efsanesi", "Herbert Interactive")

kutuphane = OyunKutuphanesi()
kutuphane.oyun_ekle(oy1)
kutuphane.oyun_ekle(oy2)
kutuphane.oyun_ekle(oy3)

assert len(kutuphane) == 3
assert "steam-1" in kutuphane
assert "steam-9" not in kutuphane

kutuphane.kirala("steam-1")
assert oy1.stokta_var == False
assert len(kutuphane.stoktaki_oyunlar()) == 2

kutuphane.iade_et("steam-1")
assert oy1.stokta_var == True

try:
    kutuphane.kirala("steam-1")
    kutuphane.kirala("steam-1")  # zaten kiralanmış
    assert False
except Exception:
    pass

print("✅ Soru 11 doğru! Tebrikler!")
