# -*- coding: utf-8 -*-
"""Python 1 Ders.py (Oyun Versiyonu)

ÖĞRENME NOTU (yönlendirme):
  Bu dosyanın amacı, veri tipleri, operatorler, string/liste islemleri,
  math modulu, kosullar, donguler, sozlukler, fonksiyonlar ve class/OOP
  konularini sirayla pratik etmektir. Veri seti gerektirmez; degisken/
  ornek baglamlari (oyun karakteri/cuzdan/oyuncu) oyun temasina
  uyarlanmistir.

## veri tipleri

int

float

string

bool
"""

skor = 24
print(type(skor))

hiz_carpani = 3.14
print(type(hiz_carpani))

silah = "AtesKilici"
print(type(silah))

gece_modu = True
hile_kodu_acik = False
print(type(gece_modu))
print(type(hile_kodu_acik))

altin = 450
elmas = 350
toplam_kaynak = altin + elmas
toplam_kaynak

toplam = altin + elmas
print(toplam)

can, zirh = 10, 3
print(can + zirh)   # toplama
print(can - zirh)   # çıkarma
print(can * zirh)   # çarpma
print(can / zirh)   # bölme
print(can // zirh)  # tam bölme
print(can % zirh)   # kalan
print(can ** zirh)  # üs alma

can, zirh = 10, 3
print(can == zirh)  # eşit mi ?
print(can != zirh)  # eşit değil mi ?
print(can > zirh)   # büyük mü ?
print(can < zirh)   # küçük mü ?
print(can >= zirh)  # büyük eşit mi ?
print(can <= zirh)  # küçük eşit mi ?

hile_acik, online_mod = True, False
print(hile_acik and online_mod)  # ve
print(hile_acik or online_mod)   # veya
print(not hile_acik)             # değil

deneyim = 10
deneyim += 5  # deneyim+5
deneyim -= 3  # deneyim-3
deneyim *= 2  # deneyim*2
deneyim /= 4  # deneyim/4
deneyim //= 2
deneyim **= 3
deneyim %= 5

"""### String işlemleri"""

s = "EfsaneOyun"
print(s.upper())
print(s.lower())
print(s.title())
print(s.capitalize())
print(s.swapcase())
print(s.replace("t", "$"))
print(s.split("f"))
print(len(s))
print(s[0:5])
print("sane" in s)

"""#### Liste İşlemleri"""

skorlar = [70, 78, 16, 53, 42, 28]

skorlar.append(24)
skorlar.insert(2, 24)
skorlar.pop()
skorlar.sort()
skorlar.reverse()
skorlar.count(24)
skorlar.index(24)
print(sum(skorlar))
print(max(skorlar))
print(min(skorlar))

skorlar

"""## Matematik İşlemler"""

import math
print(math.sqrt(16))
print(math.pow(2, 3))
print(math.pi)
print(math.factorial)
print(math.floor(3.9))
print(math.ceil(3.1))
print(math.log10(1000))
print(math.fabs(-10))

"""### Tip dönüşümleri"""

print(int("42"))
print(float("3.14"))
print(str(100))
print(bool(1))
print(bool(0))

seviye = 18
if seviye >= 18:
  print("usta oyuncu")
elif seviye >= 13:
  print("orta seviye oyuncu")
else:
  print("acemi oyuncu")

magaza_sepeti_tutari = 1320

if magaza_sepeti_tutari >= 2000:
  print("tebrikler indirim kazandınız")
  odenecek_tutar = magaza_sepeti_tutari - 200
  print("ödemeniz gereken tutar", odenecek_tutar)
else:
  print("2000 kredi altına indirim uygulanmaz")
  odenecek_tutar = magaza_sepeti_tutari
  print("ödemeniz gereken tutar", odenecek_tutar)
#

for i in range(5):

  print(i)
x = 0

while x < 3:
  print(x)
  x += 1

"""Listeler"""

envanter = ["kilic", "yay", "ok", "kalkan"]
envanter.insert(3, "zirh")
print(envanter[0])  # index e göre elemanı yazdırır
print(len(envanter))  # boyutu verir
envanter.append("iksir")  # eleman ekleme
envanter.remove("yay")  # eleman silme
envanter.pop()  # son elemanı silme

envanter

"""## Sözlükler"""

karakter = {
    "ad": "Berkay",
    "seviye": 23,
    "sunucu": "Istanbul-1",
    "sinif": "Buyucu"
}

print(karakter["ad"])
print(karakter["seviye"])
print(karakter["sunucu"])
print(karakter["sinif"])

"""## Fonksiyonlar"""

def karsila():  # karsila ile etiket
  print("merhaba oyuncu")  # fonksiyonlar belirli bir işi yapan kod parçlarını içeren kutular

karsila()

def oyuncuya_ozel_teklif(oyuncu_id):  # parametre ()
  print(f"{oyuncu_id} numaralı oyuncumuz 1000 elmas hediyesi, almak için linke tıkla")

oyuncuya_ozel_teklif(123456789)

def topla(a, b):
  return a + b  # sonucu bize teslim
toplam = topla(10, 20)
print(toplam)

def selam_ver(oyuncu="Misafir"):
  print(f"merhaba {oyuncu}")


selam_ver()

selam_ver("Tuna")

def seviye_tek_mi_cift_mi(seviye):
  if seviye % 2 == 0:
    return "Çift"
  else:
    return "Tek"
print(seviye_tek_mi_cift_mi(10))

def pozitif_skorlari_sec(liste):
  yeni_liste = []
  for skor in liste:
    if skor > 0:
      yeni_liste.append(skor)
  return yeni_liste

skorlar = [-12, 10, 0, -2, -3, 8]
print(pozitif_skorlari_sec(skorlar))

def harita_alani(en, boy):
  alan = en * boy
  cevre = 2 * (en + boy)
  return alan, cevre

a, c = harita_alani(3, 4)
print(a, c)

def magaza_indirimi(tutar):
  if tutar >= 1000:
    return tutar * 0.80
  elif tutar >= 500:
    return tutar * 0.90
  else:
    return tutar

print(magaza_indirimi(1200))
print(magaza_indirimi(800))
print(magaza_indirimi(400))

"""Class(Sınıf)"""

class OyunAraci:
  def __init__(self, uretici, model, yil):
    self.uretici = uretici
    self.model = model
    self.yil = yil

arac1 = OyunAraci("HizliMotors", "GT-Racer", 2085)
arac2 = OyunAraci("VoltTeknik", "Sirius X", 2091)

print(arac1.uretici)
print(arac1.model)
print(arac1.yil)

x = 5  # int
x = " Araç"  # str

class OyunHaritasi:
    def __init__(self, en, boy):
        self.en = en
        self.boy = boy

    def alan(self):
        return self.en * self.boy

    def cevre(self):
        return 2 * (self.en + self.boy)

    def bilgi(self):
        print(f"Alan: {self.alan()}, Çevre: {self.cevre()}")

harita = OyunHaritasi(5, 3)
harita.bilgi()
print(harita.alan())    # 15
print(harita.cevre())   # 16

# Bunların her biri ayrı bir instance:
h1 = OyunHaritasi(5, 3)    # 1. instance
h2 = OyunHaritasi(10, 7)   # 2. instance
h3 = OyunHaritasi(2, 2)    # 3. instance
print(h1.en)
print(h2.en)
print(h3.en)

h1.alan()
h2.alan()
h3.alan()

class Karakter:
    def __init__(self, isim, yas):
        self.isim = isim
        self.yas  = yas

    def saldiri(self):
        return "..."

    def tanitim(self):
        print(f"{self.isim} ({self.yas} seviye): {self.saldiri()}")


class Savasci(Karakter):   # Karakter'den türüyor
    def __init__(self, isim, yas, irk):
        super().__init__(isim, yas)
        self.irk = irk

    def saldiri(self):   # override
        return "Kılıç darbesi!"


class Buyucu(Karakter):
    def saldiri(self):
        return "Ateş topu!"


k = Savasci("Karabaş", 3, "Insan")
c = Buyucu("Pamuk", 5)

k.tanitim()
c.tanitim()

class OyuncuCuzdani:
    def __init__(self, oyuncu, altin=0):
        self.oyuncu   = oyuncu
        self._altin = altin   # dahili

    @property
    def altin(self):
        return self._altin

    @altin.setter
    def altin(self, miktar):
        if miktar < 0:
            raise ValueError("Altın negatif olamaz!")
        self._altin = miktar

    def altin_ekle(self, miktar):
        self._altin += miktar
        print(f"+{miktar} altın → Bakiye: {self._altin} altın")

    def altin_harca(self, miktar):
        if miktar > self._altin:
            print("Yetersiz altın!")
        else:
            self._altin -= miktar
            print(f"-{miktar} altın → Bakiye: {self._altin} altın")

h = OyuncuCuzdani("Ali", 1000)
h.altin_ekle(500)
h.altin_harca(200)
h.altin_harca(2000)

class Konum:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __str__(self):        # print() için
        return f"({self.x}, {self.y})"

    def __repr__(self):       # geliştirici çıktısı
        return f"Konum({self.x}, {self.y})"

    def __add__(self, other):  # + operatörü (hareket vektörlerini birleştirir)
        return Konum(self.x + other.x, self.y + other.y)

    def __len__(self):         # len() için (boyut sayısı)
        return 2

    def uzunluk(self):
        return (self.x**2 + self.y**2) ** 0.5

k1 = Konum(3, 4)
k2 = Konum(1, 2)

print(k1)           # __str__
print(k1 + k2)      # __add__
print(len(k1))      # __len__
print(k1.uzunluk()) # 5.0

class Basari:
    def __init__(self, gun, ay, yil):
        self.gun = gun
        self.ay  = ay
        self.yil = yil

    def __str__(self):
        # kullanıcı dostu — güzel görünsün
        return f"{self.gun} {self.ay} {self.yil}"

    def __repr__(self):
        # geliştirici dostu — nesneyi yeniden yaratabileyim
        return f"Basari({self.gun}, {self.ay}, {self.yil})"


basari = Basari(15, 6, 2024)
print(basari)
repr(basari)

class Oyuncu:
    sunucu = "Efsane Sunucu"    # hiçbir nesne yok şuan sadece ne var sınıfımız var
    kayitli_oyuncu_sayisi = 0   # ben sınıfa dair özellikler ekleyeceğim yer tutuyor


    def __init__(self, ad, puan): # nesneye ait herşeyi burda yazarız
        self.ad   = ad
        self.puan = puan
        Oyuncu.kayitli_oyuncu_sayisi += 1

    def bilgi(self):
        print(f"{self.ad} | {self.sunucu} | Puan: {self.puan}")


o1 = Oyuncu("Ayşe", 95)
o2 = Oyuncu("Mehmet", 78) # nesneler

o1.bilgi()
o2.bilgi()
print("Toplam oyuncu:", Oyuncu.kayitli_oyuncu_sayisi)

# Class değişkeni tüm örnekleri etkiler
Oyuncu.sunucu = "Zafer Sunucu"
o1.bilgi()   # sunucu değişti!
o2.bilgi()
