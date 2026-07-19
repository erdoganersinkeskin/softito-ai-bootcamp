# -*- coding: utf-8 -*-
"""
OPENCV GORUNTU ISLEME (STEAM KAPAK GORSELI VERSIYONU)

ÖĞRENME NOTU (yönlendirme):
  Bu dosyanın amacı, klasik bir OpenCV islem hattini (RGB kanallari ->
  kirpma/donme/olcekleme -> renk uzaylari -> histogram -> filtreleme ->
  kenar tespiti -> morfolojik islemler -> esikleme -> kontur tespiti)
  uygulamaktır. Telif hakki sorunu yasamamak icin, Kaggle'dan indirilen
  gercek bir Steam oyununun magaza kapak (header) gorseli kullaniliyor.

Kullanilan veri seti (Kaggle):
  fronkongames/steam-games-dataset
  -> Bu proje bir goruntu isleme egzersizi oldugu icin, listedeki 9 veri
     setinden gercek gorsel URL'si (header_image) iceren tek veri seti bu
     oldugundan tercih edildi (bkz. 01-fashion-mnist-cnn ile ayni gerekce).

  NOT: Kaggle kimlik dogrulamasi (kaggle.json) ve internet baglantisi
  gereklidir.
"""
import io
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.makedirs('figures', exist_ok=True)

import cv2
import pandas as pd
import requests
from PIL import Image

import warnings
warnings.filterwarnings('ignore')

plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.family'] = 'DejaVu Sans'

print("Tum kutuphaneler basariyla yuklendi!")
print(f"   NumPy:      {np.__version__}")
print(f"   OpenCV:     {cv2.__version__}")

# ============================================================================
# Bolum 1 — Steam Kapak Gorselini Kaggle Katalogundan Indirme
# ============================================================================
import kagglehub

TARGET_GAME_NAME = "Portal 2"   # Baska bir oyun denemek icin bu ismi degistirin
FALLBACK_APPID = 620            # Portal 2'nin bilinen AppID'si (katalogda bulunamazsa yedek)


def download_game_catalog():
    print("Steam oyun katalogu indiriliyor (Kaggle: fronkongames/steam-games-dataset)...")
    dataset_path = kagglehub.dataset_download("fronkongames/steam-games-dataset")
    csv_files = [f for f in os.listdir(dataset_path) if f.lower().endswith(".csv")]
    if not csv_files:
        raise FileNotFoundError(f"'{dataset_path}' icinde .csv bulunamadi.")
    catalog_path = os.path.join(dataset_path, csv_files[0])
    # NOT: Kolon adlarini once TUM basliklari okuyup normalize ederek eslestiriyoruz
    # (ornegin "Header image" gibi bosluklu/farkli sermayeli adlar da yakalanir);
    # dogrudan usecols=lambda c: c in (...) ile tam esitlik kontrolu, kolon adi
    # surumden surume degisirse (ornegin "header_image" -> "Header image")
    # sessizce hicbir satir donmemesine yol acabiliyordu.
    header_cols = pd.read_csv(catalog_path, nrows=0).columns
    def _norm(c):
        return c.strip().lower().replace(" ", "_")
    wanted = {"appid": "AppID", "name": "Name", "header_image": "header_image"}
    usecols = [c for c in header_cols if _norm(c) in wanted]
    catalog = pd.read_csv(catalog_path, usecols=usecols)
    catalog = catalog.rename(columns={c: wanted[_norm(c)] for c in usecols})
    return catalog.dropna(subset=["AppID", "header_image"])


game_catalog = download_game_catalog()
matches = game_catalog[game_catalog["Name"].astype(str).str.strip().str.lower() == TARGET_GAME_NAME.lower()]
if len(matches) > 0:
    header_image_url = matches.iloc[0]["header_image"]
else:
    # Katalogda tam isim eslesmesi bulunamazsa, bilinen CDN URL kalibina dus
    header_image_url = f"https://cdn.cloudflare.steamstatic.com/steam/apps/{FALLBACK_APPID}/header.jpg"

print(f"Kapak gorseli indiriliyor: {header_image_url}")
response = requests.get(header_image_url, timeout=15)
response.raise_for_status()
img = Image.open(io.BytesIO(response.content)).convert("RGB")
img_array = np.array(img)  # (H, W, 3) RGB, uint8

# ============================================================================
# Bolum 2 — Goruntu Yukleme ve Gosterme
# ============================================================================
plt.figure(figsize=(8, 6))
plt.imshow(img)
plt.axis('off')
plt.title(f'Yuklenen Gorsel — Steam Kapak Gorseli ({TARGET_GAME_NAME})')
plt.savefig('figures/01_yuklenen_goruntu.png', dpi=150, bbox_inches='tight')
plt.close()

# ── RGB Kanallarini Ayri Ayri Gorsellestir ────────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(18, 4))

axes[0].imshow(img)
axes[0].set_title("Orijinal RGB")
axes[0].axis('off')

channel_info = [
    (img_array[:, :, 0], 'Reds', 'R (Kirmizi) Kanali'),
    (img_array[:, :, 1], 'Greens', 'G (Yesil) Kanali'),
    (img_array[:, :, 2], 'Blues', 'B (Mavi) Kanali'),
]

for ax, (channel, cmap, title) in zip(axes[1:], channel_info):
    ax.imshow(channel, cmap=cmap)
    ax.set_title(f"{title}\nMean: {channel.mean():.1f}")
    ax.axis('off')

plt.suptitle("RGB Kanal Ayristirmasi — Her kanal bagimsiz bir gri goruntudur",
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/02_rgb_kanal_ayristirmasi.png', dpi=150, bbox_inches='tight')
plt.close()

# ── Bireysel Piksel Degerlerini Incele ───────────────────────────────────────
print("Belirli piksellerin RGB degerleri:")
print(f"  Sol ust kose    [0, 0]    : R={img_array[0, 0, 0]}, G={img_array[0, 0, 1]}, B={img_array[0, 0, 2]}")

h, w, _ = img_array.shape
center_h, center_w = h // 2, w // 2

print(f"  Merkez          [{center_h},{center_w}] : R={img_array[center_h, center_w, 0]}, G={img_array[center_h, center_w, 1]}, B={img_array[center_h, center_w, 2]}")
print(f"  Sag alt kose    [{h-1},{w-1}]: R={img_array[-1, -1, 0]}, G={img_array[-1, -1, 1]}, B={img_array[-1, -1, 2]}")

# Bir bolgeyi zoom ile goster (goruntu boyutuna orantili bir kirpma)
region_y0, region_x0 = h // 4, w // 4
region_h, region_w = min(50, h // 4), min(50, w // 4)
region = img_array[region_y0:region_y0 + region_h, region_x0:region_x0 + region_w]
print(f"\nKirpilan bolge sekli: {region.shape}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
ax1.imshow(img_array)
rect = plt.Rectangle((region_x0, region_y0), region_w, region_h, linewidth=2, edgecolor='yellow', facecolor='none')
ax1.add_patch(rect)
ax1.set_title("Orijinal — Sari bolge kirpilacak")
ax1.axis('off')

ax2.imshow(region)
ax2.set_title(f"Kirpilmis Bolge\n{region.shape[0]}x{region.shape[1]} piksel")
ax2.axis('off')

plt.tight_layout()
plt.savefig('figures/03_piksel_kirpma_bolge.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================================
# Oyun Logosu/Baslik Bolgesini Kirpma (Bolge Secimi)
# ============================================================================
# Izgara + manuel koordinat secimi teknigiyle kapak gorselinin ust-sol
# kismindaki logo/baslik bolgesi kirpiliyor.

# 1. ADIM: Tam koordinatlari bulmak icin goruntuye izgara (grid) ekle
fig, ax = plt.subplots(figsize=(16, 8))
ax.imshow(img_array)

ax.minorticks_on()
ax.grid(which='major', color='red', linestyle='-', linewidth=1, alpha=0.5)
ax.grid(which='minor', color='yellow', linestyle=':', linewidth=0.5, alpha=0.5)

plt.title("Logo/Baslik Koordinatlarini Bulmak Icin Izgarali Gorunum (Kirmizi: Ana, Sari: Ara Cizgiler)")
plt.savefig('figures/04_plaka_izgara_koordinat.png', dpi=150, bbox_inches='tight')
plt.close()

# 2. ADIM: Izgaraya bakarak logo/baslik bolgesinin koordinatlarini gir
# Goruntu matrisleri [Y, X] yani [Satir, Sutun] seklinde calisir.
# Steam kapak gorselleri genelde 460x215 piksel civarindadir; bu yuzden
# kirpma bolgesi goruntu boyutuna oranla belirlenir (sabit deger yerine).
y_start = int(h * 0.10)
y_end = int(h * 0.55)
x_start = int(w * 0.05)
x_end = int(w * 0.55)

logo_region = img_array[y_start:y_end, x_start:x_end]

plt.figure(figsize=(8, 4))
plt.imshow(logo_region)
plt.title(f"Kirpilan Logo/Baslik Bolgesi ({logo_region.shape[0]}x{logo_region.shape[1]} piksel)")
plt.axis('off')
plt.savefig('figures/05_kirpilan_bolge.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================================
# Bolum 3 — Temel Goruntu Manipulasyonu
# ============================================================================
# Kirpma, yeniden boyutlandirma, dondurme, yatay/dikey cevirme.

# ── Kirpma (Cropping) ─────────────────────────────────────────────────────────
crop_h0, crop_h1 = int(h * 0.1), int(h * 0.9)
crop_w0, crop_w1 = int(w * 0.1), int(w * 0.9)
crop = img_array[crop_h0:crop_h1, crop_w0:crop_w1]
print(f"Kirpma sonrasi sekil: {crop.shape}")

# ── Yeniden Boyutlandirma (Resize) ────────────────────────────────────────────
img_small = cv2.resize(img_array, (320, 240))
img_large = cv2.resize(img_array, (800, 600))
print(f"Kucultulmus: {img_small.shape}")
print(f"Buyutulmus : {img_large.shape}")

# ── Dondurme (Rotation) ───────────────────────────────────────────────────────
img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

center = (w // 2, h // 2)
M = cv2.getRotationMatrix2D(center, 45, 1.0)
img_rotated_bgr = cv2.warpAffine(img_bgr, M, (w, h))
img_rotated = cv2.cvtColor(img_rotated_bgr, cv2.COLOR_BGR2RGB)

# ── Yatay Cevirme (Horizontal Flip) ──────────────────────────────────────────
img_flip_h = np.fliplr(img_array)

# ── Dikey Cevirme (Vertical Flip) ────────────────────────────────────────────
img_flip_v = np.flipud(img_array)

fig, axes = plt.subplots(2, 3, figsize=(16, 10))

axes[0, 0].imshow(img_array); axes[0, 0].set_title("Orijinal"); axes[0, 0].axis('off')
axes[0, 1].imshow(crop); axes[0, 1].set_title("Kirpma"); axes[0, 1].axis('off')
axes[0, 2].imshow(img_small); axes[0, 2].set_title("Kucultulmus 320x240"); axes[0, 2].axis('off')
axes[1, 0].imshow(img_rotated); axes[1, 0].set_title("45° Dondurme"); axes[1, 0].axis('off')
axes[1, 1].imshow(img_flip_h); axes[1, 1].set_title("Yatay Ayna"); axes[1, 1].axis('off')
axes[1, 2].imshow(img_flip_v); axes[1, 2].set_title("Dikey Ayna"); axes[1, 2].axis('off')

plt.suptitle("Temel Goruntu Manipulasyonlari", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/06_temel_manipulasyon.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================================
# Bolum 4 — Renk Uzaylari
# ============================================================================
# RGB, Grayscale, HSV, LAB donusumleri.

# ── RGB -> Grayscale Donusumu ──────────────────────────────────────────────────
gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
print(f"Grayscale sekli: {gray.shape}")

gray_manual = (0.2989 * img_array[:, :, 0] +
               0.5870 * img_array[:, :, 1] +
               0.1140 * img_array[:, :, 2])
print(f"Manuel grayscale farki (max): {abs(gray.astype(float) - gray_manual).max():.2f}")

# ── RGB -> HSV Donusumu ────────────────────────────────────────────────────────
img_bgr_orig = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
hsv = cv2.cvtColor(img_bgr_orig, cv2.COLOR_BGR2HSV)
print(f"HSV sekli: {hsv.shape}")

# ── RGB -> LAB Donusumu ────────────────────────────────────────────────────────
lab = cv2.cvtColor(img_bgr_orig, cv2.COLOR_BGR2LAB)
print(f"LAB sekli: {lab.shape}")

fig, axes = plt.subplots(2, 4, figsize=(30, 9))

axes[0, 0].imshow(img_array); axes[0, 0].set_title("RGB — Orijinal"); axes[0, 0].axis('off')
axes[0, 1].imshow(gray, cmap='gray'); axes[0, 1].set_title("Grayscale"); axes[0, 1].axis('off')
axes[0, 2].imshow(hsv[:, :, 0], cmap='hsv'); axes[0, 2].set_title("HSV — Ton (H)"); axes[0, 2].axis('off')
axes[0, 3].imshow(lab[:, :, 0], cmap='gray'); axes[0, 3].set_title("LAB — Parlaklik (L)"); axes[0, 3].axis('off')

axes[1, 0].imshow(hsv[:, :, 0], cmap='hsv'); axes[1, 0].set_title("H — Ton"); axes[1, 0].axis('off')
axes[1, 1].imshow(hsv[:, :, 1], cmap='gray'); axes[1, 1].set_title("S — Doyma"); axes[1, 1].axis('off')
axes[1, 2].imshow(hsv[:, :, 2], cmap='gray'); axes[1, 2].set_title("V — Parlaklik"); axes[1, 2].axis('off')
axes[1, 3].imshow(lab[:, :, 1], cmap='RdYlGn'); axes[1, 3].set_title("A — Kirmizi-Yesil"); axes[1, 3].axis('off')

plt.suptitle("Renk Uzayi Donusumleri", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/07_renk_uzaylari.png', dpi=150, bbox_inches='tight')
plt.close()

# ── Pratik Uygulama: HSV ile Renk Maskesi ─────────────────────────────────────
lower_blue = np.array([100, 50, 50])
upper_blue = np.array([140, 255, 255])

mask = cv2.inRange(hsv, lower_blue, upper_blue)

masked_bgr = cv2.bitwise_and(img_bgr_orig, img_bgr_orig, mask=mask)
masked_rgb = cv2.cvtColor(masked_bgr, cv2.COLOR_BGR2RGB)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].imshow(img_array); axes[0].set_title("Orijinal RGB"); axes[0].axis('off')
axes[1].imshow(mask, cmap='gray'); axes[1].set_title("Mavi Renk Maskesi\n(beyaz = mavi pikseller)"); axes[1].axis('off')
axes[2].imshow(masked_rgb); axes[2].set_title("Maskeli Sonuc\n(sadece mavi bolgeler)"); axes[2].axis('off')
plt.suptitle("HSV ile Renk Segmentasyonu (Mavi)", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/08_gri_donusum.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================================
# Bolum 5 — Histogram Analizi
# ============================================================================
hist_gray, bins = np.histogram(gray.flatten(), 256, [0, 256])

colors = ['red', 'green', 'blue']
channel_names = ['Kirmizi (R)', 'Yesil (G)', 'Mavi (B)']

fig = plt.figure(figsize=(16, 8))

ax1 = fig.add_subplot(2, 3, 1)
ax1.imshow(img_array)
ax1.set_title("Orijinal Goruntu")
ax1.axis('off')

ax2 = fig.add_subplot(2, 3, 4)
ax2.imshow(gray, cmap='gray')
ax2.set_title("Gri Goruntu")
ax2.axis('off')

ax4 = fig.add_subplot(2, 3, 2)
for i, (color, name) in enumerate(zip(colors, channel_names)):
    hist_c, _ = np.histogram(img_array[:, :, i].flatten(), 256, [0, 256])
    ax4.plot(hist_c, color=color, alpha=0.7, linewidth=1, label=name)
ax4.set_title("RGB Histogramlari")
ax4.set_xlabel("Piksel Degeri (0-255)")
ax4.legend(fontsize=8)
ax4.set_xlim([0, 256])

ax3 = fig.add_subplot(2, 3, 5)
ax3.plot(hist_gray, color='gray', linewidth=1)
ax3.fill_between(range(256), hist_gray, alpha=0.3, color='gray')
ax3.set_title("Gri Histogram")
ax3.set_xlabel("Piksel Degeri (0-255)")
ax3.set_ylabel("Piksel Sayisi")
ax3.set_xlim([0, 256])

# Sag: Histogram Esitleme — dusuk kontrastli bir bolge sec
dark_h0, dark_h1 = int(h * 0.3), int(h * 0.7)
dark_w0, dark_w1 = 0, int(w * 0.4)
dark_region = img_array[dark_h0:dark_h1, dark_w0:dark_w1]
dark_gray = cv2.cvtColor(dark_region, cv2.COLOR_RGB2GRAY)
equalized = cv2.equalizeHist(dark_gray)

ax5 = fig.add_subplot(2, 3, 3)
ax5.imshow(dark_gray, cmap='gray')
ax5.set_title("Karanlik Bolge (Orijinal)")
ax5.axis('off')

ax6 = fig.add_subplot(2, 3, 6)
ax6.imshow(equalized, cmap='gray')
ax6.set_title("Histogram Esitleme Sonrasi\n(Kontrast iyilesti)")
ax6.axis('off')

plt.suptitle("Histogram Analizi ve Esitleme", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/09_histogram_analizi.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================================
# Bolum 6 — Goruntu Filtreleme
# ============================================================================
rng = np.random.default_rng(42)
noise = rng.integers(0, 60, gray.shape, dtype=np.uint8)
noisy = np.clip(gray.astype(int) + noise, 0, 255).astype(np.uint8)

# ── 1. Ortalama (Box) Filtre ──────────────────────────────────────────────────
kernel_box = np.ones((5, 5), np.float32) / 25
blur_box = cv2.filter2D(noisy, -1, kernel_box)

# ── 2. Gaussian Bulaniklastirma ───────────────────────────────────────────────
blur_gauss = cv2.GaussianBlur(noisy, (5, 5), sigmaX=1.5)

# ── 3. Medyan Filtre ──────────────────────────────────────────────────────────
blur_median = cv2.medianBlur(noisy, 5)

# ── 4. Keskinlestirme (Sharpening) ────────────────────────────────────────────
kernel_sharp = np.array([[0, -1, 0],
                          [-1, 5, -1],
                          [0, -1, 0]])
sharp = cv2.filter2D(gray, -1, kernel_sharp)

# ── 5. Bilateral Filtre ───────────────────────────────────────────────────────
bilateral = cv2.bilateralFilter(noisy, d=9, sigmaColor=75, sigmaSpace=75)

fig, axes = plt.subplots(2, 3, figsize=(16, 10))

imgs = [gray, noisy, blur_box, blur_gauss, blur_median, bilateral]
titles = [
    "Orijinal",
    "Gurultulu (+Gaussian gurultu)",
    "Box Filter (Ortalama 5x5)",
    "Gaussian Blur (sigma=1.5)",
    "Medyan Filtre",
    "Bilateral Filtre (Kenar koruyucu)"
]

for ax, image, title in zip(axes.flatten(), imgs, titles):
    ax.imshow(image, cmap='gray')
    ax.set_title(title, fontsize=10)
    ax.axis('off')

plt.suptitle("Goruntu Filtreleme Karsilastirmasi", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/10_goruntu_filtreleme.png', dpi=150, bbox_inches='tight')
plt.close()


def calculate_psnr(original, filtered):
    """PSNR hesapla: Orijinal ve filtrelenmis goruntuyu karsilastirir."""
    mse = np.mean((original.astype(float) - filtered.astype(float)) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr


filters = {
    'Box Filter': blur_box,
    'Gaussian Blur': blur_gauss,
    'Medyan Filtre': blur_median,
    'Bilateral Filtre': bilateral,
}

print("=== PSNR Karsilastirmasi (dB) ===")
print(f"{'Filtre':<20} {'PSNR (dB)':>10}")
print("-" * 32)
for name, filtered in filters.items():
    psnr = calculate_psnr(gray, filtered)
    bar = "#" * int(psnr / 2)
    print(f"{name:<20} {psnr:>8.2f} dB  {bar}")
print("\nNot: Daha yuksek PSNR = Orijinale daha yakin (iyi) filtreleme")

# ============================================================================
# Bolum 7 — Kenar Tespiti (Edge Detection)
# ============================================================================
test_img = cv2.GaussianBlur(gray, (3, 3), 0)

# ── 1. Sobel Kenar Dedektoru ──────────────────────────────────────────────────
sobel_x = cv2.Sobel(test_img, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(test_img, cv2.CV_64F, 0, 1, ksize=3)
sobel_total = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
sobel_total = np.clip(sobel_total, 0, 255).astype(np.uint8)

# ── 2. Laplacian Kenar Dedektoru ──────────────────────────────────────────────
laplacian = cv2.Laplacian(test_img, cv2.CV_64F, ksize=3)
laplacian_abs = np.abs(laplacian).astype(np.uint8)

# ── 3. Canny Kenar Dedektoru ──────────────────────────────────────────────────
canny_strict = cv2.Canny(test_img, threshold1=100, threshold2=200)
canny_loose = cv2.Canny(test_img, threshold1=30, threshold2=100)

fig, axes = plt.subplots(2, 3, figsize=(16, 10))

axes[0, 0].imshow(test_img, cmap='gray'); axes[0, 0].set_title("Orijinal (Gri)"); axes[0, 0].axis('off')
axes[0, 1].imshow(np.abs(sobel_x).astype(np.uint8), cmap='gray'); axes[0, 1].set_title("Sobel X\n(Dikey kenarlar)"); axes[0, 1].axis('off')
axes[0, 2].imshow(np.abs(sobel_y).astype(np.uint8), cmap='gray'); axes[0, 2].set_title("Sobel Y\n(Yatay kenarlar)"); axes[0, 2].axis('off')
axes[1, 0].imshow(sobel_total, cmap='gray'); axes[1, 0].set_title("Sobel Toplam\nsqrt(Gx^2+Gy^2)"); axes[1, 0].axis('off')
axes[1, 1].imshow(laplacian_abs, cmap='gray'); axes[1, 1].set_title("Laplacian"); axes[1, 1].axis('off')
axes[1, 2].imshow(canny_strict, cmap='gray'); axes[1, 2].set_title("Canny (100/200)\n(En temiz sonuc)"); axes[1, 2].axis('off')

plt.suptitle("Kenar Tespiti Algoritmalari Karsilastirmasi", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/11_kenar_tespiti.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================================
# Bolum 8 — Morfolojik Islemler
# ============================================================================
binary = cv2.Canny(test_img, 50, 150)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

erosion = cv2.erode(binary, kernel, iterations=2)
dilation = cv2.dilate(binary, kernel, iterations=2)
opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
gradient = cv2.morphologyEx(binary, cv2.MORPH_GRADIENT, kernel)

fig, axes = plt.subplots(2, 3, figsize=(16, 10))

axes[0, 0].imshow(binary, cmap='gray'); axes[0, 0].set_title("Orijinal Binary"); axes[0, 0].axis('off')
axes[0, 1].imshow(erosion, cmap='gray'); axes[0, 1].set_title("Erosion (Asindirma)"); axes[0, 1].axis('off')
axes[0, 2].imshow(dilation, cmap='gray'); axes[0, 2].set_title("Dilation (Genisletme)"); axes[0, 2].axis('off')
axes[1, 0].imshow(opening, cmap='gray'); axes[1, 0].set_title("Opening\n(Gurultu temizle)"); axes[1, 0].axis('off')
axes[1, 1].imshow(closing, cmap='gray'); axes[1, 1].set_title("Closing\n(Delik kapat)"); axes[1, 1].axis('off')
axes[1, 2].imshow(gradient, cmap='gray'); axes[1, 2].set_title("Morphological Gradient\n(Kenar kalinligi)"); axes[1, 2].axis('off')

plt.suptitle("Morfolojik Islemler", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/12_morfolojik_islemler.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================================
# Bolum 9 — Esikleme (Thresholding)
# ============================================================================
test_h0, test_h1 = int(h * 0.2), int(h * 0.8)
test_w0, test_w1 = int(w * 0.2), int(w * 0.8)
test = gray[test_h0:test_h1, test_w0:test_w1]

# ── 1. Global (Manuel) Esikleme ─────────────────────────────────────────────
_, thresh_global = cv2.threshold(test, 127, 255, cv2.THRESH_BINARY)

# ── 2. Otsu Yontemi ──────────────────────────────────────────────────────────
otsu_val, thresh_otsu = cv2.threshold(test, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
print(f"Otsu'nun hesapladigi esik degeri: {otsu_val:.0f}")

# ── 3. Adaptif Ortalama Esikleme ─────────────────────────────────────────────
thresh_adapt_mean = cv2.adaptiveThreshold(
    test, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, blockSize=11, C=2
)

# ── 4. Adaptif Gaussian Esikleme ─────────────────────────────────────────────
thresh_adapt_gauss = cv2.adaptiveThreshold(
    test, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blockSize=11, C=2
)

fig, axes = plt.subplots(2, 3, figsize=(16, 10))

axes[0, 0].imshow(test, cmap='gray'); axes[0, 0].set_title("Orijinal Gri"); axes[0, 0].axis('off')
axes[0, 1].imshow(thresh_global, cmap='gray'); axes[0, 1].set_title("Global Esik (127)"); axes[0, 1].axis('off')
axes[0, 2].imshow(thresh_otsu, cmap='gray'); axes[0, 2].set_title(f"Otsu (esik={otsu_val:.0f})"); axes[0, 2].axis('off')
axes[1, 0].imshow(thresh_adapt_mean, cmap='gray'); axes[1, 0].set_title("Adaptif Ortalama"); axes[1, 0].axis('off')
axes[1, 1].imshow(thresh_adapt_gauss, cmap='gray'); axes[1, 1].set_title("Adaptif Gaussian"); axes[1, 1].axis('off')

axes[1, 2].hist(test.flatten(), 256, [0, 256], color='steelblue', alpha=0.7)
axes[1, 2].axvline(x=otsu_val, color='red', linewidth=2, linestyle='--', label=f"Otsu Esigi ({otsu_val:.0f})")
axes[1, 2].axvline(x=127, color='orange', linewidth=2, linestyle='--', label="Manuel Esik (127)")
axes[1, 2].set_title("Histogram + Esik Degerleri")
axes[1, 2].legend()
axes[1, 2].set_xlabel("Piksel Degeri")
axes[1, 2].set_ylabel("Frekans")

plt.suptitle("Esikleme Yontemleri Karsilastirmasi", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/13_esikleme.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================================
# Bolum 10 — Kontur Tespiti ve Nesne Analizi
# ============================================================================
prep_h0, prep_h1 = int(h * 0.05), int(h * 0.95)
prep_w0, prep_w1 = int(w * 0.05), int(w * 0.95)
prep = gray[prep_h0:prep_h1, prep_w0:prep_w1]
prep_blur = cv2.GaussianBlur(prep, (5, 5), 0)
_, prep_thresh = cv2.threshold(prep_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

contours, hierarchy = cv2.findContours(
    prep_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)
print(f"Bulunan toplam kontur sayisi: {len(contours)}")

min_area = 200
large_contours = [c for c in contours if cv2.contourArea(c) > min_area]
print(f"Alan > {min_area} px^2 olan konturlar: {len(large_contours)}")

canvas = cv2.cvtColor(prep, cv2.COLOR_GRAY2BGR)
cv2.drawContours(canvas, large_contours, -1, (0, 255, 0), 2)

for i, cnt in enumerate(large_contours[:10]):
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, closed=True)

    x, y, cw, ch = cv2.boundingRect(cnt)
    cv2.rectangle(canvas, (x, y), (x + cw, y + ch), (255, 0, 0), 1)

    moments = cv2.moments(cnt)
    if moments['m00'] != 0:
        cx = int(moments['m10'] / moments['m00'])
        cy = int(moments['m01'] / moments['m00'])
        cv2.circle(canvas, (cx, cy), 3, (0, 0, 255), -1)

canvas_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
axes[0].imshow(prep, cmap='gray'); axes[0].set_title("Orijinal"); axes[0].axis('off')
axes[1].imshow(prep_thresh, cmap='gray'); axes[1].set_title("Esiklenmis (Otsu)"); axes[1].axis('off')
axes[2].imshow(canvas_rgb); axes[2].set_title(f"Konturlar ({len(large_contours)} adet)\nYesil=Kontur Mavi=BBox Kirmizi=Merkez"); axes[2].axis('off')

plt.suptitle("Kontur Tespiti ve Nesne Analizi", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/14_kontur_tespiti.png', dpi=150, bbox_inches='tight')
plt.close()

print("\nTum bolumler tamamlandi, gorseller figures/ klasorune kaydedildi.")
