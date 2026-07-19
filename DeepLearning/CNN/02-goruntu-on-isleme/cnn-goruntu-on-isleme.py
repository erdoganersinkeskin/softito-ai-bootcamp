# -*- coding: utf-8 -*-
"""
GORUNTU ON ISLEME (OYUN IKONLARI VERSIYONU) - Veri Artirma, PCA ve t-SNE
Amac: Bir goruntu siniflandirma veri setinde CNN'e gecmeden once klasik
      ozellik muhendisligi adimlarini (augmentation, PCA, t-SNE) uygulamak
      ve bunlarin model basarisina etkisini gostermek.

ÖĞRENME NOTU (yönlendirme):
  Bu dosyanın amacı, augmentation -> MLP -> PCA -> t-SNE islem hattini
  kurmaktır. Paylasilan 9 Kaggle veri setinden hicbiri ham piksel
  goruntusu icermedigi (hepsi tablo verisi) icin, gorev tanimindaki
  "uygun veri seti yoksa baska uygun bir veri seti/yontem kullanilabilir"
  istisnasi uygulanarak SENTETIK sekiller (Daire/Kare/Ucgen/Yildiz) oyun
  ikonlarina cevrilerek uretilir: Coin (Para), Shield (Kalkan), Potion
  (Iksir), PowerUp (Guclendirme).
"""
import os
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.makedirs('figures', exist_ok=True)

IMG_SIZE = 64
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

print("Sentetik oyun ikonu veri seti uretiliyor...")

class_names = ["Coin", "Shield", "Potion", "PowerUp"]
class_to_idx = {name: i for i, name in enumerate(class_names)}
n_per_class = 150


def draw_icon(icon_name, size=64):
    """Tek bir sentetik oyun ikonu goruntusu (gri tonlama) cizer."""
    img = np.zeros((size, size), dtype=np.uint8)
    cx, cy = size // 2 + np.random.randint(-6, 6), size // 2 + np.random.randint(-6, 6)
    r = np.random.randint(size // 4, size // 2 - 4)
    thickness = np.random.choice([-1, 2, 3])  # -1: dolu, aksi halde cizgi kalinligi
    color = int(np.random.randint(180, 256))

    if icon_name == "Coin":
        cv2.circle(img, (cx, cy), r, color, thickness)
    elif icon_name == "Shield":
        cv2.rectangle(img, (cx - r, cy - r), (cx + r, cy + r), color, thickness)
    elif icon_name == "Potion":
        pts = np.array([
            [cx, cy - r],
            [cx - r, cy + r],
            [cx + r, cy + r],
        ], dtype=np.int32)
        if thickness == -1:
            cv2.fillPoly(img, [pts], color)
        else:
            cv2.polylines(img, [pts], True, color, thickness)
    elif icon_name == "PowerUp":
        pts = []
        for i in range(10):
            angle = i * np.pi / 5 - np.pi / 2
            radius = r if i % 2 == 0 else r * 0.45
            pts.append([cx + radius * np.cos(angle), cy + radius * np.sin(angle)])
        pts = np.array(pts, dtype=np.int32)
        if thickness == -1:
            cv2.fillPoly(img, [pts], color)
        else:
            cv2.polylines(img, [pts], True, color, thickness)

    # Gercekci hale getirmek icin hafif gauss gurultusu ekle
    noise = np.random.normal(0, 12, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return img


X_custom = []
y_custom = []
for class_name in class_names:
    label = class_to_idx[class_name]
    for _ in range(n_per_class):
        img = draw_icon(class_name, IMG_SIZE)
        img_flat = img.flatten() / 255.0
        X_custom.append(img_flat)
        y_custom.append(label)

X_custom = np.array(X_custom)
y_custom = np.array(y_custom)

print(f"Uretilen X_custom boyutu: {X_custom.shape}")
print(f"Uretilen y_custom boyutu: {y_custom.shape}")
print(f"Siniflar: {class_names}")

# Uretilen ikonlardan birini gorsellestir
sample_index = np.random.randint(0, len(X_custom))
plt.figure(figsize=(3, 3))
plt.imshow(X_custom[sample_index].reshape(IMG_SIZE, IMG_SIZE), cmap='gray')
plt.title(f"Ornek Ikon (Etiket: {class_names[y_custom[sample_index]]})")
plt.axis('off')
plt.savefig('figures/00_ornek_goruntu.png', dpi=150, bbox_inches='tight')
plt.close()

# ── Egitim/Test Bolmesi ───────────────────────────────────────────────────────
# X_custom ve y_custom NumPy dizileri hazir oldugundan, bunlari egitim ve
# test setlerine ayirip StandardScaler ile standartlastiriyoruz.
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# stratify=y_custom, her iki kumede de sinif oranlarinin korunmasini saglar
if len(X_custom) > 0:
    X_train_custom, X_test_custom, y_train_custom, y_test_custom = train_test_split(
        X_custom, y_custom, test_size=0.25, random_state=42, stratify=y_custom
    )

    # StandardScaler'i yalnizca egitim verisine fit ediyoruz, test verisine
    # sadece transform uyguluyoruz (veri sizintisini onlemek icin)
    scaler_custom = StandardScaler()
    X_train_sc_custom = scaler_custom.fit_transform(X_train_custom)
    X_test_sc_custom = scaler_custom.transform(X_test_custom)

    print(f"\nEgitim seti boyutu: {X_train_sc_custom.shape}")
    print(f"Test seti boyutu: {X_test_sc_custom.shape}")
    print("Ikon veri seti basariyla hazirlandi!")
else:
    print("Hata: X_custom bos oldugu icin egitim/test bolmesi yapilamadi.")
    X_train_custom = np.array([])
    X_test_custom = np.array([])
    y_train_custom = np.array([])
    y_test_custom = np.array([])
    X_train_sc_custom = np.array([])
    X_test_sc_custom = np.array([])

# ── Veri Artirma (Data Augmentation) ──────────────────────────────────────────
# Problem: Gercek projelerde etiketli veri az olur. Modeli overfit'ten korumak
# icin mevcut goruntuler donusturulerek yapay yeni ornekler uretilir.
# Neden ise yarar? Model ayni ikonun farkli aci/parlaklik/gurultu
# versiyonlarini gorunce gercek dunya varyasyonlarini ogrenir.
# Sira: Orijinal -> Dondur -> Kaydir -> Olcekle -> Gurultu ekle -> Ayna -> Parlaklik

def augment_image(img_flat_1d, image_size, seed=None):
    """Tek bir duzlestirilmis (flattened) goruntuye rastgele augmentation uygular.
    Doner: sozluk - her donusumun sonucu ayri ayri."""

    rng = np.random.default_rng(seed)  # Tekrar uretilebilir rastgelelik

    # Duzlestirilmis goruntuyu orijinal boyuta (image_size x image_size) geri getir
    img_2d = (img_flat_1d * 255.0).astype(np.uint8).reshape(image_size, image_size)

    results = {}
    results['original'] = img_flat_1d.copy()  # Orijinali sakla (duzlestirilmis halde)

    h, w = img_2d.shape

    # ── 1. Dondurme (Rotation) ──────────────────────────────────────────────
    angle = rng.uniform(-25, 25)  # -25 ile +25 derece arasi rastgele aci
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)  # Donme matrisi
    rotated = cv2.warpAffine(img_2d, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    results['rotation'] = (rotated.astype(float) / 255.0).flatten()

    # ── 2. Kaydirma (Translation) ────────────────────────────────────────────
    tx = rng.uniform(-image_size * 0.1, image_size * 0.1)
    ty = rng.uniform(-image_size * 0.1, image_size * 0.1)
    M_trans = np.float32([[1, 0, tx], [0, 1, ty]])
    shifted = cv2.warpAffine(img_2d, M_trans, (w, h), borderMode=cv2.BORDER_REPLICATE)
    results['translation'] = (shifted.astype(float) / 255.0).flatten()

    # ── 3. Yatay Ayna (Horizontal Flip) ──────────────────────────────────────
    flipped = np.fliplr(img_2d)
    results['flip'] = (flipped.astype(float) / 255.0).flatten()

    # ── 4. Gaussian Gurultu ───────────────────────────────────────────────────
    sigma = rng.uniform(5, 20)
    noise = rng.normal(0, sigma, img_2d.shape).astype(np.int16)
    noisy = np.clip(img_2d.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    results['noise'] = (noisy.astype(float) / 255.0).flatten()

    # ── 5. Parlaklik Degistirme (Brightness) ─────────────────────────────────
    factor = rng.uniform(0.6, 1.6)
    bright = np.clip(img_2d.astype(float) * factor, 0, 255).astype(np.uint8)
    results['brightness'] = (bright.astype(float) / 255.0).flatten()

    # ── 6. Zoom (Kirp + Yeniden Boyutlandir) ─────────────────────────────────
    zoom = rng.uniform(0.7, 0.95)
    crop_h = int(h * zoom)
    crop_w = int(w * zoom)
    y0 = rng.integers(0, h - crop_h + 1)
    x0 = rng.integers(0, w - crop_w + 1)
    cropped = img_2d[y0:y0 + crop_h, x0:x0 + crop_w]
    zoomed = cv2.resize(cropped, (w, h))
    results['zoom'] = (zoomed.astype(float) / 255.0).flatten()

    return results


# ── Ilk birkac ornek uzerinde augmentation goster ────────────────────────────
if len(X_custom) > 0 and len(class_names) > 0:
    sample_idx = np.random.randint(0, len(X_custom))
    sample_img_flat = X_custom[sample_idx]
    sample_label = y_custom[sample_idx]
    sample_class_name = class_names[sample_label]

    aug_results = augment_image(sample_img_flat, IMG_SIZE, seed=42)

    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    axes = axes.flatten()

    titles = ['Orijinal', 'Dondurme\n(±25°)', 'Kaydirma\n(±10%)',
              'Yatay Ayna', 'Gaussian\nGurultu', 'Parlaklik\nDegistirme',
              'Zoom\n(Kirp+Yeniden Boyutlandir)', 'Cakisma\n(Hepsi Birden)']

    for i, (key, img_flat_res) in enumerate(aug_results.items()):
        img_2d_res = img_flat_res.reshape(IMG_SIZE, IMG_SIZE)
        axes[i].imshow(img_2d_res, cmap='gray_r', interpolation='nearest')
        axes[i].set_title(titles[i], fontsize=11, fontweight='bold')
        axes[i].set_xlabel(f"Shape: {img_2d_res.shape}\nMin:{img_2d_res.min():.2f} Max:{img_2d_res.max():.2f}", fontsize=8)
        axes[i].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        axes[i].set_xticks(np.arange(-0.5, IMG_SIZE, 1), minor=True)
        axes[i].set_yticks(np.arange(-0.5, IMG_SIZE, 1), minor=True)
        axes[i].grid(which='minor', color='gray', linewidth=0.3, alpha=0.5)

    # Son panel: tum augmentasyonlarin yiginlanmis goruntusu
    combined_flat = np.mean([v for v in aug_results.values()], axis=0)
    combined_2d = combined_flat.reshape(IMG_SIZE, IMG_SIZE)
    axes[7].imshow(combined_2d, cmap='gray_r', interpolation='nearest')
    axes[7].set_title(titles[7], fontsize=11, fontweight='bold')
    axes[7].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    plt.suptitle(f"Veri Artirma (Data Augmentation) — Sinif: {sample_class_name}",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('figures/01_veri_artirma_augmentation.png', dpi=150, bbox_inches='tight')
    plt.close()
else:
    print("Hata: X_custom veya class_names bos, ornek goruntu yuklenemedi.")

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score


def build_augmented_dataset(X_flat_data, y_data, image_size, n_aug=1, seed=0):
    """Her goruntuye n_aug kez augmentation uygulayarak veri setini buyutur.
    Doner: (X_aug, y_aug) - orijinaller dahil."""

    X_list, y_list = [X_flat_data], [y_data]
    rng = np.random.default_rng(seed)

    for aug_round in range(n_aug):
        X_new = []
        for i, img_flat in enumerate(X_flat_data):
            s = rng.integers(0, 10000)
            aug = augment_image(img_flat, image_size, seed=int(s))
            # Her turda farkli bir augmentation tipi sec ('original' haric)
            key = list(aug.keys())[1 + (aug_round % (len(aug.keys()) - 1))]
            X_new.append(aug[key])
        X_list.append(np.array(X_new))
        y_list.append(y_data)

    return np.vstack(X_list), np.concatenate(y_list)


print("Augmentation uygulaniyor...")

if len(X_train_custom) > 0 and len(y_train_custom) > 0:
    X_aug, y_aug = build_augmented_dataset(X_train_custom, y_train_custom, IMG_SIZE, n_aug=1)
    print(f"Orijinal egitim seti  : {len(y_train_custom)} ornek")
    print(f"Augmented egitim seti : {len(y_aug)} ornek  ({(len(y_aug) / len(y_train_custom)):.0f}x buyudu)")

    sc_aug = StandardScaler()
    X_aug_sc = sc_aug.fit_transform(X_aug)

    mlp_original = MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=300, random_state=42)
    mlp_augmented = MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=300, random_state=42)

    mlp_original.fit(X_train_sc_custom, y_train_custom)
    mlp_augmented.fit(X_aug_sc, y_aug)

    acc_original = accuracy_score(y_test_custom, mlp_original.predict(X_test_sc_custom))
    acc_augmented = accuracy_score(y_test_custom, mlp_augmented.predict(X_test_sc_custom))

    print(f"\nTest Dogrulugu — Orijinal veri : {acc_original*100:.2f}%")
    print(f"Test Dogrulugu — Augmented veri: {acc_augmented*100:.2f}%")
    print(f"Augmentation kazanci            : {(acc_augmented-acc_original)*100:+.2f} puan")
else:
    print("Hata: X_train_custom veya y_train_custom bos oldugu icin augmentation yapilamadi.")

# ── PCA ile Boyut Indirgeme ───────────────────────────────────────────────────
# Problem: 64x64=4096 ozellik -> gorsellestirme imkansiz.
# PCA: Veriyi en fazla varyansi koruyan eksenlere yansitir.
# Her oz vektor bir "eigenimage" (oz goruntu) gibi dusunulebilir.
from sklearn.decomposition import PCA
from matplotlib.gridspec import GridSpec

if len(X_custom) > 0:
    pca_full = PCA()
    pca_full.fit(X_custom)

    cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
    n90 = np.searchsorted(cumulative_variance, 0.90) + 1
    n95 = np.searchsorted(cumulative_variance, 0.95) + 1
    n99 = np.searchsorted(cumulative_variance, 0.99) + 1

    print(f"%90 varyans icin gerekli bilesen: {n90}")
    print(f"%95 varyans icin gerekli bilesen: {n95}")
    print(f"%99 varyans icin gerekli bilesen: {n99}")
    print(f"Orijinal boyut: {X_custom.shape[1]}  ->  Sikistirma orani (%90): {X_custom.shape[1]/n90:.1f}x")

    # ── Eigenicons (PCA bilesenlerini gorsellestir) ──────────────────────────
    num_components_to_show = min(len(pca_full.components_), 21)
    num_cols = 7
    num_rows = int(np.ceil(num_components_to_show / num_cols))

    fig = plt.figure(figsize=(18, num_rows * 3 + 2))
    gs = GridSpec(num_rows + 1, num_cols, height_ratios=[0.5] + [1] * num_rows)

    ax_var = fig.add_subplot(gs[0, :])
    ax_var.bar(range(1, num_components_to_show + 1), pca_full.explained_variance_ratio_[:num_components_to_show] * 100,
               color='steelblue', alpha=0.7, label='Tekil bilesen')
    ax_var.plot(range(1, num_components_to_show + 1), cumulative_variance[:num_components_to_show] * 100,
                'ro-', markersize=4, linewidth=1.5, label='Kumulatif')
    ax_var.axhline(y=90, color='orange', linestyle='--', linewidth=1, label='%90')
    ax_var.axhline(y=95, color='red', linestyle='--', linewidth=1, label='%95')
    ax_var.set_xlabel("Bilesen Numarasi")
    ax_var.set_ylabel("Aciklanan Varyans (%)")
    ax_var.set_title("PCA — Her Bilesenin Katkisi", fontweight='bold')
    ax_var.legend(loc='right', fontsize=8)
    ax_var.set_xlim(0.5, num_components_to_show + 0.5)

    for i in range(num_components_to_show):
        ax = fig.add_subplot(gs[i // num_cols + 1, i % num_cols])
        eigenimage = pca_full.components_[i].reshape(IMG_SIZE, IMG_SIZE)
        im = ax.imshow(eigenimage, cmap='RdBu_r',
                        vmin=-np.max(np.abs(eigenimage)), vmax=np.max(np.abs(eigenimage)))
        ax.set_title(f"PC {i+1}\n({pca_full.explained_variance_ratio_[i]*100:.1f}%)", fontsize=8)
        ax.axis('off')

    fig.colorbar(im, ax=ax_var, orientation='horizontal', fraction=0.02, pad=0.04,
                 label='Bilesen agirligi (negatif <-> pozitif)')

    plt.suptitle("Eigenicons — PCA Temel Bilesenleri\n(Her goruntu bir 'temel ikon' kalibini temsil eder)",
                 fontsize=13, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('figures/02_pca_eigendigits.png', dpi=150, bbox_inches='tight')
    plt.close()
else:
    print("Hata: X_custom bos oldugu icin PCA uygulanamadi.")

# ── PCA ile Goruntu Sikistirma ve Yeniden Yapilandirma ────────────────────────
# n bilesenle sikistir -> orijinal boyuta geri dondur ve kaliteyi karsilastir
if len(X_custom) > 0 and len(class_names) > 0:
    original_dim = X_custom.shape[1]
    n_components_list = [2, 5, 10, 20, 30]
    if original_dim > 64:
        n_components_list.extend([64, 128])
    n_components_list = [n for n in n_components_list if n <= original_dim]

    sample_idx = np.random.randint(0, len(X_custom))
    original_flat = X_custom[sample_idx]
    original_label = y_custom[sample_idx]
    original_class_name = class_names[original_label]

    fig, axes = plt.subplots(1, len(n_components_list) + 1, figsize=(3 * (len(n_components_list) + 1), 3.5))

    axes[0].imshow(original_flat.reshape(IMG_SIZE, IMG_SIZE), cmap='gray_r', interpolation='nearest')
    axes[0].set_title(f"Orijinal\n({original_dim} ozellik)\nSinif: {original_class_name}", fontsize=10)
    axes[0].axis('off')

    for j, n in enumerate(n_components_list):
        pca_n = PCA(n_components=n)
        pca_n.fit(X_custom)

        compressed = pca_n.transform(X_custom[sample_idx:sample_idx + 1])
        reconstructed = pca_n.inverse_transform(compressed)

        mse = np.mean((original_flat - reconstructed[0]) ** 2)
        var_explained = np.sum(pca_n.explained_variance_ratio_) * 100

        axes[j + 1].imshow(reconstructed[0].reshape(IMG_SIZE, IMG_SIZE), cmap='gray_r', interpolation='nearest')
        axes[j + 1].set_title(f"n={n}\n%{var_explained:.0f} varyans\nMSE={mse:.2f}", fontsize=9)
        axes[j + 1].axis('off')

    plt.suptitle("PCA Goruntu Sikistirma — Bilesen Sayisi Arttikca Kalite Yukselir",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('figures/03_pca_sikistirma_kalite.png', dpi=150, bbox_inches='tight')
    plt.close()
else:
    print("Hata: X_custom bos oldugu icin PCA sikistirma uygulanamadi.")

# ── t-SNE ile Ozellik Uzayi Gorsellestirme ────────────────────────────────────
# t-SNE: Yuksek boyutlu veriyi 2D'ye indirger; benzer ornekler yakin,
# farkli olanlar uzak olur. PCA lineerdir, t-SNE non-lineerdir ve kume
# yapisini cok daha iyi ortaya cikarir.
from sklearn.manifold import TSNE
from matplotlib.offsetbox import AnnotationBbox, OffsetImage

if len(X_custom) > 0 and len(y_custom) > 0 and len(class_names) > 0:
    initial_pca_components = min(50, X_custom.shape[1] // 2)
    if initial_pca_components < 2:
        initial_pca_components = X_custom.shape[1]

    print(f"Adim 1: PCA ile {initial_pca_components} boyuta indir...")
    pca_initial = PCA(n_components=initial_pca_components, random_state=42)
    X_pca_reduced = pca_initial.fit_transform(X_custom)
    print(f"  PCA sonrasi: {X_pca_reduced.shape}")

    print("Adim 2: t-SNE ile 2 boyuta indir (bu biraz surebilir)...")
    tsne = TSNE(
        n_components=2,
        perplexity=30,
        max_iter=1000,
        random_state=42,
        learning_rate='auto',
        init='pca'
    )
    X_tsne = tsne.fit_transform(X_pca_reduced)
    print(f"  t-SNE sonrasi: {X_tsne.shape}")

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    colors = plt.cm.get_cmap('tab10', len(class_names))

    # Sol: t-SNE scatter plot
    for label_idx in range(len(class_names)):
        mask = y_custom == label_idx
        axes[0].scatter(
            X_tsne[mask, 0],
            X_tsne[mask, 1],
            c=[colors(label_idx)],
            label=class_names[label_idx],
            alpha=0.6,
            s=15,
            edgecolors='none'
        )

    axes[0].set_title("t-SNE — Tum Veri Seti\n(Her renk bir sinifi)", fontsize=13, fontweight='bold')
    axes[0].legend(title="Sinif", ncol=2, fontsize=9, loc='upper right')
    axes[0].set_xlabel("t-SNE Boyutu 1")
    axes[0].set_ylabel("t-SNE Boyutu 2")
    axes[0].set_facecolor('#f5f5f5')
    axes[0].grid(True, alpha=0.3)

    # Sag: t-SNE uzerine gercek ikonlari koy (ornekleme ile)
    axes[1].set_facecolor('#1a1a2e')
    axes[1].set_title("t-SNE — Gercek Ikonlarla\n(Her nokta gercek piksel degeri)",
                       fontsize=13, fontweight='bold', color='white')

    rng = np.random.default_rng(0)
    for label_idx in range(len(class_names)):
        mask_idx = np.where(y_custom == label_idx)[0]
        num_samples_to_show = min(15, len(mask_idx))
        if num_samples_to_show == 0:
            continue

        sample_indices = rng.choice(mask_idx, size=num_samples_to_show, replace=False)

        for idx in sample_indices:
            img_flat = X_custom[idx]
            img_2d = img_flat.reshape(IMG_SIZE, IMG_SIZE)
            x, y_pos = X_tsne[idx, 0], X_tsne[idx, 1]

            colored = colors(label_idx)[:3]

            imagebox = OffsetImage(img_2d, zoom=0.5, cmap='gray_r')
            ab = AnnotationBbox(imagebox, (x, y_pos),
                                 frameon=True,
                                 bboxprops=dict(
                                     boxstyle='round,pad=0.1',
                                     facecolor=(*colored, 0.3),
                                     edgecolor=(*colored, 0.9),
                                     linewidth=1.0
                                 ))
            axes[1].add_artist(ab)

    axes[1].set_xlim(X_tsne[:, 0].min() - 3, X_tsne[:, 0].max() + 3)
    axes[1].set_ylim(X_tsne[:, 1].min() - 3, X_tsne[:, 1].max() + 3)
    axes[1].tick_params(colors='white')
    for spine in axes[1].spines.values():
        spine.set_edgecolor('white')

    plt.tight_layout()
    plt.savefig('figures/04_tsne_gorsellestirme.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\nt-SNE'de: Birbirinden ayrisan kumeler = model kolayca ayirt edebilir demektir.")
else:
    print("Hata: X_custom, y_custom veya class_names bos oldugu icin t-SNE uygulanamadi.")
