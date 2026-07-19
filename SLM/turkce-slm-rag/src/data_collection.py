"""
Wikipedia'dan config.yaml'da tanımlı kategori ve makaleleri çeker.
Her makaleyi data/raw/<kategori>/<baslik>.txt olarak kaydeder.
Ayrıca hangi makalenin ne zaman, hangi kategoriden çekildiğini
data/metadata.json içinde tutar.

Kullanim:
    python src/data_collection.py --config configs/config.yaml
"""

import argparse
import json
import re
import time
from datetime import datetime
from pathlib import Path

import wikipediaapi
import yaml


def config_yukle(config_yolu: str) -> dict:
    with open(config_yolu, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def dosya_adi_temizle(baslik: str) -> str:
    """Makale başlığını güvenli bir dosya adına çevirir."""
    temiz = baslik.lower()
    temiz = temiz.replace(" ", "_")
    temiz = re.sub(r"[^a-z0-9ığüşöç_]", "", temiz)
    return temiz


def makale_cek(wiki: wikipediaapi.Wikipedia, baslik: str) -> str | None:
    """Tek bir makaleyi çeker. Bulunamazsa None döner."""
    sayfa = wiki.page(baslik)
    if not sayfa.exists():
        return None
    return sayfa.text


def kategorileri_isle(config: dict, cikti_dizini: Path) -> list[dict]:
    """Tüm kategorileri gezip makaleleri çeker, metadata listesi döner."""
    wiki = wikipediaapi.Wikipedia(
        language=config["wikipedia"]["dil"],
        user_agent=config["wikipedia"]["user_agent"],
    )

    metadata_listesi = []
    kategoriler = config["wikipedia"]["kategoriler"]

    for kategori_adi, basliklar in kategoriler.items():
        kategori_dizini = cikti_dizini / kategori_adi
        kategori_dizini.mkdir(parents=True, exist_ok=True)

        print(f"\nKategori: {kategori_adi} ({len(basliklar)} makale)")

        for baslik in basliklar:
            metin = makale_cek(wiki, baslik)

            if metin is None:
                print(f"  atlandi (bulunamadi): {baslik}")
                continue

            if len(metin) < config["preprocessing"]["min_makale_uzunlugu"]:
                print(f"  atlandi (cok kisa): {baslik}")
                continue

            dosya_adi = dosya_adi_temizle(baslik) + ".txt"
            dosya_yolu = kategori_dizini / dosya_adi
            dosya_yolu.write_text(metin, encoding="utf-8")

            metadata_listesi.append({
                "baslik": baslik,
                "kategori": kategori_adi,
                "dosya_yolu": str(dosya_yolu),
                "karakter_sayisi": len(metin),
                "cekilme_tarihi": datetime.now().isoformat(),
            })

            print(f"  cekildi: {baslik} ({len(metin)} karakter)")

            # Wikipedia API'ye nazik davranmak icin kucuk bir bekleme
            time.sleep(0.2)

    return metadata_listesi


def metadata_kaydet(metadata_listesi: list[dict], metadata_yolu: Path) -> None:
    metadata_yolu.parent.mkdir(parents=True, exist_ok=True)
    with open(metadata_yolu, "w", encoding="utf-8") as f:
        json.dump(metadata_listesi, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Wikipedia veri toplama")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()

    config = config_yukle(args.config)

    raw_dizin = Path(config["paths"]["raw_data_dir"])
    metadata_yolu = Path(config["paths"]["metadata_file"])

    metadata_listesi = kategorileri_isle(config, raw_dizin)
    metadata_kaydet(metadata_listesi, metadata_yolu)

    print(f"\nToplam {len(metadata_listesi)} makale basariyla cekildi.")
    print(f"Metadata kaydedildi: {metadata_yolu}")


if __name__ == "__main__":
    main()
