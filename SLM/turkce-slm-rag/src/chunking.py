"""
data/raw/ altindaki her makaleyi, RAG'de kullanilmak uzere sabit
boyutlu ve ortusmeli pasajlara (chunk) boler.

Kullanim:
    python src/chunking.py --config configs/config.yaml
"""

import argparse
import json
from pathlib import Path

import yaml

from src.preprocessing import metni_temizle


def config_yukle(config_yolu: str) -> dict:
    with open(config_yolu, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def metni_parcala(metin: str, chunk_boyutu: int, ortusme: int) -> list[str]:
    """Metni kelime bazli, ortusmeli parcalara boler."""
    kelimeler = metin.split()
    parcalar = []

    baslangic = 0
    while baslangic < len(kelimeler):
        bitis = baslangic + chunk_boyutu
        parca = " ".join(kelimeler[baslangic:bitis])
        if parca.strip():
            parcalar.append(parca)
        baslangic += chunk_boyutu - ortusme

    return parcalar


def tum_makaleleri_parcala(raw_dizin: Path, chunk_boyutu: int, ortusme: int) -> list[dict]:
    chunk_listesi = []
    chunk_id = 0

    for dosya_yolu in sorted(raw_dizin.rglob("*.txt")):
        kategori = dosya_yolu.parent.name
        baslik = dosya_yolu.stem

        metin = metni_temizle(dosya_yolu.read_text(encoding="utf-8"))
        parcalar = metni_parcala(metin, chunk_boyutu, ortusme)

        for parca in parcalar:
            chunk_listesi.append({
                "chunk_id": chunk_id,
                "baslik": baslik,
                "kategori": kategori,
                "metin": parca,
            })
            chunk_id += 1

    return chunk_listesi


def chunks_kaydet(chunk_listesi: list[dict], cikti_yolu: Path) -> None:
    cikti_yolu.parent.mkdir(parents=True, exist_ok=True)
    with open(cikti_yolu, "w", encoding="utf-8") as f:
        for chunk in chunk_listesi:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="RAG icin metin parcalama")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()

    config = config_yukle(args.config)

    raw_dizin = Path(config["paths"]["raw_data_dir"])
    chunk_boyutu = config["rag"]["chunk_boyutu_kelime"]
    ortusme = config["rag"]["chunk_ortusme_kelime"]
    cikti_yolu = Path(config["rag"]["chunks_dosyasi"])

    chunk_listesi = tum_makaleleri_parcala(raw_dizin, chunk_boyutu, ortusme)
    chunks_kaydet(chunk_listesi, cikti_yolu)

    print(f"Toplam {len(chunk_listesi)} pasaj olusturuldu.")
    print(f"Kaydedildi: {cikti_yolu}")


if __name__ == "__main__":
    main()
