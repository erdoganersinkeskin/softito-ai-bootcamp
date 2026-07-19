"""
data/raw/ altindaki tum makaleleri okur, temizler, birlestirir ve
karakter seviyeli bir tokenizer olusturur.

Kullanim:
    python src/preprocessing.py --config configs/config.yaml
"""

import argparse
import json
import re
from pathlib import Path

import yaml


def config_yukle(config_yolu: str) -> dict:
    with open(config_yolu, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def metni_temizle(metin: str) -> str:
    """Fazla bosluk, satir sonu ve referans isaretlerini temizler."""
    metin = re.sub(r"\[\d+\]", "", metin)          # [1], [23] gibi referanslar
    metin = re.sub(r"\n{2,}", "\n", metin)          # fazla bos satirlar
    metin = re.sub(r"[ \t]{2,}", " ", metin)        # fazla bosluk
    return metin.strip()


def tum_makaleleri_birlestir(raw_dizin: Path, min_uzunluk: int) -> str:
    parcalar = []
    for dosya_yolu in sorted(raw_dizin.rglob("*.txt")):
        metin = dosya_yolu.read_text(encoding="utf-8")
        temiz = metni_temizle(metin)
        if len(temiz) >= min_uzunluk:
            parcalar.append(temiz)
    return "\n".join(parcalar)


class KarakterTokenizer:
    """Basit karakter seviyeli tokenizer. Metindeki her benzersiz
    karakteri bir tamsayiya esler."""

    def __init__(self, metin: str):
        karakterler = sorted(set(metin))
        self.stoi = {ch: i for i, ch in enumerate(karakterler)}
        self.itos = {i: ch for i, ch in enumerate(karakterler)}
        self.vocab_boyutu = len(karakterler)

    def encode(self, metin: str) -> list[int]:
        # Egitim verisinde hic gorulmemis (bilinmeyen) karakterler
        # bosluk ile degistirilir, boslukta yoksa atlanir. Bu sayede
        # RAG prompt'undaki ":" gibi ozel karakterler hata vermez.
        bosluk_id = self.stoi.get(" ")
        sonuc = []
        for ch in metin:
            if ch in self.stoi:
                sonuc.append(self.stoi[ch])
            elif bosluk_id is not None:
                sonuc.append(bosluk_id)
        return sonuc

    def decode(self, id_listesi: list[int]) -> str:
        return "".join(self.itos[i] for i in id_listesi)

    def kaydet(self, yol: Path) -> None:
        veri = {"stoi": self.stoi, "vocab_boyutu": self.vocab_boyutu}
        yol.write_text(json.dumps(veri, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def yukle(cls, yol) -> "KarakterTokenizer":
        yol = Path(yol)
        veri = json.loads(yol.read_text(encoding="utf-8"))
        obj = cls.__new__(cls)
        obj.stoi = veri["stoi"]
        obj.itos = {int(i): ch for ch, i in veri["stoi"].items()}
        obj.vocab_boyutu = veri["vocab_boyutu"]
        return obj


def main():
    parser = argparse.ArgumentParser(description="Veri on isleme ve tokenizer olusturma")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()

    config = config_yukle(args.config)

    raw_dizin = Path(config["paths"]["raw_data_dir"])
    temiz_metin_yolu = Path(config["paths"]["temiz_metin_file"])
    min_uzunluk = config["preprocessing"]["min_makale_uzunlugu"]
    max_karakter = config["preprocessing"]["max_karakter"]

    birlesik_metin = tum_makaleleri_birlestir(raw_dizin, min_uzunluk)
    birlesik_metin = birlesik_metin[:max_karakter]

    temiz_metin_yolu.parent.mkdir(parents=True, exist_ok=True)
    temiz_metin_yolu.write_text(birlesik_metin, encoding="utf-8")
    print(f"Birlesik temiz metin kaydedildi: {temiz_metin_yolu} ({len(birlesik_metin)} karakter)")

    tokenizer = KarakterTokenizer(birlesik_metin)
    tokenizer_yolu = temiz_metin_yolu.parent / "tokenizer.json"
    tokenizer.kaydet(tokenizer_yolu)
    print(f"Tokenizer kaydedildi: {tokenizer_yolu} (vocab boyutu: {tokenizer.vocab_boyutu})")


if __name__ == "__main__":
    main()
