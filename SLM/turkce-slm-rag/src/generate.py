"""
Egitilmis checkpoint'i yukleyip verilen baslangic metnine gore
devam metni uretir.

Kullanim:
    python src/generate.py --prompt "Video oyunu" --uzunluk 300
"""

import argparse
from pathlib import Path

import torch

from src.preprocessing import KarakterTokenizer
from src.models.transformer import TransformerModel


def model_yukle(checkpoint_yolu: str, vocab_boyutu: int, cihaz: str) -> TransformerModel:
    checkpoint = torch.load(checkpoint_yolu, map_location=cihaz)
    config = checkpoint["config"]
    model_cfg = config["model"]

    model = TransformerModel(
        vocab_boyutu=vocab_boyutu,
        gomme_boyutu=model_cfg["gomme_boyutu"],
        blok_uzunlugu=config["dataset"]["blok_uzunlugu"],
        katman_sayisi=model_cfg["katman_sayisi"],
        baslik_sayisi=model_cfg["baslik_sayisi"],
        dropout=model_cfg["dropout"],
    )

    model.load_state_dict(checkpoint["model_durumu"])
    model.to(cihaz)
    model.eval()
    return model


def metin_uret(model: TransformerModel, tokenizer, prompt: str, uzunluk: int, sicaklik: float, cihaz: str) -> str:
    giris = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=cihaz)
    cikti = model.uret(giris, uzunluk, sicaklik=sicaklik)
    return tokenizer.decode(cikti[0].tolist())


def main():
    parser = argparse.ArgumentParser(description="SLM ile metin uretimi")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/slm_model.pt")
    parser.add_argument("--tokenizer", type=str, default="data/processed/tokenizer.json")
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--uzunluk", type=int, default=300)
    parser.add_argument("--sicaklik", type=float, default=0.8)
    args = parser.parse_args()

    cihaz = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = KarakterTokenizer.yukle(Path(args.tokenizer))
    model = model_yukle(args.checkpoint, tokenizer.vocab_boyutu, cihaz)

    uretilen_metin = metin_uret(model, tokenizer, args.prompt, args.uzunluk, args.sicaklik, cihaz)

    print("\nUretilen metin:\n")
    print(uretilen_metin)


if __name__ == "__main__":
    main()
