"""
Transformer modelini config.yaml ayarlarina gore egitir ve
checkpoint olarak kaydeder.

Kullanim:
    python src/train.py --config configs/config.yaml
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import yaml
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.dataset import dataset_hazirla
from src.models.transformer import TransformerModel


def config_yukle(config_yolu: str) -> dict:
    with open(config_yolu, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def model_olustur(config: dict, vocab_boyutu: int) -> TransformerModel:
    model_cfg = config["model"]
    return TransformerModel(
        vocab_boyutu=vocab_boyutu,
        gomme_boyutu=model_cfg["gomme_boyutu"],
        blok_uzunlugu=config["dataset"]["blok_uzunlugu"],
        katman_sayisi=model_cfg["katman_sayisi"],
        baslik_sayisi=model_cfg["baslik_sayisi"],
        dropout=model_cfg["dropout"],
    )


def dogrulama_yap(model, dogrulama_loader, cihaz):
    model.eval()
    toplam_kayip = 0.0
    adim_sayisi = 0

    with torch.no_grad():
        for girisler, hedefler in dogrulama_loader:
            girisler, hedefler = girisler.to(cihaz), hedefler.to(cihaz)
            _, kayip = model(girisler, hedefler)
            toplam_kayip += kayip.item()
            adim_sayisi += 1

    model.train()
    return toplam_kayip / max(adim_sayisi, 1)


def egit(config: dict):
    cihaz = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Kullanilan cihaz: {cihaz}")

    torch.manual_seed(config["egitim"]["seed"])

    egitim_loader, dogrulama_loader, tokenizer = dataset_hazirla(
        temiz_metin_yolu=config["paths"]["temiz_metin_file"],
        tokenizer_yolu=str(Path(config["paths"]["temiz_metin_file"]).parent / "tokenizer.json"),
        blok_uzunlugu=config["dataset"]["blok_uzunlugu"],
        batch_boyutu=config["dataset"]["batch_boyutu"],
        egitim_orani=config["dataset"]["egitim_orani"],
    )

    model = model_olustur(config, tokenizer.vocab_boyutu).to(cihaz)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=config["egitim"]["ogrenme_hizi"],
        weight_decay=config["egitim"]["weight_decay"],
    )
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    egitim_kayiplari = []
    dogrulama_kayiplari = []
    en_iyi_dogrulama_kaybi = float("inf")

    epoch_sayisi = config["egitim"]["epoch_sayisi"]

    for epoch in range(1, epoch_sayisi + 1):
        toplam_kayip = 0.0
        adim_sayisi = 0

        for girisler, hedefler in egitim_loader:
            girisler, hedefler = girisler.to(cihaz), hedefler.to(cihaz)

            _, kayip = model(girisler, hedefler)

            optimizer.zero_grad()
            kayip.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            toplam_kayip += kayip.item()
            adim_sayisi += 1

        ortalama_egitim_kaybi = toplam_kayip / max(adim_sayisi, 1)
        ortalama_dogrulama_kaybi = dogrulama_yap(model, dogrulama_loader, cihaz)

        scheduler.step(ortalama_dogrulama_kaybi)

        egitim_kayiplari.append(ortalama_egitim_kaybi)
        dogrulama_kayiplari.append(ortalama_dogrulama_kaybi)

        print(
            f"Epoch {epoch}/{epoch_sayisi} | "
            f"egitim kaybi: {ortalama_egitim_kaybi:.4f} | "
            f"dogrulama kaybi: {ortalama_dogrulama_kaybi:.4f}"
        )

        if ortalama_dogrulama_kaybi < en_iyi_dogrulama_kaybi:
            en_iyi_dogrulama_kaybi = ortalama_dogrulama_kaybi
            checkpoint_yolu = Path(config["egitim"]["cikti_checkpoint"])
            checkpoint_yolu.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model_durumu": model.state_dict(),
                "config": config,
            }, checkpoint_yolu)
            print(f"  -> yeni en iyi model kaydedildi ({checkpoint_yolu})")

    grafik_ciz(egitim_kayiplari, dogrulama_kayiplari, config["egitim"]["cikti_figure"])


def grafik_ciz(egitim_kayiplari, dogrulama_kayiplari, cikti_yolu):
    plt.figure(figsize=(8, 5))
    plt.plot(egitim_kayiplari, label="Egitim kaybi")
    plt.plot(dogrulama_kayiplari, label="Dogrulama kaybi")
    plt.xlabel("Epoch")
    plt.ylabel("Kayip")
    plt.title("Egitim / Dogrulama Kaybi")
    plt.legend()

    Path(cikti_yolu).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(cikti_yolu)
    print(f"Egitim grafigi kaydedildi: {cikti_yolu}")


def main():
    parser = argparse.ArgumentParser(description="SLM egitim script'i")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()

    config = config_yukle(args.config)
    egit(config)


if __name__ == "__main__":
    main()
