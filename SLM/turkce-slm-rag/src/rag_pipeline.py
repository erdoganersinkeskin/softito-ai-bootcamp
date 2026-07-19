"""
Retrieval (Retriever) ve generation (Transformer SLM) adimlarini
birlestirip tek bir "soru sor, kaynakli cevap al" akisi sunar.
"""

from pathlib import Path

import torch
import yaml

from src.generate import metin_uret, model_yukle
from src.preprocessing import KarakterTokenizer
from src.retrieval import Retriever


class RagPipeline:
    def __init__(self, config_yolu: str, checkpoint_yolu: str):
        with open(config_yolu, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.cihaz = "cuda" if torch.cuda.is_available() else "cpu"

        tokenizer_yolu = Path(self.config["paths"]["temiz_metin_file"]).parent / "tokenizer.json"
        self.tokenizer = KarakterTokenizer.yukle(tokenizer_yolu)

        self.model = model_yukle(checkpoint_yolu, self.tokenizer.vocab_boyutu, self.cihaz)

        self.retriever = Retriever(
            chunks_yolu=self.config["rag"]["chunks_dosyasi"],
            embedding_yolu=self.config["rag"]["embedding_dosyasi"],
            vectorizer_yolu=str(Path(self.config["rag"]["embedding_dosyasi"]).parent / "tfidf_vectorizer.pkl"),
        )

        self.top_k = self.config["rag"]["top_k"]

    def soru_sor(self, soru: str, uzunluk: int = 200, sicaklik: float = 0.7) -> dict:
        ilgili_pasajlar = self.retriever.getir(soru, top_k=self.top_k)

        baglam = "\n".join(p["metin"] for p in ilgili_pasajlar)
        prompt = f"Baglam: {baglam}\nSoru: {soru}\nCevap:"

        cevap = metin_uret(self.model, self.tokenizer, prompt, uzunluk, sicaklik, self.cihaz)

        return {
            "soru": soru,
            "cevap": cevap,
            "kaynaklar": [
                {"baslik": p["baslik"], "kategori": p["kategori"], "skor": p["benzerlik_skoru"]}
                for p in ilgili_pasajlar
            ],
        }
