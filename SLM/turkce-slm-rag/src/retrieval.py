"""
Verilen bir soru icin, onceden hesaplanmis pasaj embeddinglerinden
en alakali olanlari (cosine similarity ile) bulur.
"""

import json
import pickle
from pathlib import Path

import numpy as np


class Retriever:
    def __init__(self, chunks_yolu: str, embedding_yolu: str, vectorizer_yolu: str):
        self.chunk_listesi = self._chunks_yukle(Path(chunks_yolu))
        self.embeddings = np.load(embedding_yolu)

        with open(vectorizer_yolu, "rb") as f:
            self.vectorizer = pickle.load(f)

    @staticmethod
    def _chunks_yukle(chunks_yolu: Path) -> list[dict]:
        chunk_listesi = []
        with open(chunks_yolu, "r", encoding="utf-8") as f:
            for satir in f:
                chunk_listesi.append(json.loads(satir))
        return chunk_listesi

    def _cosine_benzerlik(self, soru_vektoru: np.ndarray) -> np.ndarray:
        norm_chunklar = np.linalg.norm(self.embeddings, axis=1)
        norm_soru = np.linalg.norm(soru_vektoru)

        # sifira bolme hatasini onlemek icin kucuk bir epsilon ekleniyor
        payda = (norm_chunklar * norm_soru) + 1e-10
        pay = self.embeddings @ soru_vektoru
        return pay / payda

    def getir(self, soru: str, top_k: int = 3) -> list[dict]:
        soru_vektoru = self.vectorizer.transform([soru]).toarray()[0]
        benzerlikler = self._cosine_benzerlik(soru_vektoru)

        en_iyi_indeksler = np.argsort(benzerlikler)[::-1][:top_k]

        sonuclar = []
        for idx in en_iyi_indeksler:
            chunk = self.chunk_listesi[idx]
            sonuclar.append({
                "baslik": chunk["baslik"],
                "kategori": chunk["kategori"],
                "metin": chunk["metin"],
                "benzerlik_skoru": float(benzerlikler[idx]),
            })

        return sonuclar
