"""
chunks.jsonl icindeki pasajlari TF-IDF ile vektorlestirir ve
diske kaydeder. Ileride "sentence-transformers" gibi daha guclu
bir embedding modeline gecmek icin config.yaml -> rag.embedding_modeli
degeri degistirilebilir.

Kullanim:
    python src/embedding.py --config configs/config.yaml
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer


def config_yukle(config_yolu: str) -> dict:
    with open(config_yolu, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def chunks_yukle(chunks_yolu: Path) -> list[dict]:
    chunk_listesi = []
    with open(chunks_yolu, "r", encoding="utf-8") as f:
        for satir in f:
            chunk_listesi.append(json.loads(satir))
    return chunk_listesi


def tfidf_embedding_olustur(chunk_listesi: list[dict]):
    metinler = [chunk["metin"] for chunk in chunk_listesi]
    vectorizer = TfidfVectorizer(max_features=5000)
    embeddings = vectorizer.fit_transform(metinler).toarray()
    return embeddings, vectorizer


def main():
    parser = argparse.ArgumentParser(description="RAG icin pasaj embedding olusturma")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()

    config = config_yukle(args.config)

    chunks_yolu = Path(config["rag"]["chunks_dosyasi"])
    embedding_yolu = Path(config["rag"]["embedding_dosyasi"])
    vectorizer_yolu = embedding_yolu.parent / "tfidf_vectorizer.pkl"

    chunk_listesi = chunks_yukle(chunks_yolu)

    if config["rag"]["embedding_modeli"] != "tfidf":
        raise NotImplementedError(
            "Su an sadece 'tfidf' destekleniyor. "
            "'sentence-transformers' secenegi ileride eklenecek."
        )

    embeddings, vectorizer = tfidf_embedding_olustur(chunk_listesi)

    embedding_yolu.parent.mkdir(parents=True, exist_ok=True)
    np.save(embedding_yolu, embeddings)

    with open(vectorizer_yolu, "wb") as f:
        pickle.dump(vectorizer, f)

    print(f"Embedding matrisi kaydedildi: {embedding_yolu} (boyut: {embeddings.shape})")
    print(f"Vectorizer kaydedildi: {vectorizer_yolu}")


if __name__ == "__main__":
    main()
