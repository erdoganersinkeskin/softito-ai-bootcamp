"""
KarakterTokenizer icin temel testler.

Kullanim:
    pytest tests/test_tokenizer.py
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.preprocessing import KarakterTokenizer


def test_encode_decode_round_trip():
    metin = "Merhaba dunya! Bu bir test metnidir."
    tokenizer = KarakterTokenizer(metin)

    kodlanmis = tokenizer.encode(metin)
    cozulmus = tokenizer.decode(kodlanmis)

    assert cozulmus == metin


def test_vocab_boyutu_dogru():
    metin = "aabbcc"
    tokenizer = KarakterTokenizer(metin)

    assert tokenizer.vocab_boyutu == 3  # sadece a, b, c


def test_turkce_karakterler_calisir():
    metin = "çğıöşü ÇĞIÖŞÜ"
    tokenizer = KarakterTokenizer(metin)

    kodlanmis = tokenizer.encode(metin)
    cozulmus = tokenizer.decode(kodlanmis)

    assert cozulmus == metin
