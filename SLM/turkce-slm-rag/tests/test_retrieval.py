"""
Retriever ve metni_parcala fonksiyonlari icin temel testler.

Kullanim:
    pytest tests/test_retrieval.py
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.chunking import metni_parcala


def test_metni_parcala_temel():
    metin = " ".join([f"kelime{i}" for i in range(100)])
    parcalar = metni_parcala(metin, chunk_boyutu=20, ortusme=5)

    assert len(parcalar) > 1
    for parca in parcalar:
        assert len(parca.split()) <= 20


def test_metni_parcala_ortusme_var():
    metin = " ".join([f"k{i}" for i in range(50)])
    parcalar = metni_parcala(metin, chunk_boyutu=10, ortusme=3)

    # ikinci parcanin basi, ilk parcanin sonuyla ortusmeli olmali
    ilk_parca_kelimeleri = parcalar[0].split()
    ikinci_parca_kelimeleri = parcalar[1].split()

    ortusen_kelimeler = set(ilk_parca_kelimeleri[-3:]) & set(ikinci_parca_kelimeleri[:3])
    assert len(ortusen_kelimeler) > 0


def test_kisa_metin_tek_parca_doner():
    metin = "kisa bir metin"
    parcalar = metni_parcala(metin, chunk_boyutu=200, ortusme=30)

    assert len(parcalar) == 1
    assert parcalar[0] == metin
