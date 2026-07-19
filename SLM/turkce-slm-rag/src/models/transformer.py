"""
Kucuk olcekli, decoder-only (GPT tarzi) Transformer dil modeli.
Self-attention kullandigi icin LSTM'e gore uzun menzilli
bagimliliklari daha iyi yakalar.
"""

import torch
import torch.nn as nn
from torch.nn import functional as F


class AttentionBaslik(nn.Module):
    """Tek bir self-attention baslik (head)."""

    def __init__(self, gomme_boyutu: int, baslik_boyutu: int, blok_uzunlugu: int, dropout: float):
        super().__init__()
        self.anahtar = nn.Linear(gomme_boyutu, baslik_boyutu, bias=False)
        self.sorgu = nn.Linear(gomme_boyutu, baslik_boyutu, bias=False)
        self.deger = nn.Linear(gomme_boyutu, baslik_boyutu, bias=False)
        self.register_buffer("mask", torch.tril(torch.ones(blok_uzunlugu, blok_uzunlugu)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        k = self.anahtar(x)
        q = self.sorgu(x)

        agirliklar = q @ k.transpose(-2, -1) * (c ** -0.5)
        agirliklar = agirliklar.masked_fill(self.mask[:t, :t] == 0, float("-inf"))
        agirliklar = F.softmax(agirliklar, dim=-1)
        agirliklar = self.dropout(agirliklar)

        v = self.deger(x)
        return agirliklar @ v


class CokBasliklAttention(nn.Module):
    def __init__(self, gomme_boyutu: int, baslik_sayisi: int, blok_uzunlugu: int, dropout: float):
        super().__init__()
        baslik_boyutu = gomme_boyutu // baslik_sayisi
        self.basliklar = nn.ModuleList([
            AttentionBaslik(gomme_boyutu, baslik_boyutu, blok_uzunlugu, dropout)
            for _ in range(baslik_sayisi)
        ])
        self.projeksiyon = nn.Linear(gomme_boyutu, gomme_boyutu)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cikti = torch.cat([baslik(x) for baslik in self.basliklar], dim=-1)
        return self.dropout(self.projeksiyon(cikti))


class IleriBesleme(nn.Module):
    def __init__(self, gomme_boyutu: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(gomme_boyutu, 4 * gomme_boyutu),
            nn.ReLU(),
            nn.Linear(4 * gomme_boyutu, gomme_boyutu),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlok(nn.Module):
    def __init__(self, gomme_boyutu: int, baslik_sayisi: int, blok_uzunlugu: int, dropout: float):
        super().__init__()
        self.attention = CokBasliklAttention(gomme_boyutu, baslik_sayisi, blok_uzunlugu, dropout)
        self.ileri_besleme = IleriBesleme(gomme_boyutu, dropout)
        self.norm1 = nn.LayerNorm(gomme_boyutu)
        self.norm2 = nn.LayerNorm(gomme_boyutu)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.norm1(x))
        x = x + self.ileri_besleme(self.norm2(x))
        return x


class TransformerModel(nn.Module):
    def __init__(
        self,
        vocab_boyutu: int,
        gomme_boyutu: int,
        blok_uzunlugu: int,
        katman_sayisi: int,
        baslik_sayisi: int,
        dropout: float,
    ):
        super().__init__()
        self.blok_uzunlugu = blok_uzunlugu
        self.token_gomme = nn.Embedding(vocab_boyutu, gomme_boyutu)
        self.pozisyon_gomme = nn.Embedding(blok_uzunlugu, gomme_boyutu)

        self.bloklar = nn.Sequential(*[
            TransformerBlok(gomme_boyutu, baslik_sayisi, blok_uzunlugu, dropout)
            for _ in range(katman_sayisi)
        ])

        self.son_norm = nn.LayerNorm(gomme_boyutu)
        self.cikis_katmani = nn.Linear(gomme_boyutu, vocab_boyutu)

    def forward(self, girisler: torch.Tensor, hedefler: torch.Tensor = None):
        b, t = girisler.shape

        token_emb = self.token_gomme(girisler)
        pozisyon_emb = self.pozisyon_gomme(torch.arange(t, device=girisler.device))
        x = token_emb + pozisyon_emb

        x = self.bloklar(x)
        x = self.son_norm(x)
        logits = self.cikis_katmani(x)

        kayip = None
        if hedefler is not None:
            b, t, v = logits.shape
            kayip = F.cross_entropy(logits.view(b * t, v), hedefler.view(b * t))

        return logits, kayip

    @torch.no_grad()
    def uret(self, giris: torch.Tensor, yeni_token_sayisi: int, sicaklik: float = 1.0):
        for _ in range(yeni_token_sayisi):
            giris_kirpilmis = giris[:, -self.blok_uzunlugu:]
            logits, _ = self(giris_kirpilmis)
            son_logit = logits[:, -1, :] / sicaklik
            olasiliklar = F.softmax(son_logit, dim=-1)
            sonraki = torch.multinomial(olasiliklar, num_samples=1)
            giris = torch.cat([giris, sonraki], dim=1)
        return giris
