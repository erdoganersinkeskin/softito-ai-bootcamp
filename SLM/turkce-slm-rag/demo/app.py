"""
Turkce SLM icin Gradio demo arayuzu.

Iki modu destekler:
  - Faz 1: sadece metin uretimi (RAG'siz)
  - Faz 2: RAG (soru sor, kaynakli cevap al)

Kullanim:
    python demo/app.py
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import gradio as gr

from src.generate import metin_uret, model_yukle
from src.preprocessing import KarakterTokenizer
from src.rag_pipeline import RagPipeline

CONFIG_YOLU = "configs/config.yaml"
CHECKPOINT_YOLU = "checkpoints/slm_model.pt"
TOKENIZER_YOLU = "data/processed/tokenizer.json"

_tokenizer = None
_model = None
_rag_pipeline = None


def _faz1_modelini_yukle():
    global _tokenizer, _model
    if _model is None:
        import torch
        cihaz = "cuda" if torch.cuda.is_available() else "cpu"
        _tokenizer = KarakterTokenizer.yukle(Path(TOKENIZER_YOLU))
        _model = model_yukle(CHECKPOINT_YOLU, _tokenizer.vocab_boyutu, cihaz)
    return _tokenizer, _model


def metin_tamamla(prompt: str, uzunluk: int, sicaklik: float) -> str:
    import torch
    cihaz = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer, model = _faz1_modelini_yukle()
    return metin_uret(model, tokenizer, prompt, uzunluk, sicaklik, cihaz)


def _rag_pipeline_yukle():
    global _rag_pipeline
    if _rag_pipeline is None:
        _rag_pipeline = RagPipeline(CONFIG_YOLU, CHECKPOINT_YOLU)
    return _rag_pipeline


def soru_cevapla(soru: str, uzunluk: int, sicaklik: float) -> str:
    pipeline = _rag_pipeline_yukle()
    sonuc = pipeline.soru_sor(soru, uzunluk=uzunluk, sicaklik=sicaklik)

    kaynak_metni = "\n".join(
        f"- {k['baslik']} ({k['kategori']}, skor: {k['skor']:.3f})"
        for k in sonuc["kaynaklar"]
    )

    return f"{sonuc['cevap']}\n\nKaynaklar:\n{kaynak_metni}"


with gr.Blocks(title="Turkce Oyun SLM Demo") as demo:
    gr.Markdown("# Turkce Kucuk Dil Modeli Demo (Oyun Dunyasi)")

    with gr.Tab("Metin Tamamlama (Faz 1)"):
        prompt_girisi = gr.Textbox(label="Baslangic metni", placeholder="Video oyunu...")
        uzunluk_kaydirici = gr.Slider(50, 500, value=300, step=10, label="Uretilecek karakter sayisi")
        sicaklik_kaydirici = gr.Slider(0.1, 1.5, value=0.8, step=0.1, label="Sicaklik (yaraticilik)")
        tamamla_buton = gr.Button("Metni Tamamla")
        tamamlama_ciktisi = gr.Textbox(label="Uretilen metin", lines=8)

        tamamla_buton.click(
            fn=metin_tamamla,
            inputs=[prompt_girisi, uzunluk_kaydirici, sicaklik_kaydirici],
            outputs=tamamlama_ciktisi,
        )

    with gr.Tab("Soru-Cevap (Faz 2 - RAG)"):
        soru_girisi = gr.Textbox(label="Sorunuz", placeholder="Video oyunu nedir?")
        rag_uzunluk_kaydirici = gr.Slider(50, 500, value=200, step=10, label="Cevap uzunlugu")
        rag_sicaklik_kaydirici = gr.Slider(0.1, 1.5, value=0.7, step=0.1, label="Sicaklik")
        soru_buton = gr.Button("Soru Sor")
        cevap_ciktisi = gr.Textbox(label="Cevap ve kaynaklar", lines=10)

        soru_buton.click(
            fn=soru_cevapla,
            inputs=[soru_girisi, rag_uzunluk_kaydirici, rag_sicaklik_kaydirici],
            outputs=cevap_ciktisi,
        )


if __name__ == "__main__":
    demo.launch()
