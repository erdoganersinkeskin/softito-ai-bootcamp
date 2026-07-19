# Prompt Engineering — Oyun Dünyası Versiyonu (Zero-shot, Few-shot, CoT, System Prompt, Temperature)

## 🎓 Bu Proje Hakkında

Bu çalışmanın amacı, 5 prompt tekniğini (zero-shot, few-shot,
chain-of-thought, system prompt, temperature) aynı ölçüm yöntemiyle
(yanıt süresi, yanıt uzunluğu) test etmektir; senaryoların konusu oyun
dünyasına uyarlanmıştır.

**Veri seti notu:** Bu proje bir Kaggle veri seti kullanmıyor, gerçek
zamanlı OpenAI API çağrılarıyla çalışıyor. Bu yüzden
paylaşılan 9 Kaggle veri setinden hiçbiri burada uygulanamaz; bunun yerine
tüm prompt senaryoları oyun endüstrisi temasına uyarlanmıştır.

## 🎯 Projenin Amacı

Aynı LLM'e (GPT-4o-mini) farklı prompt tekniklerini uygulayıp gerçek API
yanıtlarını karşılaştırmak:

| Teknik | Senaryo |
|---|---|
| Zero-shot | Mobil oyunda **oyuncu terk etme (churn)** oranını azaltma önerileri |
| Few-shot | **Oyun yorumu** duygu analizi (3 örnek) |
| Chain-of-Thought | Bir **mobil oyun stüdyosunun** kozmetik eşya satışından kâr/zarar hesabı |
| System Prompt | Kıdemli **oyun tasarımcısı** / **oyun geliştirme eğitmeni** / şüpheci **oyun eleştirmeni** rolleriyle **battle pass sistemi** tavsiyesi |
| Temperature | **Indie oyun stüdyosu** isim önerilerinin çeşitliliği |

## 🚀 Kurulum ve Çalıştırma

```bash
pip install -r requirements.txt
```

Proje köküne bir `.env` dosyası oluştur:

```
OPENAI_API_KEY=senin-api-anahtarin
```

Sonra çalıştır:

```bash
python prompt_engineering.py
```

Script 5 farklı prompt tekniğini gerçek API üzerinden çalıştırır, terminale
özet yanıtları basar ve tüm sonuçları `figures/` klasörüne kaydeder.

## 📊 Neyi Ölçüyor

Script her API çağrısında şunları kaydediyor: yanıt metni, yanıt süresi
(saniye), yanıt uzunluğu (kelime sayısı). Sonuçlar şu grafiklere dökülür:

- `figures/yanit_uzunlugu.png` — Zero-shot / Few-shot / Chain-of-Thought yanıtlarının kelime sayısı karşılaştırması
- `figures/yanit_suresi.png` — aynı 3 stratejinin API yanıt süresi karşılaştırması
- `figures/system_prompt_karsilastirma.png` — 3 persona'nın yanıt uzunluğuna etkisi
- `figures/temperature_karsilastirma.png` — Temperature (0.0 → 1.5) arttıkça yanıt uzunluğunun/çeşitliliğinin değişimi

Ayrıca tüm ham çağrılar `figures/tum_cagrilar.csv` (tablo) ve
`figures/tum_yanitlar.json` (tam yanıt metinleriyle) olarak kaydediliyor.

## ⚠️ Not

Bu script gerçek bir OpenAI API anahtarı gerektirir ve çalıştırıldığında
**gerçek API çağrıları yapar** (küçük bir maliyeti olur, `gpt-4o-mini`
kullanıldığı için bu maliyet çok düşüktür). `figures/` klasöründeki
grafikler ancak script bir kez çalıştırıldıktan sonra oluşur — bu yüzden
buraya hazır sonuç/görsel eklenmedi.

## 🛠️ Kullanılan Teknolojiler

`Python` · `OpenAI API` · `pandas` · `matplotlib` · `python-dotenv`

<p align="center"><i>Prompt engineering pratiği amaçlı, öğrenme sürecinde egzersiz olarak hazırlanmış bir versiyondur.</i></p>
