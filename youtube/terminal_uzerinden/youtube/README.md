# YouTube Yorum Duygu Analizi (Türkçe) — PyTorch (Embedding + LSTM)

YouTube video linki verince yorumları çeken, kendi tasarladığımız **Embedding +
Bidirectional LSTM + ReLU'lu sınıflandırma başlığı** mimarisiyle (PyTorch
üzerinde, hazır bir sentiment modeli KULLANILMADAN) pozitif/negatif/nötr
sınıflandırması yapan ve "neden negatif?" sorusuna kelime bazlı açıklama üreten
bir araç.

TF-IDF+MLP'nin aksine bu mimari kelime SIRASINI da dikkate alır (örn. "hiç iyi
değil" ifadesindeki "değil"in anlamı tersine çevirmesini daha iyi yakalayabilir),
bu yüzden daha "akıllı" sonuçlar vermesi beklenir.

## 1. Kurulum (Linux)

En hızlısı hazır scripti çalıştırmak:

```bash
chmod +x setup.sh
./setup.sh
```

Bu script sanal ortam (venv) oluşturur, `requirements.txt` ve `datasets`
paketini kurar. Sonrasında her yeni terminalde önce şunu çalıştırman gerekir:

```bash
source venv/bin/activate
```

Elle yapmak istersen:

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install datasets            # eğitim verisini Hugging Face'ten indirmek için
```

**GPU notu:** Ekran kartın olmadığı için `setup.sh` doğrudan CPU-only PyTorch
kuruyor (CUDA kütüphaneleri indirilmiyor, kurulum daha hafif). LSTM, MLP'ye göre
daha yavaş eğitilir (bkz. adım 3'teki süre tahminleri). Önce hızlı bir deneme
yapman için:

```bash
python train.py --max_rows 60000 --epochs 8
```

ile ~5-10 dakikada bir model çıkar, her şeyin çalıştığını doğrularsın. Sonuç
memnun ediciyse tam veriyle (parametre vermeden `python train.py`, ~1-2 saat)
daha yüksek doğruluk için tekrar eğitebilirsin — akşam çalıştırıp sabah kontrol
etmek gibi bir yol izlenebilir.

## 2. YouTube API Key nasıl alınır (ücretsiz)

1. https://console.cloud.google.com/ adresine git, Google hesabınla giriş yap.
2. Üstten "Yeni Proje" oluştur (örn. adı: `youtube-sentiment`).
3. Sol menüden **APIs & Services > Library**'e gir.
4. Arama kutusuna **YouTube Data API v3** yaz, aç ve **Enable** butonuna bas.
5. Sol menüden **APIs & Services > Credentials**'a gir.
6. **Create Credentials > API key** seç. Sana bir anahtar (uzun bir metin) verecek, onu kopyala.
7. (Önerilir, opsiyonel) Anahtarı "Restrict key" ile sadece YouTube Data API v3'e kısıtla,
   böylece anahtarın başka amaçla kötüye kullanılma riski azalır.
8. Ücretsiz kotan günlük 10.000 "unit" — yorum çekmek ~1 unit/istek olduğu için
   günlük binlerce yorum çekmek için fazlasıyla yeterli.

Anahtarı ortam değişkeni olarak ayarlamak pratik olur:

```bash
export YOUTUBE_API_KEY="BURAYA_ANAHTARIN"
```

Bu komut sadece o terminal oturumu için geçerlidir. Kalıcı yapmak istersen
`~/.bashrc` (veya kullandığın shell neyse onun rc dosyasına, örn. `~/.zshrc`)
dosyasının sonuna aynı satırı ekleyip `source ~/.bashrc` çalıştırabilirsin.

## 3. Modeli eğit

Veri seti olarak Hugging Face'teki **winvoker/turkish-sentiment-analysis-dataset**
kullanılıyor (490 bin civarı etiketli Türkçe cümle — pozitif/negatif/nötr,
ürün yorumları + tweet + wikipedia karışımı). İlk çalıştırmada otomatik indirilir.

```bash
python train.py
```

Hızlı bir deneme yapmak istersen:

```bash
python train.py --max_rows 60000 --epochs 8
```

**Not — LSTM, MLP'den daha yavaş eğitilir** çünkü her cümleyi kelime kelime
sırayla okur (paralelleştirilemeyen bir işlem). CPU'da yaklaşık süreler:
- `--max_rows 60000 --epochs 8` → ~5-10 dakika
- Tam veri (490K) → ~1-2 saat (CPU'na göre değişir)

Ayarlayabileceğin diğer parametreler:
```bash
python train.py \
  --vocab_size 20000 \    # sözlükte tutulacak en sık geçen kelime sayısı
  --max_len 40 \          # yorum başına dikkate alınacak maksimum kelime sayısı
  --embed_dim 128 \       # her kelimenin öğrenilen vektör boyutu
  --hidden_dim 128 \      # LSTM'in gizli durum boyutu
  --n_layers 1 \          # üst üste kaç LSTM katmanı (2 yaparsan daha güçlü ama yavaş olur)
  --dropout 0.3
```

Eğitim bitince şu dosyalar oluşur:
- `model/sentiment_model.pt` — eğitilmiş ağırlıklar (embedding + LSTM + sınıflandırma başlığı)
- `model/model_config.pt` — model mimarisi bilgisi (boyutlar vb.)
- `model/vocab.json` — kelime → ID sözlüğü

Kendi CSV'ini kullanmak istersen (sütunlar: `text`, `label` — label değerleri
`positive` / `negative` / `notr` olmalı):

```bash
python train.py --csv data/kendi_verim.csv
```

## 4. Bir videoyu analiz et

```bash
python analyze_video.py --url "https://www.youtube.com/watch?v=XXXXXXXXXXX" --max_comments 300
```

(YOUTUBE_API_KEY ortam değişkenini ayarlamadıysan `--api_key` parametresini de ekle.)

Çıktı örneği:
```
============================================================
GENEL SONUC
============================================================
  Pozitif : 210 yorum  ( 70.0%)
  Negatif :  60 yorum  ( 20.0%)
  Notr    :  30 yorum  ( 10.0%)

============================================================
EN BELIRGIN NEGATIF YORUMLAR (ilk 10)
============================================================
- "kargo çok geç geldi, ürün de kırık çıktı berbat bir deneyimdi"
  Guven: %94.2  |  Olasi neden(ler): berbat, kırık, geç, kötü
```

## 5. Dosya yapısı

```
preprocessing.py     Türkçe metin temizleme, basit hafif kök bulma, stopword listesi
vocab.py               Kelime dağarcığı (vocabulary) oluşturma ve metin->ID kodlama
neural_network.py     PyTorch ile yazılmış Embedding+LSTM+ReLU modeli (SentimentLSTM)
train.py               Eğitim scripti
youtube_utils.py       YouTube Data API'den yorum çekme
analyze_video.py       Ana kullanım scripti
```

## 6. Notlar / geliştirme fikirleri

- `light_stem` fonksiyonu tam bir morfolojik analiz yapmaz; daha güçlü sonuç
  istersen Zemberek-NLP (Türkçe için özel bir kütüphane) entegre edebilirsin.
- Şu anki embedding'ler sıfırdan öğreniliyor. Önceden eğitilmiş Türkçe kelime
  vektörleri (fastText Türkçe modeli gibi) ile başlatılırsa (fine-tuning),
  özellikle az veri olan durumlarda doğruluk artabilir.
- İki yönlü LSTM yerine, daha da güçlü bağlam yakalamak isteyen ileri seviye
  bir adım olarak bir "attention" katmanı eklenebilir — bu hem doğruluğu
  artırır hem de "hangi kelimeye ne kadar dikkat etti" bilgisini saliency'den
  daha doğal şekilde verir.
- `analyze_video.py` şu an konsola yazdırıyor; istersen sonuçları bir
  `pandas.DataFrame`'e çevirip CSV/Excel olarak da kaydedebiliriz.
