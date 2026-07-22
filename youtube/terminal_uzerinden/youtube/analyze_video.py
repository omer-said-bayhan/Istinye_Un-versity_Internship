# -*- coding: utf-8 -*-
"""
Kullanim:
    python analyze_video.py --url "https://www.youtube.com/watch?v=XXXXXXXXXXX" --api_key YOUR_KEY

API key'i ortam degiskeni olarak da verebilirsin:
    export YOUTUBE_API_KEY=xxxx
ve --api_key parametresini atlayabilirsin.
"""

import argparse
import os
from collections import Counter

import torch

from vocab import Vocabulary
from neural_network import SentimentLSTM
from youtube_utils import fetch_comments

INV_LABEL_MAP = {0: "Negatif", 1: "Notr", 2: "Pozitif"}
TOP_K_WORDS = 6


def load_model():
    config = torch.load("model/model_config.pt", weights_only=False)
    model = SentimentLSTM.load("model/sentiment_model.pt", config)
    vocab = Vocabulary.load("model/vocab.json")
    return model, vocab, config["max_len"]


def explain(model, vocab, text, max_len):
    ids, length, tokens = vocab.encode(text, max_len)
    ids_t = torch.tensor(ids, dtype=torch.long)
    pred_class, probs, contribution = model.saliency(ids_t, length)

    word_scores = list(zip(tokens[:length], contribution.tolist()))
    word_scores.sort(key=lambda t: t[1], reverse=True)
    top_words = [w for w, s in word_scores[:TOP_K_WORDS] if s > 0]
    return pred_class, probs.numpy(), top_words


def main(args):
    api_key = args.api_key or os.environ.get("YOUTUBE_API_KEY")
    if not api_key:
        raise SystemExit(
            "YouTube API key bulunamadi. --api_key ile ver ya da YOUTUBE_API_KEY "
            "ortam degiskenini ayarla. (README.md -> 'API key nasil alinir' bolumune bak)"
        )

    print("Model yukleniyor...")
    model, vocab, max_len = load_model()

    print(f"Yorumlar cekiliyor (en fazla {args.max_comments})...")
    comments = fetch_comments(api_key, args.url, max_comments=args.max_comments)
    print(f"{len(comments)} yorum cekildi.")

    if not comments:
        print("Hicbir yorum bulunamadi (yorumlar kapali olabilir ya da video linki hatali).")
        return

    results = []
    for c in comments:
        pred_class, probs, top_words = explain(model, vocab, c["text"], max_len)
        results.append({**c, "pred": pred_class, "probs": probs, "top_words": top_words})

    counts = Counter(r["pred"] for r in results)
    total = len(results)

    print("\n" + "=" * 60)
    print("GENEL SONUC")
    print("=" * 60)
    for cls in (2, 0, 1):
        n = counts.get(cls, 0)
        pct = 100 * n / total
        print(f"  {INV_LABEL_MAP[cls]:<8}: {n:>5} yorum  ({pct:5.1f}%)")

    negatives = [r for r in results if r["pred"] == 0]
    negatives.sort(key=lambda r: r["probs"][0], reverse=True)

    print("\n" + "=" * 60)
    print(f"EN BELIRGIN NEGATIF YORUMLAR (ilk {min(args.show_negative, len(negatives))})")
    print("=" * 60)
    for r in negatives[: args.show_negative]:
        conf = r["probs"][0] * 100
        words = ", ".join(r["top_words"]) if r["top_words"] else "-"
        text_preview = r["text"][:140].replace("\n", " ")
        print(f"\n- \"{text_preview}\"")
        print(f"  Guven: %{conf:.1f}  |  Olasi neden(ler): {words}")

    all_negative_words = Counter()
    for r in negatives:
        all_negative_words.update(r["top_words"])
    if all_negative_words:
        print("\n" + "=" * 60)
        print("YORUMLAR GENELINDE EN SIK NEGATIF TETIKLEYICI KELIMELER")
        print("=" * 60)
        for word, cnt in all_negative_words.most_common(10):
            print(f"  {word:<20} {cnt} yorumda gecti")
    positives = [r for r in results if r["pred"] == 2]
    positives.sort(key=lambda r: r["probs"][2], reverse=True)

    print("\n" + "=" * 60)
    print(f"EN BELIRGIN POZITIF YORUMLAR (ilk {min(args.show_positive, len(positives))})")
    print("=" * 60)
    for r in positives[: args.show_positive]:
        conf = r["probs"][2] * 100
        words = ", ".join(r["top_words"]) if r["top_words"] else "-"
        text_preview = r["text"][:140].replace("\n", " ")
        print(f"\n- \"{text_preview}\"")
        print(f"  Guven: %{conf:.1f}  |  Olasi neden(ler): {words}")

    all_positive_words = Counter()
    for r in positives:
        all_positive_words.update(r["top_words"])
    if all_positive_words:
        print("\n" + "=" * 60)
        print("YORUMLAR GENELINDE EN SIK POZITIF TETIKLEYICI KELIMELER")
        print("=" * 60)
        for word, cnt in all_positive_words.most_common(10):
            print(f"  {word:<20} {cnt} yorumda gecti")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", type=str, required=True, help="YouTube video linki veya ID'si")
    parser.add_argument("--api_key", type=str, default=None, help="YouTube Data API v3 anahtari")
    parser.add_argument("--max_comments", type=int, default=300)
    parser.add_argument("--show_negative", type=int, default=10)
    parser.add_argument("--show_positive", type=int, default=10)
    args = parser.parse_args()
    main(args)
