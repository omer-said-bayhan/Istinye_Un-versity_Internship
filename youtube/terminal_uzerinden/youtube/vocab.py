# -*- coding: utf-8 -*-
"""
LSTM modeli TF-IDF yerine ogrenilebilir kelime embedding'leri kullanacagi icin,
her kelimeye sabit bir tamsayi ID atayan bir sozluk (vocabulary) gerekiyor.
Bu modul o sozlugu olusturur ve metinleri sabit uzunlukta ID dizilerine cevirir.
"""

import json
from collections import Counter

from preprocessing import tokenize

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"


class Vocabulary:
    def __init__(self, max_size: int = 20000, min_freq: int = 2):
        self.max_size = max_size
        self.min_freq = min_freq
        self.word2idx = {PAD_TOKEN: 0, UNK_TOKEN: 1}

    def fit(self, texts):
        counter = Counter()
        for text in texts:
            counter.update(tokenize(text))

        words = [w for w, c in counter.items() if c >= self.min_freq]
        words.sort(key=lambda w: counter[w], reverse=True)
        words = words[: max(0, self.max_size - len(self.word2idx))]

        for w in words:
            self.word2idx[w] = len(self.word2idx)
        return self

    def __len__(self):
        return len(self.word2idx)

    def encode(self, text: str, max_len: int = 40):
        """Metni sabit uzunlukta (max_len) ID dizisine cevirir.
        Dondurdugu 3 deger: ids (padded liste), gercek_uzunluk, tokenler
        (tokenler saliency aciklamasinda kelimeleri geri gostermek icin lazim)."""
        tokens = tokenize(text)[:max_len]
        unk_id = self.word2idx[UNK_TOKEN]
        ids = [self.word2idx.get(t, unk_id) for t in tokens]

        length = len(ids)
        if length == 0:
            ids = [unk_id]
            tokens = [UNK_TOKEN]
            length = 1

        if len(ids) < max_len:
            pad_id = self.word2idx[PAD_TOKEN]
            ids = ids + [pad_id] * (max_len - len(ids))

        return ids, length, tokens

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {"word2idx": self.word2idx, "max_size": self.max_size, "min_freq": self.min_freq},
                f, ensure_ascii=False
            )

    @classmethod
    def load(cls, path: str):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        vocab = cls(max_size=data["max_size"], min_freq=data["min_freq"])
        vocab.word2idx = data["word2idx"]
        return vocab
