# -*- coding: utf-8 -*-
"""
Embedding + (Bidirectional) LSTM tabanli duygu siniflandirma modeli.

Mimari:
    Girdi: kelime ID dizisi (orn. [45, 12, 980, 3, 0, 0, ... ])  (0 = <pad>)
      -> Embedding(vocab_size, embed_dim)          her kelime -> ogrenilebilir vektor
      -> Bidirectional LSTM(embed_dim -> hidden_dim)  cumleyi sirayla, iki yonden okur
      -> son adimin ileri+geri gizli durumlari birlestirilir
      -> Linear -> ReLU -> Dropout -> Linear(n_class)   (siniflandirma basligi)

TF-IDF+MLP'ye gore fark:
    - TF-IDF, kelime SIRASINI bilmez ("hic iyi degil" ile "iyi, hic degil"
      TF-IDF icin ayni "kelime torbasi"dir). LSTM ise cumleyi soldan saga
      okuyarak sirayi ve baglami (orn. "degil" kelimesinin oncesindeki
      kelimenin anlamini tersine cevirmesini) daha iyi yakalayabilir.
    - Kelimeler artik sabit TF-IDF agirliklari degil, egitim sirasinda
      OGRENILEN yogun (dense) vektorler (embedding) ile temsil ediliyor.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence


class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128,
                 n_layers=1, n_classes=3, dropout=0.3, pad_idx=0):
        super().__init__()
        self.pad_idx = pad_idx
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)

        lstm_dropout = dropout if n_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=lstm_dropout,
        )

        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim * 2, 64)   # *2 çünkü bidirectional (ileri+geri birleşik)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, n_classes)

        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity="relu")
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def _classify_from_embeddings(self, emb, lengths):
        """emb: (batch, seq_len, embed_dim) -> logits: (batch, n_classes).
        Ayri bir fonksiyon olarak tutuyoruz cunku saliency() de embedding
        ciktisindan itibaren ayni yolu (LSTM + siniflandirma basligi) izliyor."""
        packed = pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.lstm(packed)
        # h_n: (num_layers*2, batch, hidden_dim) -> son katmanin ileri+geri durumlarini birlestir
        h_forward = h_n[-2]
        h_backward = h_n[-1]
        h_cat = torch.cat([h_forward, h_backward], dim=1)

        h = self.dropout(h_cat)
        h = self.relu(self.fc1(h))
        h = self.dropout(h)
        logits = self.fc2(h)
        return logits

    def forward(self, x, lengths):
        emb = self.embedding(x)
        return self._classify_from_embeddings(emb, lengths)

    @torch.no_grad()
    def predict_proba(self, x, lengths):
        self.eval()
        logits = self.forward(x, lengths)
        return F.softmax(logits, dim=1)

    @torch.no_grad()
    def predict(self, x, lengths):
        return torch.argmax(self.predict_proba(x, lengths), dim=1)

    def saliency(self, ids_tensor: torch.Tensor, length: int):
        """Tek bir yorum icin, her kelimenin tahmine olan (isaretli) katkisini
        hesaplar: dLogit/dEmbedding * Embedding, kelime boyutlari (embed_dim)
        uzerinden toplanir. Pozitif deger = o kelime tahmin edilen sinifi
        destekliyor demek. Bu, 'neden negatif?' aciklamasi icin kullanilir."""
        self.eval()
        x = ids_tensor.unsqueeze(0)               # (1, seq_len)
        lengths_t = torch.tensor([length])

        emb = self.embedding(x)
        emb = emb.clone().detach().requires_grad_(True)   # yeni, izlenebilir bir hesap grafiği

        logits = self._classify_from_embeddings(emb, lengths_t)
        probs = F.softmax(logits, dim=1)
        pred_class = int(torch.argmax(probs, dim=1).item())

        score = logits[0, pred_class]
        self.zero_grad()
        score.backward()

        grad = emb.grad[0]              # (seq_len, embed_dim)
        vec = emb[0].detach()           # (seq_len, embed_dim)
        contribution = (grad * vec).sum(dim=1)   # (seq_len,) -> her kelime icin tek bir skor

        return pred_class, probs[0].detach(), contribution[:length].detach()

    def save(self, path):
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path, config: dict):
        model = cls(
            vocab_size=config["vocab_size"],
            embed_dim=config["embed_dim"],
            hidden_dim=config["hidden_dim"],
            n_layers=config["n_layers"],
            n_classes=config["n_classes"],
            dropout=config["dropout"],
            pad_idx=config.get("pad_idx", 0),
        )
        model.load_state_dict(torch.load(path, map_location="cpu"))
        model.eval()
        return model
