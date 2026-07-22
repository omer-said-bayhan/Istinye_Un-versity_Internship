# -*- coding: utf-8 -*-
"""
Embedding + LSTM modelini winvoker/turkish-sentiment-analysis-dataset uzerinde egitir.

Kullanim:
    python train.py
    python train.py --max_rows 60000 --epochs 8       (hizli deneme)
    python train.py --csv data/kendi_verim.csv         (kendi verinle)
"""

import argparse
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from vocab import Vocabulary
from neural_network import SentimentLSTM

LABEL_MAP = {"negative": 0, "notr": 1, "neutral": 1, "positive": 2}
INV_LABEL_MAP = {0: "Negatif", 1: "Notr", 2: "Pozitif"}


def load_data(csv_path=None, max_rows=None):
    if csv_path:
        import pandas as pd
        df = pd.read_csv(csv_path)
    else:
        from datasets import load_dataset
        ds = load_dataset("winvoker/turkish-sentiment-analysis-dataset")
        import pandas as pd
        df = pd.concat([ds["train"].to_pandas(), ds["test"].to_pandas()], ignore_index=True)

    df["label"] = df["label"].str.lower().map(LABEL_MAP)
    df = df.dropna(subset=["text", "label"])
    df["label"] = df["label"].astype(int)
    if max_rows:
        df = df.sample(n=min(max_rows, len(df)), random_state=42)
    return df["text"].tolist(), df["label"].tolist()


class SeqDataset(Dataset):
    """Metinleri onceden encode edip sabit uzunlukta ID tensorlerine cevirir."""
    def __init__(self, texts, labels, vocab: Vocabulary, max_len: int):
        self.ids = []
        self.lengths = []
        for t in texts:
            ids, length, _ = vocab.encode(t, max_len)
            self.ids.append(ids)
            self.lengths.append(length)
        self.ids = torch.tensor(self.ids, dtype=torch.long)
        self.lengths = torch.tensor(self.lengths, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.ids[idx], self.lengths[idx], self.labels[idx]


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Cihaz: {device}")

    print("Veri yukleniyor...")
    texts, labels = load_data(args.csv, args.max_rows)
    print(f"Toplam ornek: {len(texts)}")

    rng = np.random.RandomState(42)
    idx = rng.permutation(len(texts))
    split = int(len(texts) * 0.9)
    train_idx, val_idx = idx[:split], idx[split:]
    train_texts = [texts[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    val_texts = [texts[i] for i in val_idx]
    val_labels = [labels[i] for i in val_idx]

    print("Kelime dagarcigi (vocabulary) olusturuluyor...")
    vocab = Vocabulary(max_size=args.vocab_size, min_freq=2)
    vocab.fit(train_texts)
    print(f"Sozluk boyutu: {len(vocab)}")
    vocab.save("model/vocab.json")

    print("Veri kodlaniyor (ID dizilerine cevriliyor)...")
    train_ds = SeqDataset(train_texts, train_labels, vocab, args.max_len)
    val_ds = SeqDataset(val_texts, val_labels, vocab, args.max_len)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=512, shuffle=False)

    model = SentimentLSTM(
        vocab_size=len(vocab), embed_dim=args.embed_dim, hidden_dim=args.hidden_dim,
        n_layers=args.n_layers, n_classes=3, dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max",
                                                             factor=0.5, patience=2)

    config = {
        "vocab_size": len(vocab), "embed_dim": args.embed_dim, "hidden_dim": args.hidden_dim,
        "n_layers": args.n_layers, "n_classes": 3, "dropout": args.dropout,
        "max_len": args.max_len, "pad_idx": 0,
    }

    best_val_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        total_loss = 0.0
        for xb, lb, yb in train_loader:
            xb, lb, yb = xb.to(device), lb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb, lb)
            loss = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)  # LSTM'de gradyan patlamasini onler
            optimizer.step()
            total_loss += loss.item() * xb.size(0)

        train_loss = total_loss / len(train_ds)

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for xb, lb, yb in val_loader:
                xb, lb, yb = xb.to(device), lb.to(device), yb.to(device)
                preds = torch.argmax(model(xb, lb), dim=1)
                correct += (preds == yb).sum().item()
                total += yb.size(0)
        val_acc = correct / total

        scheduler.step(val_acc)
        dt = time.time() - t0
        print(f"Epoch {epoch}/{args.epochs} - loss: {train_loss:.4f} "
              f"- val_acc: {val_acc:.4f} - ({dt:.1f}s)")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model.save("model/sentiment_model.pt")
            torch.save(config, "model/model_config.pt")
            print(f"  -> yeni en iyi model kaydedildi (val_acc={val_acc:.4f})")

    print(f"Egitim bitti. En iyi val_acc: {best_val_acc:.4f}")
    print("Model: model/sentiment_model.pt | Vocab: model/vocab.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default=None,
                         help="Yerel CSV (text,label sutunlu). Verilmezse HF'den indirilir.")
    parser.add_argument("--max_rows", type=int, default=None,
                         help="Hizli test icin veri setini kucultmek istersen (orn. 60000).")
    parser.add_argument("--vocab_size", type=int, default=20000)
    parser.add_argument("--max_len", type=int, default=40, help="Yorum basina maksimum kelime sayisi")
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--n_layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    train(args)
