
import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from model import ShapeClassifier
from data_generator import CLASSES, IMG_SIZE

DATASET_DIR = "dataset"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_dataset():
    X, y = [], []
    for idx, label in enumerate(CLASSES):
        class_dir = os.path.join(DATASET_DIR, label)
        for fname in os.listdir(class_dir):
            img = Image.open(os.path.join(class_dir, fname)).convert("L")
            arr = np.array(img, dtype=np.float32) / 255.0
            X.append(arr)              # artık flatten YOK, 2D kalıyor
            y.append(idx)
    X = np.array(X)                    # (N, 64, 64)
    X = X[:, None, :, :]               # (N, 1, 64, 64) - kanal boyutu ekle
    y = np.array(y)
    perm = np.random.permutation(len(X))
    return X[perm], y[perm]

X, y = load_dataset()
print(f"Toplam {len(X)} örnek yüklendi.")

split = int(len(X) * 0.85)
X_train, y_train = X[:split], y[:split]
X_val, y_val = X[split:], y[split:]

X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.long).to(device)
X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
y_val = torch.tensor(y_val, dtype=torch.long).to(device)

model = ShapeClassifier(num_classes=len(CLASSES)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

EPOCHS = 40
BATCH = 64

best_val_acc = 0
patience = 6
patience_counter = 0

for epoch in range(EPOCHS):
    model.train()
    perm = torch.randperm(len(X_train))
    total_loss = 0
    n_batches = 0
    for i in range(0, len(X_train), BATCH):
        idx = perm[i:i+BATCH]
        xb, yb = X_train[idx], y_train[idx]
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1

    model.eval()
    with torch.no_grad():
        val_out = model(X_val)
        val_acc = (val_out.argmax(1) == y_val).float().mean().item()
    avg_loss = total_loss / n_batches
    print(f"Epoch {epoch+1}/{EPOCHS} - avg_loss: {avg_loss:.4f} - val_acc: {val_acc:.3f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        patience_counter = 0
        torch.save(model.state_dict(), "shape_model.pth")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping - {patience} epoch boyunca iyileşme yok.")
            break

print(f"En iyi model kaydedildi (val_acc: {best_val_acc:.3f})")
