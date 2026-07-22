import torch
import torch.nn as nn
import torch.optim as optim
from data_generator import generate_dataset, CLASSES
from model import ShapeClassifier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

X, y = generate_dataset(n_per_class=1500)
split = int(len(X) * 0.85)
X_train, y_train = X[:split], y[:split]
X_val, y_val = X[split:], y[split:]

X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.long).to(device)
X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
y_val = torch.tensor(y_val, dtype=torch.long).to(device)

model = ShapeClassifier(num_classes=len(CLASSES)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

EPOCHS = 200
BATCH = 64

for epoch in range(EPOCHS):
    model.train()
    perm = torch.randperm(len(X_train))
    total_loss = 0
    for i in range(0, len(X_train), BATCH):
        idx = perm[i:i+BATCH]
        xb, yb = X_train[idx], y_train[idx]
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    model.eval()
    with torch.no_grad():
        val_out = model(X_val)
        val_acc = (val_out.argmax(1) == y_val).float().mean().item()
    print(f"Epoch {epoch+1}/{EPOCHS} - loss: {total_loss:.3f} - val_acc: {val_acc:.3f}")

torch.save(model.state_dict(), "shape_model.pth")
print("Model kaydedildi: shape_model.pth")