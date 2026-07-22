
import torch
import numpy as np
from model import ShapeClassifier
from data_generator import generate_dataset, CLASSES

device = torch.device("cpu")
model = ShapeClassifier(num_classes=len(CLASSES))
model.load_state_dict(torch.load("shape_model.pth", map_location=device))
model.eval()

X, y = generate_dataset(n_per_class=300)          # X: (N, 4096) flatten geliyor
X = X.reshape(-1, 1, 64, 64)                       # CNN formatına çevir: (N, 1, 64, 64)
X_t = torch.tensor(X, dtype=torch.float32)

with torch.no_grad():
    preds = model(X_t).argmax(1).numpy()

matrix = np.zeros((len(CLASSES), len(CLASSES)), dtype=int)
for true_label, pred_label in zip(y, preds):
    matrix[true_label][pred_label] += 1

print("Satır: gerçek, Sütun: tahmin")
print("        " + "  ".join(f"{c:>8}" for c in CLASSES))
for i, row in enumerate(matrix):
    print(f"{CLASSES[i]:>8}: " + "  ".join(f"{v:>8}" for v in row))
