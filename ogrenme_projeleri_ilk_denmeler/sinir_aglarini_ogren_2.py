                                #diğer kodun üzerine eklenmiştir


import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt  # Görselleştirme kütüphanesi!
import matplotlib
matplotlib.use('TkAgg')  

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root='./data2', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=1000, shuffle=False)

# 2. MİMARİ
class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = NeuralNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 3. EĞİTİM VE GRAFİK İÇİN VERİ TOPLAMA
epochs = 3
loss_history = []  # Her adımda loss değerlerini buraya kaydedeceğiz

print("🚀 Eğitiliyor ve grafik verisi toplanıyor...\n")

for epoch in range(epochs):
    for batch_idx, (data, targets) in enumerate(train_loader):
        outputs = model(data)
        loss = criterion(outputs, targets)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Her adımın Loss değerini listeye ekliyoruz
        loss_history.append(loss.item())

# ==========================================
# 4. GÖRSELLEŞTİRME (LOSS GRAFİĞİ)
# ==========================================
plt.figure(figsize=(10, 5))
plt.plot(loss_history, label='Eğitim Loss Değeri', color='orange', alpha=0.7)
plt.title('Eğitim Süresince Hatanın (Loss) Değişimi')
plt.xlabel('Adım Sayısı (Batch Iteration)')
plt.ylabel('Loss (Hata)')
plt.grid(True)
plt.legend()
plt.show()
# Test setinden ilk 5 resmi çekip modele soralım
dataiter = iter(test_loader)
images, labels = next(dataiter)

# Modelin tahmini
outputs = model(images)
_, preds = torch.max(outputs, 1)

# İlk 5 resmi ve yapay zekanın tahminini çizdirelim
fig, axes = plt.subplots(1, 5, figsize=(12, 3))
for i in range(5):
    axes[i].imshow(images[i].squeeze(), cmap='gray')
    axes[i].set_title(f"Tahmin: {preds[i].item()}")
    axes[i].axis('off')
plt.show()
