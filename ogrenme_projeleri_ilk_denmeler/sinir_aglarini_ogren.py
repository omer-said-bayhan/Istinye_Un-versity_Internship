import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ==========================================
# 1. VERİ HAZIRLIĞI (MNIST Veri Seti)
# ==========================================
# Resimleri sayısal matrislere dönüştürüp normalize ediyoruz
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)) # Veriyi standartlaştırma
])

# 60.000 eğitim resmini ve 10.000 test resmini indiriyoruz
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Mini-Batch mantığı! Verileri 64'erli paketler halinde yüklüyoruz
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=1000, shuffle=False)

# ==========================================
# 2. YAPAY SİNİR AĞI MİMARİSİ
# ==========================================
class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        # 28x28 piksellik resmi düzleştirip 784 girdi nöronu yapıyoruz
        self.fc1 = nn.Linear(28 * 28, 128) # 1. Katman: 784 -> 128 nöron
        self.relu = nn.ReLU()              # Aktivasyon fonksiyonu
        self.fc2 = nn.Linear(128, 64)      # 2. Katman: 128 -> 64 nöron
        self.fc3 = nn.Linear(64, 10)       # Çıktı Katmanı: 64 -> 10 rakam (0-9)

    def forward(self, x):
        x = x.view(-1, 28 * 28) # Resmi tek sıra vektör yap (Flatten)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = NeuralNet()

# ==========================================
# 3. MALIYET FONKSİYONU VE OPTIMIZER
# ==========================================
criterion = nn.CrossEntropyLoss() # Loss (Maliyet) fonksiyonumuz
optimizer = optim.Adam(model.parameters(), lr=0.001) # Learning Rate = 0.001

# ==========================================
# 4. EĞİTİM DÖNGÜSÜ (TRAINING LOOP)
# ==========================================
epochs = 3 # Tüm verinin üzerinden 3 kez geçeceğiz

print("🚀 Eğitimin Başlıyor...\n")

for epoch in range(epochs):
    running_loss = 0.0
    for batch_idx, (data, targets) in enumerate(train_loader):
        # 1. İleri Besleme (Forward Pass): Tahmin yap
        outputs = model(data)
        loss = criterion(outputs, targets)
        
        # 2. Sıfırlama: Eski gradyanları temizle
        optimizer.zero_grad()
        
        # 3. Geri Yayılım (Backpropagation): Suçluları tespit et ve gradyanı hesapla!
        loss.backward()
        
        # 4. Ağırlıkları Güncelle (Gradient Descent): Düğmeleri çevir!
        optimizer.step()
        
        running_loss += loss.item()
        
        # Her 200 pakette bir durumu ekrana yazdır
        if (batch_idx + 1) % 200 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Adım [{batch_idx+1}/{len(train_loader)}], Loss (Hata): {loss.item():.4f}")

print("\n✅ Eğitim Tamamlandı!")

# ==========================================
# 5. TEST ETME (BAŞARI ORANI)
# ==========================================
correct = 0
total = 0

with torch.no_grad(): # Test ederken gradyan hesaplamaya gerek yok
    for data, targets in test_loader:
        outputs = model(data)
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()

accuracy = 100 * correct / total
print(f"\n🎯 Modelin Hiç Görmediği Test Verisindeki Başarısı: %{accuracy:.2f}")
