import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import tkinter as tk
from PIL import Image, ImageOps

# ==========================================
# 1. MODEL MİMARİSİ
# ==========================================
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

MODEL_FILE = 'mnist_model.pth'
model = NeuralNet()

# ==========================================
# 2. MODELİ EĞİT VEYA YÜKLE
# ==========================================
if not os.path.exists(MODEL_FILE):
    print("⚠️ Kaydedilmiş model bulunamadı! Model eğitiliyor...")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 3
    for epoch in range(epochs):
        for batch_idx, (data, targets) in enumerate(train_loader):
            outputs = model(data)
            loss = criterion(outputs, targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{epochs} Tamamlandı. Loss: {loss.item():.4f}")
        
    torch.save(model.state_dict(), MODEL_FILE)
    print("✅ Model eğitildi ve 'mnist_model.pth' olarak kaydedildi!\n")
else:
    print("✅ Kaydedilmiş model dosyası bulundu! Eğitilmeden direkt yükleniyor...\n")
    model.load_state_dict(torch.load(MODEL_FILE))

model.eval() # Modeli değerlendirme (tahmin) moduna alıyoruz

# ==========================================
# 3. PAINT (ÇİZİM) ARAYÜZÜ VE TAHMİN
# ==========================================
class App:
    def __init__(self, root):
        self.root = root
        self.root.title("MNIST Rakam Tahmin Edici")
        
        # Çizim tuvali (Canvas) - MNIST gibi siyah arka plan
        self.canvas_size = 280
        self.canvas = tk.Canvas(root, width=self.canvas_size, height=self.canvas_size, bg='black')
        self.canvas.pack(pady=10)
        
        # Arkada görüntüyü işlemek için PIL resmi
        self.image = Image.new("L", (self.canvas_size, self.canvas_size), "black")
        self.draw = tk.Canvas(self.canvas)
        
        # Fare çizim olayları
        self.canvas.bind("<B1-Motion>", self.paint)
        
        # Butonlar ve Etiketler
        self.btn_confirm = tk.Button(root, text="Confirm (Tahmin Et)", command=self.predict, font=("Arial", 12, "bold"), bg="#4CAF50", fg="white")
        self.btn_confirm.pack(side=tk.LEFT, padx=20, pady=10)
        
        self.btn_clear = tk.Button(root, text="Temizle", command=self.clear, font=("Arial", 12), bg="#f44336", fg="white")
        self.btn_clear.pack(side=tk.RIGHT, padx=20, pady=10)
        
        self.label_result = tk.Label(root, text="Lütfen bir rakam çizin", font=("Arial", 16))
        self.label_result.pack(pady=15)
        
    def paint(self, event):
        # Fare ile çizim yapma (Beyaz kalem)
        x1, y1 = (event.x - 10), (event.y - 10)
        x2, y2 = (event.x + 10), (event.y + 10)
        self.canvas.create_oval(x1, y1, x2, y2, fill='white', outline='white')
        
        # PIL tarafındaki arka plan resmine de aynısını çiziyoruz
        from PIL import ImageDraw
        draw = ImageDraw.Draw(self.image)
        draw.ellipse([x1, y1, x2, y2], fill='white')

    def clear(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (self.canvas_size, self.canvas_size), "black")
        self.label_result.config(text="Lütfen bir rakam çizin")

    def predict(self):
        # 1. Çizilen resmi 28x28 boyutuna indir (MNIST formatı)
        img_resized = self.image.resize((28, 28))
        
        # 2. PyTorch Tensörüne dönüştür ve MNIST normalizasyonu uygula
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        img_tensor = transform(img_resized).unsqueeze(0) # Batch boyutu ekle (1, 1, 28, 28)
        
        # 3. Model tahmini yap
        with torch.no_grad():
            output = model(img_tensor)
            prediction = torch.argmax(output, dim=1).item()
            
        self.label_result.config(text=f"Tahmin: {prediction} 🎯")

# Uygulamayı Başlat
if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()