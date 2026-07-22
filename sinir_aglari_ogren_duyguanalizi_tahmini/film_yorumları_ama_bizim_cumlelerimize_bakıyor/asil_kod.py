import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter


# ==========================================
# 1. VERİ SETİNİ YÜKLE
# ==========================================
df = pd.read_csv('film_yorumlari.csv')
sentences = df['yorum'].values
labels = df['duygu'].values


# ==========================================
# 2. BASİT KELİME SÖZLÜĞÜ (VOCABULARY) OLUŞTURMA
# ==========================================
all_words = ' '.join(sentences).lower().split()
word_counts = Counter(all_words)
vocab = {word: i + 1 for i, (word, _) in enumerate(word_counts.items())}

def sentence_to_tensor(sentence, max_len=10):
    words = sentence.lower().split()
    ids = [vocab.get(w, 0) for w in words]
    if len(ids) < max_len:
        ids += [0] * (max_len - len(ids))
    else:
        ids = ids[:max_len]
    return torch.tensor(ids, dtype=torch.long)

X = torch.stack([sentence_to_tensor(s) for s in sentences])
y = torch.tensor(labels, dtype=torch.float32)

# ==========================================
# 3. DUYGU ANALİZİ MODELİ MİMARİSİ
# ==========================================
class SentimentNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super(SentimentNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size + 1, embed_dim, padding_idx=0)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        embedded = self.embedding(x)
        pooled = embedded.mean(dim=1)
        out = self.relu(self.fc1(pooled))
        out = self.sigmoid(self.fc2(out))
        return out.squeeze()

model = SentimentNN(vocab_size=len(vocab), embed_dim=16, hidden_dim=8)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# ==========================================
# 4. EĞİTİM DÖNGÜSÜ (EPOCH SAYISI 50'YE ÇIKARILDI)
# ==========================================
epochs = 50  # Epoch sayısını 20'den 50'ye yükselttik!
print(f"🚀 Duygu Analizi Modeli {epochs} Epoch Boyunca Eğitiliyor...\n")

for epoch in range(epochs):
    optimizer.zero_grad()
    predictions = model(X)
    loss = criterion(predictions, y)
    loss.backward()
    optimizer.step()
    
    # Her 10 epoch'ta bir loss değerini yazdıralım
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{epochs} - Loss (Hata): {loss.item():.4f}")

print("\n✅ Eğitim Tamamlandı! Şimdi Kendi Cümlelerini Test Edebilirsin.\n")

# ==========================================
# 5. CANLI KULLANICI GİRDİSİ (CANLI TEST)
# ==========================================
def tahmin_et(cumle):
    model.eval()
    with torch.no_grad():
        tensor_input = sentence_to_tensor(cumle).unsqueeze(0)
        prob = model(tensor_input).item()
        
        # Olasılık değerine göre duygu belirleme
        duygu = "Pozitif 😊" if prob > 0.5 else "Negatif 😡"
        
        print("-" * 40)
        print(f"💬 Cümle: '{cumle}'")
        print(f"🎯 Tahmin: {duygu}")
        print(f"📊 Güven Skoru: %{prob*100:.1f}")
        print("-" * 40 + "\n")

# Sonsuz döngü ile kullanıcıdan sürekli cümle alalım
while True:
    kullanici_girdisi = input("Bir film yorumu yazın (Çıkmak için 'q' yazın): ")
    
    if kullanici_girdisi.lower() == 'q':
        print("Programdan çıkılıyor. Görüşmek üzere!")
        break
    elif kullanici_girdisi.strip() == "":
        continue
        
    tahmin_et(kullanici_girdisi)