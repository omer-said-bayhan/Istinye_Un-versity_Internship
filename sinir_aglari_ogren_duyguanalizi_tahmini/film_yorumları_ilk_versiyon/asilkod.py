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

# En çok geçen kelimelerden sözlük yapıyoruz.
# 0 indeksini dolgu/padding için ayırıyoruz.
vocab = {word: i + 1 for i, (word, _) in enumerate(word_counts.items())}

def sentence_to_tensor(sentence, max_len=10):
    words = sentence.lower().split()
    # Kelimeleri ID'lerine çevir, sözlükte yoksa 0 ver
    ids = [vocab.get(w, 0) for w in words]
    # Sabit boyuta ayarla (padding / truncating)
    if len(ids) < max_len:
        ids += [0] * (max_len - len(ids))
    else:
        ids = ids[:max_len]
    return torch.tensor(ids, dtype=torch.long)

# Tüm cümleleri tensör matrisine çevirelim
X = torch.stack([sentence_to_tensor(s) for s in sentences])
y = torch.tensor(labels, dtype=torch.float32)

# ==========================================
# 3. DUYGU ANALİZİ MODELİ MİMARİSİ
# ==========================================
class SentimentNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super(SentimentNN, self).__init__()
        # Embedding: Kelime ID'lerini n-boyutlu vektörlere dönüştürür
        self.embedding = nn.Embedding(vocab_size + 1, embed_dim, padding_idx=0)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()  # Çıktıyı 0 ile 1 arasında sıkıştırır (Olasılık)

    def forward(self, x):
        embedded = self.embedding(x)  # [batch_size, max_len, embed_dim]
        # Cümledeki tüm kelime vektörlerinin ortalamasını alarak özet çıkarıyoruz
        pooled = embedded.mean(dim=1)  # [batch_size, embed_dim]
        
        out = self.relu(self.fc1(pooled))
        out = self.sigmoid(self.fc2(out))
        return out.squeeze()

# Modeli başlat
model = SentimentNN(vocab_size=len(vocab), embed_dim=16, hidden_dim=8)
criterion = nn.BCELoss()  # İkili sınıflandırma (Binary Cross Entropy)
optimizer = optim.Adam(model.parameters(), lr=0.01)

# ==========================================
# 4. EĞİTİM DÖNGÜSÜ
# ==========================================
print("🚀 Duygu Analizi Modeli Eğitiliyor...\n")
for epoch in range(20):
    optimizer.zero_grad()
    predictions = model(X)
    loss = criterion(predictions, y)
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1}/20 - Loss: {loss.item():.4f}")

print("\n✅ Eğitim Tamamlandı!\n")

# ==========================================
# 5. TEST ETME FONKSİYONU
# ==========================================
def tahmin_et(cumle):
    model.eval()
    with torch.no_grad():
        tensor_input = sentence_to_tensor(cumle).unsqueeze(0)
        prob = model(tensor_input).item()
        duygu = "Pozitif 😊" if prob > 0.5 else "Negatif 😡"
        print(f"Cümle: '{cumle}'")
        print(f"Tahmin: {duygu} (Güven Skoru: %{prob*100:.1f})\n")

# Örnek testler
tahmin_et("Bu film gerçekten mükemmel bir iş çıkarmış")
tahmin_et("Çok kötü ve sıkıcı bir film zaman kaybı")
tahmin_et("Yönetmen efsane bir atmosfer oluşturmuş")