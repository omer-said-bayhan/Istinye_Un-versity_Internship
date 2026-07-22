import pandas as pd
import random

# Pozitif ve Negatif kelime/kalıp havuzları
pozitif_taslaklar = [
    "Bu film gerçekten {harika}.", "Oyunculuklar {muhtesem} seviyedeydi.",
    "Senaryo {cok_iyi} yazılmış.", "İzlerken {cok_keyif_aldim}.",
    "Kesinlikle {tavsiye_ediyorum}.", "Görsel efektler {kusursuz} olmus.",
    "Hayatımda izlediğim en {iyi} filmlerden biriydi.", "Yönetmen {harika} bir iş çıkarmış.",
    "Müzikler ve atmosfer {cok_iyi}.", "Son zamanların en {basarili} yapımı."
]

negatif_taslaklar = [
    "Bu film tam bir {fiyasko}.", "Oyunculuklar {cok_kotu} seviyedeydi.",
    "Senaryo {berbat} yazılmış.", "İzlerken {cok_sıkıldım}.",
    "Zaman kaybı, {tavsiye_etmiyorum}.", "Görsel efektler {cok_amatorce} olmus.",
    "Hayatımda izlediğim en {kotu} filmlerden biriydi.", "Yönetmen {berbat} bir iş çıkarmış.",
    "Müzikler ve atmosfer {hic_olmamis}.", "Son zamanların en {basarisiz} yapımı."
]

harika = ["harika", "mükemmel", "efsane", "olağanüstü"]
muhtesem = ["muhteşem", "harika", "etkileyici", "kusursuz"]
cok_iyi = ["çok iyi", "akıcı", "derin", "sürükleyici"]
cok_keyif_aldim = ["çok keyif aldım", "büyülendim", "hayran kaldım", "çok eğlendim"]
tavsiye_ediyorum = ["tavsiye ediyorum", "herkese öneririm", "izlemelisiniz", "kaçırmayın"]

fiyasko = ["fiyasko", "zaman kaybı", "rezalet", "hayal kırıklığı"]
cok_kotu = ["çok kötü", "yetersiz", "yapay", "berbat"]
berbat = ["berbat", "sıkıcı", "anlamsız", "mantıksız"]
cok_sıkıldım = ["çok sıkıldım", "uyuyakaldım", "yarıda bıraktım", "pişman oldum"]
tavsiye_etmiyorum = ["tavsiye etmiyorum", "sakın izlemeyin", "yanından geçmeyin", "uzak durun"]

data = []

# 500 Pozitif Cümle Üret
for _ in range(500):
    template = random.choice(pozitif_taslaklar)
    sentence = template.format(
        harika=random.choice(harika), muhtesem=random.choice(muhtesem),
        cok_iyi=random.choice(cok_iyi), cok_keyif_aldim=random.choice(cok_keyif_aldim),
        tavsiye_ediyorum=random.choice(tavsiye_ediyorum), kusursuz="kusursuz",
        iyi="iyi", basarili="başarılı"
    )
    data.append([sentence, 1])  # 1 = Pozitif

# 500 Negatif Cümle Üret
for _ in range(500):
    template = random.choice(negatif_taslaklar)
    sentence = template.format(
        fiyasko=random.choice(fiyasko), cok_kotu=random.choice(cok_kotu),
        berbat=random.choice(berbat), cok_sıkıldım=random.choice(cok_sıkıldım),
        tavsiye_etmiyorum=random.choice(tavsiye_etmiyorum), cok_amatorce="çok amatörce",
        kotu="kötü", hic_olmamis="hiç olmamış", basarisiz="başarısız"
    )
    data.append([sentence, 0])  # 0 = Negatif

# DataFrame'e dönüştür ve karıştır (shuffle)
df = pd.DataFrame(data, columns=['yorum', 'duygu'])
df = df.sample(frac=1).reset_index(drop=True)

# CSV olarak kaydet
df.to_csv('film_yorumlari.csv', index=False, encoding='utf-8')
print("✅ 1000 cümlelik 'film_yorumlari.csv' başarıyla oluşturuldu!")