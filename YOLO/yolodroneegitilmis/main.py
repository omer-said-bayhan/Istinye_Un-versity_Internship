import os
# Arch/Wayland üzerinde Qt pencere hatasını önlemek için X11 uyumluluğunu zorluyoruz
os.environ["QT_QPA_PLATFORM"] = "xcb"

import cv2
from ultralytics import YOLO

# 1. Model ağırlığını yüklüyoruz (best.pt dosyasının bu klasörde olduğundan emin ol)
model = YOLO('bestdrone.pt')

# 2. Test etmek istediğin drone videosunun adı
# Videoyu bu script ile aynı klasöre atıp adını "drone.mp4" yapabilirsin
video_path = "drone.mp4" 

# Güvenlik kontrolü: Video gerçekten orada mı?
if not os.path.exists(video_path):
    print(f"\n❌ HATA: '{video_path}' dosyası bulunamadı!")
    print(f"Lütfen test etmek istediğin drone videosunu şu klasöre kopyala ve adını '{video_path}' yap:")
    print(f"👉 {os.getcwd()}\n")
    exit()

# 3. Videoyu OpenCV ile açıyoruz
cap = cv2.VideoCapture(video_path)
window_name = "YOLO11 Canli Drone Avcisi (CPU)"

print("\n🚀 Yapay zeka video üzerinden drone taramaya başlıyor...")
print("Çıkmak için video penceresi açıkken klavyeden 'q' tuşuna basabilirsin.\n")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Video bitti veya okunacak kare kalmadı.")
        break

    # [ÖNEMLİ]: device='cpu' ile işlemciyi zorunlu kıldık.
    # conf=0.15 yaptık; model drone'dan %15 bile emin olsa etrafına yeşil kutuyu çizecek.
    results = model(frame, stream=True, conf=0.15, device='cpu')

    # Tahmin sonuçlarını (yeşil kutuları) karenin üzerine çizdiriyoruz
    annotated_frame = frame
    for r in results:
        annotated_frame = r.plot()

    # O açılan canlı pencerede görüntüyü ve kutuları gösteriyoruz
    cv2.imshow(window_name, annotated_frame)

    # 'q' tuşuna basılırsa döngüden çık
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Program başarıyla sonlandırıldı.")