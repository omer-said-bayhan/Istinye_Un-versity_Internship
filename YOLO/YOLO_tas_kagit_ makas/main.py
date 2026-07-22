import cv2
from ultralytics import YOLO

# Modeli yükle (best.pt dosyan aynı klasörde olduğu için doğrudan isimle çağırdık)
model = YOLO("best.pt")

# Web kamerasını aç
cap = cv2.VideoCapture(0)
    
print("Kamera başladı! 'q' tuşuna basarak çıkabilirsin.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO11n ile anlık tahmin
    results = model.predict(source=frame, conf=0.4, verbose=False) # conf'u 0.4 yaparsan daha az hatalı kutu çizer

    # Sonucu çizdir
    frame = results[0].plot()

    cv2.imshow("YOLO11n Canli Test", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()