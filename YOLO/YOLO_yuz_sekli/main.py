from ultralytics import YOLO

# Modeli yükle
model = YOLO('yüz/best.pt')
# 0 varsayılan bilgisayar kamerasını ifade eder
model.predict(source="0", show=True)