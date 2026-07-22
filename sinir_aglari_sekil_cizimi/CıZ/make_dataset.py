import os
from PIL import Image
from data_generator import generate_sample, CLASSES

OUTPUT_DIR = "dataset"

CLASS_COUNTS = {
    "kup": 3000,
    "kure": 3000,
    "silindir": 3800,
    "koni": 3000,
}

def main():
    for label in CLASSES:
        class_dir = os.path.join(OUTPUT_DIR, label)
        os.makedirs(class_dir, exist_ok=True)
        n = CLASS_COUNTS[label]
        for i in range(n):
            arr = generate_sample(label)
            img = Image.fromarray((arr * 255).astype("uint8"), mode="L")
            img.save(os.path.join(class_dir, f"{label}_{i:04d}.png"))
        print(f"{label}: {n} resim kaydedildi -> {class_dir}")

    print("Bitti. Dataset klasörü:", os.path.abspath(OUTPUT_DIR))

if __name__ == "__main__":
    main()
