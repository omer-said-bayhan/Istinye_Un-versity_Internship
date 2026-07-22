import tkinter as tk
from PIL import Image, ImageDraw
import os
from data_generator import CLASSES, IMG_SIZE, crop_and_center

CANVAS_SIZE = 280
REAL_DIR = "dataset_real"

class CollectorApp:
    def __init__(self, root):
        for label in CLASSES:
            os.makedirs(os.path.join(REAL_DIR, label), exist_ok=True)

        self.canvas = tk.Canvas(root, width=CANVAS_SIZE, height=CANVAS_SIZE, bg="black")
        self.canvas.pack()
        self.image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), color=0)
        self.draw = ImageDraw.Draw(self.image)
        self.canvas.bind("<B1-Motion>", self.paint)

        self.status = tk.Label(root, text="Bir şekil çiz, sonra doğru butona bas", font=("Arial", 12))
        self.status.pack()

        btn_frame = tk.Frame(root)
        btn_frame.pack()
        for label in CLASSES:
            tk.Button(btn_frame, text=f"Bu bir {label}", width=12,
                      command=lambda l=label: self.save(l)).pack(side="left")

        tk.Button(root, text="Temizle", command=self.clear).pack()

        self.counts = {l: len(os.listdir(os.path.join(REAL_DIR, l))) for l in CLASSES}
        self.update_status()

    def paint(self, event):
        r = 5
        self.canvas.create_oval(event.x-r, event.y-r, event.x+r, event.y+r, fill="white", outline="white")
        self.draw.ellipse([event.x-r, event.y-r, event.x+r, event.y+r], fill=255)

    def clear(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), color=0)
        self.draw = ImageDraw.Draw(self.image)

    def save(self, label):
        processed = crop_and_center(self.image, out_size=IMG_SIZE, pad=15)
        idx = self.counts[label]
        processed.save(os.path.join(REAL_DIR, label, f"real_{idx:03d}.png"))
        self.counts[label] += 1
        self.clear()
        self.update_status()

    def update_status(self):
        text = "  ".join(f"{l}: {self.counts[l]}" for l in CLASSES)
        self.status.config(text=f"Toplanan: {text}")

root = tk.Tk()
root.title("Gerçek Veri Toplama")
app = CollectorApp(root)
root.mainloop()
