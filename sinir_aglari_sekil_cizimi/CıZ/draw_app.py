import tkinter as tk
from PIL import Image, ImageDraw
import torch
import numpy as np
from model import ShapeClassifier
from data_generator import CLASSES, IMG_SIZE, crop_and_center

device = torch.device("cpu")
model = ShapeClassifier(num_classes=len(CLASSES))
model.load_state_dict(torch.load("shape_model.pth", map_location=device))
model.eval()

CANVAS_SIZE = 280

class PaintApp:
    def __init__(self, root):
        self.canvas = tk.Canvas(root, width=CANVAS_SIZE, height=CANVAS_SIZE, bg="black")
        self.canvas.pack()
        self.image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), color=0)
        self.draw = ImageDraw.Draw(self.image)
        self.canvas.bind("<B1-Motion>", self.paint)

        self.label = tk.Label(root, text="Bir şekil çiz (küp, küre, silindir, koni)", font=("Arial", 12), justify="left")
        self.label.pack()

        tk.Button(root, text="Tahmin Et", command=self.predict).pack(side="left")
        tk.Button(root, text="Temizle", command=self.clear).pack(side="left")

    def paint(self, event):
        r = 5
        self.canvas.create_oval(event.x-r, event.y-r, event.x+r, event.y+r, fill="white", outline="white")
        self.draw.ellipse([event.x-r, event.y-r, event.x+r, event.y+r], fill=255)

    def clear(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), color=0)
        self.draw = ImageDraw.Draw(self.image)
        self.label.config(text="Bir şekil çiz (küp, küre, silindir, koni)")

    def predict(self):
        processed = crop_and_center(self.image, out_size=IMG_SIZE, pad=15)
        processed.resize((256, 256)).save("debug_last_input.png")

        arr = np.array(processed, dtype=np.float32) / 255.0
        tensor = torch.tensor(arr[None, None, :, :], dtype=torch.float32)
        with torch.no_grad():
            out = model(tensor)
            probs = torch.softmax(out, dim=1)[0]
            pred = probs.argmax().item()
        detail = "\n".join(f"{CLASSES[i]}: %{probs[i]*100:.0f}" for i in range(len(CLASSES)))
        self.label.config(text=f"Tahmin: {CLASSES[pred]}\n{detail}")

root = tk.Tk()
root.title("3B Şekil Tanıma")
app = PaintApp(root)
root.mainloop()
