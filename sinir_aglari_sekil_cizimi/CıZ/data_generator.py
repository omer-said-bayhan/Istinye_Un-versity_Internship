
import numpy as np
from PIL import Image, ImageDraw
import random
import math

IMG_SIZE = 64
CLASSES = ["kup", "kure", "silindir", "koni"]

def jitter(points, amount=2):
    return [(x + random.uniform(-amount, amount), y + random.uniform(-amount, amount)) for x, y in points]

def draw_kup(draw, cx, cy, size):
    off = size * random.uniform(0.25, 0.45)
    s = size * 0.5
    front = [(cx-s, cy-s), (cx+s, cy-s), (cx+s, cy+s), (cx-s, cy+s)]
    back = [(x+off, y-off) for x, y in front]
    front, back = jitter(front), jitter(back)
    draw.polygon(front, outline=255)
    draw.polygon(back, outline=255)
    for f, b in zip(front, back):
        draw.line([f, b], fill=255)

# ---------- KÜRE ÇEŞİTLERİ ----------

def kure_klasik(draw, cx, cy, r):
    """Dış çember + 1-3 rastgele enlem/boylam elipsi."""
    draw.ellipse([cx-r, cy-r, cx+r, cy+r], outline=255)
    n_lines = random.randint(1, 3)
    for _ in range(n_lines):
        squash = random.uniform(0.25, 0.55)
        if random.random() < 0.5:
            draw.ellipse([cx-r, cy-r*squash, cx+r, cy+r*squash], outline=255)
        else:
            draw.ellipse([cx-r*squash, cy-r, cx+r*squash, cy+r], outline=255)

def kure_globe(draw, cx, cy, r):
    """Dünya küresi gibi - birden fazla paralel yatay meridyen çizgisi."""
    draw.ellipse([cx-r, cy-r, cx+r, cy+r], outline=255)
    n_meridians = random.randint(3, 5)
    for i in range(1, n_meridians):
        frac = i / n_meridians
        squash = math.sin(frac * math.pi)  # ortada geniş, kenarlarda dar
        yy = cy - r + frac * 2 * r
        width = r * squash
        if width > 2:
            draw.ellipse([cx-width, yy-2, cx+width, yy+2], outline=255)
    # bir de dikey boylam çizgisi
    draw.ellipse([cx-r*0.3, cy-r, cx+r*0.3, cy+r], outline=255)

def kure_golgeli(draw, cx, cy, r):
    """Sade çember + hilal şeklinde gölge/highlight, çizgisiz."""
    draw.ellipse([cx-r, cy-r, cx+r, cy+r], outline=255)
    shade_side = random.choice([-1, 1])
    sx = cx + shade_side * r * 0.35
    sy = cy - r * 0.2
    sr = r * random.uniform(0.5, 0.7)
    start, end = (200, 340) if shade_side > 0 else (20, 160)
    draw.arc([sx-sr, sy-sr, sx+sr, sy+sr], start=start, end=end, fill=255)

def kure_noktali(draw, cx, cy, r):
    """Çember + içeride rastgele noktalarla doku/gölgelendirme hissi."""
    draw.ellipse([cx-r, cy-r, cx+r, cy+r], outline=255)
    n_dots = random.randint(25, 45)
    # noktalar bir kenara yığılsın (gölge tarafı) ki 3B hacim hissi versin
    bias_angle = random.uniform(0, 2*math.pi)
    for _ in range(n_dots):
        rr = r * random.uniform(0.1, 0.9)
        theta = bias_angle + random.uniform(-1.2, 1.2)
        px = cx + rr * math.cos(theta)
        py = cy + rr * math.sin(theta) * 0.9
        if (px-cx)**2 + (py-cy)**2 <= r*r:
            draw.point((px, py), fill=255)

def kure_izgara(draw, cx, cy, r):
    """Küre üzerinde çapraz ızgara (grid) hissi - birden fazla farklı yönlü elips."""
    draw.ellipse([cx-r, cy-r, cx+r, cy+r], outline=255)
    for squash in [0.3, 0.6]:
        draw.ellipse([cx-r, cy-r*squash, cx+r, cy+r*squash], outline=255)
    for squash in [0.4]:
        draw.ellipse([cx-r*squash, cy-r, cx+r*squash, cy+r], outline=255)

KURE_STYLES = [kure_klasik, kure_globe, kure_golgeli, kure_noktali, kure_izgara]

def draw_kure(draw, cx, cy, size):
    r = size * 0.5
    style_func = random.choice(KURE_STYLES)
    style_func(draw, cx, cy, r)

# ---------- SİLİNDİR ----------

def draw_silindir(draw, cx, cy, size):
    w = size * 0.5
    h = w * random.uniform(1.5, 2.2)
    ellipse_h = h * random.uniform(0.12, 0.2)

    top_y = cy - h
    bot_y = cy + h

    draw.ellipse([cx-w, top_y-ellipse_h, cx+w, top_y+ellipse_h], outline=255)
    draw.ellipse([cx-w, bot_y-ellipse_h, cx+w, bot_y+ellipse_h], outline=255)
    draw.line([(cx-w, top_y), (cx-w, bot_y)], fill=255)
    draw.line([(cx+w, top_y), (cx+w, bot_y)], fill=255)

    n_hatch = random.randint(0, 3)
    for _ in range(n_hatch):
        hx = random.uniform(cx-w*0.6, cx+w*0.6)
        draw.line([(hx, top_y+ellipse_h), (hx, bot_y-ellipse_h)], fill=180)

# ---------- KONİ ----------

def draw_koni(draw, cx, cy, size):
    w = size * 0.5
    h = w * random.uniform(1.3, 1.9)
    ellipse_h = h * random.uniform(0.12, 0.2)
    apex = (cx, cy-h)
    base_y = cy + h*0.3
    draw.ellipse([cx-w, base_y-ellipse_h, cx+w, base_y+ellipse_h], outline=255)
    draw.line([apex, (cx-w, base_y)], fill=255)
    draw.line([apex, (cx+w, base_y)], fill=255)

DRAW_FUNCS = {"kup": draw_kup, "kure": draw_kure, "silindir": draw_silindir, "koni": draw_koni}

def generate_sample(label):
    img = Image.new("L", (IMG_SIZE, IMG_SIZE), color=0)
    draw = ImageDraw.Draw(img)
    cx = IMG_SIZE//2 + random.randint(-3, 3)
    cy = IMG_SIZE//2 + random.randint(-3, 3)
    size = random.uniform(20, 30)
    DRAW_FUNCS[label](draw, cx, cy, size)

    angle = random.uniform(-12, 12)
    img = img.rotate(angle, fillcolor=0)

    arr = np.array(img, dtype=np.float32) / 255.0
    return arr

def generate_dataset(n_per_class=None, class_counts=None):
    if class_counts is None:
        n = n_per_class or 1500
        class_counts = {label: n for label in CLASSES}

    X, y = [], []
    for idx, label in enumerate(CLASSES):
        for _ in range(class_counts[label]):
            X.append(generate_sample(label))
            y.append(idx)
    X = np.array(X).reshape(len(X), -1)
    y = np.array(y)
    perm = np.random.permutation(len(X))
    return X[perm], y[perm]
