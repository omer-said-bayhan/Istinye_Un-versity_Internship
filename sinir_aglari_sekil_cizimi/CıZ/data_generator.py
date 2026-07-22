import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import random
import math

IMG_SIZE = 64
GEN_SIZE = 260
CLASSES = ["kup", "kure", "silindir", "koni"]

# ---------- EL TİTREMESİ SİMÜLE EDEN ÇİZİM MOTORU ----------

# Global stil yoğunluğu - generate_sample her örnek için bunu ayarlıyor.
# 0.0 = neredeyse düz/temiz çizim, 1.0 = belirgin titrek/organik çizim
_STYLE_INTENSITY = 1.0

def hand_stroke(draw, points, base_width=6, wobble_amp=3.0, closed=False):
    """points arasını, gerçek el çizimine benzeyen dalgalı/kalınlığı değişen
    bir fırça darbesi gibi çizer. Düz PIL line/ellipse yerine bunu kullanıyoruz."""
    wobble_amp = wobble_amp * _STYLE_INTENSITY
    pts = list(points) + ([points[0]] if closed else [])
    phase = random.uniform(0, 2 * math.pi)
    freq = random.uniform(1.5, 3.5)

    all_pts = []
    for i in range(len(pts) - 1):
        x0, y0 = pts[i]
        x1, y1 = pts[i + 1]
        seg_len = math.hypot(x1 - x0, y1 - y0)
        n = max(int(seg_len / 3), 2)
        for j in range(n):
            t = j / n
            x = x0 + (x1 - x0) * t
            y = y0 + (y1 - y0) * t
            all_pts.append((x, y))

    total = len(all_pts)
    for i, (x, y) in enumerate(all_pts):
        # düşük frekanslı dalga = organik titreme (rastgele gürültüden daha doğal görünür)
        wob = math.sin(phase + freq * i / max(total, 1) * 2 * math.pi) * wobble_amp
        # dalgayı çizgiye dik yönde uygula
        if i < total - 1:
            nx, ny = all_pts[i + 1]
        else:
            nx, ny = x, y
        dx, dy = nx - x, ny - y
        norm = math.hypot(dx, dy) or 1
        px, py = -dy / norm, dx / norm
        wx, wy = x + px * wob, y + py * wob

        r = base_width / 2 + random.uniform(-1, 1.5) * _STYLE_INTENSITY
        r = max(1, r)
        draw.ellipse([wx - r, wy - r, wx + r, wy + r], fill=255)

def hand_polygon(draw, points, base_width=6, wobble_amp=3.0):
    hand_stroke(draw, points, base_width=base_width, wobble_amp=wobble_amp, closed=True)

def hand_ellipse(draw, cx, cy, rx, ry, base_width=6, wobble_amp=3.0, n_points=36):
    pts = [(cx + rx * math.cos(2*math.pi*i/n_points), cy + ry * math.sin(2*math.pi*i/n_points))
           for i in range(n_points)]
    hand_stroke(draw, pts, base_width=base_width, wobble_amp=wobble_amp, closed=True)

def jitter_pts(points, amount=4):
    return [(x + random.uniform(-amount, amount), y + random.uniform(-amount, amount)) for x, y in points]

def rand_width():
    return random.uniform(4, 11)

def crop_and_center(img, out_size=IMG_SIZE, pad=15, threshold=20):
    arr = np.array(img)
    ys, xs = np.where(arr > threshold)
    if len(xs) == 0:
        return Image.new("L", (out_size, out_size), color=0)
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
    x1, y1 = min(arr.shape[1], x1 + pad), min(arr.shape[0], y1 + pad)
    cropped = img.crop((x0, y0, x1, y1))
    w, h = cropped.size
    side = max(w, h)
    square = Image.new("L", (side, side), color=0)
    square.paste(cropped, ((side - w) // 2, (side - h) // 2))
    return square.resize((out_size, out_size), Image.LANCZOS)

def random_perspective_warp(img):
    """Hafif perspektif/eğim bozulması - elle çizerken kağıdın/açının
    hafif farklı görünmesi hissi verir."""
    w, h = img.size
    m = random.uniform(-0.15, 0.15)
    coeffs = (1, m, -m*w/2, m*0.3, 1, -m*0.3*h/2)
    return img.transform((w, h), Image.AFFINE, coeffs, fillcolor=0, resample=Image.BILINEAR)

# ---------- KÜP ----------

def kup_izometrik(draw, cx, cy, size, sw):
    off = size * random.uniform(0.25, 0.45)
    s = size * 0.5
    front = jitter_pts([(cx-s, cy-s), (cx+s, cy-s), (cx+s, cy+s), (cx-s, cy+s)])
    back = jitter_pts([(x+off, y-off) for x, y in front])
    hand_polygon(draw, front, sw)
    hand_polygon(draw, back, sw)
    for f, b in zip(front, back):
        hand_stroke(draw, [f, b], sw)

def kup_ust_yuzeyli(draw, cx, cy, size, sw):
    s = size * 0.5
    off_x = size * random.uniform(0.3, 0.5)
    off_y = size * random.uniform(0.2, 0.35)
    front = jitter_pts([(cx-s, cy-s), (cx+s, cy-s), (cx+s, cy+s), (cx-s, cy+s)])
    top = jitter_pts([(cx-s, cy-s), (cx-s+off_x, cy-s-off_y), (cx+s+off_x, cy-s-off_y), (cx+s, cy-s)])
    right = jitter_pts([(cx+s, cy-s), (cx+s+off_x, cy-s-off_y), (cx+s+off_x, cy+s-off_y), (cx+s, cy+s)])
    hand_polygon(draw, front, sw)
    hand_polygon(draw, top, sw)
    hand_polygon(draw, right, sw)

def kup_sade_kenarli(draw, cx, cy, size, sw):
    s = size * 0.55
    front = jitter_pts([(cx-s, cy-s), (cx+s, cy-s), (cx+s, cy+s), (cx-s, cy+s)])
    hand_polygon(draw, front, sw)
    off = size * 0.25
    for (x, y) in front[:2]:
        hand_stroke(draw, [(x, y), (x+off, y-off)], sw)

def kup_ters_aci(draw, cx, cy, size, sw):
    off = size * random.uniform(0.25, 0.45)
    s = size * 0.5
    front = jitter_pts([(cx-s, cy-s), (cx+s, cy-s), (cx+s, cy+s), (cx-s, cy+s)])
    back = jitter_pts([(x-off, y+off) for x, y in front])
    hand_polygon(draw, front, sw)
    hand_polygon(draw, back, sw)
    for f, b in zip(front, back):
        hand_stroke(draw, [f, b], sw)

KUP_STYLES = [kup_izometrik, kup_ust_yuzeyli, kup_sade_kenarli, kup_ters_aci]

def draw_kup(draw, cx, cy, size, sw):
    random.choice(KUP_STYLES)(draw, cx, cy, size, sw)

# ---------- KÜRE ----------

def kure_klasik(draw, cx, cy, r, sw):
    hand_ellipse(draw, cx, cy, r, r, sw)
    for _ in range(random.randint(1, 3)):
        squash = random.uniform(0.25, 0.55)
        if random.random() < 0.5:
            hand_ellipse(draw, cx, cy, r, r*squash, sw*0.8)
        else:
            hand_ellipse(draw, cx, cy, r*squash, r, sw*0.8)

def kure_globe(draw, cx, cy, r, sw):
    hand_ellipse(draw, cx, cy, r, r, sw)
    n_meridians = random.randint(3, 5)
    for i in range(1, n_meridians):
        frac = i / n_meridians
        squash = math.sin(frac * math.pi)
        yy = cy - r + frac * 2 * r
        width = r * squash
        if width > 2:
            hand_ellipse(draw, cx, yy, width, r*0.06, sw*0.6)
    hand_ellipse(draw, cx, cy, r*0.3, r, sw*0.7)

def kure_golgeli(draw, cx, cy, r, sw):
    hand_ellipse(draw, cx, cy, r, r, sw)
    shade_side = random.choice([-1, 1])
    sx = cx + shade_side * r * 0.35
    sy = cy - r * 0.2
    sr = r * random.uniform(0.5, 0.7)
    n = 12
    start = 200 if shade_side > 0 else 20
    pts = [(sx + sr*math.cos(math.radians(start+i*10)), sy + sr*math.sin(math.radians(start+i*10))) for i in range(n)]
    hand_stroke(draw, pts, sw*0.8)

def kure_noktali(draw, cx, cy, r, sw):
    hand_ellipse(draw, cx, cy, r, r, sw)
    bias_angle = random.uniform(0, 2*math.pi)
    for _ in range(random.randint(25, 45)):
        rr = r * random.uniform(0.1, 0.9)
        theta = bias_angle + random.uniform(-1.2, 1.2)
        px = cx + rr * math.cos(theta)
        py = cy + rr * math.sin(theta) * 0.9
        if (px-cx)**2 + (py-cy)**2 <= r*r:
            dr = max(1, sw/3)
            draw.ellipse([px-dr, py-dr, px+dr, py+dr], fill=255)

def kure_izgara(draw, cx, cy, r, sw):
    hand_ellipse(draw, cx, cy, r, r, sw)
    for squash in [0.3, 0.6]:
        hand_ellipse(draw, cx, cy, r, r*squash, sw*0.8)
    hand_ellipse(draw, cx, cy, r*0.4, r, sw*0.8)

KURE_STYLES = [kure_klasik, kure_globe, kure_golgeli, kure_noktali, kure_izgara]

def draw_kure(draw, cx, cy, size, sw):
    r = size * 0.5
    random.choice(KURE_STYLES)(draw, cx, cy, r, sw)

# ---------- SİLİNDİR ----------

def draw_silindir(draw, cx, cy, size, sw):
    w = size * 0.5
    h = w * random.uniform(1.6, 2.4)          # daha belirgin uzun/dar oran - küple karışmasın
    eh = h * random.uniform(0.18, 0.28)        # elips daha belirgin yuvarlak (düz çizgi gibi algılanmasın)
    top_y, bot_y = cy - h, cy + h

    # üst/alt elipsler HER ZAMAN belirgin şekilde çizilsin (yuvarlaklık vurgusu için ekstra tur)
    hand_ellipse(draw, cx, top_y, w, eh, sw, wobble_amp=1.5)
    hand_ellipse(draw, cx, top_y, w*0.97, eh*0.97, sw*0.6, wobble_amp=1.0)  # ikinci ince çizgi - kalınlaştırma
    hand_ellipse(draw, cx, bot_y, w, eh, sw, wobble_amp=1.5)

    # dikey kenarlar DÜZ ve az titrek olsun - küpün eğik/kaotik kenarlarından net ayrışsın
    hand_stroke(draw, [(cx-w, top_y), (cx-w, bot_y)], sw, wobble_amp=0.8)
    hand_stroke(draw, [(cx+w, top_y), (cx+w, bot_y)], sw, wobble_amp=0.8)

    # gövde eğriliğini vurgulayan hafif orta çizgi (silindirin yuvarlaklığını hatırlatır)
    if random.random() < 0.5:
        hand_stroke(draw, [(cx-w*0.3, top_y+eh*0.3), (cx-w*0.3, bot_y-eh*0.3)], sw*0.4, wobble_amp=0.6)

    for _ in range(random.randint(0, 2)):
        hx = random.uniform(cx-w*0.6, cx+w*0.6)
        hand_stroke(draw, [(hx, top_y+eh), (hx, bot_y-eh)], sw*0.5)

# ---------- KONİ ----------

def draw_koni(draw, cx, cy, size, sw):
    w = size * 0.5
    h = w * random.uniform(1.3, 1.9)
    eh = h * random.uniform(0.12, 0.2)
    apex = (cx, cy-h)
    base_y = cy + h*0.3
    hand_ellipse(draw, cx, base_y, w, eh, sw)
    hand_stroke(draw, [apex, (cx-w, base_y)], sw)
    hand_stroke(draw, [apex, (cx+w, base_y)], sw)

DRAW_FUNCS = {"kup": draw_kup, "kure": draw_kure, "silindir": draw_silindir, "koni": draw_koni}

def add_hand_drawn_noise(img, intensity=1.0):
    if random.random() < 0.5 * intensity + 0.1:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.2, 0.9) * max(intensity, 0.3)))
    arr = np.array(img, dtype=np.float32)
    noise_std = random.uniform(3, 14) * max(intensity, 0.2)
    arr = np.clip(arr + np.random.normal(0, noise_std, arr.shape), 0, 255)
    if random.random() < 0.2 * intensity:
        h, w = arr.shape
        for _ in range(random.randint(1, 3)):
            gx, gy = random.randint(0, w-1), random.randint(0, h-1)
            gr = random.randint(1, 3)
            arr[max(0,gy-gr):gy+gr, max(0,gx-gr):gx+gr] *= random.uniform(0.2, 0.6)
    return Image.fromarray(arr.astype(np.uint8))

def generate_sample(label, augment=True):
    global _STYLE_INTENSITY
    # Örneklerin bir kısmı temiz/düzgün, bir kısmı orta, bir kısmı belirgin titrek olsun.
    # Bu üçünün karışımı modelin hem düzgün hem elle-hızlı-çizilmiş şekilleri tanımasını sağlar.
    roll = random.random()
    if roll < 0.10:
        _STYLE_INTENSITY = random.uniform(0.4, 0.7)       # az titrek (azınlık - çeşitlilik için)
    else:
        _STYLE_INTENSITY = random.uniform(0.8, 1.3)       # belirgin titrek/organik (çoğunluk - eski hal)

    img = Image.new("L", (GEN_SIZE, GEN_SIZE), color=0)
    draw = ImageDraw.Draw(img)
    cx = GEN_SIZE // 2 + random.randint(-15, 15)
    cy = GEN_SIZE // 2 + random.randint(-15, 15)
    size = random.uniform(35, 65)
    sw = rand_width()
    DRAW_FUNCS[label](draw, cx, cy, size, sw)

    angle = random.uniform(-20, 20)
    img = img.rotate(angle, fillcolor=0)

    # perspektif bozulmasını da stil yoğunluğuna bağlayalım - temiz örneklerde daha az
    if random.random() < 0.6 * _STYLE_INTENSITY + 0.1:
        img = random_perspective_warp(img)

    img = crop_and_center(img, out_size=IMG_SIZE, pad=random.randint(8, 25))

    if augment:
        # gürültü miktarını da stil yoğunluğuna göre ölçekle
        img = add_hand_drawn_noise(img, intensity=_STYLE_INTENSITY)

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
