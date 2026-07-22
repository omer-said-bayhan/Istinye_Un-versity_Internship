# -*- coding: utf-8 -*-
"""
Turkce metin on isleme.
- URL, mention, hashtag, emoji temizleme
- kucuk harfe cevirme (Turkce'ye ozel I/i sorunu duzeltilerek)
- noktalama temizleme
- stopword filtreleme
- cok basit bir suffix-stripping "hafif kokbulma" (Zemberek gibi tam morfolojik
  analiz yerine, bagimsizlik icin elle yazilmis basit bir yaklasim)
"""

import re

# En sik kullanilan Turkce stopword listesi (elle derlenmis, kucuk ama etkili)
TURKISH_STOPWORDS = set("""
acaba ama aslında az bazı belki biri birkaç birşey biz bu buna bunda
bundan bunu bunun burada çok çünkü da daha de defa diye eğer en gibi
hem hep hepsi her hiç için ile ise kez ki kim mı mu mü nasıl ne neden
nerde nerede nereye niçin niye o sanki şey siz şu tüm ve veya ya yani
yoksa çünkü ise ama fakat lakin ancak ben sen o biz siz onlar bir
bana beni benim sana seni senin ona onu onun bize bizi bizim size sizi
sizin onlara onları onların da de ki mi mu mü var yok olarak olan olur
oldu olacak diye göre kadar sonra önce şimdi hâlâ artık böyle şöyle
öyle çok daha en gene yine tekrar""".split())

URL_RE = re.compile(r"http\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#\w+")
EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001FAFF"
    "\U00002700-\U000027BF"
    "\U0001F1E0-\U0001F1FF"
    "]+", flags=re.UNICODE
)
MULTISPACE_RE = re.compile(r"\s+")
# Turkce harfler + rakam + bosluk disindaki her seyi temizle
NON_ALPHA_RE = re.compile(r"[^a-zçğıöşü0-9\s]")


def turkish_lower(text: str) -> str:
    """Python'un varsayilan .lower() metodu 'I' harfini 'i' yapar,
    Turkce'de ise 'I' -> 'ı' olmali. Bunu elle duzeltiyoruz."""
    text = text.replace("İ", "i").replace("I", "ı")
    return text.lower()


# Cok basit, kural tabanli bir "hafif govde/kok" indirgeyici.
# Tam bir morfolojik analiz yapmaz ama en yaygin cekim eklerini keser,
# boylece "guzelmis", "guzeldi", "guzeller" gibi kelimeler ayni koke yaklasir.
COMMON_SUFFIXES = [
    "lerinden", "larından", "lerinde", "larında", "lerini", "larını",
    "leriyle", "larıyla", "mişti", "muştu", "müştü", "mıştı",
    "iyor", "ıyor", "uyor", "üyor", "ecek", "acak", "meli", "malı",
    "ler", "lar", "den", "dan", "ten", "tan", "nin", "nın", "nun", "nün",
    "in", "ın", "un", "ün", "de", "da", "te", "ta", "mi", "mı", "mu", "mü",
    "im", "ım", "um", "üm", "sin", "sın", "sun", "sün", "dir", "dır",
    "dur", "dür", "tir", "tır", "tur", "tür", "miş", "mış", "muş", "müş",
    "dı", "di", "du", "dü", "tı", "ti", "tu", "tü", "cı", "ci", "cu", "cü",
]


def light_stem(word: str) -> str:
    if len(word) <= 4:
        return word
    for suf in sorted(COMMON_SUFFIXES, key=len, reverse=True):
        if word.endswith(suf) and len(word) - len(suf) >= 3:
            return word[: -len(suf)]
    return word


def clean_text(text: str) -> str:
    text = URL_RE.sub(" ", text)
    text = MENTION_RE.sub(" ", text)
    text = HASHTAG_RE.sub(" ", text)
    text = EMOJI_RE.sub(" ", text)
    text = turkish_lower(text)
    text = NON_ALPHA_RE.sub(" ", text)
    text = MULTISPACE_RE.sub(" ", text).strip()
    return text


def tokenize(text: str, use_stem: bool = True) -> list:
    text = clean_text(text)
    tokens = [t for t in text.split() if t and t not in TURKISH_STOPWORDS and len(t) > 1]
    if use_stem:
        tokens = [light_stem(t) for t in tokens]
    return tokens


if __name__ == "__main__":
    ornek = "Bu video KESİNLİKLE harikaydı!! 😍 https://youtu.be/xyz @kanaladı çok güzeldi, tekrar izleyeceğim :)"
    print(tokenize(ornek))
