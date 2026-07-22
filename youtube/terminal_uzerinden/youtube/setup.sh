#!/usr/bin/env bash
# Linux icin otomatik kurulum scripti.
# Kullanim:
#   chmod +x setup.sh
#   ./setup.sh

set -e

echo "== Python ve venv kontrolu =="
if ! command -v python3 &> /dev/null; then
    echo "python3 bulunamadi. Once python3 kur: sudo apt install python3 python3-venv python3-pip"
    exit 1
fi

if ! python3 -c "import venv" &> /dev/null; then
    echo "python3-venv modulu eksik. Kuruluyor (sudo gerekebilir)..."
    sudo apt-get update && sudo apt-get install -y python3-venv
fi

echo "== Sanal ortam olusturuluyor (venv) =="
python3 -m venv venv
source venv/bin/activate

echo "== pip guncelleniyor =="
pip install --upgrade pip

echo "== PyTorch (CPU surumu) kuruluyor =="
# Ekran karti (NVIDIA/CUDA) olmadigi icin CPU-only wheel'i kuruyoruz.
# Bu, gereksiz yere ~2GB'lik CUDA kutuphanelerinin inmesini engeller,
# kurulum daha hizli ve daha kucuk olur.
pip install torch --index-url https://download.pytorch.org/whl/cpu

echo "== Diger bagimliliklar kuruluyor =="
pip install -r requirements.txt
pip install datasets

mkdir -p model data

echo ""
echo "Kurulum tamamlandi."
echo "Sanal ortami aktif etmek icin her yeni terminalde su komutu calistir:"
echo "    source venv/bin/activate"
echo ""
echo "Sonraki adim: YouTube API key ayarla, sonra:"
echo "    python train.py"
echo "    python analyze_video.py --url \"VIDEO_LINKI\" --api_key \"SENIN_KEYIN\""
