# apt-get update && apt-get install -y tesseract-ocr
# pip install -r requirements.txt

#!/usr/bin/env bash
echo "✅ Starting render-build.sh"
apt-get update && apt-get install -y tesseract-ocr
which tesseract
pip install -r requirements.txt
