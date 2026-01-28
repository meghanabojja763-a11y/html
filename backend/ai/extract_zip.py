import os
import zipfile

# -------- CONFIG --------
ZIP_PATH = "datasets/face attendence.v1-v0-clean.yolov11.zip"
EXTRACT_TO = "datasets/face_attendance"

# ------------------------
os.makedirs(EXTRACT_TO, exist_ok=True)

if not os.path.exists(ZIP_PATH):
    print("❌ ZIP file not found:", ZIP_PATH)
    exit()

print("📦 Extracting dataset...")

with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
    zip_ref.extractall(EXTRACT_TO)

print("✅ Extraction completed!")
print("📁 Dataset extracted to:", EXTRACT_TO)
