import os
import pickle
import torch
from PIL import Image
import cv2
from facenet_pytorch import MTCNN, InceptionResnetV1

device = "cpu"

# Models
mtcnn = MTCNN(image_size=160, margin=20, device=device)
facenet = InceptionResnetV1(pretrained="vggface2").eval()

image_root = "images"   # extracted dataset images
embeddings = {}

for root, _, files in os.walk(image_root):
    for file in files:
        if file.lower().endswith((".jpg", ".png")):
            path = os.path.join(root, file)

            img = cv2.imread(path)
            if img is None:
                continue

            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            face = mtcnn(rgb)

            if face is None:
                continue

            with torch.no_grad():
                emb = facenet(face.unsqueeze(0))

            # ✅ extract roll number
            roll_no = file.split("_")[0]
            embeddings[roll_no] = emb.squeeze(0)

            print(f"Embedded: {roll_no}")

os.makedirs("embeddings", exist_ok=True)
with open("embeddings/facenet_embeddings.pkl", "wb") as f:
    pickle.dump(embeddings, f)

print("✅ Correct FaceNet embeddings created")
