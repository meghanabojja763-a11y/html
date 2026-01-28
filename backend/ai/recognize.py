import torch, pickle
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image

mtcnn = MTCNN(image_size=160)
facenet = InceptionResnetV1(pretrained='vggface2').eval()

with open("data/facenet_embeddings.pkl", "rb") as f:
    database = pickle.load(f)

def recognize_face(image: Image.Image):
    face = mtcnn(image)
    if face is None:
        return "Unknown"

    emb = facenet(face.unsqueeze(0)).detach()
    min_dist = 1.0
    identity = "Unknown"

    for name, db_emb in database.items():
        db_emb = torch.tensor(db_emb)
        dist = torch.norm(emb - db_emb).item()
        if dist < min_dist and dist < 0.9:
            min_dist = dist
            identity = name

    return identity
