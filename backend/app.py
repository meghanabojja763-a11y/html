from fastapi import FastAPI, UploadFile, File
from database import SessionLocal, engine
from models import Attendance, Base
from datetime import datetime
from ai.recognize import recognize_face
from PIL import Image
import io

Base.metadata.create_all(bind=engine)

app = FastAPI()

@app.post("/mark-attendance")
async def mark_attendance(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    name = recognize_face(image)

    if name != "Unknown":
        db = SessionLocal()
        now = datetime.now()
        record = Attendance(
            name=name,
            date=str(now.date()),
            time=now.strftime("%H:%M:%S")
        )
        db.add(record)
        db.commit()
        db.close()

    return {"name": name}

@app.get("/attendance")
def get_attendance():
    db = SessionLocal()
    data = db.query(Attendance).all()
    db.close()
    return data
