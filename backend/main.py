import streamlit as st
import cv2, pickle, torch, os
import numpy as np
import pandas as pd
from facenet_pytorch import MTCNN, InceptionResnetV1
from datetime import datetime
from torchvision import transforms

# ---------------- LOGIN CREDENTIALS ----------------
VALID_USERNAME = "admin"
VALID_PASSWORD = "admin123"

# ---------------- SESSION STATE ----------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# ==================================================
# 🔐 LOGIN PAGE
# ==================================================
if not st.session_state.logged_in:
    st.set_page_config(page_title="Login | Face Attendance", layout="centered")

    st.title("🔐 Login")
    st.markdown("### Face Attendance System")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username == VALID_USERNAME and password == VALID_PASSWORD:
            st.session_state.logged_in = True
            st.success("Login successful!")
            st.rerun()
        else:
            st.error("Invalid username or password")

    st.stop()   # 🚨 stops app execution until login success

# ==================================================
# 🎓 MAIN APP STARTS AFTER LOGIN
# ==================================================

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Face Attendance System",
    page_icon="🎓",
    layout="wide"
)

# ---------------- SIDEBAR ----------------
st.sidebar.title("🎓 Face Attendance System")
page = st.sidebar.radio(
    "Navigation",
    ["📷 Live Attendance", "📊 Attendance Records", "📈 Attendance Analysis", "ℹ️ About"]
)

# Logout button
if st.sidebar.button("🚪 Logout"):
    st.session_state.logged_in = False
    st.rerun()

# ---------------- LOAD MODELS ----------------
@st.cache_resource
def load_models():
    mtcnn = MTCNN(image_size=160, margin=20)
    facenet = InceptionResnetV1(pretrained="vggface2").eval()
    return mtcnn, facenet

mtcnn, facenet = load_models()

# ---------------- FACE TRANSFORM ----------------
face_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])
])

# ---------------- LOAD DATABASE ----------------
@st.cache_resource
def load_database():
    with open("embeddings/facenet_embeddings.pkl", "rb") as f:
        data = pickle.load(f)

    for k in data:
        if not isinstance(data[k], torch.Tensor):
            data[k] = torch.tensor(data[k])
        data[k] = data[k].detach()

    return data

database = load_database()

# ---------------- ATTENDANCE FILE ----------------
os.makedirs("attendance", exist_ok=True)
attendance_file = "attendance/attendance.xlsx"

session_marked = set()

# ---------------- FUNCTIONS ----------------
def recognize(face_tensor):
    with torch.no_grad():
        emb = facenet(face_tensor.unsqueeze(0))

    min_dist = float("inf")
    identity = "Unknown"

    for roll_no, db_emb in database.items():
        dist = torch.norm(emb - db_emb).item()
        if dist < min_dist and dist < 0.75:
            min_dist = dist
            identity = roll_no

    return identity


def mark_attendance(name):
    if name == "Unknown" or name in session_marked:
        return

    now = datetime.now()
    today = now.date()

    row = {
        "Roll No": name,
        "Date": today,
        "Time": now.strftime("%H:%M:%S")
    }

    if os.path.exists(attendance_file):
        df = pd.read_excel(attendance_file)
        df["Date"] = pd.to_datetime(df["Date"]).dt.date

        if ((df["Roll No"] == name) & (df["Date"] == today)).any():
            session_marked.add(name)
            return

        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])

    df.to_excel(attendance_file, index=False)
    session_marked.add(name)

# ==================================================
# 📷 LIVE ATTENDANCE
# ==================================================
if page == "📷 Live Attendance":
    st.title("📷 Live Face Attendance")

    run = st.checkbox("▶ Start Camera")
    frame_window = st.image([])

    if run:
        cap = cv2.VideoCapture(0)

        while run:
            ret, frame = cap.read()
            if not ret:
                st.error("Camera not accessible")
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            boxes, _ = mtcnn.detect(rgb)

            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box)
                    face = rgb[y1:y2, x1:x2]

                    if face.size == 0:
                        continue

                    face_tensor = face_transform(face)
                    name = recognize(face_tensor)
                    mark_attendance(name)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, name, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            frame_window.image(frame, channels="BGR")

        cap.release()

# ==================================================
# 📊 ATTENDANCE RECORDS
# ==================================================
elif page == "📊 Attendance Records":
    st.title("📊 Attendance Records")

    if os.path.exists(attendance_file):
        df = pd.read_excel(attendance_file)
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No attendance recorded yet")

# ==================================================
# 📈 ATTENDANCE ANALYSIS
# ==================================================
elif page == "📈 Attendance Analysis":
    st.title("📈 Attendance Analysis")

    if not os.path.exists(attendance_file):
        st.warning("No attendance data available yet")
    else:
        df = pd.read_excel(attendance_file)
        df["Date"] = pd.to_datetime(df["Date"]).dt.date

        total_students = len(database)

        daily = df.groupby("Date")["Roll No"].nunique().reset_index(name="Present")
        daily["Attendance %"] = (daily["Present"] / total_students * 100).round(2)

        st.dataframe(daily.sort_values("Date", ascending=False), use_container_width=True)
        st.bar_chart(daily.set_index("Date")["Present"])

# ==================================================
# ℹ️ ABOUT
# ==================================================
else:
    st.title("ℹ️ About")
    st.markdown("""
    ✔ Secure Login  
    ✔ Face Recognition Attendance  
    ✔ Day-wise Reports  
    ✔ Excel Export  

    🚀 **Production Ready**
    """)
