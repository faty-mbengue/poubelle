import streamlit as st
from ultralytics import YOLO
import tempfile
from PIL import Image
import cv2
import numpy as np
import time

# -------------------------
#   CONFIG INTERFACE
# -------------------------
st.set_page_config(
    page_title="Détection Poubelles - YOLOv8",
    page_icon="🗑️",
    layout="wide"
)

st.markdown("""
    <h1 style='text-align:center; color:#2E86C1;'>🗑️ Détection de Poubelles (Pleines / Vides)</h1>
    <p style='text-align:center; font-size:18px;'>
        Upload une image ou une vidéo — le modèle détecte la poubelle et indique si elle est pleine ou vide.
    </p>
""", unsafe_allow_html=True)

# -------------------------
#   CHARGEMENT DU MODELE
# -------------------------
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# -------------------------
#   FUNCTIONS
# -------------------------
def predict_image(upload):
    img = Image.open(upload).convert("RGB")
    results = model(img)[0]
    return results


def predict_video_with_controls(upload):
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(upload.read())

    cap = cv2.VideoCapture(tfile.name)
    stframe = st.empty()

    # State init
    if "paused" not in st.session_state:
        st.session_state.paused = False
    if "counts" not in st.session_state:
        st.session_state.counts = {"total": 0, "pleine": 0, "vide": 0}
    if "history" not in st.session_state:
        st.session_state.history = []
    if "captured_frames" not in st.session_state:
        st.session_state.captured_frames = []  # store thumbnails

    col1, col2 = st.columns(2)

    with col1:
        if st.button("⏸️ Pause"):
            st.session_state.paused = True

    with col2:
        if st.button("▶️ Reprendre"):
            st.session_state.paused = False

    # Video loop
    while True:
        if st.session_state.paused:
            time.sleep(0.1)
            continue

        success, frame = cap.read()
        if not success:
            break

        results = model(frame, conf=0.5)[0]
        annotated = results.plot()

        # Detection processing
        for box in results.boxes:
            cls = int(box.cls[0])
            st.session_state.counts["total"] += 1

            if cls == 0:
                label = "poubelle_vide"
                st.session_state.counts["vide"] += 1
            elif cls == 1:
                label = "poubelle_pleine"
                st.session_state.counts["pleine"] += 1

            # Save thumbnail
            thumb = cv2.resize(annotated, (320, 200))
            thumb = cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB)

            st.session_state.captured_frames.append((thumb, label))
            st.session_state.history.append(f"{label} détectée")

        # Display video frame
        stframe.image(annotated, channels="RGB", use_container_width=True)

        time.sleep(0.05)

    cap.release()


# -------------------------
#   SIDEBAR
# -------------------------
with st.sidebar:
    st.header("📤 Upload fichier")
    file = st.file_uploader(
        "Choisis une image ou une vidéo",
        type=["jpg", "jpeg", "png", "mp4", "avi"]
    )

st.markdown("---")

# Stats
st.sidebar.subheader("📊 Statistiques vidéo")
if "counts" in st.session_state:
    st.sidebar.write(f"🔢 Total détections : **{st.session_state.counts['total']}**")
    st.sidebar.write(f"🟢 Poubelle vide : **{st.session_state.counts['vide']}**")
    st.sidebar.write(f"🔴 Poubelle pleine : **{st.session_state.counts['pleine']}**")

st.sidebar.subheader("🗂️ Historique")
if "history" in st.session_state:
    for h in st.session_state.history[-10:]:
        st.sidebar.write("- " + h)

# -------------------------
#   PROCESSING
# -------------------------
if file:
    ftype = file.type

    # IMAGE
    if ftype.startswith("image"):
        st.subheader("🖼️ Image originale")
        st.image(file, use_container_width=True)

        if st.button("🚀 Lancer la détection (Image)"):
            with st.spinner("Analyse en cours..."):
                results = predict_image(file)
                annotated = results.plot()
                st.subheader("📦 Résultat")
                st.image(annotated, channels="RGB", use_container_width=True)

    # VIDEO
    elif ftype.startswith("video"):
        st.subheader("🎬 Vidéo uploadée")
        st.video(file)

        if st.button("🚀 Lancer la détection (Vidéo)"):
            # reset states
            st.session_state.paused = False
            st.session_state.counts = {"total": 0, "pleine": 0, "vide": 0}
            st.session_state.history = []
            st.session_state.captured_frames = []

            with st.spinner("Analyse vidéo en cours..."):
                predict_video_with_controls(file)

else:
    st.info("➡️ Upload une image ou une vidéo pour commencer.")


# ----------------------------------------
#   📸 AFFICHAGE DES FRAMES CAPTURÉES
# ----------------------------------------
st.markdown("---")
st.subheader("📸 Frames capturées")

if "captured_frames" in st.session_state and len(st.session_state.captured_frames) > 0:

    cols = st.columns(4)
    index = 0

    for img, label in st.session_state.captured_frames[-12:]:  # last 12 frames
        with cols[index % 4]:
            st.image(img, caption=f"Prédiction : {label}", use_container_width=True)
        index += 1

else:
    st.info("Aucune frame capturée pour l’instant.")
