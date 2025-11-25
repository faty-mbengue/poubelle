import streamlit as st
from ultralytics import YOLO
import tempfile
from PIL import Image
import cv2
import numpy as np
import os
import time

# ---------------------------
#   CONFIG PAGE
# ---------------------------
st.set_page_config(
    page_title="Détection Poubelles",
    page_icon="🗑️",
    layout="wide"
)

st.markdown("""
    <h1 style='text-align:center; color:#58a6ff;'>🗑️ Détection de Poubelles (Pleines / Vides)</h1>
    <p style='text-align:center; font-size:18px; color:#e6edf3;'>
        Analyse d’image ou vidéo — avec affichage du type de poubelle sous chaque frame détectée.
    </p>
""", unsafe_allow_html=True)


# ---------------------------
#   LOAD MODEL
# ---------------------------
@st.cache_resource
def load_model():
    if not os.path.exists("best2.pt"):
        st.error("❌ Le modèle best2.pt est introuvable.")
        st.stop()
    return YOLO("best2.pt")

model = load_model()


# ---------------------------
#   PREDICTION IMAGE
# ---------------------------
def predict_image(upload):
    img = Image.open(upload).convert("RGB")
    results = model(img, conf=0.5)[0]
    return results


# ---------------------------
#   PREDICTION VIDEO
# ---------------------------
def predict_video(upload, interval_seconds=1):
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(upload.read())
    tfile.close()

    cap = cv2.VideoCapture(tfile.name)
    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps == 0:
        st.error("Impossible de récupérer les FPS de la vidéo.")
        return

    step = int(fps * interval_seconds)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    st.session_state.counts = {"total": 0, "vide": 0, "pleine": 0}
    st.session_state.captured_frames = []

    stframe = st.empty()
    progress_bar = st.progress(0)
    status_text = st.empty()

    frame_num = 0
    analyzed_count = 0
    last_time = time.time()

    while frame_num < total_frames:
        now = time.time()
        if now - last_time < interval_seconds:
            time.sleep(interval_seconds - (now - last_time))
        last_time = time.time()

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        success, frame = cap.read()
        if not success:
            break

        results = model(frame, conf=0.5)[0]
        annotated = results.plot()

        timestamp = frame_num / fps
        minutes = int(timestamp // 60)
        seconds = int(timestamp % 60)

        labels_in_frame = []

        # Count detections
        for box in results.boxes:
            cls = int(box.cls[0])

            if cls == 0:
                labels_in_frame.append("Poubelle vide")
                st.session_state.counts["vide"] += 1

            elif cls == 1:
                labels_in_frame.append("Poubelle pleine")
                st.session_state.counts["pleine"] += 1

            st.session_state.counts["total"] += 1

        if len(labels_in_frame) == 0:
            final_label = "Aucune détection"
        elif len(set(labels_in_frame)) == 1:
            final_label = labels_in_frame[0]
        else:
            final_label = "Mix: " + ", ".join(set(labels_in_frame))

        thumb = cv2.resize(annotated, (320, 200))
        thumb = cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB)

        st.session_state.captured_frames.append(
            (thumb, final_label, f"{minutes:02d}:{seconds:02d}")
        )

        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        stframe.image(annotated_rgb, use_container_width=True)

        analyzed_count += 1
        status_text.text(f"Frame {analyzed_count} analysée à {minutes:02d}:{seconds:02d}")

        frame_num += step
        progress_bar.progress(min(frame_num / total_frames, 1))

    cap.release()
    status_text.success(f"🎉 Analyse terminée : {analyzed_count} frames analysées.")


# ---------------------------
#   SIDEBAR
# ---------------------------
with st.sidebar:
    st.header("📤 Upload fichier")
    file = st.file_uploader("Image ou vidéo", type=["jpg", "jpeg", "png", "mp4", "avi"])

    interval_seconds = st.slider("Intervalle (sec)", 1, 10, 1)

    st.markdown("---")
    st.subheader("📊 Statistiques")
    if "counts" in st.session_state:
        st.write(f"🧮 Total détecté : **{st.session_state.counts['total']}**")
        st.write(f"🟢 Vides : **{st.session_state.counts['vide']}**")
        st.write(f"🔴 Pleines : **{st.session_state.counts['pleine']}**")


# ---------------------------
#   MAIN LOGIC
# ---------------------------
st.markdown("---")

if file:
    ftype = file.type

    # IMAGE MODE
    if ftype.startswith("image"):
        st.subheader("🖼️ Image")
        st.image(file, use_container_width=True)

        if st.button("🚀 Lancer la détection (Image)"):
            with st.spinner("Analyse en cours..."):
                results = predict_image(file)
                annotated = results.plot()

                st.subheader("📦 Résultat")
                st.image(annotated, channels="RGB", use_container_width=True)

                if len(results.boxes) == 0:
                    st.warning("⚠️ Aucune détection trouvée dans l’image.")

    # VIDEO MODE
    elif ftype.startswith("video"):
        st.subheader("🎬 Vidéo uploadée")
        st.video(file)

        if st.button("🚀 Lancer la détection (Vidéo)"):
            st.warning(f"Analyse toutes les {interval_seconds} sec.")
            with st.spinner("Analyse vidéo..."):
                predict_video(file, interval_seconds)

            if "captured_frames" in st.session_state:
                st.markdown("---")
                st.subheader("📸 Frames capturées")

                cols = st.columns(4)
                for idx, (img, label, timestamp) in enumerate(st.session_state.captured_frames):
                    with cols[idx % 4]:
                        icon = "🟢" if "vide" in label else ("🔴" if "pleine" in label else "⚪")
                        st.image(img, caption=f"{icon} {label} ({timestamp})", use_container_width=True)

else:
    st.info("📥 Upload une image ou une vidéo pour commencer.")
