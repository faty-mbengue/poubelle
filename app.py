import streamlit as st
from ultralytics import YOLO
import tempfile
from PIL import Image
import cv2
import numpy as np
import os
import time

st.set_page_config(
    page_title="Détection Poubelles - YOLO",
    page_icon="🗑️",
    layout="wide"
)

st.markdown("""
    <h1 style='text-align:center; color:#2E86C1;'>🗑️ Détection de Poubelles (Pleines / Vides)</h1>
    <p style='text-align:center; font-size:18px;'>
        Upload une image ou une vidéo — le modèle détecte la poubelle et indique si elle est pleine ou vide.
    </p>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    if not os.path.exists("best2.pt"):
        st.error("❌ Modèle best2.pt introuvable")
        st.stop()
    return YOLO("best2.pt")

model = load_model()

def predict_image(upload):
    img = Image.open(upload).convert("RGB")
    results = model(img, conf=0.5)[0]
    return results

def predict_video(upload, interval_seconds=1):
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(upload.read())
    tfile.close()

    cap = cv2.VideoCapture(tfile.name)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        st.error("Impossible de récupérer FPS de la vidéo.")
        return
    step = int(fps * interval_seconds)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if "counts" not in st.session_state:
        st.session_state.counts = {"total": 0, "vide": 0, "pleine": 0}
    if "captured_frames" not in st.session_state:
        st.session_state.captured_frames = []

    st.session_state.counts = {"total": 0, "vide": 0, "pleine": 0}
    st.session_state.captured_frames = []

    stframe = st.empty()
    progress_bar = st.progress(0)
    status_text = st.empty()

    frame_num = 0
    analyzed_count = 0

    # Contrôle du temps d'affichage pour régulariser vitesse
    last_time = time.time()

    while frame_num < total_frames:
        now = time.time()
        elapsed = now - last_time
        if elapsed < interval_seconds:
            time.sleep(interval_seconds - elapsed)
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

        cv2.putText(
            annotated,
            f"Temps: {minutes:02d}:{seconds:02d}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        num_detection = len(results.boxes)
        for box in results.boxes:
            cls = int(box.cls[0])
            st.session_state.counts["total"] += 1
            if cls == 0:
                st.session_state.counts["vide"] += 1
            elif cls == 1:
                st.session_state.counts["pleine"] += 1

        if num_detection > 0:
            thumb = cv2.resize(annotated, (320, 200))
            thumb = cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB)
            st.session_state.captured_frames.append((thumb, f"{num_detection} détection(s)", f"{minutes:02d}:{seconds:02d}"))

        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        stframe.image(annotated_rgb, use_container_width=True)
        analyzed_count += 1
        status_text.text(f"Frame {analyzed_count} analysée à {minutes:02d}:{seconds:02d}")

        frame_num += step
        progress_bar.progress(min(frame_num / total_frames, 1))

    cap.release()
    status_text.success(f"✅ Analyse terminée: {analyzed_count} frames analysées")

with st.sidebar:
    st.header("📤 Upload fichier")
    file = st.file_uploader(
        "Choisis une image ou une vidéo",
        type=["jpg", "jpeg", "png", "mp4", "avi"]
    )

    interval_seconds = st.slider("Intervalle entre frames analysées (secondes)", 1, 10, 1)

    st.markdown("---")
    st.subheader("📊 Statistiques")
    if "counts" in st.session_state:
        st.write(f"🔢 Total: **{st.session_state.counts['total']}**")
        st.write(f"🟢 Vides: **{st.session_state.counts['vide']}**")
        st.write(f"🔴 Pleines: **{st.session_state.counts['pleine']}**")

st.markdown("---")

if file:
    ftype = file.type

    if ftype.startswith("image"):
        st.subheader("🖼️ Image originale")
        st.image(file, use_container_width=True)

        if st.button("🚀 Lancer la détection (Image)"):
            with st.spinner("Analyse en cours..."):
                results = predict_image(file)
                annotated = results.plot()

                st.subheader("📦 Résultat de la détection")
                st.image(annotated, channels="RGB", use_container_width=True)

                boxes = results.boxes
                if len(boxes) > 0:
                    st.subheader("📊 Détails")
                    for i, box in enumerate(boxes):
                        cls = int(box.cls[0])
                        conf = float(box.conf[0])
                        classe = model.names[cls]

                        col1, col2, col3 = st.columns([2, 1, 1])
                        with col1:
                            st.write(f"**Détection {i+1}:** {classe}")
                        with col2:
                            st.write(f"Confiance: {conf:.2%}")
                        with col3:
                            icon = "🟢" if cls == 0 else "🔴"
                            st.write(icon)

    elif ftype.startswith("video"):
        st.subheader("🎬 Vidéo uploadée")
        st.video(file)

        if st.button("🚀 Lancer la détection (Vidéo)"):
            st.warning(f"⏱️ La vidéo sera analysée toutes les {interval_seconds} secondes")
            with st.spinner("Analyse vidéo en cours..."):
                predict_video(file, interval_seconds)

            if "captured_frames" in st.session_state and len(st.session_state.captured_frames) > 0:
                st.markdown("---")
                st.subheader("📸 Frames capturées")

                cols = st.columns(4)
                for idx, (img, label, timestamp) in enumerate(st.session_state.captured_frames):
                    with cols[idx % 4]:
                        st.image(img, caption=f"{label} ({timestamp})", use_container_width=True)

else:
    st.info("➡️ Upload une image ou une vidéo pour commencer.")
