import streamlit as st
from ultralytics import YOLO
import tempfile
from PIL import Image
import cv2
import numpy as np
import time
import io
import zipfile
import os

st.set_page_config(
    page_title="Smart Bin Detection",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# En-tête avec CSS personnalisé (à compléter selon vos styles)
st.markdown("""
    <div class="title-container">
        <h1 class="main-title">♻️ Smart Bin Detection</h1>
        <p class="subtitle">Intelligence Artificielle pour la détection et classification des poubelles</p>
    </div>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    if not os.path.exists("best2.pt"):
        st.error("❌ Modèle best2.pt introuvable")
        st.stop()
    return YOLO("best2.pt")

with st.spinner("Chargement du modèle..."):
    model = load_model()
    st.success("Modèle chargé avec succès")

# Initialisation session state
for key, default in {
    "video_path": None,
    "cap": None,
    "paused": True,
    "frame_index": 0,
    "last_frame": None,
    "captured_frames": [],
    "counts": {"total": 0, "pleine": 0, "vide": 0},
    "last_saved_index": -1
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

def read_frame_at(cap, idx):
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, frame = cap.read()
    return ret, frame

def make_thumbnail(img_rgb, w=320, h=200):
    pil = Image.fromarray(img_rgb)
    pil.thumbnail((w, h))
    buf = io.BytesIO()
    pil.save(buf, format="JPEG")
    buf.seek(0)
    return buf.getvalue()

def save_captures_as_zip(captures):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for i, (img_bytes, label, idx) in enumerate(captures):
            filename = f"frame_{idx}_{label}_{i}.jpg"
            z.writestr(filename, img_bytes)
    buf.seek(0)
    return buf

col_video, col_info = st.columns([3,1])

with col_video:
    uploaded_file = st.file_uploader(
        "Choisis une image ou une vidéo",
        type=["jpg", "jpeg", "png", "mp4", "avi"]
    )

    if uploaded_file is None:
        st.info("➡️ Charge une image ou une vidéo depuis la barre latérale.")
    else:
        ftype = uploaded_file.type
        if ftype.startswith("image"):
            st.subheader("🖼️ Image")
            st.image(uploaded_file, use_container_width=True)

            if st.button("🚀 Lancer la détection (Image)"):
                with st.spinner("Analyse en cours..."):
                    results = model(Image.open(uploaded_file).convert("RGB"))[0]
                    annotated = results.plot()
                    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                    st.image(annotated_rgb, caption="Résultat", use_container_width=True)

        elif ftype.startswith("video"):
            st.subheader("🎬 Vidéo")
            st.video(uploaded_file)

            if st.button("▶️ Lancer la détection (Vidéo)"):
                st.session_state.paused = False
                st.session_state.frame_index = 0
                st.session_state.last_frame = None
                st.session_state.captured_frames = []
                st.session_state.counts = {"total": 0, "pleine": 0, "vide": 0}
                st.session_state.last_saved_index = -1

                tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                tfile.write(uploaded_file.read())
                tfile.flush()
                st.session_state.video_path = tfile.name
                st.session_state.cap = cv2.VideoCapture(st.session_state.video_path)

            c1, c2, c3 = st.columns([1,1,1])
            with c1:
                if st.button("⏸️ Pause"):
                    st.session_state.paused = True
            with c2:
                if st.button("▶️ Reprendre"):
                    st.session_state.paused = False
            with c3:
                if st.button("⏭️ Avancer d'une frame"):
                    st.session_state.paused = True
                    st.session_state.frame_index += 1

            placeholder = st.empty()
            cap = st.session_state.cap

            if cap is not None:
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                if st.session_state.frame_index >= total_frames:
                    st.success("Lecture terminée.")
                else:
                    if not st.session_state.paused:
                        ret, frame = read_frame_at(cap, st.session_state.frame_index)
                        if not ret:
                            st.warning("Impossible de lire la frame.")
                        else:
                            results = model(frame, conf=0.5)[0]
                            annotated = results.plot()
                            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                            placeholder.image(annotated_rgb, use_container_width=True)

                            has_detection = len(results.boxes) > 0
                            idx = st.session_state.frame_index

                            if has_detection and idx != st.session_state.last_saved_index:
                                thumb_bytes = make_thumbnail(annotated_rgb)
                                cls_id = int(results.boxes.cls[0])
                                label = model.names[cls_id]

                                st.session_state.captured_frames.append((thumb_bytes, label, idx))
                                st.session_state.last_saved_index = idx

                                st.session_state.counts["total"] += 1
                                if cls_id == 0:
                                    st.session_state.counts["vide"] += 1
                                elif cls_id == 1:
                                    st.session_state.counts["pleine"] += 1

                            st.session_state.last_frame = annotated_rgb
                            st.session_state.frame_index += 1

                            time.sleep(0.1)
                            st.experimental_rerun()
                    else:
                        if st.session_state.last_frame is not None:
                            placeholder.image(st.session_state.last_frame, use_container_width=True)
                        else:
                            st.info("Vidéo prête. Cliquez sur ▶️ Reprendre pour lancer la détection.")

with col_info:
    st.markdown("<div class='card'><strong>ℹ️ Infos</strong></div>", unsafe_allow_html=True)
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.write("**Statistiques (captures vidéo)**")
    st.write(f"- Total détections: **{st.session_state.counts['total']}**")
    st.write(f"- Poubelles vides: **{st.session_state.counts['vide']}**")
    st.write(f"- Poubelles pleines: **{st.session_state.counts['pleine']}**")
    st.markdown("---")
    st.write("Options:")
    st.write(f"- Seuil confidence: 0.5")
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")
st.subheader("📸 Frames capturées (dernières)")

captures = st.session_state.captured_frames[-24:] if st.session_state.captured_frames else []

if len(captures) == 0:
    st.info("Aucune capture pour l'instant (lancer la détection sur une vidéo).")
else:
    cols = st.columns(4)
    for i, (img_bytes, label, idx) in enumerate(reversed(captures)):
        col = cols[i % 4]
        with col:
            st.image(img_bytes, caption=f"{label} — frame {idx}", use_container_width=True)
            dlname = f"frame_{idx}_{label}.jpg"
            st.download_button(f"Télécharger {i+1}", data=img_bytes, file_name=dlname, mime="image/jpeg")

zip_buf = save_captures_as_zip(st.session_state.captured_frames)
st.download_button("📥 Télécharger toutes les captures (ZIP)", data=zip_buf, file_name="captures.zip", mime="application/zip")

st.markdown("<hr style='border:1px solid rgba(255,255,255,0.04)'/>", unsafe_allow_html=True)
st.caption("Développé par Faty Mbengue — YOLOv8 • LabelImg • Streamlit")
