# app.py
import streamlit as st
from ultralytics import YOLO
import tempfile
import cv2
from PIL import Image
import numpy as np
import time
import io
import zipfile
import os

# -------------------------
# CONFIG
# -------------------------
st.set_page_config(
    page_title="Détection Poubelles - YOLOv8",
    page_icon="🗑️",
    layout="wide"
)

# Chemin par défaut vers ton modèle (modifie si nécessaire)
MODEL_PATH = r"best2.pt"

# -------------------------
# CSS & THEME (sombre + animations)
# -------------------------
st.markdown(
    """
    <style>
    /* BACKGROUND & FONTS */
    .stApp {
        background: #ffffff !important;
        color: #000000 !important;
        font-family: "Segoe UI", Roboto, "Helvetica Neue", Arial;
    }

    /* CARD */
    .card {
        background: #f5f5f5;
        border-radius: 12px;
        padding: 14px;
        box-shadow: 0 4px 16px rgba(0,0,0,0.08);
        border: 1px solid #dddddd;
        margin-bottom: 12px;
    }

    /* TITRES */
    .big-title {
        font-size: 28px;
        color: #000000;
        font-weight: 700;
        margin-bottom: 6px;
    }
    .subtitle {
        color: #333333;
        margin-top: -6px;
        margin-bottom: 12px;
    }

    /* BUTTONS */
    button[kind="primary"] {
        background: #000000 !important;
        color: white !important;
        border-radius: 8px;
        padding: 8px 16px;
    }
    button[kind="primary"]:hover {
        background: #333333 !important;
        transform: translateY(-1px);
    }

    /* Animations */
    @keyframes glow {
      0% { box-shadow: 0 0 6px rgba(0,0,0,0.06); }
      50% { box-shadow: 0 0 12px rgba(0,0,0,0.12); }
      100% { box-shadow: 0 0 6px rgba(0,0,0,0.06); }
    }

    .glow {
      animation: glow 2.4s ease-in-out infinite;
      border-radius: 10px;
    }

    /* Thumbnails */
    .thumb-caption {
      color: #000000;
      font-size: 13px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------
# HEADER
# -------------------------
st.markdown(
    """
    <div class="card glow">
      <div class="big-title">🗑️ Détection de Poubelles (Pleines / Vides)</div>
      <div class="subtitle">Upload une image ou une vidéo — le modèle détecte la poubelle, la localise et prédit si elle est pleine ou vide.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# -------------------------
# LOAD MODEL (caché)
# -------------------------
@st.cache_resource
def load_model(path):
    return YOLO(path)

try:
    model = load_model(MODEL_PATH)
except Exception as e:
    st.error(f"Erreur chargement modèle : {e}")
    st.stop()

# -------------------------
# SESSION STATE INIT
# -------------------------
if "video_path" not in st.session_state:
    st.session_state.video_path = None
if "cap" not in st.session_state:
    st.session_state.cap = None
if "paused" not in st.session_state:
    st.session_state.paused = True
if "frame_index" not in st.session_state:
    st.session_state.frame_index = 0
if "last_frame" not in st.session_state:
    st.session_state.last_frame = None
if "captured_frames" not in st.session_state:
    st.session_state.captured_frames = []  # list of tuples (img_bytes, label, idx)
if "counts" not in st.session_state:
    st.session_state.counts = {"total": 0, "pleine": 0, "vide": 0}
if "last_saved_index" not in st.session_state:
    st.session_state.last_saved_index = -1

# -------------------------
# SIDEBAR: upload + options
# -------------------------
with st.sidebar:
    st.header("📤 Upload & Options")
    uploaded_file = st.file_uploader(
        "Choisis une image ou une vidéo",
        type=["jpg", "jpeg", "png", "mp4", "avi"]
    )
    st.markdown("---")
    st.subheader("⚙️ Réglages")
    fps_display = st.slider("Vitesse d'affichage (FPS simulé)", 1, 30, 8)
    conf_threshold = st.slider("Seuil de confiance", 0.1, 0.99, 0.5)
    capture_only_when_detection = st.checkbox(
        "Capturer seulement les frames qui contiennent une détection", value=True
    )
    st.markdown("---")
    st.write("Modèle chargé:")
    st.code(MODEL_PATH, language="text")

# -------------------------
# HELPERS
# -------------------------
def read_frame_at(cap, idx):
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, frame = cap.read()
    return ret, frame

def pil_from_cv2(frame_bgr):
    img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb)

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

# -------------------------
# LAYOUT PRINCIPAL
# -------------------------
col_video, col_info = st.columns([3, 1])

with col_video:
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
            # show native video player
            st.video(uploaded_file)

            # prepare capture file only once
            if st.button("▶️ Lancer la détection (Vidéo)"):
                # reset states
                st.session_state.paused = False
                st.session_state.frame_index = 0
                st.session_state.last_frame = None
                st.session_state.captured_frames = []
                st.session_state.counts = {"total": 0, "pleine": 0, "vide": 0}
                st.session_state.last_saved_index = -1

                # write upload to temp file and open VideoCapture
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                tfile.write(uploaded_file.read())
                tfile.flush()
                st.session_state.video_path = tfile.name
                st.session_state.cap = cv2.VideoCapture(st.session_state.video_path)

            # controls
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

            # process one frame if video loaded
            placeholder = st.empty()
            if st.session_state.video_path and st.session_state.cap:
                cap = st.session_state.cap

                # Bound checks
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                if st.session_state.frame_index >= total_frames:
                    st.success("Lecture terminée.")
                else:
                    # If playing (not paused), advance frame by one and re-render loop with sleep
                    if not st.session_state.paused:
                        ret, frame = read_frame_at(cap, st.session_state.frame_index)
                        if not ret:
                            st.warning("Impossible de lire la frame.")
                        else:
                            # predict
                            results = model(frame, conf=conf_threshold)[0]
                            annotated = results.plot()
                            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                            placeholder.image(annotated_rgb, use_container_width=True)

                            # capture logic: only when detection OR always (depending option)
                            has_detection = len(results.boxes) > 0
                            idx = st.session_state.frame_index

                            should_capture = True
                            if capture_only_when_detection:
                                should_capture = has_detection

                            # avoid duplicate captures for same frame index
                            if should_capture and idx != st.session_state.last_saved_index:
                                # prepare thumbnail bytes and save label
                                thumb_bytes = make_thumbnail(annotated_rgb)
                                # choose label: if multiple boxes take the first class
                                if has_detection:
                                    cls_id = int(results.boxes.cls[0])
                                    label = model.names[cls_id]
                                else:
                                    label = "no_object"

                                st.session_state.captured_frames.append((thumb_bytes, label, idx))
                                st.session_state.last_saved_index = idx

                                # update counts & history
                                if has_detection:
                                    st.session_state.counts["total"] += 1
                                    if cls_id == 0:
                                        st.session_state.counts["vide"] += 1
                                    elif cls_id == 1:
                                        st.session_state.counts["pleine"] += 1

                            st.session_state.last_frame = annotated_rgb
                            st.session_state.frame_index += 1

                            # small sleep to control display speed
                            time.sleep(1.0 / float(fps_display))

                            # rerun to update UI (safe way to create playback)
                            st.rerun()


                    else:
                        # paused: show last frame or a placeholder
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
    st.write(f"- FPS simulé: {fps_display}")
    st.write(f"- Seuil confidence: {conf_threshold:.2f}")
    st.write(f"- Capture seulement si détection: {capture_only_when_detection}")
    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------
# THUMBNAILS (EN BAS) - UNIQUEMENT POUR LA VIDÉO
# -------------------------
st.markdown("---")
st.subheader("📸 Frames capturées (dernières)")

captures = st.session_state.captured_frames[-24:] if st.session_state.captured_frames else []

if len(captures) == 0:
    st.info("Aucune capture pour l'instant (lancer la détection sur une vidéo).")
else:
    # grid 4 colonnes
    cols = st.columns(4)
    for i, (img_bytes, label, idx) in enumerate(reversed(captures)):
        col = cols[i % 4]
        with col:
            st.image(img_bytes, caption=f"{label} — frame {idx}", use_container_width=True)
            # petit bouton pour télécharger l'image
            dlname = f"frame_{idx}_{label}.jpg"
            st.download_button(f"Télécharger {i+1}", data=img_bytes, file_name=dlname, mime="image/jpeg")

    # bouton pour télécharger toutes les captures en zip
    zip_buf = save_captures_as_zip(st.session_state.captured_frames)
    st.download_button("📥 Télécharger toutes les captures (ZIP)", data=zip_buf, file_name="captures.zip", mime="application/zip")

# -------------------------
# FIN
# -------------------------
st.markdown("<hr style='border:1px solid rgba(255,255,255,0.04)'/>", unsafe_allow_html=True)
st.caption("Développé par Fatou Mbengue — YOLOv8 • LabelImg • Streamlit")
