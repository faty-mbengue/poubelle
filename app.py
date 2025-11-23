# app.py
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

# -------------------------
# CONFIG
# -------------------------
st.set_page_config(
    page_title="Détection Poubelles - YOLOv8",
    page_icon="🗑️",
    layout="wide"
)

# Chemin vers ton modèle (modifie si besoin)
MODEL_PATH = r"best2.pt"

# -------------------------
# CSS minimal (clair)
# -------------------------
st.markdown(
    """
    <style>
      .card { background: #f8f8f8; padding:14px; border-radius:10px; box-shadow: 0 6px 18px rgba(0,0,0,0.06); }
      .big-title { font-size: 24px; font-weight:700; color:#111; }
      .subtitle { color:#333; margin-top:6px; }
      .thumb-caption { font-size:13px; color:#111; }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------
# HEADER
# -------------------------
st.markdown(
    """
    <div class="card">
      <div class="big-title">🗑️ Détection de Poubelles (Pleines / Vides)</div>
      <div class="subtitle">Upload une image ou une vidéo — le modèle détecte la poubelle, la localise et prédit si elle est pleine ou vide.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# -------------------------
# LOAD MODEL
# -------------------------
@st.cache_resource
def load_model(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Modèle introuvable: {path}")
    return YOLO(path)

try:
    model = load_model(MODEL_PATH)
except Exception as e:
    st.error(f"Erreur chargement modèle : {e}")
    st.stop()

# -------------------------
# SESSION STATE INIT
# -------------------------
ses = st.session_state
if "video_path" not in ses: ses.video_path = None
if "cap" not in ses: ses.cap = None
if "playing" not in ses: ses.playing = False
if "frame_index" not in ses: ses.frame_index = 0
if "last_frame" not in ses: ses.last_frame = None
if "captured_frames" not in ses: ses.captured_frames = []  # list of tuples (bytes, label, idx)
if "counts" not in ses: ses.counts = {"total": 0, "pleine": 0, "vide": 0}
if "last_saved_index" not in ses: ses.last_saved_index = -1

# -------------------------
# HELPERS
# -------------------------
def read_frame_at(cap, idx):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
    ret, frame = cap.read()
    return ret, frame

def make_thumbnail_from_bgr(bgr, w=320, h=200):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
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
    frame_interval_sec = st.slider("Analyser 1 frame toutes les X secondes", 1, 10, 1)
    fps_display = st.slider("FPS d'affichage (simulé)", 1, 30, 8)
    conf_threshold = st.slider("Seuil de confiance", 0.05, 0.99, 0.4)
    capture_only_when_detection = st.checkbox("Capturer seulement si détection", value=True)
    st.markdown("---")
    st.write("Modèle chargé:")
    st.code(MODEL_PATH, language="text")
    st.markdown("---")
    st.markdown("ℹ️ Notes :\n- L'analyse effectue une prédiction **une fois** toutes les X secondes (paramètre ci-dessus).\n- Entre deux prédictions, l’affichage continue d’actualiser (animation).")

# -------------------------
# MAIN LAYOUT
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
                    img = Image.open(uploaded_file).convert("RGB")
                    results = model(img, conf=conf_threshold)[0]
                    annotated = results.plot()  # BGR
                    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                    st.image(annotated_rgb, caption="Résultat", use_container_width=True)

        elif ftype.startswith("video"):
            st.subheader("🎬 Vidéo")
            st.video(uploaded_file)  # native player for preview

            # Buttons row
            c1, c2, c3, c4 = st.columns([1,1,1,1])
            with c1:
                if st.button("▶ Play", key="play"):
                    # prepare if not prepared
                    if ses.video_path is None:
                        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                        tfile.write(uploaded_file.read())
                        tfile.flush()
                        ses.video_path = tfile.name
                        ses.cap = cv2.VideoCapture(ses.video_path)
                        ses.frame_index = 0
                        ses.last_frame = None
                        ses.captured_frames = []
                        ses.counts = {"total": 0, "pleine": 0, "vide": 0}
                        ses.last_saved_index = -1
                    ses.playing = True
            with c2:
                if st.button("⏸ Pause", key="pause"):
                    ses.playing = False
            with c3:
                if st.button("⏭️ Avancer d'une frame", key="step"):
                    # ensure video loaded
                    if ses.cap is None and ses.video_path is None:
                        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                        tfile.write(uploaded_file.read())
                        tfile.flush()
                        ses.video_path = tfile.name
                        ses.cap = cv2.VideoCapture(ses.video_path)
                    ses.playing = False
                    ses.frame_index += 1
            with c4:
                if st.button("⏮️ Reculer d'une frame", key="back"):
                    ses.playing = False
                    ses.frame_index = max(0, ses.frame_index - 1)

            # placeholder for the annotated video frame
            placeholder = st.empty()
            progress = st.progress(0)
            status = st.empty()

            # If video prepared, process frames
            if ses.video_path and ses.cap:
                cap = ses.cap
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                fps = cap.get(cv2.CAP_PROP_FPS) or fps_display

                # compute skip value
                skip = max(1, int(fps * frame_interval_sec))

                # if frame index beyond total, stop
                if ses.frame_index >= total_frames:
                    ses.playing = False
                    status.info("Lecture terminée.")
                else:
                    # If playing, advance frames in loop — but Streamlit requires re-run to update UI,
                    # so we display one frame per run and then call st.experimental_rerun to continue.
                    ret, frame = read_frame_at(cap, ses.frame_index)
                    if not ret:
                        status.warning("Impossible de lire la frame.")
                    else:
                        # run detection only on frames that are multiples of skip
                        do_detection = (ses.frame_index % skip == 0)

                        if do_detection:
                            results = model(frame, conf=conf_threshold)[0]
                            annotated = results.plot()
                            # update counts & capture thumbnails
                            nboxes = len(results.boxes)
                            timestamp_sec = int(ses.frame_index / fps)
                            mins = timestamp_sec // 60
                            secs = timestamp_sec % 60
                            tstr = f"{mins:02d}:{secs:02d}"

                            # update counts and maybe capture
                            if nboxes > 0:
                                # choose first box class for label
                                cls0 = int(results.boxes.cls[0])
                                label0 = model.names.get(cls0, str(cls0))
                                ses.counts["total"] += 1
                                if cls0 == 0:
                                    ses.counts["vide"] += 1
                                elif cls0 == 1:
                                    ses.counts["pleine"] += 1

                                # save thumbnail only if option allows
                                if not capture_only_when_detection or (capture_only_when_detection and nboxes > 0):
                                    thumb_bytes = make_thumbnail_from_bgr(annotated)
                                    ses.captured_frames.append((thumb_bytes, label0, ses.frame_index, tstr))

                            # draw timestamp overlay on annotated image
                            cv2.rectangle(annotated, (5,5), (200,45), (0,0,0), -1)
                            cv2.putText(annotated, f"t={tstr}", (10,32),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)

                            ses.last_frame = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                        else:
                            # if not detecting this frame, show last detection overlay (if exists) or plain frame
                            if ses.last_frame is not None:
                                ses.last_frame = ses.last_frame  # keep last annotated
                            else:
                                ses.last_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                        # show frame
                        placeholder.image(ses.last_frame, use_container_width=True)

                        # update progress & status
                        progress.progress(min(1.0, (ses.frame_index+1) / max(1, total_frames)))
                        status.info(f"Frame {ses.frame_index+1}/{total_frames} (analyze every {frame_interval_sec}s). Counts total: {ses.counts['total']}")

                        # increment frame index if playing; if paused it stays
                        if ses.playing:
                            ses.frame_index += 1
                        # else remain

                        # control display speed (simulate fps_display)
                        time.sleep(1.0 / float(fps_display))

                        # if still playing, re-run to continue loop (Streamlit will re-execute script)
                        if ses.playing:
                            st.experimental_rerun()

            else:
                st.info("Appuie sur ▶ Play pour préparer la vidéo et lancer l'analyse.")

# -------------------------
# RIGHT COLUMN (infos / stats)
# -------------------------
with col_info:
    st.markdown("<div class='card'><strong>ℹ️ Infos & Statistiques</strong></div>", unsafe_allow_html=True)
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.write("**Statistiques (captures vidéo)**")
    st.write(f"- Total détections: **{ses.counts['total']}**")
    st.write(f"- Poubelles vides: **{ses.counts['vide']}**")
    st.write(f"- Poubelles pleines: **{ses.counts['pleine']}**")
    st.markdown("---")
    st.write("Options sélectionnées:")
    st.write(f"- Intervalle d'analyse : {frame_interval_sec} s")
    st.write(f"- FPS affiché : {fps_display}")
    st.write(f"- Seuil confiance : {conf_threshold:.2f}")
    st.write(f"- Capture seulement si détection : {capture_only_when_detection}")
    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------
# THUMBNAILS (EN BAS) - UNIQUEMENT POUR LA VIDÉO
# -------------------------
st.markdown("---")
st.subheader("📸 Frames capturées (dernières)")

captures = ses.captured_frames[-24:] if ses.captured_frames else []

if len(captures) == 0:
    st.info("Aucune capture pour l'instant (lancer l'analyse sur une vidéo).")
else:
    cols = st.columns(4)
    for i, (img_bytes, label, idx, tstr) in enumerate(reversed(captures)):
        col = cols[i % 4]
        with col:
            st.image(img_bytes, caption=f"{label} — frame {idx} @ {tstr}", use_container_width=True)
            dlname = f"frame_{idx}_{label}.jpg"
            st.download_button(f"Télécharger {i+1}", data=img_bytes, file_name=dlname, mime="image/jpeg")

    # bouton pour télécharger toutes les captures en zip
    zip_buf = save_captures_as_zip(ses.captured_frames)
    st.download_button("📥 Télécharger toutes les captures (ZIP)", data=zip_buf, file_name="captures.zip", mime="application/zip")

# Footer
st.markdown("<hr/>")
st.caption("Développé par Fatou Mbengue — YOLOv8 • LabelImg • Streamlit")

