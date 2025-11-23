import streamlit as st
from ultralytics import YOLO
import tempfile
from PIL import Image
import cv2
import numpy as np
import os
import time

st.set_page_config(
    page_title="Smart Bin Detection",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé pour un design moderne (votre CSS ici)

# En-tête
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

def predict_image(upload):
    img = Image.open(upload).convert("RGB")
    results = model(img, conf=0.5)[0]
    return results

def predict_video_live(upload, frame_interval=1, stats_placeholder=None):
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(upload.read())
    tfile.close()

    cap = cv2.VideoCapture(tfile.name)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    col1, col2 = st.columns(2)
    with col1:
        st.info(f"Vidéo : {total_frames} frames @ {fps:.1f} FPS")
    with col2:
        st.info(f"Prédiction : 1 frame / {frame_interval}s")

    control_col1, control_col2, control_col3 = st.columns([1, 1, 3])
    
    with control_col1:
        if st.button("▶️ Play", use_container_width=True, key="play_btn"):
            st.session_state.video_paused = False
    
    with control_col2:
        if st.button("⏸️ Pause", use_container_width=True, key="pause_btn"):
            st.session_state.video_paused = True
    
    with control_col3:
        speed_options = {"0.5x": 0.5, "1x": 1.0, "2x": 2.0}
        selected_speed = st.select_slider(
            "Vitesse",
            options=list(speed_options.keys()),
            value="1x"
        )
        playback_speed = speed_options[selected_speed]

    stframe = st.empty()
    progress_bar = st.progress(0)
    status_text = st.empty()

    if "counts" not in st.session_state:
        st.session_state.counts = {"total": 0, "vide": 0, "pleine": 0}
    if "video_paused" not in st.session_state:
        st.session_state.video_paused = False

    st.session_state.counts = {"total": 0, "vide": 0, "pleine": 0}

    detect_every = int(fps * frame_interval)
    frame_id = 0
    last_detection = None
    analyzed_count = 0

    while True:
        if st.session_state.video_paused:
            time.sleep(0.1)
            continue
        
        ret, frame = cap.read()
        if not ret:
            break

        timestamp = frame_id / fps
        minutes = int(timestamp // 60)
        seconds = int(timestamp % 60)

        if frame_id % detect_every == 0:
            last_detection = model(frame, conf=0.5)[0]
            analyzed_count += 1

            num_detections = len(last_detection.boxes)
            
            for box in last_detection.boxes:
                cls = int(box.cls[0])
                st.session_state.counts["total"] += 1
                
                if cls == 0:
                    st.session_state.counts["vide"] += 1
                else:
                    st.session_state.counts["pleine"] += 1
            
            if num_detections > 0:
                annotated = last_detection.plot()
                thumb = cv2.resize(annotated, (320, 200))
                thumb = cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB)
                st.session_state.captured_frames.append((thumb, f"{num_detections} détection(s)", f"{minutes:02d}:{seconds:02d}"))
            
            status_text.success(f"Frame {analyzed_count} | {minutes:02d}:{seconds:02d} | Détections: {num_detections}")

        display_frame = frame.copy()
        
        if last_detection is not None:
            for box in last_detection.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                color = (0, 255, 0) if cls == 0 else (255, 0, 0)

                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 3)
                
                label_text = f"{model.names[cls]} {conf:.2f}"
                (text_width, text_height), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(display_frame, (x1, y1-25), (x1+text_width+10, y1), color, -1)
                cv2.putText(display_frame, label_text, (x1+5, y1-8), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        timestamp_text = f"{minutes:02d}:{seconds:02d}"
        cv2.rectangle(display_frame, (5, 5), (150, 45), (0, 0, 0), -1)
        cv2.putText(display_frame, timestamp_text, (10, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        stframe.image(frame_rgb, channels="RGB", use_container_width=True)

        if stats_placeholder and frame_id % detect_every == 0:
            with stats_placeholder.container():
                st.markdown(f"""
                    <div class="stat-box">
                        <div class="stat-number">{st.session_state.counts['total']}</div>
                        <div class="stat-label">Détections totales</div>
                    </div>
                """, unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Vides", st.session_state.counts["vide"])
                with col2:
                    st.metric("Pleines", st.session_state.counts["pleine"])

        frame_id += 1
        progress_bar.progress(frame_id / total_frames)
        
        if playback_speed < 1.0:
            time.sleep(0.01 / playback_speed)

    cap.release()
    status_text.success(f"✅ Analyse terminée : {analyzed_count} frames analysées | Total détections: {st.session_state.counts['total']}")

# Sidebar et upload de fichiers
with st.sidebar:
    st.markdown("### 📤 Upload")
    file = st.file_uploader(
        "Glissez votre fichier ici",
        type=["jpg", "jpeg", "png", "mp4", "avi"],
        help="Formats supportés: JPG, PNG, MP4, AVI"
    )
    
    if file and file.type.startswith("video"):
        st.markdown("---")
        st.markdown("### ⚙️ Configuration")
        frame_interval = st.slider(
            "Intervalle d'analyse (secondes)",
            min_value=1,
            max_value=60,
            value=1,
            help="Plus l'intervalle est court, plus l'analyse est précise mais lente"
        )
    else:
        frame_interval = 1
    
    st.markdown("---")
    st.markdown("### 📊 Statistiques en temps réel")
    
    stats_placeholder = st.empty()
    
    with stats_placeholder.container():
        if "counts" in st.session_state and st.session_state.counts["total"] > 0:
            st.markdown(f"""
                <div class="stat-box">
                    <div class="stat-number">{st.session_state.counts['total']}</div>
                    <div class="stat-label">Détections totales</div>
                </div>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Vides", st.session_state.counts['vide'])
            with col2:
                st.metric("Pleines", st.session_state.counts['pleine'])
        else:
            st.info("Aucune détection pour le moment")
    
    st.markdown("---")
    st.markdown("### ℹ️ Performances du modèle")
    st.markdown("""
        **mAP50:** 85.7%  
        **mAP50-95:** 43.5%  
        **Précision:** 85.9%  
        **Recall:** 79.4%  
        **Version:** YOLOv11n
    """)

if file:
    ftype = file.type

    if ftype.startswith("image"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📷 Image originale")
            st.image(file, use_container_width=True)

        with col2:
            st.markdown("### 🎯 Zone de détection")
            if st.button("🚀 Lancer l'analyse", use_container_width=True):
                with st.spinner("🔍 Analyse en cours..."):
                    results = predict_image(file)
                    annotated = results.plot()
                    
                    st.image(annotated, channels="RGB", use_container_width=True)
                    
                    boxes = results.boxes
                    if len(boxes) > 0:
                        st.success(f"{len(boxes)} poubelle(s) détectée(s)")
        
        if 'results' in locals():
            boxes = results.boxes
            if len(boxes) > 0:
                st.markdown("---")
                st.markdown("### 📋 Détails des détections")
                
                for i, box in enumerate(boxes):
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    classe = model.names[cls]
                    
                    icon = "🟢" if cls == 0 else "🔴"
                    color = "#10b981" if cls == 0 else "#ef4444"
                    
                    st.markdown(f"""
                        <div class="detection-card" style="border-left-color: {color}">
                            <h4>{icon} Détection {i+1}: {classe}</h4>
                            <p style="margin: 0.5rem 0;">
                                <strong>Confiance:</strong> {conf:.2%}
                            </p>
                            <div style="background: {color}20; padding: 0.5rem; border-radius: 5px; margin-top: 0.5rem;">
                                <div style="background: {color}; width: {conf*100}%; height: 8px; border-radius: 4px;"></div>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)

    elif ftype.startswith("video"):
        st.markdown("### 🎬 Vidéo uploadée")
        st.video(file)

        if st.button("🚀 Lancer l'analyse vidéo", use_container_width=True):
            st.warning(f"La vidéo sera analysée toutes les {frame_interval} seconde(s)")
            
            with st.spinner("Analyse de la vidéo en cours..."):
                predict_video_live(file, frame_interval, stats_placeholder)
            if "captured_frames" in st.session_state and len(st.session_state.captured_frames) > 0:
                st.markdown("---")
                st.markdown("### 📸 Frames capturées")
                
                cols = st.columns(4)
                for idx, (img, label, timestamp) in enumerate(st.session_state.captured_frames):
                    with cols[idx % 4]:
                        st.image(img, use_container_width=True)
                        st.markdown(f"""
                            <div class="frame-caption">
                                {label}<br>
                                {timestamp}
                            </div>
                        """, unsafe_allow_html=True)

else:
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
            <div class="card">
                <div class="icon">📷</div>
                <h3>Images</h3>
                <p>Détection instantanée sur vos photos</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class="card">
                <div class="icon">🎥</div>
                <h3>Vidéos</h3>
                <p>Analyse frame par frame de vos vidéos</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
            <div class="card">
                <div class="icon">📊</div>
                <h3>Statistiques</h3>
                <p>Rapports détaillés en temps réel</p>
            </div>
        """, unsafe_allow_html=True)
        
    st.info("👆 Uploadez une image ou une vidéo dans la barre latérale pour commencer")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: white; padding: 2rem;'>
        <p style='font-size: 0.9rem;'>
            Développé par <strong>Faty Mbengue</strong> | Propulsé par YOLOv11 & Streamlit
        </p>
    </div>
""", unsafe_allow_html=True)
