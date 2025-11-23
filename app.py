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

# CSS personnalisé pour un design moderne
st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
    
    /* Style général */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        font-family: 'Poppins', sans-serif;
    }
    
    /* Titre principal */
    .title-container {
        background: rgba(255, 255, 255, 0.95);
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        margin-bottom: 2rem;
        backdrop-filter: blur(10px);
    }
    
    .main-title {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3.5rem;
        font-weight: 700;
        text-align: center;
        margin: 0;
        line-height: 1.2;
    }
    
    .subtitle {
        color: #666;
        text-align: center;
        font-size: 1.2rem;
        margin-top: 0.5rem;
        font-weight: 300;
    }
    
    /* Cartes */
    .card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.07);
        margin: 1rem 0;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 15px rgba(0, 0, 0, 0.1);
    }
    
    /* Stats boxes */
    .stat-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    .stat-number {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
    }
    
    .stat-label {
        font-size: 0.9rem;
        opacity: 0.9;
        margin-top: 0.5rem;
    }
    
    /* Boutons personnalisés */
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        font-weight: 600;
        border-radius: 10px;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    
    /* Sidebar */
    .css-1d391kg {
        background: rgba(255, 255, 255, 0.95);
    }
    
    /* Upload zone */
    .uploadedFile {
        border: 2px dashed #667eea;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
    }
    
    /* Progress bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Detection card */
    .detection-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
        border-left: 4px solid #667eea;
    }
    
    /* Frame caption */
    .frame-caption {
        background: rgba(102, 126, 234, 0.1);
        padding: 0.5rem;
        border-radius: 8px;
        margin-top: 0.5rem;
        text-align: center;
        font-weight: 500;
    }
    
    /* Icons */
    .icon {
        font-size: 2rem;
        margin-bottom: 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)

# En-tête
st.markdown("""
    <div class="title-container">
        <h1 class="main-title">♻️ Smart Bin Detection</h1>
        <p class="subtitle">Intelligence Artificielle pour la détection et classification des poubelles</p>
    </div>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    try:
        if not os.path.exists("best2.pt"):
            st.error("❌ Modèle best2.pt introuvable")
            st.stop()
        return YOLO("best2.pt")
    except Exception as e:
        st.error(f"❌ Erreur de chargement: {e}")
        st.stop()

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

    # Contrôles vidéo modernes
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

    # Reset stats
    if "counts" not in st.session_state:
        st.session_state.counts = {"total": 0, "vide": 0, "pleine": 0}
    if "captured_frames" not in st.session_state:
        st.session_state.captured_frames = []
    if "video_paused" not in st.session_state:
        st.session_state.video_paused = False
    
    st.session_state.counts = {"total": 0, "vide": 0, "pleine": 0}
    st.session_state.captured_frames = []

    detect_every = int(fps * frame_interval)
    frame_id = 0
    last_detection = None
    analyzed_count = 0

    while True:
        # Gestion pause
        if st.session_state.video_paused:
            time.sleep(0.1)
            continue
        
        ret, frame = cap.read()
        if not ret:
            break

        timestamp = frame_id / fps
        minutes = int(timestamp // 60)
        seconds = int(timestamp % 60)

        # Détection toutes les X secondes
        if frame_id % detect_every == 0:
            last_detection = model(frame, conf=0.5)[0]
            analyzed_count += 1

            num_detections = len(last_detection.boxes)
            
            for box in last_detection.boxes:
                cls = int(box.cls[0])
                st.session_state.counts["total"] += 1
                
                if cls == 0:
                    st.session_state.counts["vide"] += 1
                    label = "poubelle_vide"
                else:
                    st.session_state.counts["pleine"] += 1
                    label = "poubelle_pleine"
            
            # Sauvegarder thumbnail
            if num_detections > 0:
                annotated = last_detection.plot()
                thumb = cv2.resize(annotated, (320, 200))
                thumb = cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB)
                st.session_state.captured_frames.append((thumb, f"{num_detections} détection(s)", f"{minutes:02d}:{seconds:02d}"))
            
            status_text.success(f"Frame {analyzed_count} | {minutes:02d}:{seconds:02d} | Détections: {num_detections}")

        # Dessin des boxes
        display_frame = frame.copy()
        
        if last_detection is not None:
            for box in last_detection.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                color = (0, 255, 0) if cls == 0 else (255, 0, 0)

                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 3)
                
                # Background pour le texte
                label_text = f"{model.names[cls]} {conf:.2f}"
                (text_width, text_height), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(display_frame, (x1, y1-25), (x1+text_width+10, y1), color, -1)
                cv2.putText(display_frame, label_text, (x1+5, y1-8), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Timestamp avec fond
        timestamp_text = f"{minutes:02d}:{seconds:02d}"
        cv2.rectangle(display_frame, (5, 5), (150, 45), (0, 0, 0), -1)
        cv2.putText(display_frame, timestamp_text, (10, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Convertir et afficher
        frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        stframe.image(frame_rgb, channels="RGB", use_container_width=True)

        # Mettre à jour stats
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
        
        # Délai selon la vitesse
        if playback_speed < 1.0:
            time.sleep(0.01 / playback_speed)

    cap.release()
    status_text.success(f"✅ Analyse terminée : {analyzed_count} frames analysées | Total détections: {st.session_state.counts['total']}")

# Sidebar
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

# Corps principal
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
    # Landing page
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
