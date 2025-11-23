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

def predict_video(upload, frame_interval=1, stats_placeholder=None, show_all_frames=False):
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(upload.read())
    tfile.close()
    
    cap = cv2.VideoCapture(tfile.name)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if "counts" not in st.session_state:
        st.session_state.counts = {"total": 0, "vide": 0, "pleine": 0}
    if "captured_frames" not in st.session_state:
        st.session_state.captured_frames = []
    
    st.session_state.counts = {"total": 0, "vide": 0, "pleine": 0}
    st.session_state.captured_frames = []
    
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"Vidéo : {total_frames} frames @ {fps:.1f} FPS")
    with col2:
        st.info(f"Prédiction : 1 frame / {frame_interval}s")
    
    stframe = st.empty()
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    frame_count = 0
    analyzed_count = 0
    frame_skip = int(fps * frame_interval)
    last_prediction = None
    
    while True:
        success, frame = cap.read()
        if not success:
            break
        
        timestamp = frame_count / fps
        minutes = int(timestamp // 60)
        seconds = int(timestamp % 60)
        
        # Faire une prédiction toutes les X secondes
        if frame_count % frame_skip == 0:
            results = model(frame, conf=0.5)[0]
            last_prediction = results
            
            num_detections = len(results.boxes)
            
            for box in results.boxes:
                cls = int(box.cls[0])
                st.session_state.counts["total"] += 1
                
                if cls == 0:
                    st.session_state.counts["vide"] += 1
                    label = "poubelle_vide"
                elif cls == 1:
                    st.session_state.counts["pleine"] += 1
                    label = "poubelle_pleine"
            
            analyzed_count += 1
            status_text.success(f"Frame {analyzed_count} | {minutes:02d}:{seconds:02d} | Détections: {num_detections}")
            
            # Mettre à jour les stats
            if stats_placeholder:
                with stats_placeholder.container():
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
            
            # Sauvegarder thumbnail
            if num_detections > 0:
                annotated = results.plot()
                thumb = cv2.resize(annotated, (320, 200))
                thumb = cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB)
                st.session_state.captured_frames.append((thumb, f"{num_detections} détection(s)", f"{minutes:02d}:{seconds:02d}"))
        
        # AFFICHER TOUTES LES FRAMES
        display_frame = frame.copy()
        
        # Si on a une prédiction récente, dessiner les boxes
        if last_prediction is not None:
            for box in last_prediction.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                
                # Couleur selon la classe
                color = (0, 255, 0) if cls == 0 else (255, 0, 0)
                label_text = f"{model.names[cls]} {conf:.2f}"
                
                # Dessiner la box
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(display_frame, label_text, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Ajouter timestamp
        cv2.putText(
            display_frame,
            f"Temps: {minutes:02d}:{seconds:02d}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )
        
        # Convertir BGR to RGB
        display_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        
        # AFFICHER LA FRAME
        stframe.image(display_frame, channels="RGB", use_container_width=True)
        
        # Petit délai pour que l'animation soit visible
        time.sleep(1/min(fps, 30))  # Max 30 FPS pour l'affichage
        
        frame_count += 1
        progress = frame_count / total_frames
        progress_bar.progress(progress)
    
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
        
        show_all_frames = st.checkbox(
            "Afficher toutes les frames (animation fluide)",
            value=False,
            help="Affiche toutes les frames entre les analyses pour une animation fluide"
        )
    else:
        frame_interval = 1
        show_all_frames = False
    
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
                predict_video(file, frame_interval, stats_placeholder, show_all_frames)
            
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
