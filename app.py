import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import tempfile
import cv2

st.set_page_config(
    page_title="Détection Poubelles",
    page_icon="♻️",
    layout="wide"
)

st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .stApp {
        background-color: transparent;
    }
    h1 {
        color: white;
        text-align: center;
        font-size: 3rem;
        margin-bottom: 0;
    }
    .subtitle {
        color: white;
        text-align: center;
        font-size: 1.2rem;
        margin-top: 0;
        margin-bottom: 2rem;
    }
    .stat-box {
        background: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
    .detection-card {
        background: white;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1>♻️ Détection Poubelles</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Intelligence artificielle pour détecter si une poubelle est pleine ou vide</p>", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    st.markdown("### 📤 Charger une image")
    uploaded_file = st.file_uploader(
        "",
        type=["jpg", "jpeg", "png"],
        help="Formats acceptés: JPG, JPEG, PNG"
    )
    
    confidence_threshold = st.slider(
        "🎯 Seuil de confiance",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05
    )

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.markdown("### 📷 Image originale")
        st.image(img, use_column_width=True)
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        img.save(tmp.name)
        temp_path = tmp.name
    
    with st.spinner("🔍 Analyse en cours..."):
        results = model.predict(
            source=temp_path,
            conf=confidence_threshold,
            imgsz=640
        )
    
    result_img = results[0].plot()
    result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
    
    with col_right:
        st.markdown("### 🎯 Détections")
        st.image(result_img_rgb, use_column_width=True)
    
    boxes = results[0].boxes
    names = model.names
    
    if len(boxes) > 0:
        st.markdown("---")
        st.markdown("### 📊 Résultats détaillés")
        
        stats_col1, stats_col2, stats_col3 = st.columns(3)
        
        with stats_col1:
            st.markdown(f"""
            <div class='stat-box'>
                <h2 style='color: #667eea; margin: 0;'>{len(boxes)}</h2>
                <p style='margin: 0; color: #666;'>Détections</p>
            </div>
            """, unsafe_allow_html=True)
        
        vide_count = sum(1 for box in boxes if int(box.cls[0]) == 0)
        pleine_count = sum(1 for box in boxes if int(box.cls[0]) == 1)
        
        with stats_col2:
            st.markdown(f"""
            <div class='stat-box'>
                <h2 style='color: #10b981; margin: 0;'>{vide_count}</h2>
                <p style='margin: 0; color: #666;'>Poubelles vides</p>
            </div>
            """, unsafe_allow_html=True)
        
        with stats_col3:
            st.markdown(f"""
            <div class='stat-box'>
                <h2 style='color: #ef4444; margin: 0;'>{pleine_count}</h2>
                <p style='margin: 0; color: #666;'>Poubelles pleines</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("### 🔍 Détails des détections")
        
        for i, box in enumerate(boxes):
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            classe = names[cls]
            
            if classe == "poubelle_vide":
                icon = "✅"
                color = "#10b981"
            else:
                icon = "⚠️"
                color = "#ef4444"
            
            conf_percentage = conf * 100
            
            st.markdown(f"""
            <div class='detection-card'>
                <h3 style='margin: 0; color: {color};'>{icon} Détection {i+1}</h3>
                <p style='margin: 5px 0;'><strong>Type:</strong> {classe}</p>
                <div style='background: #f0f0f0; border-radius: 5px; height: 20px; overflow: hidden;'>
                    <div style='background: {color}; width: {conf_percentage}%; height: 100%; display: flex; align-items: center; justify-content: center; color: white; font-size: 12px;'>
                        {conf:.2%}
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.warning("⚠️ Aucune détection trouvée. Essayez de réduire le seuil de confiance.")
else:
    st.info("👆 Chargez une image pour commencer l'analyse")

st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: white; padding: 20px;'>
        <p>Développé avec ❤️ using YOLO11 & Streamlit</p>
    </div>
""", unsafe_allow_html=True)
