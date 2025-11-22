import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import os
import tempfile

st.set_page_config(page_title="Détection Poubelles", layout="centered")

# Charger le modèle YOLO (ton modèle final)
model = YOLO("best.pt")   # <-- remplace par ton modèle entraîné

st.title("♻️ Détection Poubelle (pleine/vide) avec YOLOv8")
st.write("Upload une image pour obtenir la prédiction.")


uploaded_file = st.file_uploader("Choisir une image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Afficher l'image uploadée
    img = Image.open(uploaded_file)
    st.image(img, caption="Image uploadée", use_column_width=True)

    # Sauvegarder temporairement l’image
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp_path = tmp.name
        img.save(tmp_path)

    # Prédiction
    st.write("🔍 **Détection en cours...**")
    results = model(tmp_path)

    # Afficher le résultat YOLO
    result_img = results[0].plot()

    st.image(result_img, caption="Résultat YOLO", use_column_width=True)

    # Récupérer les infos textuelles
    names = model.names
    dets = results[0].boxes

    if len(dets) == 0:
        st.warning("Aucun objet détecté.")
    else:
        st.subheader("📌 Résultats")
        for box in dets:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            st.write(f"- **Classe :** {names[cls]} — **Confiance :** {conf:.2f}")
