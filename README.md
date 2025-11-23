st.markdown("""
<style>

.doc-card {
    background: #161b22;
    border: 1px solid #30363d;
    padding: 25px;
    border-radius: 18px;
    margin-bottom: 25px;
    box-shadow: 0 0 20px rgba(0,0,0,0.4);
    animation: fadeIn 1s ease-in-out;
    color: #e6edf3;
    font-size: 17px;
}

.doc-card h1, .doc-card h2, .doc-card h3 {
    color: #58a6ff !important;
    font-family: 'Segoe UI';
}

.doc-card ul {
    line-height: 1.7;
}

.doc-title {
    font-size: 40px;
    text-align: center;
    color: #58a6ff;
    font-weight: bold;
    margin-bottom: 30px;
    text-shadow: 0 0 25px #0d6efd;
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}

</style>

<div class="doc-card">

<div class="doc-title">🗑️ Détection & Classification de Poubelles (Pleine / Vide)</div>

<h2>📌 Reconnaissance d’objets avec YOLOv8 + Application Web Streamlit</h2>

<h2>📖 1. Description du Projet</h2>
<p>
Ce projet a pour objectif de développer un système complet de détection d’objets permettant :
<ul>
<li>de détecter automatiquement une poubelle dans une image,</li>
<li>de localiser la poubelle via une bounding box,</li>
<li>de déterminer si elle est pleine ou vide.</li>
</ul>
L’utilisateur peut envoyer une image via l’application Streamlit, et le modèle renvoie une image annotée avec la prédiction finale.
</p>

<h2>🤖 2. Modèle d’IA utilisé : YOLOv8</h2>
<p>
YOLOv8 est un modèle récent et performant pour la détection d’objets.  
Il effectue :
<ul>
<li><b>Détection</b> – identifier l’objet,</li>
<li><b>Localisation</b> – tracer une bounding box,</li>
<li><b>Classification</b> – prédire si la poubelle est pleine ou vide.</li>
</ul>
YOLOv8 utilise :
<ul>
<li><b>IoU</b> (Intersection over Union) pour vérifier la précision des boxes,</li>
<li><b>NMS</b> (Non-Maximum Suppression) pour filtrer les meilleures détections.</li>
</ul>
</p>

<h2>📝 3. Annotation des images (LabelImg)</h2>
<p>
Les images ont été annotées avec <b>LabelImg</b> :
<ul>
<li>tracé manuel des bounding boxes,</li>
<li>attribution d’une classe (poubelle_vide / poubelle_pleine),</li>
<li>export automatique en format YOLO : (cls, bx, by, bw, bh).</li>
</ul>
Cela garantit un dataset propre pour l’entraînement.
</p>

<h2>📂 4. Dataset</h2>
<p>Structure finale :</p>
<pre>
dataset_final/
 ├── train/
 │   ├── images/
 │   └── labels/
 ├── val/
 │   ├── images/
 │   └── labels/
 ├── test/
 │   ├── images/
 │   └── labels/
 └── data.yaml
</pre>

<p><b>Classes :</b><br>
0 : poubelle_vide<br>
1 : poubelle_pleine</p>

<h2>🧠 5. Entraînement du modèle</h2>

<pre>
from ultralytics import YOLO
model = YOLO("yolov8n.pt")
model.train(
    data="dataset_final/data.yaml",
    epochs=20,
    imgsz=640,
    batch=8
)
</pre>

<p>Le modèle final est enregistré sous :  
<b>runs/detect/train/weights/best.pt</b></p>

<h2>🔍 6. Test & Prédiction</h2>

<pre>
from ultralytics import YOLO
model = YOLO("best.pt")
results = model("test.jpg")[0]
results.show()
</pre>

<h2>🌐 7. Application Web Streamlit</h2>
<p>
Fonctionnalités :
<ul>
<li>Uploader une image ou une vidéo</li>
<li>Détection avec YOLOv8</li>
<li>Image annotée + prédiction affichée</li>
<li>Statistiques en direct</li>
<li>Capture automatique des frames détectées</li>
</ul>
</p>

<h2>📦 8. Installation</h2>

<pre>
git clone https://github.com/faty-mbengue/poubelle.git
cd poubelle
pip install -r requirements.txt
streamlit run app.py
</pre>

<h2>📁 9. Structure du dépôt GitHub</h2>
<pre>
poubelle/
 ├── app.py
 ├── best.pt
 ├── requirements.txt
 ├── README.md
</pre>

<h2>🚀 10. Déploiement</h2>
<p>
Déploiement Streamlit Cloud :
<ul>
<li>Push du projet sur GitHub</li>
<li>Connexion à https://streamlit.io/cloud</li>
<li>Choisir le repo → Lancer</li>
<li>L'application devient publique</li>
</ul>
</p>

</div>
""", unsafe_allow_html=True)
