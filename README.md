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


<div class='doc-card'>

<div class='doc-title'>🗑️ Détection & Classification de Poubelles (Pleine / Vide)</div>

<h2>📌 Reconnaissance d’objets avec YOLOv8 + Streamlit</h2>

<p>
Ce projet a pour objectif de développer une application complète capable :
</p>

<ul>
<li>🟦 de détecter automatiquement une poubelle dans une image,</li>
<li>🟦 de localiser la poubelle via une bounding box,</li>
<li>🟦 de déterminer si la poubelle est pleine ou vide,</li>
<li>🟦 d’afficher un résultat annoté clair et précis.</li>
</ul>

<h2>🤖 1. Modèle d’IA utilisé : YOLOv8</h2>

<p>
YOLOv8 est un modèle puissant pour :
</p>
<ul>
<li>🔍 Détection d’objets</li>
<li>📦 Localisation (bounding boxes)</li>
<li>🏷️ Classification (poubelle vide / pleine)</li>
</ul>

<p>Le modèle utilise :</p>

<ul>
<li>📐 <b>IoU</b> pour évaluer la qualité des prédictions,</li>
<li>🚫 <b>NMS</b> pour supprimer les doublons,</li>
<li>⚡ Une architecture rapide et optimisée pour le temps réel.</li>
</ul>

<h2>📝 2. Annotation des Images (LabelImg)</h2>

<p>
Les images ont été annotées manuellement via <b>LabelImg</b> :
</p>

<ul>
<li>✏️ tracé des bounding boxes,</li>
<li>🏷️ assignation d’une classe (poubelle_vide ou poubelle_pleine),</li>
<li>📄 export automatique des labels au format YOLO.</li>
</ul>

<h2>📂 3. Structure du Dataset</h2>

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

Classes disponibles :
<ul>
<li>0 → 🟩 poubelle_vide</li>
<li>1 → 🟥 poubelle_pleine</li>
</ul>

<h2>🧠 4. Entraînement du modèle</h2>

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

Le modèle final est sauvegardé dans :
<b>runs/detect/train/weights/best.pt</b>

<h2>🔍 5. Test & Prédiction</h2>

<pre>
model = YOLO("best.pt")
results = model("test.jpg")[0]
results.show()
</pre>

<h2>🌐 6. Application Web Streamlit</h2>

Fonctionnalités :
<ul>
<li>🖼️ Upload d'image</li>
<li>🤖 Prédiction YOLOv8</li>
<li>🟦 Affichage de l’image annotée</li>
<li>📊 Classification pleine / vide</li>
</ul>

<h2>📦 7. Installation</h2>

<pre>
git clone https://github.com/faty-mbengue/poubelle.git
cd poubelle
pip install -r requirements.txt
streamlit run app.py
</pre>

<h2>🚀 8. Déploiement</h2>

Déployable en quelques clics via :
<b>Streamlit Cloud</b>

<ul>
<li>Push sur GitHub</li>
<li>Connexion à streamlit.io/cloud</li>
<li>Déploiement automatique ✔️</li>
</ul>

</div>

""", unsafe_allow_html=True)
