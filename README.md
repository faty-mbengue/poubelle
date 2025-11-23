<div class="doc-card">

<div class="doc-title">🗑️ Détection & Classification de Poubelles (Pleine / Vide)</div>

<h2>📌 Reconnaissance d’objets avec yolo11n + Application Web </h2>

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

<h2>🤖 2. Modèle d’IA utilisé : YOLOv11n</h2>
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

model = YOLO("yolo11n.pt")

results = model.train(
    data=r"data.yaml",
    epochs=20,
    imgsz=416,
    batch=2,
    workers=0,
    amp=False,
    cache=False,
    patience=20,
    
    # AUGMENTATION FORTE
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=15,
    translate=0.1,
    scale=0.5,
    fliplr=0.5,
    mosaic=1.0,
    mixup=0.1,
    copy_paste=0.1,
    
    project="runs/detect",
    name="poubelles_augmented",
)

print("\nModèle avec augmentation: runs/detect/poubelles_augmented/weights/best.pt")
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
### 📊 Performances du Modèle (YOLOv11n)

- **mAP50:** 85.7%  
- **mAP50-95:** 43.5%  
- **Précision:** 85.9%  
- **Recall:** 79.4%  
- **Version du modèle:** YOLOv11n

<h2>🌐 7. Application Web Streamlit</h2>
<p>
Fonctionnalités :
<ul>
<li>Uploader une image ou une vidéo</li>
<li>Détection avec YOLOv11n</li>
<li>Image annotée + prédiction affichée</li>
<li>Statistiques </li>
<li>Capture automatique des frames détectées</li>
</ul>
</p>

<h2>📁 8. Structure du dépôt GitHub</h2>
<pre>
poubelle/
 ├── app.py
 ├── best.pt
 ├── requirements.txt
 ├── README.md
</pre>

<h2>🚀 9. Déploiement</h2>
<p>
Déploiement Streamlit Cloud 
</p>

</div>

