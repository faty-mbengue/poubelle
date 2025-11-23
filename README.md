🗑️ Détection et Classification de Poubelles (Pleine / Vide)
📌 Reconnaissance d’objets avec YOLOv8 + Application Web Streamlit
📖 1. Description du Projet
Ce projet a pour objectif de développer un système complet de détection d’objets permettant :
de détecter automatiquement une poubelle dans une image,
de localiser la poubelle via une bounding box,
de déterminer si elle est pleine ou vide.
L’utilisateur peut envoyer une image via l’application Streamlit, et le modèle renvoie une image annotée avec la prédiction finale.

🤖 2. Modèle d’IA utilisé : YOLOv8

Le modèle utilisé est YOLOv8, un modèle récent et performant pour la détection d’objets.
YOLOv8 réalise trois tâches  :
Détection : identifier la présence d’un objet.
Localisation : générer une bounding box autour de l’objet détecté.
Classification : prédire la classe (poubelle vide ou poubelle pleine).
YOLOv8 applique ensuite :
IoU (Intersection over Union) pour évaluer la qualité des boxes,
NMS (Non-Maximum Suppression) pour ne garder que les détections fiables.
Cela permet d’avoir un système rapide, optimisé et précis, parfaitement adapté à une application web en temps réel.

📝 3. Annotation des images (LabelImg)

Les images du dataset ont été annotées manuellement avec LabelImg, un outil open-source permettant :
de tracer les bounding boxes,
d’assigner une classe à chaque objet,
d’exporter automatiquement les labels au format YOLO.
Cela garantit un dataset propre et adapté à l'entraînement.

Annotation des images
Les images du dataset ont été annotées manuellement à l'aide de LabelImg﻿, un outil open-source populaire pour l'annotation d'images. Ce processus inclut :
Le tracé des bounding boxes﻿ (boîtes englobantes) autour des objets d'intérêt ;
L'assignation d'une classe à chaque objet détecté, ici les classes sont poubelle-vide﻿ et poubelle-plein﻿ ;
L'export automatique des annotations au format YOLO, où chaque fichier .txt contient pour chaque objet : la présence de l'objet, les coordonnées normalisées du centre de la boîte (bx, by), ainsi que la largeur et la hauteur (bw, bh) ;

Le fichier cla

📂 4. Dataset

Le dataset provient de deux sources :
un dataset téléchargé (poubelles pleines / vides),
des annotations réalisées manuellement avec LabelImg.
Structure finale :

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

Classes :

0 : poubelle_vide  
1 : poubelle_pleine

🧠 5. Entraînement du modèle
Le modèle YOLOv8 est entraîné grâce à la librairie Ultralytics :
from ultralytics import YOLO
model = YOLO("yolov8n.pt")
model.train(
    data="dataset_final/data.yaml",
    epochs=20,
    imgsz=640,
    batch=8
)


Le modèle final est enregistré sous :
runs/detect/train/weights/best.pt

🔍 6. Test & Prédiction

Exemple de prédiction sur une image :
from ultralytics import YOLO
model = YOLO("best.pt")
results = model("test.jpg")[0]
results.show()

🌐 7. Application Web Streamlit

Une interface utilisateur simple et ergonomique permet d’envoyer une image et d’obtenir la prédiction.

➡️ Fichier : app.py

Fonctionnalités :
upload d’image,
passage dans le modèle YOLOv8,
affichage de l’image annotée,
prédiction textuelle (poubelle vide / pleine).

📦 8. Installation
1️⃣ Cloner le dépôt
git clone https://github.com/faty-mbengue/poubelle.git
cd poubelle

2️⃣ Installer les dépendances
pip install -r requirements.txt

3️⃣ Lancer l’application
streamlit run app.py

📁 9. Structure du dépôt GitHub
poubelle/
 ├── app.py
 ├── best.pt
 ├── requirements.txt
 ├── README.md
 └── 

🚀 10. Déploiement

Le projet peut être déployé facilement sur Streamlit Cloud :
Push du projet sur GitHub
Connexion à https://streamlit.io/cloud
L’application devient accessible via une URL publique

