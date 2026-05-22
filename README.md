#  Détection de l'Endométriose par Deep Learning

> **Projet encadré par M. Abdallah Khemais — Module Deep Learning**


##  Description du Problème

L'**endométriose** est une maladie gynécologique chronique touchant environ **10% des femmes** en âge de procréer dans le monde. Le diagnostic est souvent retardé de **7 à 10 ans** en moyenne, car les symptômes sont difficiles à distinguer d'autres pathologies.

### Objectif
Développer un système de **classification binaire** d'images laparoscopiques basé sur le Deep Learning pour détecter automatiquement la présence ou l'absence d'endométriose.

| Classe | Description |
|--------|-------------|
| `endometriosis` | Image laparoscopique avec présence d'endométriose |
| `no_endometriosis` | Image laparoscopique sans endométriose |

---

##  Jeu de Données

**Source :** [Kaggle — Dataset Endometriosis and Non-Endometriosis]

| Caractéristique | Valeur |
|-----------------|--------|
| Total images | **1 022** |
| Endométriose | **533 images (52.2%)** |
| No-Endométriose | **489 images (47.8%)** |
| Résolution | **640 × 360 px** |
| Format | JPG |
| Type | Images laparoscopiques médicales |

### Distribution

```
Endométriose    ████████████████████  533 (52.2%)
No-Endométriose ███████████████████   489 (47.8%)
```

>  Dataset **équilibré** — pas de biais de classe significatif

### Split Train / Validation / Test

| Split | Images | Pourcentage |
|-------|--------|-------------|
| Train | 715 | 70% |
| Validation | 153 | 15% |
| Test | 154 | 15% |

---

##  Architecture Choisie

### Modèle Final : **VGG16 avec Transfer Learning**

Après comparaison de 3 architectures, **VGG16** a été sélectionné comme meilleur modèle.

```
Input (224×224×3)
       ↓
VGG16 Base — poids ImageNet (couches convolutives gelées)
       ↓
GlobalAveragePooling2D
       ↓
Dense(256) → BatchNormalization → ReLU → Dropout(0.5)
       ↓
Dense(1, activation='sigmoid')
       ↓
Output : Endométriose (1) / No-Endométriose (0)
```

### Pourquoi VGG16 ?

-  Architecture simple et efficace pour les petits datasets
-  Poids pré-entraînés sur ImageNet (1M images)
-  Meilleure généralisation que CNN from scratch
-  Moins de paramètres à entraîner → moins d'overfitting

### Stratégie d'Entraînement

| Phase | Couches | Learning Rate | Époques |
|-------|---------|---------------|---------|
| Phase 1 | Base gelée | 1e-3 | 20 |
| Phase 2 (Fine-tuning) | 4 dernières couches dégelées | 1e-5 | 50 |

---

##  Pipeline Deep Learning

### 1. Prétraitement
- Redimensionnement : **224×224 px** (standard VGG16/ResNet50)
- Normalisation : pixels ÷ 255 → valeurs ∈ [0, 1]
- Split stratifié : **70% / 15% / 15%**

### 2. Augmentation des Données (Train uniquement)

```python
ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.15,
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=[0.8, 1.2],
    shear_range=0.1
)
```

### 3. Techniques Anti-Overfitting
- **Dropout(0.5)** dans la couche Dense
- **BatchNormalization** après chaque couche Dense
- **EarlyStopping** (patience=10, monitor=val_loss)
- **ReduceLROnPlateau** (factor=0.5, patience=5)
- **GlobalAveragePooling2D** au lieu de Flatten
- **Class Weights** pour compenser le déséquilibre

### 4. Compilation

| Composant | Choix | Justification |
|-----------|-------|---------------|
| Loss | Binary Crossentropy | Classification binaire |
| Optimizer | Adam (lr=1e-4) | Convergence rapide et adaptative |
| Métriques | Accuracy, AUC, Precision, Recall | Évaluation complète |

---

##  Performances Obtenues

### Comparaison des Architectures

| Modèle | Accuracy | Precision | Recall | F1-Score | AUC | Remarque |
|--------|----------|-----------|--------|----------|-----|----------|
| CNN Baseline | 48.1% | 23.1% | 48.1% | 31.2% | 74.6% | Overfitting sévère |
| ResNet50 | 48.1% | 23.1% | 48.1% | 31.2% | 63.3% | Dataset trop petit |
| **VGG16**  | **92.9%** | **93.2%** | **92.9%** | **92.9%** | **98.9%** | **Meilleur modèle** |

### Résultats VGG16 sur le Test Set

```
              precision    recall  f1-score   support
endometriosis      0.97      0.89      0.93        80
no_endometriosis   0.89      0.97      0.93        74
accuracy                               0.93       154
```

### Matrice de Confusion

```
                 Prédit: Endo   Prédit: No-Endo
Vraie: Endo         72               9 
Vraie: No-Endo       2              71 
```

| Métrique Médicale | Valeur |
|-------------------|--------|
| **Sensibilité (Recall)** | **97.3%** — 97% des malades détectés |
| **Spécificité** | **88.7%** — 89% des sains identifiés |
| **Faux Négatifs** | **2** — seulement 2 cas manqués |

---

##  Structure du Dépôt

```
deep_learning_project/
│
├── 📓 Notebooks/
│   ├── 01_exploration.ipynb          ← Analyse et visualisation du dataset
│   ├── 02_preprocessing.ipynb        ← Prétraitement et augmentation
│   ├── 03_cnn_baseline.ipynb         ← CNN from scratch (baseline)
│   ├── 04_transfer_learning.ipynb    ← VGG16 + ResNet50
│   └── 05_gradcam_analyse.ipynb      ← Grad-CAM + Analyse finale
│
├──  presentation.pptx              ← Slides de soutenance (7-8 slides)
│
├──  app.py                         ← Application Streamlit
│
├── requirements.txt               ← Dépendances Python
│
└──  README.md                      ← Ce fichier
```

---



### Installation

```bash
# 1. Cloner le dépôt
git clone https://github.com/hedil1/deep_learning_project.git
cd deep_learning_project

# 2. Créer un environnement virtuel
python -m venv venv

# Activer (Windows)
venv\Scripts\activate

# Activer (Mac/Linux)
source venv/bin/activate

# 3. Installer les dépendances
pip install -r requirements.txt
```

### Exécuter les Notebooks


### Ordre d'Exécution

```
01_exploration.ipynb       → Analyser le dataset
       ↓
02_preprocessing.ipynb     → Préparer les données
       ↓
03_cnn_baseline.ipynb      → Entraîner CNN baseline
       ↓
04_transfer_learning.ipynb → Entraîner VGG16 + ResNet50
       ↓
05_gradcam_analyse.ipynb   → Visualiser Grad-CAM
```

### Lancer l'Application Streamlit

```bash
# Installer les dépendances
pip install -r requirements.txt

# Lancer l'application
streamlit run app.py

``
https://deeplearningproject-9fcbxfotrgyh6fmmtnnzex.streamlit.app/

---

## 🌐 Application Streamlit

L'application permet de **prédire automatiquement** la présence d'endométriose à partir d'une image laparoscopique uploadée.

### Fonctionnalités

| Page | Description |
|------|-------------|
|  Accueil | Présentation du projet et de l'endométriose |
|  Diagnostic | Upload image → Prédiction + Score + Grad-CAM |
|  Performance | Métriques, matrice de confusion, comparaison |

### Déploiement en Ligne

 https://deeplearningproject-9fcbxfotrgyh6fmmtnnzex.streamlit.app/

---

##  Interprétation des Résultats

### Grad-CAM — Visualisation de l'Attention

Le Grad-CAM (**Gradient-weighted Class Activation Mapping**) permet de visualiser les zones de l'image sur lesquelles le modèle concentre son attention pour prendre sa décision.

```
🔴 Rouge = Zone très importante pour la décision
🟡 Jaune = Zone modérément importante
🔵 Bleu  = Zone peu importante
```

### Analyse des Courbes d'Apprentissage

| Observation | Interprétation |
|-------------|----------------|
| Train loss ↘ / Val loss ↗ | Overfitting (CNN Baseline) |
| Train ≈ Val accuracy | Bon apprentissage (VGG16) |
| Gap < 2% | Excellente généralisation |

### Choix Justifiés

- **VGG16 vs ResNet50** : VGG16 plus adapté aux petits datasets
- **Binary Crossentropy** : tâche de classification binaire
- **Adam** : convergence plus rapide que SGD
- **Dropout(0.5)** : régularisation efficace contre l'overfitting
- **GlobalAvgPooling** : moins de paramètres que Flatten → moins d'overfitting

---

##  Dépendances

```
streamlit==1.32.0
tensorflow==2.15.1
opencv-python-headless==4.9.0.80
pillow==10.2.0
numpy==1.26.4
matplotlib==3.8.3
pandas==2.2.1
seaborn==0.13.2
gdown==5.1.0
scikit-learn==1.4.0
```

---

##  Auteur

**Rouatbi Hedil **
Module Deep Learning — Polytechnique-Sousse
Encadrant : **M. Abdallah Khemais**

---


