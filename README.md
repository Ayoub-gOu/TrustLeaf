# TrustLeaf

Application Streamlit de **diagnostic des maladies foliaires** avec **explicabilité** (XAI).
Elle classe une photo de feuille parmi 37 classes PlantVillage (sans Soybean) avec deux modèles,
**ResNet50** et **ViT-B/16**, et explique visuellement la décision.

## Fonctionnalités

- Import d'une photo ou capture par caméra
- Choix du modèle : ResNet50, ViT-B/16, ou les deux avec détection de consensus / désaccord
- Diagnostic, confiance, écart top-1 / top-2, top 5 des classes
- Seuil de confiance réglable avec alerte si le diagnostic est peu fiable
- Explications visuelles : **Grad-CAM** (les deux modèles) et **carte d'attention** (ViT)
- Explication textuelle en français via **Groq** (conseils sur la maladie prédite)

## Modèles et entraînement

- Données : PlantVillage (versions couleur), 38 classes réduites à 37 en retirant Soybean
- Modèles : ResNet50 et ViT-B/16 pré-entraînés (timm), tête de classification remplacée
- Augmentation : flip horizontal, rotation (20°), variation de luminosité, contraste et saturation
- Entraînement : 5 époques, AdamW (lr = 1e-4), `ReduceLROnPlateau`, split 80/20 (seed 42)

Le notebook (`notebook_plant_diseases_classifica.ipynb`) contient aussi la comparaison des méthodes
d'explicabilité : cartes d'attention, Grad-CAM, LIME et SHAP.

> **Limite importante** : PlantVillage est composé de photos de laboratoire (feuille seule, fond uni).
> Les performances peuvent baisser sur des photos prises au champ.
> TrustLeaf est une aide à la décision, pas un diagnostic professionnel.

## Installation

Prérequis : [uv](https://docs.astral.sh/uv/) et Python 3.14.

```bash
git clone https://github.com/<ton-user>/trustleaf.git
cd trustleaf
uv sync
```

## Configuration

1. **Poids des modèles** : place les fichiers issus de l'entraînement dans `weights/`
   - `resnet50_maroc_best.pth`
   - `vit_maroc_best.pth`

2. **Ordre des classes** (recommandé) : exporte `classes.json` depuis le notebook et place-le à la racine.
   Sans ce fichier, l'application utilise une liste par défaut.

3. **Clé Groq** : copie `.env.example` en `.env` et renseigne ta clé
   (à créer sur [console.groq.com](https://console.groq.com)).

   ```
   GROQ_API_KEY=gsk_...
   # Optionnel
   GROQ_MODEL=openai/gpt-oss-120b
   GROQ_TEMPERATURE=0.3
   ```

## Lancement

```bash
uv run streamlit run app.py
```

## Structure

```
trustleaf/
├── app.py                                   # application Streamlit
├── notebook_plant_diseases_classifica.ipynb # entraînement et XAI
├── classes.json                             # ordre des 37 classes (optionnel)
├── weights/                                 # poids .pth (non versionnés)
├── .streamlit/
│   └── config.toml                          # thème
├── .env.example                             # modèle de configuration
└── pyproject.toml
```

## Limites de l'explicabilité

- Grad-CAM et l'attention montrent **où** le modèle a regardé, pas **pourquoi** il a décidé.
- L'attention du ViT n'est pas toujours fidèle à la décision finale.
- L'explication Groq porte sur la maladie prédite : le modèle de langage ne voit pas l'image.

## Auteurs

- **Ayoub GOUAHSSOUN**
- **Khalil Abboud**, en collaboration, notamment pour l'entraînement des modèles

Master Data Science & IA (SDIA), ENSA El Jadida.