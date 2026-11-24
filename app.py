"""TrustLeaf — inférence + explicabilité pour la classification des maladies des plantes.

Modèles : ResNet50 et ViT-B/16 (timm), fine-tunés sur PlantVillage sans Soybean (37 classes).
Lancement : streamlit run app.py
"""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn.functional as F
import timm
from PIL import Image
from dotenv import load_dotenv
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

st.set_page_config(page_title="TrustLeaf", page_icon="🌿", layout="wide")

BASE_DIR = Path(__file__).parent
load_dotenv(BASE_DIR / ".env")  # charge GROQ_API_KEY depuis .env
WEIGHTS_DIR = Path(os.getenv("TRUSTLEAF_WEIGHTS_DIR", BASE_DIR / "weights"))
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")  # surchargeable via .env
GROQ_TEMPERATURE = float(os.getenv("GROQ_TEMPERATURE", "0.3"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODELS = {
    "ResNet50": {"arch": "resnet50", "file": "resnet50_maroc_best.pth"},
    "ViT-B/16": {"arch": "vit_base_patch16_224", "file": "vit_maroc_best.pth"},
}

# Ordre = sorted(os.listdir(dataset_path)) du notebook, sans "Soybean___healthy".
# Si ton ordre diffère, dépose un classes.json (liste de 37 noms) à côté de app.py.
DEFAULT_CLASSES = [
    "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew", "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight", "Corn_(maize)___healthy",
    "Grape___Black_rot", "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot", "Peach___healthy",
    "Pepper,_bell___Bacterial_spot", "Pepper,_bell___healthy",
    "Potato___Early_blight", "Potato___Late_blight", "Potato___healthy",
    "Raspberry___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch", "Strawberry___healthy",
    "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Late_blight",
    "Tomato___Leaf_Mold", "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite", "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]


def _load_classes():
    path = BASE_DIR / "classes.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return DEFAULT_CLASSES


CLASSES = _load_classes()
NUM_CLASSES = len(CLASSES)

# Mêmes transformations que la validation du notebook
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ── Utilitaires ──────────────────────────────────────────────────────────────
def pretty(label: str) -> str:
    crop, _, cond = label.partition("___")
    clean = lambda s: s.replace("_", " ").replace("  ", " ").strip()
    return f"{clean(crop)} — {clean(cond)}"


def is_healthy(label: str) -> bool:
    return label.endswith("healthy")


@st.cache_resource(show_spinner="Chargement du modèle…")
def load_model(name: str):
    cfg = MODELS[name]
    path = WEIGHTS_DIR / cfg["file"]
    if not path.exists():
        raise FileNotFoundError(f"Poids introuvables : {path}")
    model = timm.create_model(cfg["arch"], pretrained=False, num_classes=NUM_CLASSES)
    state = torch.load(path, map_location=DEVICE)
    state = {k.removeprefix("module."): v for k, v in state.items()}
    model.load_state_dict(state)
    return model.to(DEVICE).eval()


def reshape_transform_vit(tensor, height=14, width=14):
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
    return result.transpose(2, 3).transpose(1, 2)


def predict(model, x):
    with torch.no_grad():
        return torch.softmax(model(x), dim=1)[0].cpu()


def gradcam_overlay(name, model, x, target, img_np):
    if name == "ResNet50":
        layers, rt = [model.layer4[-1]], None
    else:
        layers, rt = [model.blocks[-1].norm1], reshape_transform_vit
    with GradCAM(model=model, target_layers=layers, reshape_transform=rt) as cam:
        heat = cam(input_tensor=x, targets=[ClassifierOutputTarget(target)])[0]
    return show_cam_on_image(img_np, heat, use_rgb=True)


def vit_attention_overlay(model, x, img_np):
    """Attention du token CLS vers les patches (dernier bloc), moyenne des têtes."""
    store = {}
    attn_module = model.blocks[-1].attn

    def hook(_, __, out):
        b, n, _ = out.shape
        h = attn_module.num_heads
        qkv = out.reshape(b, n, 3, h, -1).permute(2, 0, 3, 1, 4)
        q, k = qkv[0], qkv[1]
        store["a"] = ((q @ k.transpose(-2, -1)) * q.shape[-1] ** -0.5).softmax(-1).detach().cpu()

    handle = attn_module.qkv.register_forward_hook(hook)
    try:
        with torch.no_grad():
            model(x)
    finally:
        handle.remove()

    cls_to_patches = store["a"][0].mean(0)[0, 1:]            # [196]
    grid = cls_to_patches.reshape(1, 1, 14, 14).float()
    up = F.interpolate(grid, size=(224, 224), mode="bilinear", align_corners=False)[0, 0].numpy()
    up = (up - up.min()) / (up.max() - up.min() + 1e-8)
    return show_cam_on_image(img_np, up, use_rgb=True)


@st.cache_data(show_spinner="Génération du conseil…", ttl=3600)
def llm_advice(label: str, conf: float, model_name: str, _api_key: str) -> str:
    from groq import Groq

    healthy = is_healthy(label)
    task = (
        "La plante semble saine : donne 4 phrases de conseils préventifs."
        if healthy else
        "En 4 phrases simples pour un agriculteur marocain : 1) c'est quoi cette maladie, "
        "2) symptômes visibles sur la feuille, 3) risque pour la récolte, "
        "4) traitement concret et accessible au Maroc."
    )
    system_prompt = (
        "Tu es expert en phytopathologie au Maroc. Réponds toujours en français, "
        "en langage simple pour un agriculteur, sans jargon inutile. "
        "Ne présente jamais le diagnostic comme certain : il vient d'un modèle d'IA. "
        "Termine par une phrase recommandant de confirmer avec un conseiller agricole."
    )
    user_prompt = (
        f"Diagnostic du modèle : {pretty(label)}\n"
        f"Confiance : {conf:.1%} (modèle {model_name}).\n{task}"
    )
    resp = Groq(api_key=_api_key).chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=GROQ_TEMPERATURE,
    )
    return resp.choices[0].message.content


def get_groq_key(manual: str) -> str:
    if manual:
        return manual
    try:
        if "GROQ_API_KEY" in st.secrets:
            return st.secrets["GROQ_API_KEY"]
    except Exception:
        pass
    return os.getenv("GROQ_API_KEY", "")


# ── Interface ────────────────────────────────────────────────────────────────
st.title("TrustLeaf")
st.caption("Diagnostic des maladies foliaires avec explications visuelles — "
           "ResNet50 & ViT-B/16 · PlantVillage (37 classes)")

with st.sidebar:
    st.header("Paramètres")
    choice = st.selectbox("Modèle", ["ResNet50", "ViT-B/16", "Les deux (consensus)"])
    selected = list(MODELS) if choice.startswith("Les deux") else [choice]

    xai_options = ["Grad-CAM"] + (["Attention ViT"] if "ViT-B/16" in selected else [])
    xai = st.radio("Explication visuelle", xai_options + ["Aucune"])

    threshold = st.slider("Seuil de confiance", 0.3, 0.99, 0.80, 0.01,
                          help="En dessous, TrustLeaf signale un diagnostic peu fiable.")

    st.divider()
    use_llm = st.toggle("Explication Groq", value=True)
    manual_key = (
        st.text_input("Clé API Groq", type="password",
                      help="Ou définis GROQ_API_KEY (variable d'environnement / .streamlit/secrets.toml).")
        if use_llm else ""
    )
    st.caption(f"Device : `{DEVICE}`")

tab_up, tab_cam = st.tabs(["Importer", "Caméra"])
with tab_up:
    uploaded = st.file_uploader("Photo de feuille", type=["jpg", "jpeg", "png", "webp"])
with tab_cam:
    snapshot = st.camera_input("Prendre une photo")

source = uploaded or snapshot
if source is None:
    st.info("Importe ou prends en photo une feuille unique, bien centrée, sur fond uni si possible.")
    st.stop()

image = Image.open(source).convert("RGB")
x = TRANSFORM(image).unsqueeze(0).to(DEVICE)
img_np = np.asarray(image.resize((224, 224)), dtype=np.float32) / 255.0

left, right = st.columns([1, 3])
left.image(image, caption="Image analysée", width="stretch")

results = {}
with right:
    cols = st.columns(len(selected))
    for col, name in zip(cols, selected):
        with col:
            st.subheader(name)
            try:
                model = load_model(name)
            except FileNotFoundError as e:
                st.error(f"{e}\n\nPlace le fichier `.pth` dans `{WEIGHTS_DIR}`.")
                continue

            probs = predict(model, x)
            top = torch.topk(probs, min(5, NUM_CLASSES))
            idx, conf = top.indices[0].item(), top.values[0].item()
            margin = conf - top.values[1].item()
            results[name] = (CLASSES[idx], conf)

            st.metric("Diagnostic", pretty(CLASSES[idx]))
            c1, c2 = st.columns(2)
            c1.metric("Confiance", f"{conf:.1%}")
            c2.metric("Écart top-1 / top-2", f"{margin:.1%}")

            if conf < threshold:
                st.warning("Confiance faible : refais la photo ou demande un avis d'expert.")
            elif is_healthy(CLASSES[idx]):
                st.success("Feuille probablement saine.")
            else:
                st.error("Maladie probable détectée.")

            df = pd.DataFrame({
                "Classe": [pretty(CLASSES[i]) for i in top.indices.tolist()],
                "Probabilité": top.values.tolist(),
            }).set_index("Classe")
            st.bar_chart(df, horizontal=True)

            if xai != "Aucune":
                with st.spinner("Calcul de l'explication…"):
                    if xai == "Attention ViT" and name == "ViT-B/16":
                        overlay = vit_attention_overlay(model, x, img_np)
                        cap = "Attention du token CLS (moyenne des têtes)"
                    else:
                        overlay = gradcam_overlay(name, model, x, idx, img_np)
                        cap = "Grad-CAM — zones qui ont le plus pesé"
                st.image(overlay, caption=cap, width="stretch")

if len(results) == 2:
    (n1, (c1_, p1)), (n2, (c2_, p2)) = results.items()
    if c1_ == c2_:
        st.success(f"✅ Consensus : les deux modèles prédisent **{pretty(c1_)}**.")
    else:
        st.warning(f"⚠️ Désaccord : {n1} → {pretty(c1_)} ({p1:.0%}) · {n2} → {pretty(c2_)} ({p2:.0%}).")

if use_llm and results:
    best_model, (best_label, best_conf) = max(results.items(), key=lambda kv: kv[1][1])
    key = get_groq_key(manual_key)
    st.divider()
    st.subheader("🤖 Conseils agronomiques")
    if not key:
        st.info("Renseigne une clé Groq (sidebar, `st.secrets` ou variable `GROQ_API_KEY`).")
    else:
        try:
            st.write(llm_advice(best_label, best_conf, best_model, key))
        except Exception as e:
            st.error(f"Explication indisponible : {e}")

st.divider()
st.caption("Les modèles sont entraînés sur des photos de laboratoire (PlantVillage) : "
           "les performances peuvent chuter sur des photos prises au champ. "
           "TrustLeaf est une aide à la décision, pas un diagnostic professionnel.")