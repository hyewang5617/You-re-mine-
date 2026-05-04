"""
Pokemon Classifier - Streamlit Demo
=====================================
Run: streamlit run app.py

Requires trained models from train.py:
  python train.py --data_dir ./PokemonData --all
"""

import json
from pathlib import Path

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms

RESULTS_DIR = Path('results')

# --------------------------------------------------------------------------- #
# Page config
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title='Pokemon Classifier',
    page_icon='poke_ball',
    layout='wide',
)

# --------------------------------------------------------------------------- #
# Model utilities (mirrors train.py build_model, no pretrained weights needed)
# --------------------------------------------------------------------------- #
def _build_model_shell(backbone: str, num_classes: int) -> nn.Module:
    """Build model architecture (no pretrained weights - loaded from checkpoint)."""
    if backbone == 'resnet34':
        model = models.resnet34()
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif backbone == 'resnet50':
        model = models.resnet50()
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif backbone == 'vgg16':
        model = models.vgg16()
        model.classifier[6] = nn.Linear(4096, num_classes)
    elif backbone == 'efficientnet_b0':
        model = models.efficientnet_b0()
        model.classifier[1] = nn.Linear(
            model.classifier[1].in_features, num_classes
        )
    else:
        raise ValueError(f'Unknown backbone: {backbone}')
    return model


@st.cache_resource
def load_model(exp_name: str):
    """Load trained model checkpoint (cached across Streamlit reruns)."""
    ckpt_path = RESULTS_DIR / exp_name / 'best_model.pth'
    if not ckpt_path.exists():
        return None, None
    ckpt  = torch.load(ckpt_path, map_location='cpu')
    model = _build_model_shell(ckpt['backbone'], ckpt['num_classes'])
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    return model, ckpt['class_names']


def preprocess(image: Image.Image, img_size: int = 224) -> torch.Tensor:
    """Same preprocessing as evaluate transform in train.py."""
    tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return tf(image.convert('RGB')).unsqueeze(0)


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
EXP_LABELS = {
    'exp1': 'ResNet-34 | pretrained | frozen (feature extraction)',
    'exp2': 'ResNet-34 | pretrained | full fine-tuning',
    'exp3': 'ResNet-34 | scratch (no pretrained)',
    'exp4': 'EfficientNet-B0 | pretrained | frozen (feature extraction)',
}

with st.sidebar:
    st.title('Pokemon Classifier')
    st.markdown('**Homework #6 - Transfer Learning**')
    st.markdown('---')

    available = [e for e in EXP_LABELS
                 if (RESULTS_DIR / e / 'best_model.pth').exists()]

    if not available:
        st.error(
            'No trained models found.\n\n'
            'Run:\n```\npython train.py --data_dir ./PokemonData --all\n```'
        )
        st.stop()

    selected_exp = st.selectbox(
        'Select Model',
        options=available,
        format_func=lambda e: f'{e}: {EXP_LABELS.get(e, e)}',
    )

    top_k = st.slider('Top-K predictions', min_value=1, max_value=10, value=5)

    # Show stored test metrics
    metrics_path = RESULTS_DIR / selected_exp / 'metrics.json'
    if metrics_path.exists():
        with open(metrics_path) as f:
            m = json.load(f)
        st.markdown('---')
        st.markdown('**Test Performance**')
        col_a, col_b = st.columns(2)
        col_a.metric('Accuracy',  f'{m["test_acc"]:.4f}')
        col_b.metric('Precision', f'{m["test_precision"]:.4f}')
        col_a.metric('Recall',    f'{m["test_recall"]:.4f}')
        col_b.metric('F1',        f'{m["test_f1"]:.4f}')

    # Show comparison chart if all experiments ran
    cmp_path = RESULTS_DIR / 'comparison.png'
    if cmp_path.exists():
        st.markdown('---')
        with st.expander('All Experiments Comparison'):
            st.image(str(cmp_path))


# --------------------------------------------------------------------------- #
# Main area
# --------------------------------------------------------------------------- #
st.title('Pokemon Classifier')
st.caption(
    'Upload a Pokemon image and the model will identify which Pokemon it is.'
)

model, class_names = load_model(selected_exp)
if model is None:
    st.error(f'Model for {selected_exp} not found. Please train it first.')
    st.stop()

st.markdown(f'**Active model:** `{selected_exp}` - {EXP_LABELS.get(selected_exp, "")}')
st.markdown('---')

uploaded = st.file_uploader(
    'Upload a Pokemon image',
    type=['jpg', 'jpeg', 'png', 'gif', 'bmp', 'webp'],
)

if uploaded is not None:
    image = Image.open(uploaded)

    col_img, col_pred = st.columns([1, 1])

    with col_img:
        st.subheader('Input Image')
        st.image(image, width=300)

    with col_pred:
        st.subheader(f'Top-{min(top_k, len(class_names))} Predictions')

        with st.spinner('Classifying...'):
            tensor = preprocess(image)
            with torch.no_grad():
                logits = model(tensor)
                probs  = F.softmax(logits, dim=1)[0]

        k_actual             = min(top_k, len(class_names))
        top_probs, top_idxs  = torch.topk(probs, k=k_actual)
        top_probs            = top_probs.tolist()
        top_idxs             = top_idxs.tolist()

        rank_emojis = ['1st', '2nd', '3rd', '4th', '5th',
                       '6th', '7th', '8th', '9th', '10th']

        for i, (prob, idx) in enumerate(zip(top_probs, top_idxs)):
            label = class_names[idx]
            rank  = rank_emojis[i] if i < len(rank_emojis) else f'{i+1}th'
            st.markdown(f'**{rank}  {label}**  —  `{prob * 100:.2f}%`')
            st.progress(prob)

    # Learning curve for selected experiment
    curve_path = RESULTS_DIR / selected_exp / 'learning_curve.png'
    if curve_path.exists():
        st.markdown('---')
        with st.expander('Training History (Learning Curve)'):
            st.image(str(curve_path))
