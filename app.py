import streamlit as st
import numpy as np
import cv2
import os
import tempfile
import torch
from torchvision import transforms
from PIL import Image
import pandas as pd

from load_triple_net import load_triple_model
from my_models import extract_faces_from_video, image_to_graph

# ------------------------------------------------------------
# Streamlit Page Config
# ------------------------------------------------------------
st.set_page_config(page_title="Deepfake Analyzer", layout="wide")

# 💅 Custom CSS Theme
st.markdown("""
    <style>
    body { background-color: #fff0f5; }
    body, .stTextInput, .stFileUploader label, .stMarkdown, .stText {
        color: #880e4f;
    }
    .main {
        background-color: #fff0f5;
        padding: 0;
    }
    .pink-card {
        background-color: #ffe6f0;
        padding: 25px;
        border-radius: 12px;
        box-shadow: 0 4px 10px rgba(216, 27, 96, 0.2);
        border: 1px solid #f8bbd0;
        color: #880e4f;
    }
    .taskbar {
        background-color: #ffe6f0;
        padding: 15px 30px;
        border-radius: 0 0 12px 12px;
        color: #880e4f;
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-family: 'Segoe UI', sans-serif;
        border-bottom: 2px solid #f8bbd0;
    }
    .taskbar a {
        color: #880e4f;
        margin: 0 10px;
        font-size: 18px;
        text-decoration: none;
        font-weight: 600;
        position: relative;
        transition: color 0.2s ease-in-out;
    }
    .taskbar a:not(:last-child)::after {
        content: "|";
        color: #d81b60;
        margin-left: 15px;
        margin-right: 10px;
        font-weight: 400;
    }
    .taskbar a:hover {
        text-decoration: underline;
        color: #d81b60;
    }
    </style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------
# Load TripleNet Model
# ------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
st.write(f"🔌 Using device: **{device}**")

model = load_triple_model("triplenet_best.pth", device=device)
model.eval()

# Transform for CNN
_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ------------------------------------------------------------
# Taskbar
# ------------------------------------------------------------
st.markdown("""
    <div class="taskbar">
        <div><strong style='font-size: 20px;'>💖 Deepfake Analyzer — TRIPLENET</strong></div>
        <div>
            <a href="#upload">Upload</a>
            <a href="#results">Results</a>
            <a href="#about">About</a>
        </div>
    </div>
""", unsafe_allow_html=True)


# ------------------------------------------------------------
# Title Section
# ------------------------------------------------------------
st.markdown("<h1 style='color:#880e4f;'>Deepfake Detector (CNN + GNN + RNN)</h1>", unsafe_allow_html=True)
st.write("Upload a video to begin deepfake analysis.")


# ------------------------------------------------------------
# Upload / Results
# ------------------------------------------------------------
col1, spacer, col2 = st.columns([2, 0.5, 2])

with col1:
    st.markdown("""
        <div class="pink-card" id="upload">
            <h3 style='color:#880e4f;'>Upload Video</h3>
            <p style='color:#880e4f; font-weight:bold;'>Select a video file (MP4, AVI, MOV)</p>
        </div>
    """, unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload", label_visibility="collapsed", type=["mp4", "avi", "mov"])

with col2:
    st.markdown("""
        <div class="pink-card" id="results">
            <h3 style='color:#880e4f;'>Analysis Results</h3>
            <p style='color:#880e4f; font-weight:bold;'>Output will appear below</p>
        </div>
    """, unsafe_allow_html=True)


# ------------------------------------------------------------
# Video Processing & TripleNet Inference (Correct)
# ------------------------------------------------------------
if uploaded_file is not None:

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded_file.read())
        video_path = tmp.name

    with st.spinner("Extracting faces & running TripleNet..."):

        frames = extract_faces_from_video(video_path, max_faces=8)

        if len(frames) == 0:
            st.error("❌ No faces detected.")
            os.remove(video_path)
            st.stop()

        st.write(f"### 🎞 Extracted {len(frames)} Frames (Temporal Sequence)")
        cols = st.columns(4)
        for i, frame in enumerate(frames):
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cols[i % 4].image(rgb, caption=f"Frame {i+1}", use_column_width=True)

        # Prepare model inputs
        imgs_list, graphs_list = [], []

        for frame in frames:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)

            img_tensor = _transform(pil)
            imgs_list.append(img_tensor)

            graph = image_to_graph(img_tensor)
            graphs_list.append(graph)

        imgs_tensor = torch.stack(imgs_list).unsqueeze(0).to(device)
        graphs_batch = [graphs_list]

        # Model prediction
        with torch.no_grad():
            logits = model(imgs_tensor, graphs_batch)
            prob_fake = torch.softmax(logits, dim=1)[0, 1].item()

        # Per-frame CNN-only probabilities (visual explanation)
        cnn_probs = []
        cnn_model = model.cnn
        cnn_model.eval()

        for t in imgs_list:
            x = t.unsqueeze(0).to(device)
            with torch.no_grad():
                feat = cnn_model(x)
                p = torch.sigmoid(feat.mean()).item()
                cnn_probs.append(p)

        # Accuracy & F1 (self-comparison)
        pred_label = 1 if prob_fake > 0.5 else 0
        preds = [1 if p > 0.5 else 0 for p in cnn_probs]
        truth = [pred_label] * len(preds)

        accuracy = np.mean(np.array(preds) == np.array(truth))
        f1 = 2 * accuracy / (accuracy + 1e-6)

        # Final TripleNet probability
        final_prob = prob_fake

        # Average per-frame probability
        avg_frame_prob = float(np.mean(cnn_probs))

        st.write("### 📌 Final Model Probability")
        if avg_frame_prob > 0.469:
            st.error(f"⚠ FINAL RESULT: DEEPFAKE ")
        else:
            st.success(f"✔ FINAL RESULT: REAL ")

        

        st.write("---")
        st.write("### 📊 Per-frame Probabilities Table")

        df = pd.DataFrame({
            "Frame": list(range(1, len(cnn_probs)+1)),
            "Fake Probability": [round(p, 3) for p in cnn_probs]
        })
        st.table(df)

        st.write("### 📈 Probability Chart")
        st.bar_chart({"Fake Probability": cnn_probs})

        st.write("---")
    os.remove(video_path)



