import zipfile, os, shutil, cv2
import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, global_max_pool
import torch.nn as nn
import torch.nn.functional as F
from facenet_pytorch import MTCNN





def image_to_graph(image_tensor, k=9, patch_size=32, debug=True):
    """
    Converts an image tensor [3,H,W] into a graph where
    each node = flattened patch of size (3*patch_size*patch_size).
    """

    C, H, W = image_tensor.shape
    if H < patch_size or W < patch_size:
        if debug:
            print(f"⚠️ Skipping tiny frame: {image_tensor.shape}")
        return None

    # Extract patches
    patches = image_tensor.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
    patches = patches.permute(1, 2, 0, 3, 4).contiguous()   # [num_h, num_w, C, ps, ps]
    patches = patches.view(-1, C * patch_size * patch_size)  # [num_patches, features]

    if patches.size(0) < 2:
        if debug:
            print(f"⚠️ Too few patches: {patches.shape}")
        return None

    # Normalize patches for cosine similarity
    patches_norm = F.normalize(patches, p=2, dim=1)  # [num_patches, features]

    # Compute similarity (matrix)
    similarity = torch.matmul(patches_norm, patches_norm.T)  # [num_patches, num_patches]

    # Build k-NN graph
    edge_index = []
    for i in range(similarity.size(0)):
        # Top-k neighbors (excluding self)
        k_eff = min(k + 1, similarity.size(1))  # don’t ask for more than available
        indices = torch.topk(similarity[i], k_eff).indices.tolist()
        indices = [j for j in indices if j != i]  # remove self
        edge_index += [(i, j) for j in indices]

    if len(edge_index) == 0:
        if debug:
            print("⚠️ No edges created")
        num_nodes = patches.size(0)
        edge_index = [(i, j) for i in range(num_nodes) for j in range(num_nodes) if i != j]

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    x = patches.float()  # node features
    batch = torch.zeros(x.size(0), dtype=torch.long)  # all belong to same graph

    return Data(x=x, edge_index=edge_index, batch=batch)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
detector = MTCNN(keep_all=False, device=device, post_process=False)
 #Initializes the MTCNN face detector
def extract_faces_from_video(video_path, max_faces=8):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames == 0:
        return []

    frame_indices = np.linspace(0, total_frames - 1, max_faces).astype(int)
    frames = []

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        # facenet-pytorch MTCNN
        boxes, probs = detector.detect(frame)

        if boxes is None or probs is None:
            continue

        # take the highest probability detection
        best_idx = np.argmax(probs)
        x1, y1, x2, y2 = boxes[best_idx].astype(int)

        # ensure bounds
        h, w, _ = frame.shape
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(w, x2); y2 = min(h, y2)

        face_img = frame[y1:y2, x1:x2]

        try:
            face_img = cv2.resize(face_img, (64, 64))
            frames.append(face_img)
        except:
            continue

    cap.release()
    return frames

# -------------------------------------------
# NEW: Sequential frame extractor for TripleNet
# -------------------------------------------

def extract_face_sequence(video_path, T=16, resize=(224, 224)):
    """
    Extracts T evenly spaced face crops from the video.
    Ensures chronological sequence for RNN.
    """

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ Cannot open video:", video_path)
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < 2:
        print("❌ Too few frames in video.")
        return []

    # Evenly spaced frame indices
    indices = np.linspace(0, total_frames - 1, T).astype(int)

    sequence = []
    last_valid_face = None

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            if last_valid_face is not None:
                sequence.append(last_valid_face)
            continue

        detections = detector.detect_faces(frame)

        if len(detections) == 0:
            if last_valid_face is not None:
                sequence.append(last_valid_face)
            continue

        # choose largest face
        det = max(detections, key=lambda d: d['box'][2] * d['box'][3])
        x, y, w, h = det['box']
        x, y = max(0, x), max(0, y)

        face = frame[y:y+h, x:x+w]

        try:
            face = cv2.resize(face, resize)
            face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            last_valid_face = face_rgb
            sequence.append(face_rgb)
        except:
            if last_valid_face is not None:
                sequence.append(last_valid_face)

    cap.release()

    # If shorter than T, fill the rest
    while len(sequence) < T:
        sequence.append(last_valid_face)

    return sequence
