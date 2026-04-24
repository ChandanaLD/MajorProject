import os
import cv2
import torch
import numpy as np
from facenet_pytorch import MTCNN
from PIL import Image
from tqdm import tqdm
from torchvision import transforms

from my_models import image_to_graph

# -------------------------------
# CONFIG
# -------------------------------
ROOT = r"C:\Users\chand\DeepfakeDataset\celebdf"
OUT  = r"C:\Users\chand\DeepfakeDataset\processed"
T = 8      # frames per video
IMG_SIZE = 224

# GPU device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# GPU MTCNN detector
mtcnn = MTCNN(
    keep_all=False,
    device=device,
    post_process=False,
    image_size=IMG_SIZE
)

# torchvision transform (same as training)
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# -------------------------------
# Extract T evenly spaced frames
# -------------------------------
def extract_t_frames(cap, T):
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < T:
        return None
    idx = np.linspace(0, total_frames - 1, T, dtype=int)
    return idx


# -------------------------------
# Extract a face using GPU MTCNN
# -------------------------------
def detect_face(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)

    boxes, probs = mtcnn.detect(pil)

    if boxes is None:
        return None

    x1,y1,x2,y2 = boxes[0].astype(int)
    face = rgb[y1:y2, x1:x2]

    if face.size == 0:
        return None

    return cv2.resize(face, (IMG_SIZE, IMG_SIZE))


# -------------------------------
# Process a single video
# -------------------------------
def process_video(video_path, out_dir, label):
    cap = cv2.VideoCapture(video_path)
    idxs = extract_t_frames(cap, T)

    if idxs is None:
        return False  # too few frames

    frames = []

    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            continue

        face = detect_face(frame)
        if face is None:
            continue

        frames.append(face)

    cap.release()

    if len(frames) < T:
        return False

    os.makedirs(out_dir, exist_ok=True)

    # Save frames + graphs
    for i, face in enumerate(frames):
        # save image
        fpath = os.path.join(out_dir, f"frame_{i:03d}.png")
        cv2.imwrite(fpath, cv2.cvtColor(face, cv2.COLOR_RGB2BGR))

        # convert to tensor
        pil = Image.fromarray(face)
        tensor = transform(pil)

        # build graph
        graph = image_to_graph(tensor)
        torch.save(graph, os.path.join(out_dir, f"graph_{i:03d}.pt"))

    # save label
    with open(os.path.join(out_dir, "label.txt"), "w") as f:
        f.write(str(label))

    return True


# -------------------------------
# Process split (train or val)
# -------------------------------
def process_split(csv_file, split_name):
    import pandas as pd

    df = pd.read_csv(csv_file)
    print(f"Processing {split_name} ({len(df)} videos)...")

    success = 0
    fail = 0

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        video = os.path.join(ROOT, row["filename"])
        label = row["label"]

        out_dir = os.path.join(OUT, f"{split_name}_{idx:05d}")

        ok = process_video(video, out_dir, label)
        if ok:
            success += 1
        else:
            fail += 1

    print(f"\n✔ Done {split_name}: {success} success, {fail} failed.")


# -------------------------------
# Run whole preprocess
# -------------------------------
if __name__ == "__main__":
    process_split(os.path.join(ROOT, "train.csv"), "train")
    process_split(os.path.join(ROOT, "val.csv"), "val")
