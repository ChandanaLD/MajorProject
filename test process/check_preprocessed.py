import pickle
import os

root = r"C:\Users\chand\DeepfakeDataset\celebdf\preprocessed"

files = [
    "train_videos_preprocessed.pkl",
    "train_labels_preprocessed.pkl",
    "val_videos_preprocessed.pkl",
    "val_labels_preprocessed.pkl"
]

for f in files:
    path = os.path.join(root, f)
    print(f"\nChecking: {f}")
    print("Exists:", os.path.exists(path))

    if os.path.exists(path):
        data = pickle.load(open(path, "rb"))
        print("Type:", type(data))
        print("Length:", len(data))
