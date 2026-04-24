import os
import csv

root = r"C:\Users\chand\DeepfakeDataset\celebdf"

real_dir = os.path.join(root, "real")
fake_dir = os.path.join(root, "fake")

csv_path = os.path.join(root, "labels.csv")

with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["filename", "label"])

    # REAL videos -> label 0
    for v in sorted(os.listdir(real_dir)):
        if v.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
            writer.writerow([os.path.join("real", v), 0])

    # FAKE videos -> label 1
    for v in sorted(os.listdir(fake_dir)):
        if v.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
            writer.writerow([os.path.join("fake", v), 1])

print("labels.csv created at:", csv_path)

