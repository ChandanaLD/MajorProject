import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torch_geometric
import torch.nn.functional as F
from torchvision import transforms


class VideoSequenceDataset(Dataset):
    def __init__(self, video_dirs, labels, T=8):
        """
        video_dirs: list of folder paths (train_00000, val_00001, etc)
        labels: 0 or 1
        """
        self.video_dirs = video_dirs
        self.labels = labels
        self.T = T

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __len__(self):
        return len(self.video_dirs)

    def __getitem__(self, idx):
        vid_dir = self.video_dirs[idx]
        label = self.labels[idx]

        # Load frames
        frames = []
        for t in range(self.T):
            img_path = os.path.join(vid_dir, f"frame_{t:03d}.png")
            img = Image.open(img_path).convert("RGB")
            img = self.transform(img)
            frames.append(img)

        frames = torch.stack(frames, dim=0)  # [T, 3, 224, 224]

        # Load graphs
        graphs = []
        for t in range(self.T):
            g_path = os.path.join(vid_dir, f"graph_{t:03d}.pt")
            g = torch.load(g_path)
            graphs.append(g)

        return frames, graphs, torch.tensor(label, dtype=torch.long)


def sequence_collate_fn(batch):
    imgs_batch = []
    graphs_batch = []
    labels_batch = []

    for imgs, graphs, label in batch:
        imgs_batch.append(imgs)
        graphs_batch.append(graphs)
        labels_batch.append(label)

    imgs_batch = torch.stack(imgs_batch, dim=0)  # [B, T, 3, 224,224]
    labels_batch = torch.stack(labels_batch)

    return imgs_batch, graphs_batch, labels_batch
