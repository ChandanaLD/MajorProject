import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import glob

class ProcessedDataset(Dataset):
    def __init__(self, root, split="train", T=8):
        """
        root = C:/Users/chand/DeepfakeDataset/processed
        split = train or val
        """
        self.T = T
        self.root = os.path.join(root, split)

        # list all sample folders
        self.samples = sorted(os.listdir(self.root))
        self.samples = [s for s in self.samples if os.path.isdir(os.path.join(self.root, s))]

        self.transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        folder = os.path.join(self.root, self.samples[idx])

        # Load label
        label = int(open(os.path.join(folder, "label.txt")).read().strip())

        imgs = []
        graphs = []

        # Load T frames + T graphs
        for t in range(self.T):
            img_path = os.path.join(folder, f"frame_{t:03d}.png")
            graph_path = os.path.join(folder, f"graph_{t:03d}.pt")

            img = Image.open(img_path).convert("RGB")
            img = self.transform(img)
            imgs.append(img)

            g = torch.load(graph_path)
            graphs.append(g)

        imgs = torch.stack(imgs)  # [T, 3, 224, 224]

        return imgs, graphs, torch.tensor(label)
