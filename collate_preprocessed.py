
import torch
def collate_fn(batch):
    imgs = []
    graphs = []
    labels = []

    for b in batch:
        imgs.append(b[0])      # [T, 3, H, W]
        graphs.append(b[1])    # list of T graphs
        labels.append(b[2])

    imgs = torch.stack(imgs)   # [B, T, 3, H, W]
    labels = torch.stack(labels)

    return imgs, graphs, labels
