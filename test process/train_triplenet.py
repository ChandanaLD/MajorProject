import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from triple_net import TripleNet
from load_fusion_weights import load_fusion_weights
from video_sequence_dataset import VideoSequenceDataset, sequence_collate_fn

from sklearn.metrics import roc_auc_score, f1_score
import numpy as np
import os


# -----------------------------------------------
# 1. Evaluation Function (AUC + F1)
# -----------------------------------------------

def evaluate(model, dataloader, device):
    model.eval()
    preds = []
    trues = []

    with torch.no_grad():
        for imgs, graphs, labels in dataloader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            logits = model(imgs, graphs)   # [B,2]
            prob_fake = torch.softmax(logits, dim=1)[:, 1]

            preds.extend(prob_fake.cpu().numpy())
            trues.extend(labels.cpu().numpy())

    # Safe metrics
    try:
        auc = roc_auc_score(trues, preds)
    except:
        auc = 0.5

    preds_binary = (np.array(preds) > 0.5).astype(int)
    f1 = f1_score(trues, preds_binary, zero_division=0)

    return {"auc": auc, "f1": f1}


# -----------------------------------------------
# 2. Training Logic
# -----------------------------------------------

def train_triplenet(
    train_videos, train_labels,
    val_videos, val_labels,
    batch_size=2, T=8,
    lr_stage1=1e-3, lr_stage2=1e-4,
    num_epochs=20,
    freeze_epochs=4
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Dataset
    train_ds = VideoSequenceDataset(train_videos, train_labels, T=T)
    val_ds   = VideoSequenceDataset(val_videos, val_labels, T=T)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=sequence_collate_fn, drop_last=True
    )

    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=sequence_collate_fn
    )

    # Model
    model = TripleNet().to(device)

    # Load pretrained CNN + GNN
    model = load_fusion_weights(model, "funet_a_full.pth")

    # Stage 1: Freeze CNN + GNN
    for p in model.cnn.parameters():
        p.requires_grad = False
    for p in model.gnn.parameters():
        p.requires_grad = False

    # Optimizer stage 1: only RNN + FC train
    optimizer = optim.AdamW(
        [
            {"params": model.rnn.parameters(), "lr": lr_stage1},
            {"params": model.fc.parameters(),  "lr": lr_stage1},
        ],
        weight_decay=1e-5
    )

    criterion = nn.CrossEntropyLoss()

    best_auc = 0
    patience = 5
    no_improve = 0

    for epoch in range(1, num_epochs + 1):

        # -------------------------
        # Stage transition
        # -------------------------
        if epoch == freeze_epochs + 1:
            print("\n🔓 Unfreezing CNN + GNN for fine-tuning...")

            for p in model.cnn.parameters():
                p.requires_grad = True
            for p in model.gnn.parameters():
                p.requires_grad = True

            optimizer = optim.AdamW(
                [
                    {"params": model.cnn.parameters(), "lr": lr_stage2},
                    {"params": model.gnn.parameters(), "lr": lr_stage2},
                    {"params": model.rnn.parameters(), "lr": lr_stage1},
                    {"params": model.fc.parameters(),  "lr": lr_stage1},
                ],
                weight_decay=1e-6
            )

        # -------------------------
        # Training Loop
        # -------------------------
        model.train()
        total_loss = 0

        for imgs, graphs, labels in train_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(imgs, graphs)
            loss = criterion(logits, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # -------------------------
        # Evaluation
        # -------------------------
        metrics = evaluate(model, val_loader, device)
        auc, f1 = metrics['auc'], metrics['f1']

        print(f"Epoch {epoch}/{num_epochs} | Loss={avg_loss:.4f} | AUC={auc:.4f} | F1={f1:.4f}")

        # -------------------------
        # Checkpoint
        # -------------------------
        if auc > best_auc:
            best_auc = auc
            no_improve = 0
            torch.save(model.state_dict(), "triplenet_best.pth")
            print("🔥 Saved new best model!")
        else:
            no_improve += 1

        # Early stopping
        if no_improve >= patience:
            print("⛔ Early stopping...")
            break

    return model


if __name__ == "__main__":
    import pandas as pd

    root = r"C:\Users\chand\DeepfakeDataset\celebdf"

    train_df = pd.read_csv(root + r"\train.csv")
    val_df   = pd.read_csv(root + r"\val.csv")

    train_videos = [os.path.join(root, v) for v in train_df["filename"].tolist()]
    train_labels = train_df["label"].tolist()

    val_videos = [os.path.join(root, v) for v in val_df["filename"].tolist()]
    val_labels = val_df["label"].tolist()

    print("Train videos:", len(train_videos))
    print("Val videos:", len(val_videos))

    # you can adjust T depending on performance (8, 12, 16)
    model = train_triplenet(
        train_videos=train_videos,
        train_labels=train_labels,
        val_videos=val_videos,
        val_labels=val_labels,
        T=8,              # sequence length
        batch_size=2,     # adjust based on GPU
        num_epochs=20,
        freeze_epochs=4
    )


