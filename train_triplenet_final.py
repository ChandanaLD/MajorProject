# train_triplenet_final.py (updated)
import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from triple_net import TripleNet
from load_fusion_weights import load_fusion_weights
from video_sequence_dataset import VideoSequenceDataset, sequence_collate_fn

from sklearn.metrics import roc_auc_score, f1_score
import numpy as np

import csv

# --------------------------
# Config
# --------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PROCESSED_ROOT = r"C:\Users\chand\DeepfakeDataset\processed"
BATCH_SIZE = 4                # RTX 4060 can usually handle 4; reduce to 2 if OOM
T = 8
NUM_EPOCHS = 20
FREEZE_EPOCHS = 3             # freeze CNN+GNN for first N epochs
LR_RNN_FC = 1e-4
LR_FINE_TUNE = 1e-5
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 4               # increase if you have faster disk/CPU
PIN_MEMORY = True
CLIP_NORM = 5.0

print("Using device:", DEVICE)


# --------------------------
# Utilities
# --------------------------
def evaluate(model, loader):
    model.eval()
    preds, trues = [], []

    with torch.no_grad():
        for imgs, graphs, labels in loader:
            imgs = imgs.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            logits = model(imgs, graphs)
            probs = torch.softmax(logits, dim=1)[:, 1]

            preds.extend(probs.cpu().numpy())
            trues.extend(labels.cpu().numpy())

    try:
        auc = roc_auc_score(trues, preds)
    except Exception:
        auc = 0.5

    f1 = f1_score(trues, (np.array(preds) > 0.5).astype(int), zero_division=0)
    return auc, f1


# --------------------------
# Training
# --------------------------
def train():
    # --- prepare data folders & labels
    train_dirs = sorted(glob.glob(os.path.join(PROCESSED_ROOT, "train_*")))
    val_dirs   = sorted(glob.glob(os.path.join(PROCESSED_ROOT, "val_*")))

    if len(train_dirs) == 0:
        raise RuntimeError(f"No train_* folders found in {PROCESSED_ROOT}")

    train_labels = []
    for d in train_dirs:
        with open(os.path.join(d, "label.txt"), "r") as f:
            train_labels.append(int(f.read().strip()))

    val_labels = []
    for d in val_dirs:
        with open(os.path.join(d, "label.txt"), "r") as f:
            val_labels.append(int(f.read().strip()))

    print(f"Train samples: {len(train_dirs)}, Val samples: {len(val_dirs)}")

    # --- datasets / loaders
    train_ds = VideoSequenceDataset(train_dirs, train_labels, T=T)
    val_ds   = VideoSequenceDataset(val_dirs, val_labels, T=T)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=sequence_collate_fn,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=sequence_collate_fn,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=False
    )

    # --- model + weights
    model = TripleNet().to(DEVICE)
    model = load_fusion_weights(model, "funet_a_full.pth")

    # Freeze cnn + gnn at start
    for p in model.cnn.parameters():
        p.requires_grad = False
    for p in model.gnn.parameters():
        p.requires_grad = False

    # optimizer (initial: only rnn + fc)
    optimizer = optim.AdamW(
        [
            {"params": model.rnn.parameters(), "lr": LR_RNN_FC},
            {"params": model.fc.parameters(), "lr": LR_RNN_FC},
        ],
        weight_decay=WEIGHT_DECAY
    )

    criterion = nn.CrossEntropyLoss()

    # AMP scaler
    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE.type == "cuda"))

    best_auc = 0.0

    log_file = open("training_log.csv", "w", newline="")
    log_writer = csv.writer(log_file)
    log_writer.writerow(["epoch", "train_loss", "val_auc", "val_f1"])

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0.0

        # Unfreeze CNN + GNN at correct epoch
        if epoch == FREEZE_EPOCHS + 1:
            print("\n🔓 Unfreezing CNN + GNN for fine-tuning...")
            for p in model.cnn.parameters():
                p.requires_grad = True
            for p in model.gnn.parameters():
                p.requires_grad = True

            optimizer = optim.AdamW(
                [
                    {"params": model.cnn.parameters(), "lr": LR_FINE_TUNE},
                    {"params": model.gnn.parameters(), "lr": LR_FINE_TUNE},
                    {"params": model.rnn.parameters(), "lr": LR_RNN_FC},
                    {"params": model.fc.parameters(),  "lr": LR_RNN_FC},
                ],
                weight_decay=WEIGHT_DECAY / 10.0
            )

        # Training loop
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}/{NUM_EPOCHS}", ncols=120)
        for batch_idx, (imgs, graphs, labels) in pbar:
            imgs = imgs.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=(DEVICE.type == "cuda")):
                logits = model(imgs, graphs)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            pbar.set_postfix(loss=f"{(total_loss/(batch_idx+1)):.4f}")

        avg_loss = total_loss / len(train_loader)
        auc, f1 = evaluate(model, val_loader)

        print(f"\nEpoch {epoch}/{NUM_EPOCHS} | Loss={avg_loss:.4f} | AUC={auc:.4f} | F1={f1:.4f}")

        # ⬅⬅⬅ NEW: save metrics to CSV
        log_writer.writerow([epoch, avg_loss, auc, f1])
        log_file.flush()

        # Save best model
        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), "triplenet_best.pth")
            print("🔥 Saved new best model!")

    print("Training finished.")


if __name__ == "__main__":
    train()
