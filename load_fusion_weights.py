# load_fusion_weights.py

import torch
from model_definitions import FuNetA
from triple_net import TripleNet

def load_fusion_weights(triple_model, funet_path):
    """
    Loads pretrained CNN and GNN weights from FuNetA into TripleNet.
    """

    print(f"\n🔄 Loading FuNetA weights from: {funet_path}")

    ckpt = torch.load(funet_path, map_location="cpu")

    # Detect checkpoint format
    if isinstance(ckpt, dict) and 'model_state' in ckpt:
        sd = ckpt['model_state']
        print("🟡 Detected checkpoint dictionary with 'model_state'")
    elif isinstance(ckpt, dict) and 'state_dict' in ckpt:
        sd = ckpt['state_dict']
        print("🟡 Detected checkpoint dictionary with 'state_dict'")
    else:
        sd = ckpt
        print("🟢 Detected raw state_dict")

    # TripleNet parameters
    target_sd = triple_model.state_dict()

    copied = 0
    skipped = 0

    for k, v in sd.items():
        if k.startswith("cnn.") or k.startswith("gnn."):
            if k in target_sd and target_sd[k].shape == v.shape:
                target_sd[k] = v
                copied += 1
            else:
                skipped += 1

    triple_model.load_state_dict(target_sd)

    print(f"✅ Copied {copied} matching CNN/GNN weights.")
    print(f"⚠️ Skipped {skipped} mismatched keys (expected for fc).")

    return triple_model


if __name__ == "__main__":
    print("\nTesting weight transfer...")

    # Create TripleNet
    model_triple = TripleNet()

    # Load weights
    load_fusion_weights(model_triple, "funet_a_full.pth")
