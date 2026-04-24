import torch
from triple_net import TripleNet
from load_fusion_weights import load_fusion_weights

def load_triple_model(weights_path="triplenet_best.pth", device="cpu"):
    model = TripleNet().to(device)
    model = load_fusion_weights(model, "funet_a_full.pth")  # load CNN+GNN pretrained
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    return model
