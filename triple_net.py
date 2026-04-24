# triple_net.py

import torch
import torch.nn as nn
from torch_geometric.data import Batch

# import your existing CNN/GNN branches
from model_definitions import CNNBranch, HierarchicalGNNBranch

class TripleNet(nn.Module):
    def __init__(self, fusion_dim=600, rnn_hidden=256, rnn_layers=1, bidirectional=True):
        super().__init__()

        # Reuse your CNN and GNN
        self.cnn = CNNBranch()
        self.gnn = HierarchicalGNNBranch()

        self.fusion_dim = fusion_dim

        # RNN for temporal modeling
        self.rnn = nn.LSTM(
            input_size=fusion_dim,
            hidden_size=rnn_hidden,
            num_layers=rnn_layers,
            batch_first=True,
            bidirectional=bidirectional
        )

        rnn_output_dim = rnn_hidden * (2 if bidirectional else 1)

        # Final classifier
        self.fc = nn.Linear(rnn_output_dim, 2)

    def forward(self, imgs, graphs):
        """
        imgs:   [B, T, 3, H, W]
        graphs: list of lists; graphs[b][t] = Data object for frame t
        """

        B, T, _, _, _ = imgs.shape
        device = next(self.parameters()).device

        frame_embeddings = []

        for t in range(T):
            # CNN pathway
            img_batch = imgs[:, t].to(device)      # [B, 3, H, W]
            cnn_out = self.cnn(img_batch)          # [B, 600]

            # GNN pathway (batch graphs for all samples at time t)
            data_list = []
            for b in range(B):
                g = graphs[b][t]

                # if graph is None, build a dummy graph
                if g is None:
                    from torch_geometric.data import Data
                    g = Data(
                        x=torch.zeros((1, 3072)).to(device),
                        edge_index=torch.zeros((2,0), dtype=torch.long).to(device),
                        batch=torch.zeros(1, dtype=torch.long).to(device)
                    )
                else:
                    g = g.to(device)

                data_list.append(g)

            batched_graph = Batch.from_data_list(data_list)
            gnn_out = self.gnn(batched_graph)       # [B, 600]

            # FuNetA-style fusion (same as your existing model)
            fused = cnn_out + gnn_out               # [B, 600]
            frame_embeddings.append(fused.unsqueeze(1))

        seq = torch.cat(frame_embeddings, dim=1)     # [B, T, 600]

        rnn_output, _ = self.rnn(seq)                # [B, T, hidden*2]

        final_representation = rnn_output[:, -1, :]  # last time step

        logits = self.fc(final_representation)       # [B, 2]

        return logits
