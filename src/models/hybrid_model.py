import torch
import torch.nn as nn
from cnn import Net
from gru import BiGRUModel

class Hybrid_Model(nn.Module):
    def __init__(self, num_classes=5, gru_hidden_size=32):
        super(Hybrid_Model, self).__init__()

        self.cnn = Net(in_channels=1)

        self.bigru = BiGRUModel(
            input_size=64, 
            hidden_size=gru_hidden_size, 
            num_layers=1
        )

        fusion_dim = (19 * gru_hidden_size * 2) + 3 

        self.dense = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x_signal_cleaned, x_rr_features):

        ## Input: (Batch, 1, 300)
        cnn_features = self.cnn(x_signal_cleaned)
        ## Output: (Batch, 64, 19)

        ## Input: (Batch, 19, 64)
        bigru_features = self.bigru(cnn_features)
        ## Output: (Batch, 19 * gru_hidden_size * 2)

        ## Input: (Batch, flattened)
        bigru_rr_features = torch.cat([bigru_features, x_rr_features], dim=1)
        ## Output: (Batch, (19 * gru_hidden_size * 2) + 3)

        ## Input: (Batch, (19 * gru_hidden_size * 2) + 3)
        result = self.dense(bigru_rr_features)
        ## Output: (Batch, 5)
        
        return result