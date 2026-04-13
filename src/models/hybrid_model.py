import torch
import torch.nn as nn
from src.models.cnn import Net
from src.models.gru import BiGRUModel

#########################################################################################################################

class Hybrid_Model(nn.Module):

    def __init__(self, num_classes=5, gru_hidden_size=32, gru_num_layer=1):
        super(Hybrid_Model, self).__init__()

        self.cnn = Net(in_channels=1)

        self.bigru = BiGRUModel(
            input_size=64, 
            hidden_size=gru_hidden_size, 
            num_layers=1
        )

        ## Kich thuoc sau khi output BiGRU flatten va concat voi rr_features
        fusion_dim = (19 * gru_hidden_size * 2) + 3 

        ## Dense layer 
        self.dense = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x_signal_cleaned, x_rr_features):

        batch_size = x_signal_cleaned.size(0)

        ## Input: (Batch, 1, 300)
        cnn_features = self.cnn(x_signal_cleaned)
        ## Output: (Batch, 64, 19)

        ## Input: (Batch, 19, 64)
        bigru_features = self.bigru(cnn_features)
        ## Output: (Batch, 19, gru_hidden_size)

        ## Flatten bigru output
        ## Input: (Batch, 19, hidden_size)
        bigru_flattened = bigru_features.reshape(batch_size, -1)
        ## Output: (Batch, 18 * gru_hidden_size * 2)

        ## Input: (Batch, 19 * gru_hidden_size * 2)
        fusion_vector = torch.cat([bigru_flattened, x_rr_features], dim=1)
        ## Output: (Batch, (19 * gru_hidden_size * 2) + 3)

        ## Input: (Batch, (19 * gru_hidden_size * 2) + 3)
        result = self.dense(fusion_vector)
        ## Output: (Batch, 5)
        
        return result