import torch
import torch.nn as nn

class BiGRUModel(nn.Module):
    def __init__(self, input_size=64, hidden_size=32, num_layers=2):
        super(BiGRUModel, self).__init__()
        
        # Khởi tạo GRU với bidirectional=True 
        self.bigru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )


    def forward(self, x):

        x_permuted = x.permute(0, 2, 1) 

        out, hidden = self.bigru(x_permuted)
        
        batch_size = out.size(0)
        out_flattened = out.reshape(batch_size, -1)
        
        return out_flattened