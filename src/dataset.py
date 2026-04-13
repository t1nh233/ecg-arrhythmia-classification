import torch
from torch.utils.data import Dataset

#########################################################################################################################

class ecg_arrhythmia_dataset(Dataset):
    
    def __init__(self, X_beats, X_rr_features, y):

        ## Vì đầu vào của 1D CNN (Batch, channel, length) nhưng hiện tại (Batch, Length)
        self.X_beats = torch.tensor(X_beats, dtype=torch.float32).unsqueeze(1) 
        self.X_rr = torch.tensor(X_rr_features, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)


    def __len__(self):

        return len(self.y)


    def __getitem__(self, idx):

        return self.X_beats[idx], self.X_rr[idx], self.y[idx]