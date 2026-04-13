import torch
import torch.nn as nn

#########################################################################################################################

class Net(nn.Module):

    def __init__(self, in_channels=1):
        super(Net, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=16, kernel_size=21, padding=10),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=23, padding=11),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=25, padding=12),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )

        ## Giu nguyen outchannel=64 de dua vao BiGRU
        self.conv4 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=27, padding=13),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )


    def forward(self, x):

        x1 = self.conv1(x)
        x2 = self.conv2(x1) 
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)

        return x4