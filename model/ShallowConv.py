import torch
import torch.nn as nn
import torch.nn.functional as F

class shallowconv(nn.Module):
    def __init__(self, 
                 length = 500,
                 enc_dim = 4,
                 dropout= 0, 
                 ):
        super(shallowconv, self).__init__()

        self.l_seq = length
        self.enc_dim = enc_dim

        self.conv1 = nn.Conv1d(self.enc_dim, 128, kernel_size=40, dilation=1)
        self.conv1_norm_block = nn.Sequential(
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.BatchNorm1d(128)
        )

        self.conv2 = nn.Conv1d(128, 128, kernel_size=1, dilation=1)
        self.conv2_norm_block = nn.Sequential(
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.BatchNorm1d(128)
        )

        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv1d(128, 32, kernel_size=1, dilation=1)
        self.conv3_norm_block = nn.Sequential(
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.BatchNorm1d(32)
        )

        self.conv4 = nn.Conv1d(32, 64, kernel_size=30, dilation=4)
        self.conv4_norm_block = nn.Sequential(
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.BatchNorm1d(64)
        )

        self.hidden_linear_1 = nn.Sequential(
            nn.Linear(114*64, 128),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.BatchNorm1d(128)
        )

        self.hidden_linear_2 = nn.Sequential(
            nn.Linear(128, 32),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.BatchNorm1d(32)
        )

        self.output = nn.Linear(32, 1)

    def forward(self, x):
        o = x.permute(0,2,1)
        #print(f'x shape:{o.shape}')
        o = self.conv1_norm_block(self.conv1(o))
        #print(f'conv1 shape:{o.shape}')
        o = self.conv2_norm_block(self.conv2(o))
        #print(f'conv2 shape:{o.shape}')
        o = self.pool(o)
        o = self.conv3_norm_block(self.conv3(o))
        #print(f'conv3 shape:{o.shape}')
        o = self.conv4_norm_block(self.conv4(o))
        print(f'conv4 shape:{o.shape}')
        o = o.view(x.shape[0],-1)

        o = self.hidden_linear_1(o)
        o = self.hidden_linear_2(o)

        return self.output(o)