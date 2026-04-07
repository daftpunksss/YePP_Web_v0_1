import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Linear(channel, channel // reduction, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(channel // reduction, channel, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Squeeze and Excitation block
        avg_pool = torch.mean(x, dim=-1, keepdim=False)  # Global Average Pooling
        x = self.fc1(avg_pool)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x * x


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, kernel_size):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.se_block = SEBlock(out_channels)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        out = self.se_block(out).unsqueeze(-1)
        
        st = self.shortcut(x)
        out = out + st
        out = self.relu(out)
        return out

class UTRNet(nn.Module):
    def __init__(self):
        super(UTRNet, self).__init__()

        self.res_block1 = ResBlock(4, 64, stride=2, kernel_size=9)  # stride=2, kernel_size=5
        self.res_block2 = ResBlock(64, 128, stride=4, kernel_size=9)
        self.res_block3 = ResBlock(128, 256, stride=4, kernel_size=9)
        self.res_block4 = ResBlock(256, 512, stride=4, kernel_size=9)
        self.res_block5 = ResBlock(512, 512, stride=4, kernel_size=9)
        self.res_block6 = ResBlock(512, 512, stride=4, kernel_size=9)
        self.res_block7 = ResBlock(512, 256, stride=4, kernel_size=9)
        self.res_block8 = ResBlock(256, 128, stride=4, kernel_size=9)
    
        self.fc1 = nn.Linear(512, 128)
        self.batch_norm4 = nn.BatchNorm1d(128)
        self.dropout4 = nn.Dropout(p=0.25)

        self.fc2 = nn.Linear(128, 1) 

    def forward(self, input):
        #print(input.shape)
        # ResNet part for DNA sequence
        x = input.permute(0, 2, 1) 
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = self.res_block4(x)
        # x = self.res_block5(x)
        # x = self.res_block6(x)
        # x = self.res_block7(x)
        # x = self.res_block8(x)
        x = x.permute(0, 2, 1).squeeze()
        #print(x.shape)
        x = self.fc1(x)
        x = self.batch_norm4(x)
        x = torch.relu(x)
        x = self.dropout4(x)

        x = self.fc2(x)
        
        return x