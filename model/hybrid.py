import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridNet(nn.Module):
    def __init__(self):
        super(HybridNet, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=256, kernel_size=13, stride=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(p=0.3)
        self.batchnorm1 = nn.BatchNorm1d(256)

        self.conv2 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=13, stride=1)
        self.dropout2 = nn.Dropout(p=0.3)
        self.batchnorm2 = nn.BatchNorm1d(256)

        self.conv3 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=13, stride=1)
        self.dropout3 = nn.Dropout(p=0.3)
        self.batchnorm3 = nn.BatchNorm1d(256)
        
        self.conv1_2 = nn.Conv1d(in_channels=4, out_channels=256, kernel_size=13, stride=1)
        self.relu_2 = nn.ReLU()
        self.maxpool_2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout1_2 = nn.Dropout(p=0.3)
        self.batchnorm1_2 = nn.BatchNorm1d(256)

        self.conv2_2 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=13, stride=1)
        self.dropout2_2 = nn.Dropout(p=0.3)
        self.batchnorm2_2 = nn.BatchNorm1d(256)

        self.conv3_2 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=13, stride=1)
        self.dropout3_2 = nn.Dropout(p=0.3)
        self.batchnorm3_2 = nn.BatchNorm1d(256)
        

        # LSTM layer
        self.lstm1 = nn.LSTM(input_size=256, hidden_size=128, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=128, hidden_size=128, batch_first=True)

        # Fully connected layers
        self.fc1 = nn.Linear(in_features=128, out_features=64)
        self.dropout4 = nn.Dropout(p=0.2)
        self.batchnorm4 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(in_features=64, out_features=1)

    def forward(self, x):
        # Convolutional layers
        #print(x.shape)
        
        x = x.permute(0,2,1)
        #print(x.shape)
        x_1 = x[:,:,:-640]
        x_2 = x[:,:,-640:]
        bs, n, l = x.shape
        #print(x_1.shape)
        #print(x_2.shape)
        x_1 = self.conv1(x_1)
        x_1 = self.relu(x_1)
        x_1 = self.maxpool(x_1)
        x_1 = self.dropout1(x_1)
        x_1 = self.batchnorm1(x_1)

        x_1 = self.conv2(x_1)
        x_1 = self.relu(x_1)
        x_1 = self.maxpool(x_1)
        x_1 = self.dropout2(x_1)
        x_1 = self.batchnorm2(x_1)

        x_1 = self.conv3(x_1)
        x_1 = self.relu(x_1)
        x_1 = self.maxpool(x_1)
        x_1 = self.dropout3(x_1)
        x_1 = self.batchnorm3(x_1)
        
        # x_2 = self.conv1_2(x_2)
        # x_2 = self.relu_2(x_2)
        # x_2 = self.maxpool_2(x_2)
        # x_2 = self.dropout1_2(x_2)
        # x_2 = self.batchnorm1_2(x_2)

        # x_2 = self.conv2_2(x_2)
        # x_2 = self.relu_2(x_2)
        # x_2 = self.maxpool_2(x_2)
        # x_2 = self.dropout2_2(x_2)
        # x_2 = self.batchnorm2_2(x_2)

        # x_2 = self.conv3_2(x_2)
        # x_2 = self.relu_2(x_2)
        # x_2 = self.maxpool_2(x_2)
        # x_2 = self.dropout3_2(x_2)
        # x_2 = self.batchnorm3_2(x_2)
        
        # print(x_1.shape)
        # print(x_2.shape)
        # x = self.conv5(x)
        # x = self.relu(x)
        # x = self.maxpool(x)
        # x = self.dropout5(x)
        # x = self.batchnorm5(x)
        #print(x_1.shape)
        x_2 = x_2.reshape(bs, 256,10)
        

        # Preparing for LSTM
        x = torch.cat([x_1, x_2],dim=2)
        x = x.permute(0, 2, 1)  # (batch_size, seq_len, features)
        #x = torch.cat([x_1, x_2],dim=1)
        
        # LSTM layer
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)

        # Taking the last time step output for the fully connected layers
        x = x[:, -1, :]
        #print(x_1.shape)
        
        
        # print(x.shape)
        # Fully connected layers
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout4(x)
        x = self.batchnorm4(x)

        x = self.fc2(x)

        return x
    
    
class SCHybridNet(nn.Module):
    def __init__(self):
        super(SCHybridNet, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=256, kernel_size=13, stride=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(p=0.3)
        self.batchnorm1 = nn.BatchNorm1d(256)

        self.conv2 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=13, stride=1)
        self.dropout2 = nn.Dropout(p=0.3)
        self.batchnorm2 = nn.BatchNorm1d(256)

        self.conv3 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=13, stride=1)
        self.dropout3 = nn.Dropout(p=0.3)
        self.batchnorm3 = nn.BatchNorm1d(256)
        
        self.conv1_2 = nn.Conv1d(in_channels=4, out_channels=256, kernel_size=13, stride=1)
        self.relu_2 = nn.ReLU()
        self.maxpool_2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout1_2 = nn.Dropout(p=0.3)
        self.batchnorm1_2 = nn.BatchNorm1d(256)

        self.conv2_2 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=13, stride=1)
        self.dropout2_2 = nn.Dropout(p=0.3)
        self.batchnorm2_2 = nn.BatchNorm1d(256)

        self.conv3_2 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=13, stride=1)
        self.dropout3_2 = nn.Dropout(p=0.3)
        self.batchnorm3_2 = nn.BatchNorm1d(256)
        

        # LSTM layer
        self.lstm1 = nn.LSTM(input_size=256, hidden_size=128, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=128, hidden_size=128, batch_first=True)

        # Fully connected layers
        self.fc1 = nn.Linear(in_features=128, out_features=64)
        self.dropout4 = nn.Dropout(p=0.2)
        self.batchnorm4 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(in_features=64, out_features=1)

    def forward(self, x):
        # Convolutional layers
        #print(x.shape)
        
        x = x.permute(0,2,1)
        #print(x.shape)
        # x_1 = x[:,:,:-640]
        # x_2 = x[:,:,-640:]
        # bs, n, l = x.shape
        #print(x_1.shape)
        #print(x_2.shape)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.dropout1(x)
        x = self.batchnorm1(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.dropout2(x)
        x = self.batchnorm2(x)

        x = self.conv3(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.dropout3(x)
        x = self.batchnorm3(x)
        
        #print(x_1.shape)
        # x = x.reshape(bs, 256,10)
        

        # Preparing for LSTM
        # x = torch.cat([x_1, x_2],dim=2)
        x = x.permute(0, 2, 1)  # (batch_size, seq_len, features)
        #x = torch.cat([x_1, x_2],dim=1)
        
        # LSTM layer
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)

        # Taking the last time step output for the fully connected layers
        x = x[:, -1, :]
        #print(x_1.shape)
        
        
        # print(x.shape)
        # Fully connected layers
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout4(x)
        x = self.batchnorm4(x)

        x = self.fc2(x)

        return x
    
class CssHybridNet(nn.Module):
    def __init__(self):
        super(CssHybridNet, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=256, kernel_size=13, stride=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(p=0.3)
        self.batchnorm1 = nn.BatchNorm1d(256)

        self.conv2 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=13, stride=1)
        self.dropout2 = nn.Dropout(p=0.3)
        self.batchnorm2 = nn.BatchNorm1d(256)

        self.conv3 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=13, stride=1)
        self.dropout3 = nn.Dropout(p=0.3)
        self.batchnorm3 = nn.BatchNorm1d(256)
        
        self.conv1_2 = nn.Conv1d(in_channels=4, out_channels=256, kernel_size=13, stride=1)
        self.relu_2 = nn.ReLU()
        self.maxpool_2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout1_2 = nn.Dropout(p=0.3)
        self.batchnorm1_2 = nn.BatchNorm1d(256)

        self.conv2_2 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=13, stride=1)
        self.dropout2_2 = nn.Dropout(p=0.3)
        self.batchnorm2_2 = nn.BatchNorm1d(256)

        self.conv3_2 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=13, stride=1)
        self.dropout3_2 = nn.Dropout(p=0.3)
        self.batchnorm3_2 = nn.BatchNorm1d(256)
        

        # LSTM layer
        self.lstm1 = nn.LSTM(input_size=256, hidden_size=128, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=128, hidden_size=128, batch_first=True)

        # Fully connected layers
        self.fc1 = nn.Linear(in_features=128, out_features=64)
        self.dropout4 = nn.Dropout(p=0.2)
        self.batchnorm4 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(in_features=64, out_features=1)

    def forward(self, x):
        # Convolutional layers
        #print(x.shape)
        
        x = x.permute(0,2,1)
        x = x[:,:,:-640]
        #print(x.shape)
        # x_1 = x[:,:,:-640]
        # x_2 = x[:,:,-640:]
        bs, n, l = x.shape
        #print(x_1.shape)
        #print(x_2.shape)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.dropout1(x)
        x = self.batchnorm1(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.dropout2(x)
        x = self.batchnorm2(x)

        x = self.conv3(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.dropout3(x)
        x = self.batchnorm3(x)
        
        # x_2 = self.conv1_2(x_2)
        # x_2 = self.relu_2(x_2)
        # x_2 = self.maxpool_2(x_2)
        # x_2 = self.dropout1_2(x_2)
        # x_2 = self.batchnorm1_2(x_2)

        # x_2 = self.conv2_2(x_2)
        # x_2 = self.relu_2(x_2)
        # x_2 = self.maxpool_2(x_2)
        # x_2 = self.dropout2_2(x_2)
        # x_2 = self.batchnorm2_2(x_2)

        # x_2 = self.conv3_2(x_2)
        # x_2 = self.relu_2(x_2)
        # x_2 = self.maxpool_2(x_2)
        # x_2 = self.dropout3_2(x_2)
        # x_2 = self.batchnorm3_2(x_2)
        
        # print(x_1.shape)
        # print(x_2.shape)
        # x = self.conv5(x)
        # x = self.relu(x)
        # x = self.maxpool(x)
        # x = self.dropout5(x)
        # x = self.batchnorm5(x)
        #print(x_1.shape)
        #x_2 = x_2.reshape(bs, 256,10)
        

        # Preparing for LSTM
        #x = torch.cat([x_1, x_2],dim=2)
        x = x.permute(0, 2, 1)  # (batch_size, seq_len, features)
        #x = torch.cat([x_1, x_2],dim=1)
        
        # LSTM layer
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)

        # Taking the last time step output for the fully connected layers
        x = x[:, -1, :]
        #print(x_1.shape)
        
        
        # print(x.shape)
        # Fully connected layers
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout4(x)
        x = self.batchnorm4(x)

        x = self.fc2(x)

        return x
    
import torch
import torch.nn as nn

class DeepExpressionNet(nn.Module):
    def __init__(self):
        super(DeepExpressionNet, self).__init__()

        # 卷积层部分 (CNN)
        # 第一层卷积：kernel_size=10, filters=32
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=32, kernel_size=40, stride=1, dilation=2)
        self.batch_norm1 = nn.BatchNorm1d(32)
        self.dropout1 = nn.Dropout(p=0.25)  # dropout rate

        # 第二层卷积：kernel_size=20, filters=64
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=30, stride=1, dilation=2)
        self.batch_norm2 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(p=0.25)

        # 第三层卷积：kernel_size=30, filters=128
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=20, stride=1, dilation=2)
        self.batch_norm3 = nn.BatchNorm1d(128)
        self.dropout3 = nn.Dropout(p=0.25)

        # 池化层
        self.pool = nn.MaxPool1d(kernel_size=4, stride=2)

        # 全连接层 (FC)
        self.fc1 = nn.Linear(23912, 128)  # Dense layer, output size 64, 128*80是卷积层输出展平后的大小
        self.batch_norm4 = nn.BatchNorm1d(128)
        self.dropout4 = nn.Dropout(p=0.25)

        self.fc2 = nn.Linear(128, 1)  # 输出层，回归任务，输出为1

    def forward(self, input):
        #x = input
        x = input[:, :500, :]  # 获取前500维，DNA序列部分
        #print(x.shape)
        cod = input[:, 500:, :]  # 获取最后17维，codon_frequencies部分
        #emb = input[:, 518:, :]
        
        #卷积层1
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = torch.relu(x)
        x = self.dropout1(x)

        # 卷积层2
        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = torch.relu(x)
        x = self.dropout2(x)

        # 卷积层3
        x = self.conv3(x)
        x = self.batch_norm3(x)
        x = torch.relu(x)
        x = self.dropout3(x)

        # 池化层
        x = self.pool(x)
        # print(x.shape)
        # 展平层，将卷积层的输出展平
        x = x.view(x.size(0), -1)  # 展平后，batch_size * feature_size
        cod = cod.view(cod.size(0), -1)
        #print(x.shape)
        #print(cod.shape)
        x = torch.cat([x, cod],dim=1)
        #print(x.shape)
        # 全连接层
        x = self.fc1(x)
        x = self.batch_norm4(x)
        x = torch.relu(x)
        x = self.dropout4(x)

        x = self.fc2(x)
        
        return x
    
import torch
import torch.nn as nn

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
        
        # Add SEBlock after second convolution layer
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
        
        # Apply SEBlock to adjust channel weights
        out = self.se_block(out).unsqueeze(-1)
        
        # Add the shortcut connection
        st = self.shortcut(x)
        out = out + st
        out = self.relu(out)
        return out


class ResNTEmbeddingNet(nn.Module):
    def __init__(self):
        super(ResNTEmbeddingNet, self).__init__()

        ## 添加深度ResNet Block用于提取启动子部分的特征
        self.res_block1 = ResBlock(4, 64, stride=2, kernel_size=9)  # stride=2, kernel_size=5
        self.res_block2 = ResBlock(64, 128, stride=4, kernel_size=9)
        self.res_block3 = ResBlock(128, 256, stride=4, kernel_size=9)
        self.res_block4 = ResBlock(256, 512, stride=4, kernel_size=9)
        #self.res_block5 = ResBlock(512, 512, stride=4, kernel_size=9, padding=9//2)

        self.fc1 = nn.Linear(4104, 128)  # Dense layer, output size 128
        self.batch_norm4 = nn.BatchNorm1d(128)
        self.dropout4 = nn.Dropout(p=0.25)

        self.fc2 = nn.Linear(128, 1)  # 输出层，回归任务，输出为1

    def forward(self, input):
        x = input[:, :500, :]  # 获取前500维，DNA序列部分
        cod = input[:, 500:, :]  # 获取最后17维，codon_frequencies部分

        # ResNet part for DNA sequence
        x = x.permute(0, 2, 1)  # 改变维度以适应1D卷积输入，(batch_size, channels, length)
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = self.res_block4(x)
        #x = self.res_block5(x)
        #x = self.res_block6(x)
        #print(x.shape)
        x = x.view(x.size(0), -1)  # 展平后，batch_size * feature_size
        cod = cod.view(cod.size(0), -1)
        
        x = torch.cat([x, cod], dim=1)

        x = self.fc1(x)
        x = self.batch_norm4(x)
        x = torch.relu(x)
        x = self.dropout4(x)

        x = self.fc2(x)
        
        return x
    
class TENet(nn.Module):
    def __init__(self):
        super(TENet, self).__init__()

        self.res_block1 = ResBlock(4, 64, stride=2, kernel_size=9)  # stride=2, kernel_size=5
        self.res_block2 = ResBlock(64, 128, stride=4, kernel_size=9)
        self.res_block3 = ResBlock(128, 256, stride=4, kernel_size=9)
        self.res_block4 = ResBlock(256, 512, stride=4, kernel_size=9)
        self.res_block5 = ResBlock(512, 512, stride=4, kernel_size=9)
        self.res_block6 = ResBlock(512, 512, stride=4, kernel_size=9)

        self.fc1 = nn.Linear(272, 128)
        self.batch_norm4 = nn.BatchNorm1d(128)
        self.dropout4 = nn.Dropout(p=0.25)

        self.fc2 = nn.Linear(128, 1) 

    def forward(self, input):
        #print(input.shape)
        # x = input[:, :, :100]
        cod = input[:, :, :] 
        

        # ResNet part for DNA sequence
        #x = x.permute(0, 2, 1) 
        # x = self.res_block1(x)
        # x = self.res_block2(x)
        # x = self.res_block3(x)
        # x = self.res_block4(x)
        # x = self.res_block5(x)
        # x = self.res_block6(x)
        # # print(x.shape)
        # # print(cod.shape)
        # x = x.reshape(x.size(0), -1)     # [256, 512 × 2 = 1024]
        cod = cod.reshape(cod.size(0), -1)  # [256, 4 × 768 = 3072]

        #x = torch.cat([x, cod], dim=1)

        x = self.fc1(cod)
        x = self.batch_norm4(x)
        x = torch.relu(x)
        x = self.dropout4(x)

        x = self.fc2(x)
        
        return x
    
    
class FFNet(nn.Module):
    def __init__(self):
        super(FFNet, self).__init__()

        self.fc1 = nn.Linear(272, 128)
        self.batch_norm4 = nn.BatchNorm1d(128)
        self.dropout4 = nn.Dropout(p=0.25)

        self.fc2 = nn.Linear(128, 1) 

    def forward(self, input):
        
        x = self.fc1(input)
        x = self.batch_norm4(x)
        x = torch.relu(x)
        x = self.dropout4(x)

        x = self.fc2(x)
        
        return x



