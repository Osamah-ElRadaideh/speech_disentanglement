import torch.nn as nn
import numpy as np
import torch

class Conv_block(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=3,stride=1,padding='same',act=True):
        super().__init__()
        self.act = act
        self.conv = nn.Conv1d(in_channels,out_channels,kernel_size=kernel_size,stride=stride
                              ,padding=padding)

        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout()

    def forward(self,x):
    
        x = self.conv(x)
        x = self.bn(x)

        if self.act:
            x = self.act(x)
            x = self.dropout(x)
        return x 
        

class Content_encoder(nn.Module):

    def __init__(
            self,
            output_size = 39,
            sampling_rate=16000, k=5,

    ):
        super().__init__()
        input_size = 80 
        hidden_size = 512
        last_act = 256
        self.sampling_rate = sampling_rate
        self.conv1 = Conv_block(input_size, hidden_size, kernel_size=k, stride=1, padding='same')
        self.conv2 = Conv_block(hidden_size, hidden_size, kernel_size=k, stride=1, padding='same')
        self.conv3 = Conv_block(hidden_size, hidden_size, kernel_size=k, stride=1, padding='same')
        self.conv4 = Conv_block(hidden_size, hidden_size, kernel_size=k, stride=1, padding='same')
        self.conv5 = Conv_block(hidden_size, hidden_size, kernel_size=k, stride=1, padding='same')
        self.conv6 = Conv_block(hidden_size, last_act, kernel_size=k, stride=1, padding='same',act=False)




    def forward(self, x):
        # x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        return x.permute(0, 2 ,1)
