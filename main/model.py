import torch.nn as nn
import numpy as np
import torch
from speaker_encoder import Speaker_encoder, Conv_block as C2D
from content_encoder import Content_encoder

class Conv_block(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=3,stride=1,padding='same',act=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.act = act
        self.conv = nn.Conv1d(in_channels,out_channels,kernel_size=kernel_size,stride=stride
                              ,padding=padding)

        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.LeakyReLU(inplace=True, negative_slope=0.1)
        self.dropout = nn.Dropout()

    def forward(self,x):
        res = x
        x = self.conv(x)
        x = self.bn(x)

        if self.act:
            x = self.act(x)
            x = self.dropout(x)
        if self.in_channels == self.out_channels:
            x += res
        return x


class Generator(nn.Module):
    def __init__(self,mel_bins=80, hidden_dim=2048):
        super().__init__()
        self.content_enc = Content_encoder()
        self.speaker_enc = Speaker_encoder()
        self.conv1 = Conv_block(512,hidden_dim)
        self.conv2 = Conv_block(hidden_dim,hidden_dim)
        self.conv3 = Conv_block(hidden_dim,hidden_dim)
        self.conv4 = Conv_block(hidden_dim,hidden_dim)
        self.conv5 = Conv_block(hidden_dim,hidden_dim)
        self.conv6 = Conv_block(hidden_dim,hidden_dim // 2)
        self.conv7 = Conv_block(hidden_dim // 2,hidden_dim // 4)
        self.conv8 = Conv_block(hidden_dim // 4, mel_bins, act=False)
       
        
       
    def stack(self, content_batch, dvec_batch):
        # append dvectors framewise to each element in it's batch
        stacked = []
        for index, batch in enumerate(content_batch):
            dvec = dvec_batch[index]
            b = []
            for c in batch:
                b.append(torch.cat([c, dvec]))
            stacked.append(torch.stack(b))
        return torch.stack(stacked)
    
    def get_embeddings(self, x):
        return self.speaker_enc(x)
    
    def get_content (self, x):
        return self.content_enc(x)
    
    def forward(self, spec1, spec2=None):
        if spec2 is None:
            spec2 = spec1
        content = self.content_enc(spec1)
        embs = self.speaker_enc(spec2)
        x = self.stack(content,embs)
        x = x.permute(0 , 2, 1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
     
        return x
    



class Discriminator(nn.Module):
    def __init__(self, h=64):
        super().__init__()
        self.conv1 = C2D(1, h,kernel_size=5,)
        self.conv2 = C2D(h, h * 2,padding=0, kernel_size=5)
        self.conv3 = C2D( h * 2, h * 4, kernel_size=5)
        self.conv4 = C2D(h * 4, h * 8,padding=0, kernel_size=2)
        self.fc = nn.Linear(h * 8, 1)
    def forward(self,x):
        features = []
        x = x.unsqueeze(dim=1)
        x = self.conv1(x)
        features.append(x)
        x = self.conv2(x)
        features.append(x)
        x = self.conv3(x)
        features.append(x)
        x = self.conv4(x)
        features.append(x)
        x = x.view(x.shape[0], -1)
        
        return self.fc(x), features
    


def gen_loss(fake_outs):
    loss = 0.5 * torch.mean((fake_outs - 1) ** 2)

    return loss


def disc_loss(real_outs, fake_outs):
    d_loss = 0.5 * torch.mean((real_outs - 1)**2)
    g_loss = 0.5 * torch.mean(fake_outs ** 2)
    return d_loss + g_loss

    

