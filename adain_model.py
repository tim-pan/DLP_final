from io import open
import unicodedata
import string
import re
import random
import time
import math
import torch
import torch.nn as nn
from torch import optim
import torchvision
import torch.nn.functional as F
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import numpy as np
from os import system
from torchsummary import summary
'''
in this file, the output of decoder and adain layer are list!
'''
#the 'f' in paper(encoder)
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.net = torchvision.models.vgg19(pretrained=True)
        self.net = self.net.features[:21]

        self.net = nn.ModuleList([*self.net])#用list才能插入

        self.net = nn.Sequential(*self.net)
        for p in self.net.parameters():
            p.requires_grad = False
            
            
        self.layer_name_mapping = [1, 6, 11, 20]

    def forward(self, x, only_last):
        outs = []
        for name, module in self.net._modules.items():
            #here name is the number in sequential, so in fact name is a number
            x = module(x)
            if int(name) in self.layer_name_mapping:
                outs.append(x)
        if only_last:
            return [outs[-1]]#return a list of[(bs, c, h, w)]
        else:
            return outs#return a list of[(bs, c, h, w), ..., (bs, c, h, w)]

#middle layer(adain)
class adain_layer(nn.Module):
    def __init__(self):
        super(adain_layer, self).__init__()

    def IN(self, x):#input is a (bs, c, h, w) matrix
        bs, c, h, w = x.size()
        x = x.view(bs, c, -1)
        mu_x = x.mean(dim=2).view(bs, c, 1, 1)#(bs, c, 1, 1)
        std_x = x.std(dim=2).view(bs, c, 1, 1)#(bs, c, 1 ,1)
        return mu_x, std_x

    def adain(self, content, style, eps = 1e-8):#c:a fetaure map of content shape(bs, c, h, w)
        bs, c, h, w = content.size()
        size = [bs, c, h, w]
        mu_c, std_c = self.IN(content)#(bs, c, 1, 1)
        mu_s, std_s = self.IN(style)#(bs, c, 1, 1)
        ada = std_s * ((content - mu_c) / (std_c + eps)) + mu_s
        return ada#(bs, c, h, w)

    def forward(self, feat_maps_c, feat_maps_s):
        '''
        input:
        feat_maps_c(s): a list with len=4(4 layers), and 
        each element with shape(bs, c, h, w)

        return:
        outs:a list with len=4(4 layers), and 
        each element with shape(h, w)
        [(bs, h, w), (bs, h, w), (bs, h, w), (bs, h, w)]
        '''
        #input[bs, c, h, w], please squeeze matrix before use
        outs = []

        for feat_map_c, feat_map_s in zip(feat_maps_c, feat_maps_s):
            #calculate the outcome layer by layer
            out = self.adain(feat_map_c, feat_map_s)
            outs.append(out)

        return outs# a list
        #outs [(bs, c, h, w), (bs, c, h, w), (bs, c, h, w), (bs, c, h, w)] for 4 layers
   
    
#the 'g' in decoder
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.layer1 = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=0), 
            nn.ReLU(inplace=True)   
        )

        self.layer2 = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=0), 
            nn.ReLU(inplace=True),
            
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=0), 
            nn.ReLU(inplace=True),
            
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=0), 
            nn.ReLU(inplace=True),
            
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=0), 
            nn.ReLU(inplace=True),
        )


        self.layer3 = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=0), 
            nn.ReLU(inplace=True),
            
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=0), 
            nn.ReLU(inplace=True),
        )

        self.layer4 = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=0), 
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 3, kernel_size=(3, 3), stride=(1, 1), padding=0), 
#             nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.layer1(x)
        x = F.interpolate(x, scale_factor=2)

        x = self.layer2(x)
        x = F.interpolate(x, scale_factor=2)

        x = self.layer3(x)
        x = F.interpolate(x, scale_factor=2)
        
        out = self.layer4(x)

        return out

    
class StyleTranserNetwork(nn.Module):
    '''
    this model combine encoder, adain, decoder in the model
    '''
    def __init__(self):
        super(StyleTranserNetwork, self).__init__()
        self.encoder = Encoder()
        self.adain_layer = adain_layer()
        self.decoder = Decoder()
        
    def content_loss(self, out_features, t):
        '''
        out-features=f(g(t))=>decode t and then encode t
        , thus we have f(g(t))
        The content loss is the Euclidean distance 
        between the target features and the 
        features of the output image.
        '''   
        return F.mse_loss(out_features, t)

    def style_loss(self, content_feats, style_feats):
        '''
        input:
        content_feats is a list with feat extraction from 4 layers
        style_feats is also a list with feat extraction from 4 layers
        every feats of 4 layers
        
        let mean c close to mean s, and the same is std
        '''
        loss = 0
        for c, s in zip(content_feats, style_feats):
            mu_c, std_c = self.adain_layer.IN(c)
            mu_s, std_s = self.adain_layer.IN(s)
            loss_mix = F.mse_loss(mu_c, mu_s) + F.mse_loss(std_c, std_s)
            loss += loss_mix

        return loss
    
    
    def forward(self, c_imgs, s_imgs, alpha=1, lam=10):
        clist, slist = self.encoder(c_imgs, only_last=True), self.encoder(s_imgs, only_last=True)
        #return two list with len = 1 both, because the para only_last=true
        outlist = self.adain_layer(clist, slist)
        ada = outlist[0]
        t =(1 - alpha) * clist[0] + alpha * ada
        out = self.decoder(t)
        fgt = self.encoder(out, only_last=True)  # a list
        c_loss = self.content_loss(fgt[0], t)#fgt[0] is just because it is a list
        
        
        c_middle, s_middle = self.encoder(out, only_last = False), self.encoder(s_imgs, only_last = False)
#         c_middle, s_middle = self.encoder(c_imgs, only_last = False), self.encoder(s_imgs, only_last = False)

        #return two list with len = 4 both, because the para only_last=false
        s_loss = self.style_loss(c_middle, s_middle)
        
        loss = c_loss + lam * s_loss
        
        return loss, c_loss, s_loss
    
    def generate(self, c_imgs, s_imgs, alpha=1):
        clist, slist = self.encoder(c_imgs, only_last=True), self.encoder(s_imgs, only_last=True)
        #return two list with len = 1 both, because the para only_last=true
        outlist = self.adain_layer(clist, slist)
        t = outlist[0]
        t =(1 - alpha) * clist[0] + alpha * t
        out = self.decoder(t)
#         print(out)
        return out
        


     