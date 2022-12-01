from torchcrf import CRF
import os
import os.path as osp
import nltk
import random
# nltk.download('stopwords')
# nltk.download('punkt')
# from nltk.corpus import stopwords
# english_stopwords = stopwords.words("english")
import numpy as np
import pandas as pd
import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
import time
from torch.utils.data import DataLoader, Dataset 
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")

class BertEmbedding(nn.Module):
  def __init__(self, pretrained_model):
    super().__init__()
    self.model = AutoModel.from_pretrained(pretrained_model, output_hidden_states = True)
  

  def forward(self, x):
    # print("Input to BertEmbedding: ", x.shape)
    outputs = self.model(x)
    hidden_states = outputs[2]
    embedding = torch.cat((hidden_states[-1],hidden_states[-2],hidden_states[-3],hidden_states[-4]), dim = 2)
    # print("Output from BertEmbedding: ", embedding.shape)
    return embedding


class ParaEncoderForContext(nn.Module):
    def __init__(self, bilayers = 1, input_dim = 3072, hidden_size = 512):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_size
        self.lstm = nn.LSTM(
                input_size=input_dim, hidden_size=hidden_size,
                num_layers=bilayers, batch_first=True, bidirectional=True)

        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                nn.init.constant(param, 0.0)
            elif 'weight' in name:
                nn.init.kaiming_normal_(param)
     

    def forward(self, x): # (B*T(T=1+2*context), tokens, input_dim)
        # print("Input to Encoder: ",x.shape)
        outputs, _ = self.lstm(x) # (B*T, tokens, 2*hidden_dim)
        # print("After LSTM: ", outputs.shape)
        first = outputs[:, 0, self.hidden_dim:]
        second = outputs[:, -1, :self.hidden_dim]
        para_embed = torch.cat((second,first), dim = 1) #(B*T, 2*hidden_dim)

        # print("Output from Encoder", para_embed.shape)
        return para_embed #(B*T, 2*hidden_dim)


class ParaDecoderTriplet(nn.Module):
  def __init__(self, input_size, output_size = 1):
    super().__init__()
    self.linear = nn.Linear(input_size, 3, bias= True)
    # self.layers = nn.Sequential(nn.Linear(input_size, output_size, bias = True), 
    #                             nn.BatchNorm1d(output_size), 
    #                             nn.ReLU(inplace = True), 
    #                             nn.Linear(output_size, 1, bias = True))
    
    # for mod in self.modules():
    #   if isinstance(mod, nn.BatchNorm1d):
    #     nn.init.constant_(mod.weight.data, 1)
    #     if(mod.bias is not None):
    #       nn.init.constant_(mod.bias.data, 0)

  def forward(self, x): # #(B, T, 2*hidden_dim)
    # print("Input to decoder: ", x.shape) 
    s0,s1,s2 = x.shape
    x = x.reshape(s0,-1) #concat main and context para embeddings
    # print("Input to linear layer in decoder:", xv.shape) 
    return self.linear(x) #(B,1)


class EncoderDecoderTriplet(nn.Module):
    def __init__(self, embed_model, decoder_output_size = 1, encoder_bilayers = 1, encoder_input_dim = 3072, encoder_hidden_size = 512, context = 1, freeze_bert = True):
        super().__init__()
        self.para_encoder = ParaEncoderForContext(bilayers = encoder_bilayers, input_dim = encoder_input_dim, hidden_size = encoder_hidden_size)
        self.para_decoder = ParaDecoderTriplet(input_size = encoder_hidden_size*2*(1+2*context))
        self.embed_model = embed_model
        #NO freeze bert embedding layer
        if (freeze_bert):
            for param in self.embed_model.parameters():
                param.requires_grad = False

  

    def extract(self, x):
        output = self.forward(x)
        return torch.argmax(output.detach(), axis=1).cpu().tolist()
        

    def forward(self, x): # (B, 2*context+1, tokens_per_para)
        # print("Input to model: ", x.shape)
        s0, s1, s2 = x.shape
        xv = x.view(s0*s1, s2)
        embeds = self.embed_model(xv)
        para_vec = self.para_encoder(embeds)
        pvv = para_vec.view(s0, s1, -1)
        # print("Input to decoder: ", pvv.shape)
        return self.para_decoder(pvv)





class ParaDecoderBiLstmCRF(nn.Module):
    def __init__(self, input_dim, hidden_size, bilayers = 1):
        super().__init__()
        # self.input_dim = input_dim
        # self.hidden_dim = hidden_size
        self.lstm = nn.LSTM(
                input_size=input_dim, hidden_size=hidden_size,
                num_layers=bilayers, batch_first=True, bidirectional=True)

        self.linear = nn.Linear(2*hidden_size, 3, bias= True)
        
        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                nn.init.constant(param, 0.0)
            elif 'weight' in name:
                nn.init.kaiming_normal_(param)

    
    def forward(self, x):  #(B, T, 2*encoder.hidden_dim)
    # print("Input to decoder: ", x.shape) 
        outputs, _ = self.lstm(x)   #out = (B, T, 2*decoder.hidden_dim)

        s0, s1, s2 = outputs.shape
        op = outputs.reshape(s0*s1, s2) # (B*T, 2*decoder.hidden_dim)

        op2 = self.linear(op)

        op3 = op2.view(s0, s1, -1)

        return op3 #(B,T,3) #emissions



class EncoderDecoderBiLstmCRF(nn.Module):
    def __init__(self, embed_model, num_tags, encoder_bilayers = 1, encoder_input_dim = 3072, encoder_hidden_size = 512, decoder_bilayers = 1, decoder_hidden_size = 512, freeze_bert = True):
        super().__init__()
        self.para_encoder = ParaEncoderForContext(bilayers = encoder_bilayers, input_dim = encoder_input_dim, hidden_size = encoder_hidden_size)
        self.para_decoder = ParaDecoderBiLstmCRF(input_dim = encoder_hidden_size*2, hidden_size = decoder_hidden_size, bilayers = decoder_bilayers)
        self.crf_model = CRF(num_tags = num_tags, batch_first = True)
        self.embed_model = embed_model
        
        if(freeze_bert):
            for param in self.embed_model.parameters():
                param.requires_grad = False

    def decode(self, emission):
        return self.crf_model.decode(emission)

    def extract(self, x):
        s0, s1, s2 = x.shape
        xv = x.view(s0*s1, s2)
        embeds = self.embed_model(xv)
        para_vec = self.para_encoder(embeds)
        pvv = para_vec.view(s0, s1, -1) #(B, T, 2*hidden_dim)
        # print("Input to decoder: ", pvv.shape)
        emission = self.para_decoder(pvv) #(B,T,3) #emissions
        return self.decode(emission) 
        


    def forward(self, x, y): # (B, 2*context+1, tokens_per_para)
    # print("Input to model: ", x.shape)
        s0, s1, s2 = x.shape
        xv = x.view(s0*s1, s2)
        embeds = self.embed_model(xv)
        para_vec = self.para_encoder(embeds)
        pvv = para_vec.view(s0, s1, -1) #(B, T, 2*hidden_dim)
        # print("Input to decoder: ", pvv.shape)
        emission = self.para_decoder(pvv) #(B,T,3) #emissions
        log_likelihood = self.crf_model(emission, y, reduction='mean') 
        return -log_likelihood, emission

#########################################################
## Para Models ###

class ParaDecoderBiLstmCRF(nn.Module):
    def __init__(self, input_dim, hidden_size, bilayers = 1):
        super().__init__()
        # self.input_dim = input_dim
        # self.hidden_dim = hidden_size
        self.lstm = nn.LSTM(
                input_size=input_dim, hidden_size=hidden_size,
                num_layers=bilayers, batch_first=True, bidirectional=True)

        self.linear = nn.Linear(2*hidden_size, 3, bias= True)
        
        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                nn.init.constant(param, 0.0)
            elif 'weight' in name:
                nn.init.kaiming_normal_(param)

    
    def forward(self, x):  #(B, T, 768)
    # print("Input to decoder: ", x.shape) 
        outputs, _ = self.lstm(x)   #out = (B, T, 2*decoder.hidden_dim)

        s0, s1, s2 = outputs.shape
        op = outputs.reshape(s0*s1, s2) # (B*T, 2*decoder.hidden_dim)

        op2 = self.linear(op)

        op3 = op2.view(s0, s1, -1)

        return op3 #(B,T,3) #emissions



class ParaEncoderDecoderBiLstmCRF(nn.Module):
    def __init__(self, num_tags, embedding_input_dim = 768, decoder_bilayers = 1, decoder_hidden_size = 512, freeze_bert = True):
        super().__init__()
        # self.para_encoder = ParaEncoderForContext(bilayers = encoder_bilayers, input_dim = encoder_input_dim, hidden_size = encoder_hidden_size)
        self.para_decoder = ParaDecoderBiLstmCRF(input_dim = embedding_input_dim, hidden_size = decoder_hidden_size, bilayers = decoder_bilayers)
        self.crf_model = CRF(num_tags = num_tags, batch_first = True)
        # self.embed_model = embed_model
        

    def decode(self, emission):
        return self.crf_model.decode(emission)

    def forward(self, x, y): # (B, para_seq_len, 768)
        emission = self.para_decoder(x) #out: (B,T,3) #emissions
        log_likelihood = self.crf_model(emission, y, reduction='mean') 
        return -log_likelihood, emission