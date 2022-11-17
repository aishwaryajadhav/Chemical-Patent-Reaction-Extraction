
# In[2]:


import os

os.environ["CUDA_VISIBLE_DEVICES"]="1"
import os.path as osp
import nltk
import random
# nltk.download('stopwords')
# nltk.download('punkt')
# from nltk.corpus import stopwords
# english_stopwords = stopwords.words("english")
import numpy as np
import re
import seaborn as sns
sns.set_theme(style="whitegrid")
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt
import pandas as pd
import string
import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F

import datasets
from datasets import load_dataset
import pickle
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import BertTokenizer, BertModel
import multiprocessing
import time
from torch.utils.data import DataLoader, Dataset 
import sys
from tqdm import tqdm
from utils import get_span_perf

import warnings
warnings.filterwarnings("ignore")


# In[3]:


test_file = "test_data_iob.csv"
val_file = "val_data_iob.csv"
train_file = "train_data_iob.csv"


# In[4]:


pretrained_model = "recobo/chemical-bert-uncased-pharmaceutical-chemical-classifier"
batch_size = 32
max_para_length = 128

# Check if cuda is available and set device
cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")

# Make sure you choose suitable num_worker, otherwise it will result in errors
num_workers = 8 if cuda else 0

print("Cuda = ", str(cuda), " with num_workers = ", str(num_workers),  " system version = ", sys.version)


# In[5]:


# ## Fixed Bert word Embeddings, BiLSTM encoder, Triplet Decoder

# In[7]:


class BertEmbedding(nn.Module):
  def __init__(self, pretrained_model):
    super().__init__()
    self.model = BertModel.from_pretrained(pretrained_model, output_hidden_states = True)
    # for param in self.model.bert.parameters():
    #   param.requires_grad = False
    # print(sum(p.numel() for p in self.model.parameters()))

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
            num_layers=1, batch_first=True, bidirectional=True)
    
    for name, param in self.lstm.named_parameters():
      if 'bias' in name:
        nn.init.constant(param, 0.0)
      elif 'weight' in name:
        nn.init.orthogonal(param)
     

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
  def __init__(self, embed_model, decoder_output_size = 1, encoder_bilayers = 1, encoder_input_dim = 3072, encoder_hidden_size = 512, context = 1):
    super().__init__()
    self.para_encoder = ParaEncoderForContext(bilayers = encoder_bilayers, input_dim = encoder_input_dim, hidden_size = encoder_hidden_size)
    self.para_decoder = ParaDecoderTriplet(input_size = encoder_hidden_size*2*(1+2*context))
    self.embed_model = embed_model
    #freeze bert embedding layer
    for param in self.embed_model.parameters():
      param.requires_grad = False

  def forward(self, x): # (B, 2*context+1, tokens_per_para)
    # print("Input to model: ", x.shape)
    s0, s1, s2 = x.shape
    xv = x.view(s0*s1, s2)
    embeds = self.embed_model(xv)
    para_vec = self.para_encoder(embeds)
    pvv = para_vec.view(s0, s1, -1)
    # print("Input to decoder: ", pvv.shape)
    return self.para_decoder(pvv)



model = EncoderDecoderTriplet(embed_model = BertEmbedding(pretrained_model))
# model.load_state_dict(torch.load('./bert_base_triplet_iob/model_model_params_0.9371448069961689.pth'))
# if torch.cuda.device_count() > 1:
#   print("Let's use", torch.cuda.device_count(), "GPUs!")
#   model = nn.DataParallel(model)
model = model.to(device)

total_params = sum(p.numel() for p in model.parameters())
trainable_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
non_trainable_total_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
print("Total params: ", total_params)
print("Trainable params: ", trainable_total_params)
print("Non Trainable params: ", non_trainable_total_params)




class ContextEmbeddingDataset(Dataset):
    def __init__(self, csv_file, context, pretrained_model):
      df = pd.read_csv(csv_file)

      self.context = context
      self.tokenizer = BertTokenizer.from_pretrained(pretrained_model, do_lower=True)    

      # Tokenize the paragraphs
      self.df = df["para"].apply(self.preprocess)
      self.y = df['label']
  
     
    def preprocess(self, examples):
      return self.tokenizer(examples, truncation=True, 
                     padding="max_length", max_length=max_para_length,
                     return_token_type_ids=False)['input_ids']

    def __len__(self):
      return len(self.y) - (2*self.context)
    
    def __getitem__(self,index):
      return torch.LongTensor(list(self.df[index:(index + 2*self.context+1)])), self.y[index+self.context]
      
      # self.embed_model.eval()
      # Generate BERT embeddings for the tokens in each para
      # with torch.no_grad():
      #   x = torch.LongTensor(list(self.df[index:(index + 2*self.context+1)])).to(device)
      #   print(x.shape)    
      #   outputs = self.embed_model(x)
      #   print(outputs.shape) # (3, tokens(128), input_dim(3072))
       
      # return outputs.cpu(), self.y[index+self.context]


# In[6]:


val_data = ContextEmbeddingDataset(val_file, context = 1, pretrained_model = pretrained_model)
train_data = ContextEmbeddingDataset(train_file, context = 1, pretrained_model = pretrained_model)
test_data = ContextEmbeddingDataset(test_file, context = 1, pretrained_model = pretrained_model)

train_args = dict(shuffle=True, batch_size=batch_size, num_workers=8, pin_memory=True, drop_last=False) if cuda else dict(shuffle=True, batch_size=batch_size, drop_last=False)
train_loader = DataLoader(train_data, **train_args)

val_args = dict(shuffle=False, batch_size=batch_size, num_workers=8, pin_memory=True, drop_last=False) if cuda else dict(shuffle=False, batch_size=batch_size, drop_last=False)
val_loader = DataLoader(val_data, **val_args)


test_args = dict(shuffle=False, batch_size=batch_size, num_workers=8, pin_memory=True, drop_last=False) if cuda else dict(shuffle=False, batch_size=batch_size, drop_last=False)
test_loader = DataLoader(test_data, **test_args)


# ## Train and Validate Functions

# In[8]:


def train(para_model, data_loader, device, criterion, optimizer, scheduler):
    para_model.train()
    # crf_model.train()

    avg_loss = []
    start = time.time()
    all_predictions = []
    all_targets = []
    
    for i, (x, y) in enumerate(tqdm(data_loader, desc="Epoch", leave=False)):
        optimizer.zero_grad()
        y  = y.to(device) 
        x = x.to(device)

        output = para_model(x)

        # print("Output from model: ", output.shape)  

        loss = criterion(output, y.long())
        avg_loss.extend([loss.item()]*len(y))

        # output = nn.Sigmoid()(output)

        loss.backward()
        optimizer.step()
        scheduler.step()
        
        all_predictions.extend(torch.argmax(output, axis=1).cpu().tolist())
        all_targets.extend(y.cpu().tolist())

        del y
        torch.cuda.empty_cache()
 
        
    
    end = time.time()
    avg_loss = np.mean(avg_loss)
    print('learning_rate: {}'.format(scheduler.get_last_lr()))
    print('Training loss: {:.2f}, Time: {}'.format(avg_loss, end-start))

    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    scores = precision_recall_fscore_support(all_targets, all_predictions, 
                                            average="weighted", zero_division=0.)

    test_scores={
      "eval_accuracy": (all_predictions == all_targets).sum() / len(all_predictions),
      "eval_precision": scores[0],
      "eval_recall": scores[1],
      "eval_f-1": scores[2]
    }
    print(test_scores)


def validate(para_model, data_loader, device, criterion, optimizer, scheduler):
    para_model.eval()
    # crf_model.eval()

    avg_loss = []
    all_predictions = []
    all_targets = []
    start = time.time()

    for i, (x, y) in enumerate(tqdm(data_loader, desc="Epoch", leave=False)):
        y = y.to(device)
        x = x.to(device)

        with torch.no_grad():
            output = para_model(x)

            loss = criterion(output, y.long())
            avg_loss.extend([loss.item()]*len(y))

            # output = nn.Sigmoid()(output)

            all_predictions.extend(torch.argmax(output, axis=1).cpu().tolist())
            all_targets.extend(y.cpu().tolist())

            
            del y
            torch.cuda.empty_cache()
       
      
      
    end = time.time()
    avg_loss = np.mean(avg_loss)
    print('learning_rate: {}'.format(scheduler.get_last_lr()))
    print('Validation loss: {:.2f}, Time: {}'.format(avg_loss, end-start))

    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    scores = precision_recall_fscore_support(all_targets, all_predictions, 
                                            average="weighted", zero_division=0.)

    test_scores={
      "eval_accuracy": (all_predictions == all_targets).sum() / len(all_predictions),
      "eval_precision": scores[0],
      "eval_recall": scores[1],
      "eval_f-1": scores[2]
    }
    print(test_scores)
    return test_scores["eval_f-1"], all_predictions



# In[9]:


def save(model, acc, best=""):
    if not os.path.exists('./chem_bert_base_triplet_iob_no_ft/'):
        os.mkdir('./chem_bert_base_triplet_iob_no_ft/')

    torch.save(model.state_dict(), './chem_bert_base_triplet_iob_no_ft/'+'/{}model_params_{}.pth'.format(best, acc))
    


# ## Main

# In[10]:




# In[ ]:


epochs = 20
lamda = 1e-3  #L2 regularization
learning_rate = 1e-3

criterion = nn.CrossEntropyLoss()
criterion = criterion.to(device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=lamda)
# optimizer.load_state_dict(torch.load('./bert_base_triplet/optimizer_model_params_0.9409211846833226.pth'))    

# scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[i for i in range(4,20,4)], gamma=0.75)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(len(train_loader) * epochs))


# In[ ]:


best_val_f1 = 0
val_df = pd.read_csv(val_file)

for epoch in range(epochs):
    print('Epoch #{}'.format(epoch+1))

    train(model, train_loader, device, criterion, optimizer, scheduler)
    val_f1, val_preds = validate(model, val_loader, device, criterion, optimizer, scheduler)
    try:
        get_span_perf(val_df, val_preds)
        save(model, best_val_f1, best = "model_")
        save(optimizer, best_val_f1, best = "optimizer_")
    except Exception as error:
        print(error)
    

print("Done!")