

# In[1]:


import os
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

import warnings
warnings.filterwarnings("ignore")


# In[2]:


test_file = "test_data_iob.csv"
val_file = "val_data_iob.csv"
train_file = "train_data_iob.csv"


# In[3]:


pretrained_model = "bert-base-uncased"
batch_size = 256
max_para_length = 128

# Check if cuda is available and set device
cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")

# Make sure you choose suitable num_worker, otherwise it will result in errors
num_workers = 8 if cuda else 0

print("Cuda = ", str(cuda), " with num_workers = ", str(num_workers),  " system version = ", sys.version)


# In[4]:


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


# In[5]:


val_data = ContextEmbeddingDataset(val_file, context = 1, pretrained_model = pretrained_model)
train_data = ContextEmbeddingDataset(train_file, context = 1, pretrained_model = pretrained_model)
test_data = ContextEmbeddingDataset(test_file, context = 1, pretrained_model = pretrained_model)

train_args = dict(shuffle=True, batch_size=batch_size, num_workers=8, pin_memory=True, drop_last=False) if cuda else dict(shuffle=True, batch_size=batch_size, drop_last=False)
train_loader = DataLoader(train_data, **train_args)

val_args = dict(shuffle=False, batch_size=batch_size, num_workers=8, pin_memory=True, drop_last=False) if cuda else dict(shuffle=False, batch_size=batch_size, drop_last=False)
val_loader = DataLoader(val_data, **val_args)


test_args = dict(shuffle=False, batch_size=batch_size, num_workers=8, pin_memory=True, drop_last=False) if cuda else dict(shuffle=False, batch_size=batch_size, drop_last=False)
test_loader = DataLoader(test_data, **test_args)


# ## Fixed Bert word Embeddings, BiLSTM encoder, Triplet Decoder

# In[6]:


class BertEmbedding(nn.Module):
  def __init__(self, pretrained_model):
    super().__init__()
    self.model = BertModel.from_pretrained(pretrained_model, output_hidden_states = True)
  

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
  def __init__(self, embed_model, decoder_output_size = 1, encoder_bilayers = 1, encoder_input_dim = 3072, encoder_hidden_size = 512, context = 1, freeze_bert = True):
    super().__init__()
    self.para_encoder = ParaEncoderForContext(bilayers = encoder_bilayers, input_dim = encoder_input_dim, hidden_size = encoder_hidden_size)
    self.para_decoder = ParaDecoderTriplet(input_size = encoder_hidden_size*2*(1+2*context))
    self.embed_model = embed_model
    #NO freeze bert embedding layer
    if (freeze_bert):
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


# ## Train and Validate Functions

# In[7]:


def train(para_model, data_loader):
  para_model.train()

  avg_loss = []
  all_predictions = []
  all_targets = []
  start = time.time()

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
    
    all_predictions.extend(torch.argmax(output.detach(), axis=1).cpu().tolist())
    all_targets.extend(y.detach().cpu().tolist())
     
    
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
  return test_scores["eval_f-1"]


# In[8]:


def validate(para_model, data_loader):
  para_model.eval()
  
  avg_loss = []
  all_predictions = []
  all_targets = []
  start = time.time()

  for i, (x, y) in enumerate(tqdm(data_loader, desc="Epoch", leave=False)):
    # optimizer.zero_grad()

    y = y.to(device)
    x = x.to(device)

    with torch.no_grad():
      output = para_model(x)

      loss = criterion(output, y.long())
      avg_loss.extend([loss.item()]*len(y))

      # output = nn.Sigmoid()(output)

      all_predictions.extend(torch.argmax(output.detach(), axis=1).cpu().tolist())
      all_targets.extend(y.detach().cpu().tolist())

    
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
    if not os.path.exists('./bert_base_triplet_iob_bert_ft/'):
        os.mkdir('./bert_base_triplet_iob_bert_ft/')

    torch.save(model.state_dict(), './bert_base_triplet_iob_bert_ft/'+'/{}model_params_{}.pth'.format(best, acc))

def load_pretrained_weights(model, pretrained_path):
    pretrained_dict = torch.load(pretrained_path)
#     pretrained_dict = {k: v for k, v in pretrained_dict.items() if k[:13] == "para_encoder."}
    # print(pretrained_dict.keys())
    model_dict = model.state_dict()
    model_dict.update(pretrained_dict) 
    model.load_state_dict(model_dict)
    return model 


# ## Main

# In[13]:


model = EncoderDecoderTriplet(embed_model = BertEmbedding(pretrained_model), freeze_bert = False)
model = load_pretrained_weights(model, 'model_model_params_0.9371448069961689.pth')

if torch.cuda.device_count() > 1:
  print("Let's use", torch.cuda.device_count(), "GPUs!")
  model = nn.DataParallel(model)
model = model.to(device)

total_params = sum(p.numel() for p in model.parameters())
trainable_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
non_trainable_total_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
print("Total params: ", total_params)
print("Trainable params: ", trainable_total_params)
print("Non Trainable params: ", non_trainable_total_params)


# In[14]:


epochs = 10
lamda = 1e-3  #L2 regularization
learning_rate = 5e-5 ## Greatly reduces LR for bert finetuning

criterion = nn.CrossEntropyLoss()
criterion = criterion.to(device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=lamda)
# optimizer.load_state_dict(torch.load('./bert_base_triplet/optimizer_model_params_0.9409211846833226.pth'))    

# scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[i for i in range(4,20,4)], gamma=0.75)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(len(train_loader) * epochs))


# In[15]:


best_val_f1 = 0
for epoch in range(epochs):
  print('Epoch #{}'.format(epoch+1))
  
  train_f1 = train(model, train_loader)
  val_f1, _ = validate(model, val_loader)
  
  if val_f1 > best_val_f1:
    best_val_f1 = val_f1
    save(model, best_val_f1, best = "model_")
    save(optimizer, best_val_f1, best = "optimizer_")


# In[ ]:


# Test on Test Set


# In[ ]:


_, predictions = validate(model, test_loader)


# In[ ]:


temp = predictions
print(len(predictions))
predictions = np.concatenate([[0], predictions, [0]])
print(len(predictions))


# In[ ]:


test_df = pd.read_csv(test_file)
test_df['predictions'] = predictions
test_df.to_csv("bert_embed_triplet_iob_pred_bert_ft.csv")


# In[ ]:


error = test_df[test_df['label'] != test_df['predictions']]
print((len(test_df)- len(error)) / len(test_df))


# In[ ]:


print(len(error))
error.to_csv("errors_bert_embed_triplet_bert_ft.csv")


# In[ ]:


pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_rows', None)


# In[ ]:


error 
# 1. errors are on table rows which contain chemical name independently (without a lot of other content) 
# just like chemical name headings for reaction starts
# 2. Reaction headings that do not contain chemical names: EXAMPLE 3. Selective deprotection of position 6.\n
# 3. Errors on B tags (label 2) 3A: Sometimes paras that are like this:
# Example 15: N-(5-(4-(5-bromo-3-methyl-2-oxo-2,3-dihydro-1H-benzo[d]imidazole -1-yl)pyrimidin-2-ylamino)-2-((2-(dimethylamino)ethyl)(methyl)amino)-4-methoxyphenyl) acrylamide hydrochloride\n
# are tagged as outer (0) or Beginning (2) in the gold standard. Model probably needs more context
# 3B : Headings (tag B (2)) such as 1H NMR of GLP-111: DMSO-d6, δ 1.56-1.57 (br, m, 9H, 3CH2), 1.61-1.63 (br, m, 2H, CH2), 2.01 (brs, 4H, CH2), 2.36 (br, 1H, NH—CH2), 3.22 (brs, 2H, CH2), 5.77 (s, 1H, OH), 6.44-6.46 (dd, 1H, Arom-H), 6.92-6.93 (dd, 1H, Arom-H) 7.06-7.09 (t, 1H, Arom-H), 7.21-7.22 (t, 1H, Arom-H), 9.40 (br, s, 1H, NH), 9.74 (br, s, 1H, NH).\n
# that contain properties of chemicals are classified as 0 because in most cases, the paras that just contain the properites
# are not tagged as reaction paras
# 4. Long paras containing a lot of chemical names are tagged as 2 even tho they are 0
# 5. Example 4m\n	 : tagged as 2 even tho they are 0
# 6. Tables inside reactions are not recognized as reactions (tagged as 0 instead of 1)


# In[ ]:




