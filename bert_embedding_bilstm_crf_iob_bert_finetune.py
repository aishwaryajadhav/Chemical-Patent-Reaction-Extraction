

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
import math
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
batch_size = 49
max_para_length = 128
para_seq_len = 16  #number of paras to be encoded and decoded together (hyperparameter)
# Check if cuda is available and set device
cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")

# Make sure you choose suitable num_worker, otherwise it will result in errors
num_workers = 8 if cuda else 0

print("Cuda = ", str(cuda), " with num_workers = ", str(num_workers),  " system version = ", sys.version)


# In[4]:


class CRFEmbeddingDataset(Dataset):
    def __init__(self, csv_file, para_seq_len, pretrained_model, stride = 1,  is_test=False):
      df = pd.read_csv(csv_file)

      self.para_seq_len = para_seq_len
      self.tokenizer = BertTokenizer.from_pretrained(pretrained_model, do_lower=True)    

      # Tokenize the paragraphs
      self.df = df["para"].apply(self.preprocess)
      self.y = df['label']
      # self.test = is_test
      if(is_test):
        self.stride = self.para_seq_len
      else:
        self.stride = stride
  
     
    def preprocess(self, examples):
      return self.tokenizer(examples, truncation=True, 
                     padding="max_length", max_length=max_para_length,
                     return_token_type_ids=False)['input_ids']

    def __len__(self):
      # if(self.test):
      #   # print(math.ceil(len(self.y)/self.para_seq_len))
      #   return math.ceil(len(self.y)/self.para_seq_len)
      l = math.ceil((len(self.y) - self.para_seq_len + 1) / self.stride)
      # print(len(self.y))  
      # print(l)
      return l
    
    def __getitem__(self,index):
      return torch.LongTensor(list(self.df[index*self.stride: (index*self.stride + self.para_seq_len)])), torch.LongTensor(self.y[index*self.stride: (index*self.stride + self.para_seq_len)].tolist())
      


# In[5]:


train_data = CRFEmbeddingDataset(train_file, para_seq_len = para_seq_len, pretrained_model = pretrained_model, stride = 2)
val_data = CRFEmbeddingDataset(val_file, para_seq_len = para_seq_len, pretrained_model = pretrained_model, is_test= True)
test_data = CRFEmbeddingDataset(test_file, para_seq_len = para_seq_len, pretrained_model = pretrained_model, is_test= True)

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



class ParaDecoderBiLstmCRF(nn.Module):
  def __init__(self, input_dim, hidden_size, bilayers = 1):
    super().__init__()
    # self.input_dim = input_dim
    # self.hidden_dim = hidden_size
    self.lstm = nn.LSTM(
            input_size=input_dim, hidden_size=hidden_size,
            num_layers=1, batch_first=True, bidirectional=True)
    
    for name, param in self.lstm.named_parameters():
      if 'bias' in name:
        nn.init.constant(param, 0.0)
      elif 'weight' in name:
        nn.init.orthogonal(param)
     
    self.linear = nn.Linear(2*hidden_size, 3, bias= True)
    
    
  def forward(self, x):  #(B, T, 2*encoder.hidden_dim)
    # print("Input to decoder: ", x.shape) 
    outputs, _ = self.lstm(x)   #out = (B, T, 2*decoder.hidden_dim)
    
    s0, s1, s2 = outputs.shape
    op = outputs.reshape(s0*s1, s2) # (B*T, 2*decoder.hidden_dim)
    
    op2 = self.linear(op)
    
    op3 = op2.view(s0, s1, -1)
    
    return op3 #(B,T,3) #emissions



class EncoderDecoderBiLstmCRF(nn.Module):
  def __init__(self, embed_model, encoder_bilayers = 1, encoder_input_dim = 3072, encoder_hidden_size = 512, decoder_bilayers = 1, decoder_hidden_size = 512):
    super().__init__()
    self.para_encoder = ParaEncoderForContext(bilayers = encoder_bilayers, input_dim = encoder_input_dim, hidden_size = encoder_hidden_size)
    self.para_decoder = ParaDecoderBiLstmCRF(input_dim = encoder_hidden_size*2, hidden_size = decoder_hidden_size, bilayers = decoder_bilayers)
    # self.crf_model = crf_model
    self.embed_model = embed_model
    #DO NOT freeze bert embedding layer : Fine Tune BERT
    # for param in self.embed_model.parameters():
    #   param.requires_grad = False

  def decode(self, emissions):
    return self.crf_model.decode(emissions)

  def forward(self, x): # (B, 2*context+1, tokens_per_para)
    # print("Input to model: ", x.shape)
    s0, s1, s2 = x.shape
    xv = x.view(s0*s1, s2)
    embeds = self.embed_model(xv)
    para_vec = self.para_encoder(embeds)
    pvv = para_vec.view(s0, s1, -1) #(B, T, 2*hidden_dim)
    # print("Input to decoder: ", pvv.shape)
    decoder_result = self.para_decoder(pvv) #(B,T,3) #emissions
    return decoder_result


# ## Train and Validate Functions

# In[7]:


def train(para_model, crf_model, data_loader):
  para_model.train()
  # crf_model.train()
    
  avg_loss = []
  # all_predictions = []
  # all_targets = []
  start = time.time()

  for i, (x, y) in enumerate(tqdm(data_loader, desc="Epoch", leave=False)):
    optimizer.zero_grad()
    y  = y.to(device) 
    x = x.to(device)

    emission = para_model(x) 
    del x

    log_likelihood = crf_model(emission, y, reduction='mean') 
    loss = -log_likelihood

    avg_loss.extend([loss.item()]*len(y))

    loss.backward()
    optimizer.step()
    scheduler.step()
   
    del y
    del emission
    torch.cuda.empty_cache()
    
   
 
    #do not decode during training to save time
#     decoded_list = crf_model.decode(output)
#     for l in decoded_list:
#         all_predictions.extend(l)
        
#     all_targets.extend(torch.flatten(y).cpu().tolist())
    
    
    
  end = time.time()
  avg_loss = np.mean(avg_loss)
  print('learning_rate: {}'.format(scheduler.get_last_lr()))
  print('Training loss: {:.2f}, Time: {}'.format(avg_loss, end-start))
  
#   all_predictions = np.array(all_predictions)
#   # print(all_predictions.shape)
#   all_targets = np.array(all_targets)
#   scores = precision_recall_fscore_support(all_targets, all_predictions, 
#                                             average="weighted", zero_division=0.)
  
#   test_scores={
#       "eval_accuracy": (all_predictions == all_targets).sum() / len(all_predictions),
#       "eval_precision": scores[0],
#       "eval_recall": scores[1],
#       "eval_f-1": scores[2]
#   }
#   print(test_scores)
#   return test_scores["eval_f-1"]


# In[8]:


def validate(para_model, crf_model, data_loader):
  para_model.eval()
  # crf_model.eval()
  
  avg_loss = []
  all_predictions = []
  all_targets = []
  start = time.time()

  for i, (x, y) in enumerate(tqdm(data_loader, desc="Epoch", leave=False)):
    # optimizer.zero_grad()

    y = y.to(device)
    x = x.to(device)

    with torch.no_grad():
      emissions = para_model(x)
      del x
      log_likelihood = crf_model(emissions, y)  #think of crf as softmax_cross_entropy_loss (activation + loss)
      loss = -log_likelihood
  
      avg_loss.extend([loss.item()]*len(y))

      decoded_list = crf_model.decode(emissions)
      for l in decoded_list:
        all_predictions.extend(l)
      
      all_targets.extend(torch.flatten(y).cpu().tolist())
      del emissions
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
    if not os.path.exists('./bert_iob_bilstm_crf_bert_finetune/'):
        os.mkdir('./bert_iob_bilstm_crf_bert_finetune/')

    torch.save(model.state_dict(), './bert_iob_bilstm_crf_bert_finetune/'+'/{}model_params_{}.pth'.format(best, acc))

def load_pretrained_weights(model, pretrained_path):
    pretrained_dict = torch.load(pretrained_path)
#     pretrained_dict = {k: v for k, v in pretrained_dict.items() if k[:13] == "para_encoder."}
    print(pretrained_dict.keys())
    model_dict = model.state_dict()
    model_dict.update(pretrained_dict) 
    model.load_state_dict(model_dict)
    return model    


# ## Main

# In[10]:


os.system("export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5")


# In[11]:


torch.cuda.empty_cache()
crf_model = CRF(num_tags = 3, batch_first = True)
crf_model = crf_model.to(device)
# model.load_state_dict(torch.load('./bert_iob_bilstm_crf/model_model_params_0.9428545098368426.pth'))
# model = load_pretrained_weights(model, '/home/anjadhav/Chemical-Patent-Reaction-Extraction/model_model_params_0.9428545098368426.pth')

model = EncoderDecoderBiLstmCRF(embed_model = BertEmbedding(pretrained_model))
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


# In[12]:


epochs = 20 #changed from 10
lamda = 1e-3  #L2 regularization (prev : 1e-4)
learning_rate = 1e-2 #changed from 1e-2 

# criterion = nn.CrossEntropyLoss()
# criterion = criterion.to(device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=lamda)
# optimizer.load_state_dict(torch.load('./bert_base_triplet/optimizer_model_params_0.9409211846833226.pth'))    

# scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[i for i in range(4,20,4)], gamma=0.75)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(len(train_loader) * epochs))


# In[13]:


torch.cuda.empty_cache()
best_val_f1 = 0
for epoch in range(epochs):
  print('Epoch #{}'.format(epoch+1))
  
  train(model, crf_model, train_loader)
  val_f1, _ = validate(model,  crf_model, val_loader)
  
  if val_f1 > best_val_f1:
    best_val_f1 = val_f1
    save(model, best_val_f1, best = "enc_dec_model_")
    save(crf_model, best_val_f1, best = "crf_model_")
    save(optimizer, best_val_f1, best = "optimizer_")


# In[ ]:


# Test on Test Set


# In[ ]:


_, predictions = validate(model, test_loader)


# In[ ]:


# Store predictions


# In[74]:


test_df = pd.read_csv(test_file)
print(len(test_df))
test_df = test_df[:][:len(predictions)]
test_df['predictions'] = predictions
test_df.to_csv("bert_embed_iob_bilstm_crf_pred.csv")
print(len(test_df))


# In[75]:


# Span Retrieval Results
test_df = test_df.reset_index(drop=False)
print(test_df.columns)


# In[76]:


test_df.columns = ['index', 'para', 'label', 'document', 'predictions']

orig = set()
i = 0
while i < len(test_df):
    if(test_df['label'][i] == 2):
        st = test_df['index'][i]
        i +=1
        while(i < len(test_df) and test_df['label'][i] == 1):
            i+=1
        orig.add((st, i-1))
    else:
        i+=1

pred = set()
i = 0
while i < len(test_df):
    if(test_df['predictions'][i] == 2):
        st = test_df['index'][i]
        i +=1
        while(i < len(test_df) and test_df['predictions'][i] == 1):
            i+=1
        pred.add((st, i-1))
    else:
        i+=1
        
strict_match_spans = orig.intersection(pred)
fuzzy_cnt = 0
for o in orig:
    if ((o in pred) or ((o[0]+1,o[1]) in pred) or ((o[0]+1,o[1]-1) in pred) or ((o[0]+1,o[1]+1) in pred) 
        or ((o[0]-1,o[1]) in pred) or ((o[0]-1,o[1]+1) in pred) or ((o[0]-1,o[1]-1) in pred) or ((o[0],o[1]+1) in pred)
        or ((o[0],o[1]-1) in pred)):
        fuzzy_cnt+=1
  

miss_start_end = 0
miss_start = 0
miss_end = 0

for o in orig:
    if(o in pred):
        continue 
    elif(((o[0]-1,o[1]+1) in pred) or ((o[0]-1,o[1]-1) in pred) or ((o[0]+1,o[1]-1) in pred) or ((o[0]+1,o[1]+1) in pred)):
        miss_start_end += 1
    elif(((o[0]+1,o[1]) in pred) or ((o[0]-1,o[1]) in pred)):
        miss_start += 1
    elif(((o[0],o[1]+1) in pred) or ((o[0],o[1]-1) in pred)):
        miss_end+=1


# In[77]:


print("Total original spans: ", len(orig))
print("Total predicted spans: ", len(pred))
print("Total number of original spans correctly predicted acc to strict match: ", len(strict_match_spans))
print("Percent of original spans correctly predicted acc to strict match: ", len(strict_match_spans)/len(orig)*100)

print("Total number of original spans correctly predicted acc to fuzzy match: ", fuzzy_cnt)
print("Percent of original spans correctly predicted acc to fuzzy match: ", fuzzy_cnt/len(orig)*100)

fuzzy_matched_only = miss_start_end+miss_start+miss_end
assert(fuzzy_matched_only == fuzzy_cnt - len(strict_match_spans))
print("Count of fuzzy matched spans: ", miss_start_end+miss_start+miss_end)
print("Count of spans with misaligned begin and end: {} ({:.2f}%) ".format(miss_start_end, miss_start_end/fuzzy_matched_only*100))
print("Count of spans with misaligned begin: {} ({:.2f}%) ".format(miss_start, miss_start/fuzzy_matched_only*100))
print("Count of spans with misaligned end: {} ({:.2f}%) ".format(miss_end, miss_end/fuzzy_matched_only*100))


# In[ ]:


# Store error cases


# In[ ]:


error = test_df[test_df['label'] != test_df['predictions']]
print((len(test_df)- len(error)) / len(test_df))
print(len(error))
error.to_csv("errors_bert_embed_iob_bilstm_crf.csv")


# In[ ]:




