from torch.utils.data import DataLoader, Dataset 
import pandas as pd
import math
import torch
from transformers import BertTokenizer
from utils import load_config
import pickle

class CRFEmbeddingDataset(Dataset):
    def __init__(self, csv_file, para_seq_len, pretrained_model, stride = 1):
      df = pd.read_csv(csv_file)
      self.config_dict = load_config()
      print(self.config_dict['max_para_length'])
      self.para_seq_len = para_seq_len
      self.tokenizer = BertTokenizer.from_pretrained(pretrained_model, do_lower=True)    

      # Tokenize the paragraphs
      self.df = df["para"].apply(self.preprocess)
      self.y = df['label']
      # self.test = is_test
      self.stride = stride
      
      
  
     
    def preprocess(self, examples):
      return self.tokenizer(examples, truncation=True, 
                     padding="max_length", max_length=self.config_dict['max_para_length'],
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
      

class ParaEmbeddings(Dataset):
  def __init__(self, pickle_file, para_seq_len, stride = 1):
    self.para_seq_len = para_seq_len
    self.stride = stride

    with open(pickle_file, "rb") as fIn:
      stored_data = pickle.load(fIn)
      self.embeddings = stored_data['embeddings']
      self.y = stored_data['label']

  def __len__(self):
      l = math.ceil((len(self.y) - self.para_seq_len + 1) / self.stride)
      return l

  def __getitem__(self,index):
      return torch.FloatTensor(self.embeddings[index*self.stride: (index*self.stride + self.para_seq_len)].tolist()), torch.LongTensor(self.y[index*self.stride: (index*self.stride + self.para_seq_len)])



class ContextEmbeddingDataset(Dataset):
    def __init__(self, csv_file, context, pretrained_model, max_para_length):
      df = pd.read_csv(csv_file)

      self.context = context
      self.tokenizer = BertTokenizer.from_pretrained(pretrained_model, do_lower=True)    
      self.max_para_length = max_para_length
      # Tokenize the paragraphs
      self.df = df["para"].apply(self.preprocess)
      self.y = df['label']
  
     
    def preprocess(self, examples):
      return self.tokenizer(examples, truncation=True, 
                     padding="max_length", max_length=self.max_para_length,
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

