from utils import *
import os
config_dict = load_config()

os.environ["CUDA_VISIBLE_DEVICES"]=config_dict["CUDA_VISIBLE_DEVICES"]


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
from datasets import load_dataset
import pickle
from transformers import AutoTokenizer, AutoModel
from transformers import BertTokenizer, BertModel
import multiprocessing
import time
from torch.utils.data import DataLoader, Dataset 
import sys
from tqdm import tqdm
from dataset import *
from models import *
from torchcrf import CRF
import pickle

import warnings
warnings.filterwarnings("ignore")


if(config_dict['enable_wandb']):
    import wandb
    wandb.init(name=config_dict['name'], 
    project="ChemIR_New") 
    
else:
    wandb = None

pretrained_model = config_dict['pretrained_model']

print(config_dict['name'])
print("Using pretrained model: ", pretrained_model)
print("Freeze Bert: ", config_dict['freeze_bert'])
print(config_dict)


batch_size = config_dict['batch_size']
max_para_length = config_dict['max_para_length']
para_seq_len = config_dict['para_seq_len']  #number of paras to be encoded and decoded together (hyperparameter)

# Check if cuda is available and set device
cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")

# Make sure you choose suitable num_worker, otherwise it will result in errors
num_workers = 8 if cuda else 0

print("Cuda = ", str(cuda), " with num_workers = ", str(num_workers),  " system version = ", sys.version)

test_file = config_dict['test_file']
val_file = config_dict['val_file']
train_file = config_dict['train_file']


model = EncoderDecoderBiLstmCRF(embed_model = BertEmbedding(pretrained_model), num_tags = 3, freeze_bert=config_dict['freeze_bert'], encoder_bilayers = int(config_dict['encoder_bilstm_layers']), decoder_bilayers = int(config_dict['decoder_bilstm_layers']))

if(config_dict['pretrained_full_model_path']!=""):
    # model.load_state_dict(torch.load(config_dict['pretrained_full_model_path']))
    model = load_pretrained_weights(model, config_dict['pretrained_full_model_path'])

# if torch.cuda.device_count() > 1:
#   print("Let's use", torch.cuda.device_count(), "GPUs!")
#   model = nn.DataParallel(model)


total_params = sum(p.numel() for p in model.parameters())
trainable_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
non_trainable_total_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
print("Total params: ", total_params)
print("Trainable params: ", trainable_total_params)
print("Non Trainable params: ", non_trainable_total_params)



tokenizer = AutoTokenizer.from_pretrained(pretrained_model, do_lower = True)    
print(tokenizer.all_special_tokens) # --> ['[UNK]', '[SEP]', '[PAD]', '[CLS]', '[MASK]']
print(tokenizer.all_special_ids)    # --> [100, 102, 0, 101, 103]

num_added_toks = tokenizer.add_tokens(['[CHEM]'], special_tokens=True) 

print(model.embed_model.model.config.vocab_size)
model.embed_model.model.resize_token_embeddings(len(tokenizer))
print(model.embed_model.model.config.vocab_size)

print(tokenizer.convert_tokens_to_ids('[CHEM]'))
print(tokenizer.all_special_tokens) 

model = model.to(device)


train_data = CRFEmbeddingDataset(train_file, para_seq_len = para_seq_len, tokenizer = tokenizer, stride = config_dict['stride'])
val_data = CRFEmbeddingDataset(val_file, para_seq_len = para_seq_len, tokenizer = tokenizer, stride = para_seq_len)
test_data = CRFEmbeddingDataset(test_file, para_seq_len = para_seq_len, tokenizer = tokenizer, stride = para_seq_len)


train_args = dict(shuffle=True, batch_size=batch_size, num_workers=8, pin_memory=True, drop_last=False) if cuda else dict(shuffle=True, batch_size=batch_size, drop_last=False)
train_loader = DataLoader(train_data, **train_args)

val_args = dict(shuffle=False, batch_size=batch_size, num_workers=8, pin_memory=True, drop_last=False) if cuda else dict(shuffle=False, batch_size=batch_size, drop_last=False)
val_loader = DataLoader(val_data, **val_args)


test_args = dict(shuffle=False, batch_size=batch_size, num_workers=8, pin_memory=True, drop_last=False) if cuda else dict(shuffle=False, batch_size=batch_size, drop_last=False)
test_loader = DataLoader(test_data, **test_args)

print("Train dataset len", train_data.__len__())
print("Val dataset len", val_data.__len__())
print("Test dataset len", test_data.__len__())
print("Train Loader len", len(train_loader))
print("Val Loader len", len(val_loader))
print("Test Loader Len len", len(test_loader))


epochs = int(config_dict['epochs']) #changed from 10
lamda = float(config_dict['lamda'])  #L2 regularization (prev : 1e-4)
learning_rate = float(config_dict['learning_rate']) #changed from 1e-2   ## Greatly reduces LR for bert finetuning
print("Epochs: ",epochs)
print("Lamda: ",lamda)
print("Learning Rate: ",learning_rate)
# criterion = nn.CrossEntropyLoss()
# criterion = criterion.to(device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=lamda)
# optimizer.load_state_dict(torch.load('./bert_base_triplet/optimizer_model_params_0.9409211846833226.pth'))    

# scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[i for i in range(4,20,4)], gamma=0.75)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(len(train_loader) * epochs))


torch.cuda.empty_cache()
best_val_f1 = 0
val_df = pd.read_csv(val_file)

wandb_str = ""
if(wandb is not None):
    wandb_str = "validation"

for epoch in range(epochs):
    print('Epoch #{}'.format(epoch+1))

    train_loss, train_scores = train(model, train_loader, device, optimizer, scheduler)
    val_f1, val_preds, eval_scores = validate(model, val_loader, device, optimizer, scheduler, wandb = wandb_str)
    if(wandb is not None):
        wandb.log({"Train Loss": train_loss})
        wandb.log(train_scores)
        wandb.log(eval_scores)

    try:
        _, perf_scores = get_span_perf(val_df, val_preds, wandb = wandb_str)
        if(wandb is not None):
            wandb.log(perf_scores)

        save(model, val_f1, model_save_path = config_dict['model_save_path'], best = "model_")
        save(optimizer, val_f1, model_save_path= config_dict['model_save_path'], best = "optimizer_")
    except Exception as error:
        print(error)

    if(epoch == 0):
        with open(os.path.join(config_dict['model_save_path'],'config.pkl'), 'wb') as f:
            pickle.dump(config_dict, f)


test_df = pd.read_csv(test_file)
_, test_predictions = validate(model, test_loader, device, optimizer, scheduler)
get_span_perf(test_df, test_predictions)


print("Done!")
