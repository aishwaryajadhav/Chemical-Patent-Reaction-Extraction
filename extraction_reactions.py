import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

from utils import *
from dataset  import *
from models import *
import sys


test_file = "gtest_iob_modified.csv"


# In[3]:


pretrained_model = "recobo/chemical-bert-uncased-pharmaceutical-chemical-classifier"
# pretrained_model = "bert-base-uncased"
batch_size = 4
max_para_length = 128
para_seq_len = 16  #number of paras to be encoded and decoded together (hyperparameter)
# Check if cuda is available and set device
cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")

# Make sure you choose suitable num_worker, otherwise it will result in errors
num_workers = 8 if cuda else 0

print("Cuda = ", str(cuda), " with num_workers = ", str(num_workers),  " system version = ", sys.version)


model = EncoderDecoderBiLstmCRF(embed_model = BertEmbedding(pretrained_model), num_tags = 3, freeze_bert=True)

model.load_state_dict(torch.load('./chem_bert_iob_bilstm_crf/model_model_params_0.9539811224849719.pth'))


# model.load_state_dict(torch.load('./bert_base_triplet_iob_bert_ft/model_model_params_0.9334563821030115.pth'))
# model = load_pretrained_weights_modified(model, './bert_base_triplet_iob_bert_ft/model_model_params_0.9334563821030115.pth')

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


# In[4]:



# test_data = ContextEmbeddingDataset(test_file, context = 1, pretrained_model = pretrained_model, max_para_length = max_para_length)
# test_args = dict(shuffle=False, batch_size=batch_size, num_workers=8, pin_memory=True, drop_last=False) if cuda else dict(shuffle=False, batch_size=batch_size, drop_last=False)
# test_loader = DataLoader(test_data, **test_args)


test_data = CRFEmbeddingDataset(test_file, para_seq_len = para_seq_len, pretrained_model = pretrained_model, stride = para_seq_len)
test_args = dict(shuffle=False, batch_size=batch_size, num_workers=8, pin_memory=True, drop_last=False) if cuda else dict(shuffle=False, batch_size=batch_size, drop_last=False)
test_loader = DataLoader(test_data, **test_args)


# In[5]:


print(test_data.__len__())
print(len(test_loader))


# ## Model 1
# ./chem_bert_iob_bilstm_crf_bert_finetune/model_model_params_0.9507615640930901.pth
# 
# pretrained_model = "recobo/chemical-bert-uncased-pharmaceutical-chemical-classifier"
# 
# batch_size = 4
# 
# max_para_length = 128
# 
# para_seq_len = 16  #number of paras to be encoded and decoded together (hyperparameter)
# 

# In[6]:


# In[ ]:


test_df = pd.read_csv(test_file)
test_predictions = extract(model, test_loader, device)
test_df, pred_spans = get_spans(test_df, test_predictions)

print(test_df.head(10))
print("Prediction Spans:", pred_spans)

# In[ ]:


test_df.to_csv("gtest_predictions_modified_chem_bert_no_ft.csv")

print(test_df.head())

