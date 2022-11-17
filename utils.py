import yaml
import os
import torch
import time
from sklearn.metrics import precision_recall_fscore_support
import numpy as np
from tqdm import tqdm

def load_config():
    with open("config.yaml", "r") as configfile:
        config_dict = yaml.load(configfile, Loader=yaml.FullLoader)
    # print(config_dict)
    return config_dict


def save(model, acc, model_save_path, best=""):
    if not os.path.exists(model_save_path):
        os.mkdir(model_save_path)

    torch.save(model.state_dict(), model_save_path+'/{}model_params_{}.pth'.format(best, acc))


def load_pretrained_weights(model, pretrained_path):
    pretrained_dict = torch.load(pretrained_path)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k[:12] != "embed_model."}

    print(pretrained_dict.keys())
    model_dict = model.state_dict()
    model_dict.update(pretrained_dict) 
    print("=============")
    print(model_dict.keys())
    model.load_state_dict(model_dict)
    exit()
    return model   

def load_pretrained_weights_modified(model, pretrained_path):
    pretrained_dict = torch.load(pretrained_path)
    pretrained_dict = {k[7:]: v for k, v in pretrained_dict.items()}
    # print(pretrained_dict.keys())
    model_dict = model.state_dict()
    model_dict.update(pretrained_dict) 
    model.load_state_dict(model_dict)
    return model    

def get_spans(pred_df, predictions):
    pred_df = pred_df[['para','label','document']][:len(predictions)]
    pred_df['predictions'] = predictions
    # pred_df.to_csv("bert_embed_iob_bilstm_crf_pred.csv")
    # print(len(pred_df)) 
    print(pred_df.columns)
    pred_df = pred_df.reset_index(drop=False)
    pred_df.columns = ['index', 'para', 'label', 'document', 'predictions']
    pred = set()
    i = 0
    while i < len(pred_df):
        if(pred_df['predictions'][i] == 2):
            st = pred_df['index'][i]
            i +=1
            while(i < len(pred_df) and pred_df['predictions'][i] == 1):
                i+=1
            pred.add((st, pred_df['index'][i-1]))
        else:
            i+=1
    # Include predicted spans that do not have 2 (beginning) Tag beginning as para before first 1. 
    # Exclude this from the strict match count
    i = 0
    while i < len(pred_df)-1:
        if(pred_df['predictions'][i] == 0 and pred_df['predictions'][i+1] == 1):
            st = pred_df['index'][i]
            i +=1
            while(i < len(pred_df) and pred_df['predictions'][i] == 1):
                i+=1
            pred.add((st, pred_df['index'][i-1]))
        else:
            i+=1
    
    return pred_df, pred



def get_span_perf(pred_df, predictions):
    # print(len(pred_df))
    pred_df = pred_df[['para','label','document']][:len(predictions)]
    pred_df['predictions'] = predictions
    # pred_df.to_csv("bert_embed_iob_bilstm_crf_pred.csv")
    # print(len(pred_df)) 
    print(pred_df.columns)
    pred_df = pred_df.reset_index(drop=False)
    pred_df.columns = ['index', 'para', 'label', 'document', 'predictions']

    orig = set()
    i = 0
    while i < len(pred_df):
        if(pred_df['label'][i] == 2):
            st = pred_df['index'][i]
            i +=1
            while(i < len(pred_df) and pred_df['label'][i] == 1):
                i+=1
            orig.add((st, pred_df['index'][i-1]))
        else:
            i+=1

    pred = set()
    i = 0
    while i < len(pred_df):
        if(pred_df['predictions'][i] == 2):
            st = pred_df['index'][i]
            i +=1
            while(i < len(pred_df) and pred_df['predictions'][i] == 1):
                i+=1
            pred.add((st, pred_df['index'][i-1]))
        else:
            i+=1

    
   
    strict_match_spans = orig.intersection(pred)
    recall_strict = len(strict_match_spans)/len(orig)
    precision_strict = len(strict_match_spans)/len(pred)
    
    # Include predicted spans that do not have 2 (beginning) Tag beginning as para before first 1. 
    # Exclude this from the strict match count
    i = 0
    while i < len(pred_df)-1:
        if(pred_df['predictions'][i] == 0 and pred_df['predictions'][i+1] == 1):
            st = pred_df['index'][i]
            i +=1
            while(i < len(pred_df) and pred_df['predictions'][i] == 1):
                i+=1
            pred.add((st, pred_df['index'][i-1]))
        else:
            i+=1

    

    fuzzy_cnt = 0
    for o in orig:
        if ((o in pred) or ((o[0]+1,o[1]) in pred) or ((o[0]+1,o[1]-1) in pred) or ((o[0]+1,o[1]+1) in pred) 
            or ((o[0]-1,o[1]) in pred) or ((o[0]-1,o[1]+1) in pred) or ((o[0]-1,o[1]-1) in pred) or ((o[0],o[1]+1) in pred)
            or ((o[0],o[1]-1) in pred)):
            fuzzy_cnt+=1
            
    fuzzy_cnt_precision = 0
    for o in pred:
        if ((o in orig) or ((o[0]+1,o[1]) in orig) or ((o[0]+1,o[1]-1) in orig) or ((o[0]+1,o[1]+1) in orig) 
            or ((o[0]-1,o[1]) in orig) or ((o[0]-1,o[1]+1) in orig) or ((o[0]-1,o[1]-1) in orig) or ((o[0],o[1]+1) in orig)
            or ((o[0],o[1]-1) in orig)):
            fuzzy_cnt_precision+=1
            
    recall_fuzzy = fuzzy_cnt/len(orig)
    precision_fuzzy = fuzzy_cnt_precision/len(pred)
    
    miss_start_end = 0
    miss_start_end_spans = []
    miss_start = 0
    miss_start_spans = []
    miss_end = 0
    miss_end_spans = []
    
    no_match_spans = []

    for o in orig:
        if(o in pred):
            continue 
        elif(((o[0]-1,o[1]+1) in pred) or ((o[0]-1,o[1]-1) in pred) or ((o[0]+1,o[1]-1) in pred) or ((o[0]+1,o[1]+1) in pred)):
            miss_start_end += 1
            miss_start_end_spans.append(o)
        elif(((o[0]+1,o[1]) in pred) or ((o[0]-1,o[1]) in pred)):
            miss_start += 1
            miss_start_spans.append(o)
        elif(((o[0],o[1]+1) in pred) or ((o[0],o[1]-1) in pred)):
            miss_end+=1
            miss_end_spans.append(o)
        else:
            no_match_spans.append(o)

    
    
    print("Total original spans: ", len(orig))
    print("Total predicted spans: ", len(pred))
    print("Total number of original spans correctly predicted acc to strict match: ", len(strict_match_spans))
    print("Percent of original spans correctly predicted acc to strict match:(Recall strict) ", recall_strict)
    print("Percent of original spans correctly predicted acc to strict match:(Precision strict) ", precision_strict)
    print("F1 strict match: ", 2*precision_strict*recall_strict / (recall_strict+precision_strict))

    print("Total number of original spans correctly predicted acc to fuzzy match: ", fuzzy_cnt)
    print("Percent of original spans correctly predicted acc to fuzzy match: (Recall Fuzzy)", recall_fuzzy)
    print("Percent of original spans correctly predicted acc to fuzzy match: (Precision Fuzzy)", precision_fuzzy)
    print("F1 fuzzy match: ", 2*precision_fuzzy*recall_fuzzy / (recall_fuzzy+precision_fuzzy))

    fuzzy_matched_only = miss_start_end+miss_start+miss_end
#     assert(fuzzy_matched_only == fuzzy_cnt - len(strict_match_spans))
    print("Count of fuzzy matched spans: ", miss_start_end+miss_start+miss_end)
    print("Count of spans with misaligned begin and end: {} ({:.2f}%) ".format(miss_start_end, miss_start_end/fuzzy_matched_only*100))
    print("Count of spans with misaligned begin: {} ({:.2f}%) ".format(miss_start, miss_start/fuzzy_matched_only*100))
    print("Count of spans with misaligned end: {} ({:.2f}%) ".format(miss_end, miss_end/fuzzy_matched_only*100))
    print("Count of missed spans: {} ({:.2f}%) ".format(len(no_match_spans), len(no_match_spans)/len(orig)*100))


    return pred_df



def train(para_model, data_loader, device, optimizer, scheduler):
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

        loss, emission = para_model(x, y)
        del x

        avg_loss.extend([loss.item()]*len(y))
        
        decoded_list = para_model.decode(emission)
        for l in decoded_list:
            all_predictions.extend(l)


        all_targets.extend(torch.flatten(y.detach().cpu()).tolist())
        
        loss.backward()
        optimizer.step()
        scheduler.step()

        del y
        del emission
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


def validate(para_model, data_loader, device, optimizer, scheduler):
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
            loss, emission = para_model(x, y) 
            del x
            
            avg_loss.extend([loss.item()]*len(y))

            decoded_list = para_model.decode(emission)
            
            for l in decoded_list:
                all_predictions.extend(l)

            all_targets.extend(torch.flatten(y.detach().cpu()).tolist())
            del emission
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


def extract(para_model, data_loader, device):
    para_model.eval()
    # crf_model.eval()

    all_predictions = []
    start = time.time()

    for i, (x, y) in enumerate(tqdm(data_loader, desc="Epoch", leave=False)):
        x = x.to(device)

        with torch.no_grad():
            decoded_list = para_model.extract(x) 
            del x
            
            for l in decoded_list:
                all_predictions.extend(l)
            # all_predictions.extend(decoded_list)
    
            torch.cuda.empty_cache()
             
    end = time.time()
    print('Time: {}'.format(end-start))
    all_predictions = np.array(all_predictions)
    return all_predictions
