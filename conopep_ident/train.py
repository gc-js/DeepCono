import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
import os
join = os.path.join
from utils import setup_seed, calculate_metrics, tag_to_ix, do_nothing
from model import Model
from dataset import MyDataset, collate_fn
from early_stopping_pytorch import EarlyStopping
import yaml

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

config = load_config('config.yaml')
seed = config['seed']
device = config['device']
path = config['train_data']
log_file_path = config['log']
base_model_path = config['base_model_path']
adapter = config['adapter']
model_save = config['model_path']
lr = config['lr']
patience=config['early_stop_patience']
epochs = config['epochs']
batch_size = config['batch_size']
kfold = config['KFold']

os.makedirs(model_save, exist_ok=True)
df = pd.read_csv(path)
Sequences = df["sequence"].tolist()
Labels = df["label"].tolist()

fold = 1
kf = KFold(n_splits=kfold, shuffle=True, random_state=seed)

with open(log_file_path, 'w') as log_file:
    for train_idx, test_idx in kf.split(Sequences):
        setup_seed(seed)
        print(f"=====================================================fold:{fold}=====================================================")
        train_dict = {"text":np.array(Sequences)[train_idx],'labels':np.array(Labels)[train_idx]}
        test_dict = {"text":np.array(Sequences)[test_idx],'labels':np.array(Labels)[test_idx]}
        train_data=MyDataset(train_dict)
        train_dataloader=DataLoader(train_data,batch_size=batch_size,shuffle=True,collate_fn=collate_fn)
        test_data=MyDataset(test_dict)
        test_dataloader=DataLoader(test_data,batch_size=batch_size,shuffle=False,collate_fn=collate_fn)

        # training
        model = Model(len(tag_to_ix), base_model_path, adapter, prob=False)
        model = model.to(device)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        
        train_epochs_loss_all = []
        valid_epochs_acc_all = []
        best_f1 = 0

        early_stopping = EarlyStopping(patience=patience, verbose=False, trace_func= do_nothing)

        for i in range(epochs):
            model.train()
            train_epoch_loss = []
            for index, batch in enumerate(train_dataloader):
                batchs = {k: v for k, v in batch.items()}
                optimizer.zero_grad()
                input_ids = batchs['input_ids'].to(device)
                attention_mask = batchs['attention_mask'].to(device)
                labels = batchs['labels'].to(device)
                loss = model(input_ids, attention_mask, labels)
                loss.backward()
                optimizer.step()
                train_epoch_loss.append(loss.item())
            train_epoch_loss_ = np.average(train_epoch_loss)

            model.eval()
            with torch.no_grad():
                valid_epochs_acc = []
                valid_epochs_precision = []
                valid_epochs_recall = []
                valid_epochs_f1 = []
                
                for index, batch in enumerate(test_dataloader):
                    batchs = {k: v for k, v in batch.items()}
                    input_ids = batchs['input_ids'].to(device)
                    attention_mask = batchs['attention_mask'].to(device)
                    tags = model(input_ids, attention_mask)
                    labels = batchs['labels'].cpu().numpy()
                    attention_mask_np = attention_mask.cpu().numpy()
                    
                    for j in range(len(tags)):
                        actual_length = attention_mask_np[j].sum().item()
                        
                        if actual_length >= 2:

                            new_tag = tags[j][1:actual_length-1]
                            new_label = labels[j][1:actual_length-1]
                            
                            min_len = min(len(new_label), len(new_tag))
                            if min_len > 0:
                                new_label = new_label[:min_len]
                                new_tag = new_tag[:min_len]
                                
                                acc, precision, recall, f1 = calculate_metrics(new_label, new_tag)
                                valid_epochs_acc.append(acc)
                                valid_epochs_precision.append(precision)
                                valid_epochs_recall.append(recall)
                                valid_epochs_f1.append(f1)
                
            valid_epochs_acc_ = np.average(valid_epochs_acc)
            valid_epochs_precision_ = np.average(valid_epochs_precision)
            valid_epochs_recall_ = np.average(valid_epochs_recall)
            valid_epochs_f1_ = np.average(valid_epochs_f1)

            log_file.write(f'epoch:{i+1}, train_loss:{train_epoch_loss_:.4f}, val_acc:{valid_epochs_acc_:.4f}, precision:{valid_epochs_precision_:.4f}, recall:{valid_epochs_recall_:.4f}, f1:{valid_epochs_f1_:.4f}\n')
            if valid_epochs_f1_ >= best_f1:
                best_f1 = valid_epochs_f1_
                torch.save(model.state_dict(), join(model_save, f"best_model_fold_{fold}.pth"))
            print(f'epoch:{i+1}, train_loss:{train_epoch_loss_:.4f}, val_acc:{valid_epochs_acc_:.4f}, precision:{valid_epochs_precision_:.4f}, recall:{valid_epochs_recall_:.4f}, f1:{valid_epochs_f1_:.4f}')
            
            early_stopping(-valid_epochs_f1_, model)
            if early_stopping.early_stop:
                break

        fold += 1
