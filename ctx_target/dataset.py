import pandas as pd
import torch
import math
from transformers import AutoTokenizer
from utils import set_my_seed
import numpy as np
from torch.utils.data.sampler import Sampler
import itertools
from utils import cysteine_position
from utils import calculate_RSA
set_my_seed(4)
from torch.utils.data import Dataset, DataLoader
import os
join = os.path.join
import yaml

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config
Config = load_config('ctx_target/config.yaml')

class TwoStreamBatchSampler(Sampler):
    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size
        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in  zip(grouper(primary_iter, self.primary_batch_size),
                    grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size

def iterate_once(iterable):
    return np.random.permutation(iterable)

def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())

def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    args = [iter(iterable)] * n
    return zip(*args)

tokenizer = AutoTokenizer.from_pretrained(Config['model_checkpoint'])

def prepare_data():
    # 读取数据
    df_train = pd.read_csv(Config['train_path'])
    df_unlabel = pd.read_csv(Config['unlabel_path'])
    df_val = pd.read_csv(Config['val_path'])
    
    # 获取有标签训练数据
    labeled_sequences = df_train["train_sequences"].tolist()
    labeled_labels = df_train["train_labels"].tolist()
    labeled_structure = df_train["train_structure"].tolist()
    
    # 获取无标签数据（没有标签）
    unlabeled_sequences = df_unlabel["unlabeled_sequences"].tolist()
    unlabeled_structure = df_unlabel["unlabeled_structure"].tolist()

    # 获取验证数据（有标签）
    val_sequences = df_val["val_sequences"].tolist()
    val_labels = df_val["val_labels"].tolist()
    val_structure = df_val["val_structure"].tolist()
    
    return {
        'train_labeled_sequences': labeled_sequences,
        'train_labeled_labels': labeled_labels,
        'train_labeled_structure': labeled_structure,
        'train_unlabeled_sequences': unlabeled_sequences,
        'train_unlabeled_structure': unlabeled_structure,
        'val_sequences': val_sequences,
        'val_labels': val_labels,
        'val_structure': val_structure,
        'len_labeled': len(labeled_sequences),
        'len_unlabeled': len(unlabeled_sequences)
    }

# 数据批处理函数
def collate_fn(batch):
    sequences = [item[0] for item in batch]
    pt_batch = tokenizer(sequences, padding="max_length", truncation=True, 
                        max_length=Config['max_len'], return_tensors='pt')
    labels = [item[1] for item in batch]
    cysteine_feature = [item[2] for item in batch]
    RSA_feature = [item[3] for item in batch]
    
    return {
        'input_ids': pt_batch['input_ids'].to(Config['device']),
        'attention_mask': pt_batch['attention_mask'].to(Config['device']),
        'labels': torch.tensor(labels).to(Config['device']),
        'RSA_feature': torch.tensor(RSA_feature).to(Config['device']),
        'cysteine_feature': torch.tensor(cysteine_feature).to(Config['device'])
    }

class MyDataset(Dataset):
    def __init__(self, sequences, labels, Structure_index, pdb_path, max_len=50, file_format='pdb'):
        self.file_format = file_format
        self.sequences = sequences
        self.labels = labels
        self.Structure_index = Structure_index
        self.cysteine_position = cysteine_position(sequences, max_len)
        self.RSA = calculate_RSA(Structure_index, pdb_path, max_len, file_format, sequences)

    def __len__(self):
        return len(self.sequences)
        
    def __getitem__(self, idx):
        # rsa_int = [math.floor(x * 100) for x in self.RSA[idx]]
        label = self.labels[idx] if self.labels is not None else -1
        return self.sequences[idx], \
                label, \
                self.cysteine_position[idx],\
                self.RSA[idx]
                # rsa_int

def create_data_loaders():

    data = prepare_data()

    train_labeled_dataset = MyDataset(
        data['train_labeled_sequences'], 
        data['train_labeled_labels'], 
        data['train_labeled_structure'], 
        Config['pdb_path'], 
        Config['max_len'],
        Config['file_format']
    )

    train_unlabeled_dataset = MyDataset(
        data['train_unlabeled_sequences'], 
        None, 
        data['train_unlabeled_structure'], 
        Config['pdb_path'], 
        Config['max_len'],
        Config['file_format']
    )

    val_dataset = MyDataset(
        data['val_sequences'], 
        data['val_labels'], 
        data['val_structure'], 
        Config['pdb_path'], 
        Config['max_len'],
        Config['file_format']
    )

    train_dataset = torch.utils.data.ConcatDataset([train_labeled_dataset, train_unlabeled_dataset])

    labeled_idxs = list(range(len(train_labeled_dataset)))  # 有标签数据的索引
    unlabeled_idxs = list(range(len(train_labeled_dataset), len(train_dataset)))  # 无标签数据的索引

    sampler = TwoStreamBatchSampler(
        primary_indices=labeled_idxs,  # 主数据流（有标签）
        secondary_indices=unlabeled_idxs,  # 次数据流（无标签）
        batch_size= Config['batch_size'],
        secondary_batch_size = Config['unlabeled_batch_size']
    )

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=sampler,
        shuffle=False,
        collate_fn=collate_fn,
        drop_last=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config['batch_size'],
        shuffle=False,
        collate_fn=collate_fn,
        drop_last=False
    )

    return train_loader, val_loader

def create_test_loaders():
    df_test = pd.read_csv(Config['test_path'])
    test_sequences = df_test["Seq"].tolist()
    test_structure = df_test["ID"].tolist()

    test_dataset = MyDataset(
        test_sequences, 
        None, 
        test_structure, 
        Config['test_pdb_path'], 
        Config['max_len'],
        Config['file_format']
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config['batch_size'],
        shuffle=False,
        collate_fn=collate_fn,
        drop_last=False
    )

    return test_loader

if __name__ == '__main__':
    train_loader, val_loader = create_data_loaders()
    for i, batch in enumerate(train_loader):
        print(batch)
