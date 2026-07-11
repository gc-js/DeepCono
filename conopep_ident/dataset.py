from torch.utils.data import Dataset
from model import tokenizer
from utils import tag_to_ix
import torch

class MyDataset(Dataset):
    def __init__(self,dict_data) -> None:
        super(MyDataset,self).__init__()
        self.data=dict_data
    def __getitem__(self, index):
        return [self.data['text'][index],self.data['labels'][index]]
    def __len__(self):
        return len(self.data['text'])
    
def collate_fn(batch):
    sequences = [b[0] for b in batch]
    labels = [b[1] for b in batch]
    pt_batch = tokenizer(
        sequences, 
        padding=True, 
        truncation=True, 
        return_tensors='pt'
    )
    
    batch_size = len(batch)
    seq_length = pt_batch['input_ids'].shape[1]
    
    labels_tensor = torch.full((batch_size, seq_length), tag_to_ix["[PAD]"], dtype=torch.long)
    
    for i, label_str in enumerate(labels):
        label_indices = [tag_to_ix[char] for char in label_str]
        actual_content_len = min(len(label_indices), seq_length - 1)

        labels_tensor[i, 1:1+actual_content_len] = torch.tensor(label_indices[:actual_content_len])

    return {
        'labels': labels_tensor,
        'input_ids': pt_batch['input_ids'],
        'attention_mask': pt_batch['attention_mask']
    }
