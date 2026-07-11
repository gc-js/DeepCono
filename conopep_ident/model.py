import torch
import torch.nn as nn
from torchcrf import CRF
from peft import PeftModel
from transformers import EsmForMaskedLM, AutoTokenizer
import yaml

def load_config(config_path='conopep_ident/config.yaml'):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config
    
config = load_config()
base_model_path = config['base_model_path']

tokenizer = AutoTokenizer.from_pretrained(base_model_path)

class Model(nn.Module):
    def __init__(self,tag_num, base_model_path, adapter, prob=False):
        super().__init__()
        print("Loading base model...")
        base_model = EsmForMaskedLM.from_pretrained(base_model_path, torch_dtype=torch.float16)
        print("Loading adapter...")
        peft_model = PeftModel.from_pretrained(base_model, adapter, torch_dtype=torch.float16)
        print("Merging model...")
        self.bert = peft_model.merge_and_unload().esm
        self.config = self.bert.config
        self.lstm = nn.LSTM(bidirectional=True, num_layers=2, input_size=self.config.hidden_size, hidden_size=self.config.hidden_size//2, dropout=0.1, batch_first=True)
        self.crf = CRF(tag_num, batch_first=True)
        self.dropout = nn.Dropout(0)
        self.fc = nn.Linear(self.config.hidden_size,tag_num)
        self.prob = prob
    def forward(self, input_ids, attention_mask, labels=None):
        with torch.no_grad():
            bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask) 
        sequence_output = bert_output.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        sequence_output = sequence_output.float()
        lstm_output, _ = self.lstm(sequence_output)
        logit = self.fc(lstm_output)
        logits = torch.softmax(logit, dim=2)
        probability = logits[0][:, 1]
        attention_mask = attention_mask.bool()
        if labels is not None:
            loss = -self.crf(logit, labels, mask=attention_mask, reduction="mean")
            return loss
        else:
            if not self.prob:
                return  self.crf.decode(logit, mask=attention_mask)
            else:
                return self.crf.decode(logit, mask=attention_mask), probability

