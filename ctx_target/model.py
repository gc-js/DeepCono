import torch
import torch.nn as nn
import math
import os
join=os.path.join
import torch.nn.functional as F
from peft import PeftModel
from transformers import EsmForMaskedLM, AutoTokenizer

from dataset import load_config
Config = load_config('ctx_target/config.yaml')
device = Config['device']
class GaussianRBF(nn.Module):
    def __init__(self, n_rbf=640, start=0.0, stop=1.0):
        super().__init__()
        offset = torch.linspace(start, stop, n_rbf).float()
        self.register_buffer("offset", offset)
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2

    def forward(self, x):
        x = x.float().unsqueeze(-1)
        diff = x - self.offset
        return torch.exp(self.coeff * diff ** 2).float()

class RBFEmbeddingForTransformer(nn.Module):
    def __init__(self,
                 n_rbf: int = 640,
                 start: float = 0.0,
                 stop: float = 1.0):
        super().__init__()
        self.rbf = GaussianRBF(
            n_rbf=n_rbf,
            start=start,
            stop=stop,
        )

    def forward(self, x):
        rbf = self.rbf(x)
        return rbf

class RSA_Transformer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, num_encoder_layers):
        super(RSA_Transformer, self).__init__()

        self.rbf_embedding = RBFEmbeddingForTransformer(
            n_rbf=d_model,
            start=0.0,
            stop=1.0
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=0.1,
            batch_first=True
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers
        )
    
    def forward(self, x, pos_embeddings, src_key_padding_mask):
        token_embeddings = self.rbf_embedding(x).float()
        x = token_embeddings + pos_embeddings
        x = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
        return x

def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    return nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))

def clip_loss(similarity: torch.Tensor) -> torch.Tensor:
    seq_loss = contrastive_loss(similarity)
    rsa_loss = contrastive_loss(similarity.t())
    return (seq_loss + rsa_loss) / 2.0

class conoCLIP(nn.Module):
    def __init__(self, d_model, max_len):
        super().__init__()
        self.d_model = d_model
        self.seq_embedder = nn.Linear(d_model, d_model)
        self.struc_embedder = nn.Linear(d_model, d_model)

    def forward(self, pep_input, struc_input):
        pep_input = pep_input.mean(dim=1)
        struc_input = struc_input.mean(dim=1)

        pep_clip = self.seq_embedder(pep_input) / self.seq_embedder(pep_input).norm(p=2, dim=-1, keepdim=True)
        rsa_clip = self.struc_embedder(struc_input) / self.struc_embedder(struc_input).norm(p=2, dim=-1, keepdim=True)
        
        logits = torch.matmul(pep_clip, rsa_clip.T) 
        loss = clip_loss(logits)
        return pep_clip, rsa_clip, loss
    
class MyTransformerModel(nn.Module):
    def __init__(self, vocab_size, 
                 d_model, nhead, 
                 num_encoder_layers, 
                 dim_feedforward, 
                 max_len, 
                 dropout=0.1):
        super(MyTransformerModel, self).__init__()
        self.cys_emb = nn.Embedding(51, d_model)
        base_model = EsmForMaskedLM.from_pretrained(Config['model_checkpoint'], torch_dtype=torch.float16)
        peft_model = PeftModel.from_pretrained(base_model, Config['adapter'], torch_dtype=torch.float16)
        self.bert = peft_model.merge_and_unload().esm

        self.bn1 = nn.LayerNorm(256)
        self.bn2 = nn.LayerNorm(128)
        self.bn3 = nn.LayerNorm(64)
        self.gelu = nn.GELU()
        self.fc1 = nn.Linear(d_model*2, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.output_layer = nn.Linear(64, 5)
        self.dropout_layer = nn.Dropout(dropout)
        self.conoCLIP = conoCLIP(d_model, max_len)
        self.Seq2RSA = nn.MultiheadAttention(d_model, nhead, batch_first = True)
        self.RSA2Seq = nn.MultiheadAttention(d_model, nhead, batch_first = True)
        self.RSA_Transformer = RSA_Transformer(d_model, nhead, dim_feedforward, num_encoder_layers)
        
    def forward(self, x):
        bs = x['input_ids'].shape[0]
        length = x['input_ids'].shape[1]
        input_ids = x['input_ids'].to(device)
        attention_mask = x['attention_mask'].to(device)
        cys_mask = x['cysteine_feature'].to(device)
        RSA_feature =x['RSA_feature'].to(device)
        
        base_seq = torch.arange(1, length+1, device=device)
        matrix = torch.tile(base_seq, (bs, 1))
        result = torch.where(cys_mask == 1, torch.tensor(0, device=device), matrix.to(device))
        pos_embedding_rsa = self.cys_emb(result)

        src_key_padding_mask = (attention_mask == 0)
        
        with torch.no_grad():
            bert_output = self.bert(input_ids=input_ids,attention_mask=attention_mask) 
            output_feature = self.dropout_layer(bert_output.last_hidden_state)
            output_feature = output_feature.float()

        RSA_transformer_output = self.RSA_Transformer(RSA_feature, pos_embedding_rsa, src_key_padding_mask)

        Seq2RSA, _ = self.Seq2RSA(
            output_feature, 
            RSA_transformer_output, 
            RSA_transformer_output
        )

        RSA2Seq, _ = self.RSA2Seq(
            RSA_transformer_output, 
            output_feature,
            output_feature
        )

        Seq_clip, RSA_clip, clip_loss = self.conoCLIP(Seq2RSA, RSA2Seq)
        combined_feature = torch.cat((RSA_clip, Seq_clip), dim=1)
        output_feature = self.dropout_layer(self.gelu(self.bn1(self.fc1(combined_feature))))
        output_feature = self.dropout_layer(self.gelu(self.bn2(self.fc2(output_feature))))
        output_feature = self.dropout_layer(self.gelu(self.bn3(self.fc3(output_feature))))
        output_feature = self.dropout_layer(self.output_layer(output_feature))

        return output_feature, clip_loss
