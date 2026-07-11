import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, EsmForMaskedLM, DataCollatorForLanguageModeling, get_scheduler
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
import random
import os
from transformers import set_seed
import matplotlib.pyplot as plt
from peft import get_peft_model, LoraConfig
from early_stopping_pytorch import EarlyStopping
from accelerate import Accelerator

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    set_seed(seed)
setup_seed(4)

accelerator = Accelerator(gradient_accumulation_steps=2, mixed_precision='fp16')

lr = 1e-3
batch_size = 64
num_train_epochs = 1000
best_eval_loss = 1.5

model_output = "./mlm_conoserver_lora"
img_output = "./loss_curves.png"
checkpoint = "facebook/esm2_t30_150M_UR50D"
seq_path = "./conoserver_data.csv"

df = pd.read_csv(seq_path)
sequences = df["Seq"].tolist()

os.makedirs(model_output, exist_ok=True)

early_stopping = EarlyStopping(patience=100, verbose=False)
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

peft_config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=["query","key","value","dense"],
    lora_dropout=0.1,
    bias="none",
)

model = EsmForMaskedLM.from_pretrained(checkpoint)
model = get_peft_model(model, peft_config)

class SequenceMLMDataset(Dataset):
    def __init__(self, sequences, tokenizer, max_length=128):
        self.sequences = sequences
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        inputs = self.tokenizer(sequence, return_tensors="pt", max_length=self.max_length, truncation=True, padding="max_length")
        inputs = {key: val.squeeze(0) for key, val in inputs.items()}
        return inputs

train_sequences, eval_sequences = train_test_split(sequences, test_size=0.1, random_state=42)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)
train_dataset = SequenceMLMDataset(train_sequences, tokenizer)
eval_dataset = SequenceMLMDataset(eval_sequences, tokenizer)

train_dataloader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=data_collator,
    num_workers=32,
    pin_memory=True
)

eval_dataloader = DataLoader(
    eval_dataset, 
    batch_size=batch_size,
    collate_fn=data_collator,
    num_workers=32,
    pin_memory=True
)

optimizer = AdamW(model.parameters(), lr=lr)

num_update_steps_per_epoch = len(train_dataloader)
num_training_steps = num_train_epochs * num_update_steps_per_epoch

lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)

progress_bar = tqdm(range(num_training_steps))

train_losses = []
eval_losses = []

model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
)

for epoch in range(num_train_epochs):
    model.train()
    total_train_loss = 0.0
    for batch in train_dataloader:
        with accelerator.accumulate(model):
            outputs = model(**batch)
            loss = outputs.loss
            reduced_loss = accelerator.reduce(loss, reduction="mean")
            total_train_loss += reduced_loss.item()
            accelerator.backward(loss)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)
    avg_train_loss = total_train_loss / len(train_dataloader)
    train_losses.append(avg_train_loss)

    model.eval()
    total_eval_loss = 0.0
    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            outputs = model(**batch)
        loss = outputs.loss
        reduced_loss = accelerator.reduce(loss, reduction="mean")
        total_eval_loss += reduced_loss.item()
    avg_eval_loss = total_eval_loss / len(eval_dataloader)
    eval_losses.append(avg_eval_loss)

    if avg_eval_loss < best_eval_loss:
        best_eval_loss = avg_eval_loss
        accelerator.wait_for_everyone()
        accelerator.print(f"average training loss: {avg_train_loss:.4f}, New best evaluation loss: {best_eval_loss:.4f}. Saving model...")
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(model_output)
        tokenizer.save_pretrained(model_output)
        
    early_stopping(avg_eval_loss, model)
    if early_stopping.early_stop:
        accelerator.print("Early stopping triggered")
        break

actual_epochs = len(train_losses)
plt.figure(figsize=(10, 6))
plt.plot(range(1, actual_epochs + 1), train_losses, label='Training Loss')
plt.plot(range(1, actual_epochs + 1), eval_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
xticks = np.arange(0, actual_epochs + 1, max(1, actual_epochs // 20))
plt.xticks(xticks)
plt.legend()
plt.grid(True)
plt.savefig(img_output)
plt.close()
