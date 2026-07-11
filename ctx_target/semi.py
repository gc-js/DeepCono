import torch
import torch.nn as nn
import random
from tqdm import tqdm
from transformers import get_scheduler
from model import MyTransformerModel
from utils import set_my_seed, softmax_kl_loss, softmax_mse_loss, calculate_metrics
import ramps
import warnings
warnings.filterwarnings("ignore")
from dataset import load_config, tokenizer, create_data_loaders
Config = load_config('ctx_target/config.yaml')
device = Config['device']
set_my_seed(4)
NO_LABEL = -1

class MeanTeacherModel:
    def __init__(self, student_model, teacher_model, optimizer, lr_scheduler):
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.student_model = student_model
        self.teacher_model = teacher_model

        self.supervised_criterion = nn.CrossEntropyLoss(reduction='sum', ignore_index=NO_LABEL)
        self.consistency_criterion = softmax_mse_loss

    def update_ema_variables(self, model, ema_model, alpha, global_step):
        alpha = min(1 - 1 / (global_step + 1), alpha)
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)

    def get_current_consistency_weight(self,epoch):
        return Config['consistency'] * ramps.sigmoid_rampup(epoch, Config['consistency_rampup'])

    def transform(self, batch):
        stu = {k: v.clone() if torch.is_tensor(v) else v for k, v in batch.items()}
        tea = {k: v.clone() if torch.is_tensor(v) else v for k, v in batch.items()}

        input_ids = batch['input_ids']
        mask = (input_ids >= 4) & (input_ids <= 28)

        for i in range(input_ids.size(0)):
            eligible_positions = torch.nonzero(mask[i]).squeeze()
            if eligible_positions.numel() == 0:
                continue

            num_to_replace_stu = max(1, int(0.1 * len(eligible_positions)))
            positions_to_replace_stu = random.sample(eligible_positions.tolist(), num_to_replace_stu)
            for pos in positions_to_replace_stu:
                stu['input_ids'][i, pos] = random.randint(4, 28)

            remaining_positions = list(set(eligible_positions.tolist()) - set(positions_to_replace_stu))
            if not remaining_positions:
                remaining_positions = eligible_positions.tolist()
            num_to_replace_tea = max(1, int(0.1 * len(remaining_positions)))
            positions_to_replace_tea = random.sample(remaining_positions, num_to_replace_tea)
            for pos in positions_to_replace_tea:
                tea['input_ids'][i, pos] = random.randint(4, 28)

        return stu, tea

    def _consistency_loss(self, train_batch):
        unlabeled_mask = (train_batch['labels'] == NO_LABEL)
        if unlabeled_mask.sum() == 0:
            return torch.tensor(0.0, device=Config['device'])

        batch_unlabeled = {
            k: v[unlabeled_mask] if torch.is_tensor(v) else v
            for k, v in train_batch.items()
        }

        stu_unlabeled, tea_unlabeled = self.transform(batch_unlabeled)

        self.teacher_model.eval()
        with torch.no_grad():
            teacher_logits, _ = self.teacher_model(tea_unlabeled)

        student_logits, _ = self.student_model(stu_unlabeled)

        return self.consistency_criterion(student_logits, teacher_logits)

    def train_step(self, train_batch, epoch, global_step):
        train_batch = {
            k: v.to(Config['device']) if torch.is_tensor(v) else v
            for k, v in train_batch.items()
        }

        self.student_model.train()
        self.teacher_model.train()

        stu_output, clip_loss = self.student_model(train_batch)
        labels = train_batch["labels"]

        labeled_mask = (labels != NO_LABEL)
        train_minibatch_size = labeled_mask.sum()
        if train_minibatch_size > 0:
            supervised_loss = self.supervised_criterion(stu_output, labels) / train_minibatch_size
        else:
            supervised_loss = torch.tensor(0.0, device=Config['device'])

        consistency_weight = self.get_current_consistency_weight(epoch)
        consistency_loss = self._consistency_loss(train_batch)

        total_loss = supervised_loss + consistency_weight * consistency_loss + Config['clip_loss_weight'] * clip_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.student_model.parameters(), max_norm=1)
        self.optimizer.step()
        self.lr_scheduler.step()

        self.update_ema_variables(
            model=self.student_model,
            ema_model=self.teacher_model,
            alpha=Config['ema_alpha'],
            global_step=global_step
        )

        preds = stu_output.argmax(dim=1)
        correct = (preds[labeled_mask] == labels[labeled_mask]).float().sum()

        return supervised_loss, correct, train_minibatch_size

    def validate(self, val_batch):
        val_batch = {k: v.to(Config['device']) if torch.is_tensor(v) else v for k, v in val_batch.items()}
        self.student_model.eval()
        with torch.no_grad():
            outputs, _ = self.student_model(val_batch)
            labeled_minibatch_size = len(val_batch["labels"])
            loss = self.supervised_criterion(outputs, val_batch["labels"]) / labeled_minibatch_size

            preds = outputs.argmax(dim=1)
            correct = (preds == val_batch["labels"]).sum().item()

            pred = outputs.argmax(dim=1).cpu().numpy()
            label = val_batch["labels"].cpu().numpy()

        return loss, correct, pred, label

def main():
    model_args = {
        "vocab_size": tokenizer.vocab_size,
        "d_model": Config['d_model'],
        "nhead": Config['nhead'],
        "num_encoder_layers": Config['num_encoder_layers'],
        "dim_feedforward": Config['dim_feedforward'],
        "max_len": Config['max_len'],
        "dropout": Config['dropout']
    }

    train_loader, val_loader = create_data_loaders()

    def create_model(ema=False):
        net = MyTransformerModel(**model_args).to(Config['device'])
        if ema:
            for param in net.parameters():
                param.requires_grad = False
        return net

    stu_model = create_model()
    ema_model = create_model(ema=True)

    optimizer = torch.optim.AdamW(stu_model.parameters(),
                               lr=float(Config['learning_rate']),
                                weight_decay=float(Config['weight_decay'])
                                )

    num_update_steps_per_epoch = len(train_loader)
    num_training_steps = Config['epochs'] * num_update_steps_per_epoch
    lr_scheduler = get_scheduler("cosine",optimizer=optimizer, num_warmup_steps=100,num_training_steps=num_training_steps)

    model = MeanTeacherModel(stu_model, ema_model, optimizer, lr_scheduler)

    best_val_f1 = 0 

    global_step = 0
    for epoch in range(Config['epochs']):
        print(f"\nEpoch {epoch + 1}/{Config['epochs']}")

        train_correct = 0
        train_size = 0
        train_epoch_loss = []
        train_iter = iter(train_loader)
        for _ in range(len(train_loader)):
            global_step += 1
            train_batch = next(train_iter)
            supervised_loss, correct, train_minibatch_size = model.train_step(train_batch, epoch, global_step)

            train_epoch_loss.append(supervised_loss.item())
            train_correct += correct.item()
            train_size += train_minibatch_size

        train_acc = train_correct/train_size

        val_correct = 0
        valid_epoch_loss = []
        preds_all = []
        labels_all = []
        val_iter = iter(val_loader)
        for _ in range(len(val_loader)):
            val_batch = next(val_iter)
            loss, correct, pred, label = model.validate(val_batch)

            preds_all.extend(pred)
            labels_all.extend(label)

            valid_epoch_loss.append(loss.item())
            val_correct += correct

        acc, precision, recall, f1 = calculate_metrics(labels_all, preds_all)

        if f1 >= best_val_f1:
            best_val_f1 = f1
            torch.save({
                'model_state_dict': model.student_model.state_dict()
            }, Config['best_model'])
            print(f'epoch:{epoch},{round(acc,4)}, {round(precision,4)},{round(recall,4)},{round(f1,4)},{round(best_val_f1,4)}')
    print(best_val_f1)
     
if __name__ == "__main__":
    main()
