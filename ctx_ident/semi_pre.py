import torch
import pandas as pd
from tqdm import tqdm
from model import MyTransformerModel
from utils import set_my_seed
import warnings
warnings.filterwarnings("ignore", message="The PyTorch API of nested tensors is in prototype stage")
import torch.nn.functional as F
from dataset import load_config, tokenizer, create_test_loaders
Config = load_config('ctx_ident/config.yaml')
device = Config['device']
set_my_seed(4)

def validate(val_batch, student_model):
    val_batch = {k: v.to(Config['device']) if torch.is_tensor(v) else v for k, v in val_batch.items()}
    student_model.eval()
    with torch.no_grad():
        logits, _ = student_model(val_batch)
        probs = F.softmax(logits, dim=-1)
        preds = logits.argmax(dim=1)
    return preds, probs

def load_stu_model(model_path):
    model_args = {
        "vocab_size": tokenizer.vocab_size,
        "d_model": Config['d_model'],
        "nhead": Config['nhead'],
        "num_encoder_layers": Config['num_encoder_layers'],
        "dim_feedforward": Config['dim_feedforward'],
        "max_len": Config['max_len'],
        "dropout": Config['dropout']
    }

    stu_model = MyTransformerModel(**model_args).to(Config['device'])
    checkpoint = torch.load(model_path, map_location=Config['device'])
    stu_model.load_state_dict(checkpoint['model_state_dict'])
    stu_model.eval()
    return stu_model

def main():
    model = load_stu_model(Config['best_model'])
    test_loader = create_test_loaders()
    all_preds = []
    all_probs = []
    for i, test_batch in tqdm(enumerate(test_loader)):
        preds, probs = validate(test_batch, model) 
        for pred, prob in zip(preds.cpu().tolist(), probs.cpu().tolist()):
            label_str = "CTX" if pred == 1 else "Non-CTX"
            pred_prob = prob[pred]
            print(label_str) 
            all_preds.append(label_str)
            all_probs.append(pred_prob)

    results_df = pd.DataFrame({
        'Prediction': all_preds,
        'Probability': all_probs
    })

    results_df.to_csv(Config['output'], index=False)

if __name__ == "__main__":
    main()