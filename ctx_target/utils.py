import os
join=os.path.join
import torch.nn.functional as F
import numpy as np
import torch
from Bio.PDB import PDBParser, SASA, MMCIFParser
import pandas as pd
import random
import subprocess
from transformers import set_seed
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from torch.utils.data import Dataset
import sys
sys.path.append(".")

three_to_one = {'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
                'GLU': 'E', 'GLN': 'Q', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 
                'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
                'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'}

MaxASA = {
    'ALA': 129.0, 'ARG': 274.0, 'ASN': 195.0, 'ASP': 193.0, 'CYS': 167.0,
    'GLU': 223.0, 'GLN': 225.0, 'GLY': 104.0, 'HIS': 224.0, 'ILE': 197.0,
    'LEU': 201.0, 'LYS': 236.0, 'MET': 224.0, 'PHE': 240.0, 'PRO': 159.0,
    'SER': 155.0, 'THR': 172.0, 'TRP': 285.0, 'TYR': 263.0, 'VAL': 174.0
}

def set_my_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    set_seed(seed)

def calculate_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    return accuracy, precision, recall, f1

ESM_API_URL = "https://api.esmatlas.com/foldSequence/v1/pdb/"

def fold(seq):
    r = subprocess.run(["curl", "-X", "POST", "-k", "--data", seq, ESM_API_URL], capture_output=True, text=True)
    return r.stdout

def cysteine_position(seqs,max_len):
    cysteine_positions_all = []
    for seq in seqs:
        cysteine_positions = [0] * max_len
        for i, char in enumerate(seq[0:max_len]):
            if char == 'C':
                cysteine_positions[i] = 1
        cysteine_positions_all.append(cysteine_positions)
    return cysteine_positions_all

def calculate_RSA(Structure_index, pdb_path, max_len, file_format="pdb", sequences=None):
    sasa_vector_all = []
    seq_vector_all = []
    pdb_vector_all = []
    
    for i, index in enumerate(Structure_index):
        
        if file_format == "pdb":
            parser = PDBParser()
            pdb_file = join(pdb_path, f"{index}.pdb")
        elif file_format == "cif":
            parser = MMCIFParser()
            pdb_file = join(pdb_path, f"{index}"+"_model"+".cif")
        else:
            raise ValueError("file_format must be 'pdb' or 'cif'")

        if not os.path.exists(pdb_file):
            if sequences and i < len(sequences):
                print(f"  PDB not found for {index}, predicting structure via ESMFold...")
                pdb_text = fold(sequences[i].strip())
                os.makedirs(pdb_path, exist_ok=True)
                with open(pdb_file, "w") as f:
                    f.write(pdb_text)
            else:
                raise FileNotFoundError(f"PDB file not found: {pdb_file}")

        struct = parser.get_structure('peptide', pdb_file)
        ori_seq = struct.get_residues()
        ori_seq = [three_to_one[residue.resname] for residue in ori_seq]
        ori_seq =''.join(ori_seq)
        sr = SASA.ShrakeRupley()
        sr.compute(struct, level="S")
        sasa_vector = [-1.0] * max_len
        for i, residue in enumerate(struct.get_residues()):
            if i < max_len:
                residue_sasa = sum(atom.sasa for atom in residue.get_atoms())
                RSA = (residue_sasa / MaxASA[residue.resname])
                if RSA > 1:
                    RSA = 1
                sasa_vector[i] = RSA
            else:
                pass
        sasa_vector_all.append(sasa_vector)
        seq_vector_all.append(ori_seq)
        pdb_vector_all.append(index)
    return sasa_vector_all

def softmax_kl_loss(input_logits, target_logits):
    assert input_logits.size() == target_logits.size()
    input_log_softmax = F.log_softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    return F.kl_div(input_log_softmax, target_softmax, reduction='sum')

def softmax_mse_loss(input_logits, target_logits):
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    num_classes = input_logits.size()[1]
    return F.mse_loss(input_softmax, target_softmax, reduction='sum') / num_classes

def symmetric_mse_loss(input1, input2):
    assert input1.size() == input2.size()
    num_classes = input1.size()[1]
    return torch.sum((input1 - input2)**2) / num_classes
