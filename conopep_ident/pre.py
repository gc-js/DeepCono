import os
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, AutoConfig
from torchcrf import CRF
import pandas as pd
import re
from collections import OrderedDict
import numpy as np
import subprocess 
from tqdm import tqdm
import random
from utils import device, setup_seed, C_framework, ix_to_tag, tag_to_ix, BioTool, model_prediction, filter_sequences, remove_outliers_zscore, clean_value, sort_output_by_field, process_prediction_data, predict_ptm_modifications
from model import Model, tokenizer
import yaml

def load_config(config_path='conopep_ident/config.yaml'):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def DeepCono_conopeptide(query_file, file_name, max_len, min_len, e_value, model, save_path, signal):
    if signal:
        cono_dict = BioTool(query_file, max_len, e_value)
        Precursors = cono_dict["precursors"]
        Familys = cono_dict["family_results"]["target_names"]
        Superfamilys = cono_dict["superfamily_results"]["target_names"]
    else:
        Precursors = []
        with open(query_file, 'r') as file:
            current_sequence = ""
            for line in file:
                line = line.strip()
                if line.startswith('>'):
                    if current_sequence:
                        Precursors.append(current_sequence)
                        current_sequence = ""
                else:
                    current_sequence += line
            if current_sequence:
                Precursors.append(current_sequence)

    output_seq, output_Measure, Position_Confidence = model_prediction(
        Precursors, tokenizer, model, device, ix_to_tag, max_len
    )

    Conopeptides, unique_confidences, confidences_avg = process_prediction_data(output_Measure, Position_Confidence, min_len)

    framework_all = C_framework(Conopeptides)

    PTM_pyroglu, PTM_amidation, PTM_gla = predict_ptm_modifications(Precursors, Conopeptides)

    if signal:
        for i in range(len(Conopeptides)):
            if len(Conopeptides[i]) > 0 and Conopeptides[i] != "-":
                if len(Familys[i]) == 0:
                    Familys[i] = "Unknown"
                if len(Superfamilys[i]) == 0:
                    Superfamilys[i] = "Unknown"

    Clean_Conopeptides = []
    for i in range(len(Conopeptides)):
        pep = Conopeptides[i].strip('*')
        if len(set(pep)) == 1 or "CVCVC" in pep or "CCCC" in pep or len(pep) < min_len:
            Clean_Conopeptides.append('')
        else:
            Clean_Conopeptides.append(pep)

    output = OrderedDict()
    output['Precursor'] = Precursors

    if signal:
        output['Evalue'] = cono_dict["evalue"]
        output['Familys'] = Familys
        output['Familys_evalue'] = cono_dict["family_results"]["evalue_domains"]
        output['Superfamilys'] = Superfamilys
        output['Superfamilys_evalue'] = cono_dict["superfamily_results"]["evalue_domains"]

    output['Conopeptides'] = Conopeptides
    output['Position_confidence'] =  unique_confidences
    output['Position_confidence_avg'] =  confidences_avg

    output['Frameworks'] = framework_all
    output['PTM_pyroglu'] = PTM_pyroglu
    output['PTM_amidation'] = PTM_amidation
    output['PTM_gla'] = PTM_gla
    output['Clean_Conopeptides'] = Clean_Conopeptides
    
    for key in output:
        output[key] = [clean_value(x) for x in output[key]]

    if signal:
        output = sort_output_by_field(output, 'Evalue', reverse=False)
    
    summary_df = pd.DataFrame(pd.DataFrame.from_dict(output, orient='index').values.T, columns=list(output.keys()))
    summary_df.to_csv(os.path.join(save_path, file_name.split(".")[0]+".csv"), index=False)

if __name__ == "__main__":
    config = load_config()
    
    setup_seed(config['seed'])
    max_len = config['max_len']
    min_len = config['min_len']
    e_value = config['evalue']
    signal = config['signal']
    base_model_path = config['base_model_path']
    adapter = config['adapter']
    model_path = config['best_model_path']
    input_path = config['input_path']
    save_path = config['save_path']
    os.makedirs(save_path, exist_ok=True)

    model = Model(len(tag_to_ix), base_model_path, adapter, prob=True)
    model.load_state_dict(torch.load(model_path), strict=False)
    model.to(device)

    query_files = os.listdir(input_path)
    for file_name in tqdm(query_files):
        query_file = os.path.join(input_path,file_name)
        DeepCono_conopeptide(query_file, file_name, max_len, min_len, e_value, model, save_path, signal)
