import torch
from transformers import set_seed
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import random
import subprocess
import pandas as pd
import re
from collections import OrderedDict
import yaml

def load_config(config_path='conopep_ident/config.yaml'):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

config = load_config()
device = config['device']
signal_db = config['signal_db']
published_db = config['published_db']
superfamily_db = config['superfamily_db']
family_db = config['family_db']

tag_to_ix = {"O":0, "M":1, "[PAD]":2}
ix_to_tag = {0: "O", 1: "M", 2: "[PAD]"}

def setup_seed(seed):
    set_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def calculate_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro', zero_division=1)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=1)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=1)
    return accuracy, precision, recall, f1

def do_nothing(*args, **kwargs):
    pass

def run_mmseqs_search(query_file, database, output_file, e_value):
    cmd = [
        'mmseqs', 'easy-search', query_file,
        database, 
        output_file, 
        'tmp', '-s', '7.5', '--format-output', 'query,qseq,target,tseq,evalue', 
        '--comp-bias-corr', '0', '-e', f'{e_value}'
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    col_names = "query,qseq,target,tseq,evalue".split(',')
    df = pd.read_table(output_file, names=col_names)
    return df

def process_mmseqs_results(df, max_len):
    sorted_df = df.sort_values(by='query')
    sorted_df.reset_index(drop=True, inplace=True)
    df_min_evalue = sorted_df.loc[sorted_df.groupby('query')['evalue'].idxmin()]

    unique_precursors = df_min_evalue["qseq"].tolist()
    superfamily = df_min_evalue["target"].tolist()
    evalue = df_min_evalue["evalue"].tolist()

    filtered_indices = [i for i in range(len(unique_precursors)) if len(unique_precursors[i]) <= max_len]

    unique_precursors = [unique_precursors[i] for i in filtered_indices]
    superfamily = [superfamily[i] for i in filtered_indices]
    evalue = [evalue[i] for i in filtered_indices]

    return unique_precursors, superfamily, evalue

def run_hmmscan(hmm_db, output_file, query_file):
    cmd = ['hmmscan', '--tblout', output_file, '--noali', hmm_db, query_file]
    result = subprocess.run(cmd, capture_output=True, text=True)

def parse_hmmscan_results(output_file, seq_count):
    results = {f"query_{i}": {"target_name": "", "evalue_domain": ""} for i in range(seq_count)}
    
    query_names = []
    target_names = []
    evalue_domains = []
    
    with open(output_file, 'r') as f_in:
        seen_query_names = set()
        lines = f_in.readlines()
        for line in lines:
            if line.startswith('#'):
                continue
            fields = line.split()
            query_name = fields[2]
            if query_name in seen_query_names:
                continue
            else:
                seen_query_names.add(query_name)

            if query_name in results:
                target_name = fields[0]
                evalue_domain = fields[4]
                results[query_name] = {"target_name": target_name, "evalue_domain": evalue_domain}
        
        for query_name, data in results.items():
            query_names.append(query_name)
            target_names.append(data['target_name'])
            evalue_domains.append(data['evalue_domain'])
            
    return query_names, target_names, evalue_domains

def write_query_fasta(seqs, output_file):
    with open(output_file, 'w') as file:
        for i in range(len(seqs)):
            file.write(f">query_{i}")
            file.write('\n')
            file.write(seqs[i])
            file.write('\n')

def BioTool(query_file, max_len, e_value):
    df_signal = run_mmseqs_search(
        query_file,
        signal_db,
        'conopep_ident/DB/mmseqs_signal.txt',
        e_value
    )
    precursors, superfamily, evalue_list = process_mmseqs_results(df_signal, max_len)

    df_published = run_mmseqs_search(
        query_file,
        published_db,
        'conopep_ident/DB/mmseqs_published.txt',
        e_value
    )
    precursors_published, superfamily_published, evalue_published = process_mmseqs_results(df_published, max_len)

    for seq, superfamily_pub, evalue_pub in zip(precursors_published, superfamily_published, evalue_published):
        if seq not in precursors and len(seq) <= max_len:
            precursors.append(seq)
            superfamily.append(superfamily_pub)
            evalue_list.append(evalue_pub)
    
    seqs = precursors

    write_query_fasta(seqs, "conopep_ident/DB/query.fasta")

    run_hmmscan(
        family_db,
        'conopep_ident/DB/familys.csv',
        'conopep_ident/DB/query.fasta'
    )

    run_hmmscan(
        superfamily_db,
        'conopep_ident/DB/superfamilys.csv',
        'conopep_ident/DB/query.fasta'
    )

    query_name_familys, target_name_familys, evalue_domain_familys = parse_hmmscan_results(
        'conopep_ident/DB/familys.csv',
        len(seqs)
    )
    
    query_name_superfamilys, target_name_superfamilys, evalue_domain_superfamilys = parse_hmmscan_results(
        'conopep_ident/DB/superfamilys.csv',
        len(seqs)
    )
    
    return {
        'precursors': seqs,
        'superfamily': superfamily,
        'evalue': evalue_list,
        'family_results': {
            'query_names': query_name_familys,
            'target_names': target_name_familys,
            'evalue_domains': evalue_domain_familys
        },
        'superfamily_results': {
            'query_names': query_name_superfamilys,
            'target_names': target_name_superfamilys,
            'evalue_domains': evalue_domain_superfamilys
        }
    }

def predict_ptm_modifications(full_sequences, mature_regions):

    PTM_pyroglu = []
    PTM_amidation = []
    PTM_gla = []
    
    for i in range(len(full_sequences)):
        full_sequence = full_sequences[i]
        mature_region = mature_regions[i]

        if mature_region == "-":
            PTM_pyroglu.append('')
            PTM_amidation.append('')
            PTM_gla.append('')
        else:
            start_index = full_sequence.find(mature_region)

            pre_region = full_sequence[:start_index]
            post_region = full_sequence[start_index + len(mature_region):]

            if pre_region and mature_region.startswith('Q'):
                PTM_pyroglu.append('Y')
            else:
                PTM_pyroglu.append('N')

            if post_region and post_region.startswith('G'):
                PTM_amidation.append('Y')
            else:
                PTM_amidation.append('N')

            pattern = r'[KR].{2,3}[ACGILMFSV].{3,4}[KRN]'

            if re.search(pattern, pre_region) or re.search(pattern, post_region):
                PTM_gla.append('Y')
            else:
                PTM_gla.append('N')
                
    return PTM_pyroglu, PTM_amidation, PTM_gla

def process_prediction_data(output_Measure, Position_Confidence, min_len):
    dict_aa = [{measure: confidence} for measure, confidence in zip(output_Measure, Position_Confidence)]
    
    peptides = []
    confidences = []
    confidences_avg = []
    
    for item in dict_aa:
        measure = list(item.keys())[0]
        confidence = list(item.values())[0]

        peptide_only = measure.strip('*')
        peptides.append("-" if peptide_only == "" else peptide_only)

        start = 0
        for i, char in enumerate(confidence):
            if char != '*':
                start = i
                break

        end = len(confidence)
        for i in range(len(confidence) - 1, -1, -1):
            if confidence[i] != '*':
                end = i + 1
                break

        confidence_only = confidence[start:end]
        if all(c == '*' for c in confidence_only):
            confidences.append('-')
            confidences_avg.append('-')
        else:
            confidence_only_temp = [-1 if x == '*' else x for x in confidence_only]
            rounded_conf = np.round(confidence_only_temp, 3)
            confidences.append(rounded_conf)
            avg_conf = np.format_float_positional(np.average(rounded_conf), precision=4)
            confidences_avg.append(avg_conf)
    
    for i in range(len(peptides)):
        if str(confidences[i]) != "-":
            peptides[i], confidences[i] = filter_sequences([peptides[i]], confidences[i].tolist(), min_len)
            confidences[i], peptides[i] = remove_outliers_zscore(confidences[i], peptides[i], threshold=-1)
            confidences_avg[i] = np.round(np.average(np.round(confidences[i],3)),3)

    return peptides, confidences, confidences_avg

def model_prediction(seqs, tokenizer, model, device, ix_to_tag, max_len):
    output_seq = []
    output_Measure = []
    Position_Confidence = []
    
    for seq in seqs:
        processed_seq = seq.replace('*', '-')
        
        tokenizer_test = tokenizer(processed_seq, return_tensors='pt', max_length=max_len, truncation=True)
        
        with torch.no_grad():
            input_ids = tokenizer_test['input_ids'].to(device)
            attention_mask = tokenizer_test['attention_mask'].to(device)
            tags, probability = model(input_ids, attention_mask)

            Measure = []
            probability_Measure = []
            char_index = 0
            
            predicted_tags = [ix_to_tag[x] for x in tags[0]]
            probability_list = probability.tolist()
            
            for idx, tag in enumerate(predicted_tags[1:len(processed_seq)+1]):
                if tag == "O" or tag == "<pad>":
                    Measure.append("*")
                    probability_Measure.append("*")
                elif tag == "M":
                    if processed_seq[char_index] == '-':
                        Measure.append("*")
                        probability_Measure.append("*")
                    else:
                        probability_Measure.append(probability_list[idx+1])
                        Measure.append(processed_seq[char_index])
                char_index += 1
            
            Position_Confidence.append(probability_Measure)
            output_seq.append(processed_seq)
            output_Measure.append(''.join(Measure))
    
    return output_seq, output_Measure, Position_Confidence

def filter_sequences(sequences, confidences, min_len_peptide):
    filtered_sequences = []
    filtered_confidences = []

    for seq in sequences:
        segments = re.findall(r'[A-Za-z]+', seq)
        
        segment_confidences = []
        for segment in segments:
            segment_start = seq.find(segment)
            segment_end = segment_start + len(segment)
            segment_confidences.append(confidences[segment_start:segment_end])

        filtered_segments = []
        for segment in segments:
            if len(segment) < min_len_peptide:
                filtered_segments.append('*' * len(segment))
            else:
                filtered_segments.append(segment)

        filtered_seq = re.sub(r'[A-Za-z]+', lambda m: filtered_segments.pop(0), seq)
        filtered_sequences.append(filtered_seq)

        filtered_conf = []
        seg_idx = 0
        for char in filtered_seq:
            if char == '*':
                filtered_conf.append(-1)
            else:
                filtered_conf.append(segment_confidences[seg_idx].pop(0))
                if not segment_confidences[seg_idx]:
                    seg_idx += 1

        filtered_confidences.append(filtered_conf)
    
    first_non_star_index = next((i for i, char in enumerate(filtered_sequences[0]) if char != '*'), len(filtered_sequences[0]))

    # Slice the sequences and confidences to remove the leading * characters
    out_sequences = filtered_sequences[0][first_non_star_index:]
    out_confidences = filtered_confidences[0][first_non_star_index:]

    return out_sequences, out_confidences

def remove_outliers_zscore(Confidence, Measure, threshold=-1):
    if len(Confidence) != 0 and -1 not in Confidence and not Measure.endswith('C'):
        Confidence_arr = np.array(Confidence)
        Measure_arr = np.array(list(Measure))
        z_scores = (Confidence_arr - np.mean(Confidence_arr)) / np.std(Confidence_arr)
        mask = z_scores >= threshold
        n = len(Confidence_arr)
        check_positions = min(4, n)

        last_valid = n
        for i in range(n-1, max(n-1-check_positions, -1), -1):
            if not mask[i]:
                last_valid = i
        
        if last_valid < n:
            Confidence = Confidence_arr[:last_valid].tolist()
            Measure = "".join(Measure_arr[:last_valid])
        else:
            Confidence = Confidence_arr.tolist()
            Measure = "".join(Measure_arr)
            
        return Confidence, Measure
    else:
        return Confidence, Measure

def clean_value(x):
    if isinstance(x, (list, np.ndarray)):
        if len(x) == 0 or (len(x) == 1 and x[0] == "-"):
            return ""
        elif len(x) == 1:
            return x[0]
        else:
            return x
    elif isinstance(x, str) and (x == "-" or x == ""):
        return ""
    return x

def sort_output_by_field(output, field_name, reverse=False):
    rows = list(zip(*output.values()))
    field_idx = list(output.keys()).index(field_name)
    rows.sort(key=lambda x: float(x[field_idx]), reverse=reverse)
    sorted_output = OrderedDict()
    keys = list(output.keys())
    for i, key in enumerate(keys):
        sorted_output[key] = [row[i] for row in rows]
    
    return sorted_output

def C_framework(peptides_all):
    framework_dict = {'C[^C]+C[^C]+C[^C]+C[^C]+C[^C]+C[^C]+C[^C]+C[^C]+C[^C]+C[^C]+C[^C]+C':'XXXIII',
                    'C[^C]+C[^C]+C[^C]+C[^C]+C[^C]+C[^C]+C[^C]+C[^C]+C[^C]+C':'VIII',
                    'C[^C]+C[^C]+C[^C]+C[^C]+C[^C]+C[^C]+C[^C]+C':'XXII',
                    'C[^C]+C[^C]+C[^C]+C[^C]+C[^C]+C':'IX',
                    'C[^C]+C[^C]+C[^C]+C':'XIV',
                    'CC[^C]+C[^C]+C':'I',
                    'C[^C]+C[^C]+C[^C]+C[^C]+CC[^C]+C[^C]+C':'XII',
                    'C[^C]+C[^C]+C[^C]+C[^C]+CC[^C]+CC':'XXVI',
                    'C[^C]+C[^C]+C[^C]+C[^C]+CC':'XXV',
                    'C[^C]+C[^C]+C[^C]+CC[^C]+C[^C]+C[^C]+C[^C]+C[^C]+C':'XXVIII',
                    'C[^C]+C[^C]+C[^C]+CC[^C]+C[^C]+C[^C]+C':'XIII',
                    'C[^C]+C[^C]+C[^C]+CC[^C]+C':'XXIII',
                    'C[^C]+C[^C]+C[^C]+CCC[^C]+C[^C]+C[^C]+C[^C]+C':'XIX',
                    'C[^C]+C[^C]+C[^C]+CCC[^C]+C[^C]+C':'XXVII',
                    'C[^C]+C[^C]+CC[^C]+C[^C]+C[^C]+C[^C]+C':'XV',
                    'C[^C]+C[^C]+CC[^C]+C[^C]+C':'VI/VII',
                    'C[^C]+C[^C]+CC[^C]+C[^C]+CC[^C]+C':'XVII',
                    'C[^C]+C[^C]+CC[^C]+CC[^C]+C[^C]+C':'XI',
                    'C[^C]+C[^C]+CC[^C]+CC':'XVIII',
                    'C[^C]+C[^C]+CC':'XVI',
                    'C[^C]+C[^C]+CCC[^C]+C[^C]+C[^C]+C[^C]+CC':'XXX',
                    'C[^C]+CC[^C]+C[^C]+C[^C]+C':'XXXII',
                    'C[^C]+CC[^C]+C[^C]+CC[^C]+C[^C]+C[^C]+C[^C]+C':'XX',
                    'C[^C]+CC[^C]+C':'XXIV',
                    'CC[^C]+C[^C]+[PO]C':'X',
                    'CC[^C]+C[^C]+C[^C]+C[^C]+C':'IV',
                    'CC[^C]+C[^C]+C[^C]+C[^C]+CC[^C]+C[^C]+C[^C]+C':'XXI',
                    'CC[^C]+C[^C]+C[^C]+CC':'III',
                    'CC[^C]+CC':'V',
                    'CCC[^C]+C[^C]+C[^C]+C':'II',
                    'CCC[^C]+C[^C]+CC[^C]+C[^C]+C':'XXIX',
                    '[^C]+C[^C]+C[^C]+C[^C]+C[^C]+C[^C]+C[^C]+C[^C]+C[^C]+C[^C]+C[^C]+C[^C]+C':'XXXIII',
                    '[^C]+C[^C]+C[^C]+C[^C]+C[^C]+C[^C]+C[^C]+C[^C]+C[^C]+C[^C]+C':'VIII',
                    '[^C]+C[^C]+C[^C]+C[^C]+C[^C]+C[^C]+C[^C]+C[^C]+C':'XXII',
                    '[^C]+C[^C]+C[^C]+C[^C]+C[^C]+C[^C]+C':'IX',
                    '[^C]+C[^C]+C[^C]+C[^C]+C':'XIV',
                    '[^C]+CC[^C]+C[^C]+C':'I',
                    '[^C]+C[^C]+C[^C]+C[^C]+C[^C]+CC[^C]+C[^C]+C':'XII',
                    '[^C]+C[^C]+C[^C]+C[^C]+C[^C]+CC[^C]+CC':'XXVI',
                    '[^C]+C[^C]+C[^C]+C[^C]+C[^C]+CC':'XXV',
                    '[^C]+C[^C]+C[^C]+C[^C]+CC[^C]+C[^C]+C[^C]+C[^C]+C[^C]+C':'XXVIII',
                    '[^C]+C[^C]+C[^C]+C[^C]+CC[^C]+C[^C]+C[^C]+C':'XIII',
                    '[^C]+C[^C]+C[^C]+C[^C]+CC[^C]+C':'XXIII',
                    '[^C]+C[^C]+C[^C]+C[^C]+CCC[^C]+C[^C]+C[^C]+C[^C]+C':'XIX',
                    '[^C]+C[^C]+C[^C]+C[^C]+CCC[^C]+C[^C]+C':'XXVII',
                    '[^C]+C[^C]+C[^C]+CC[^C]+C[^C]+C[^C]+C[^C]+C':'XV',
                    '[^C]+C[^C]+C[^C]+CC[^C]+C[^C]+C':'VI/VII',
                    '[^C]+C[^C]+C[^C]+CC[^C]+C[^C]+CC[^C]+C':'XVII',
                    '[^C]+C[^C]+C[^C]+CC[^C]+CC[^C]+C[^C]+C':'XI',
                    '[^C]+C[^C]+C[^C]+CC[^C]+CC':'XVIII',
                    '[^C]+C[^C]+C[^C]+CC':'XVI',
                    '[^C]+C[^C]+C[^C]+CCC[^C]+C[^C]+C[^C]+C[^C]+CC':'XXX',
                    '[^C]+C[^C]+CC[^C]+C[^C]+C[^C]+C':'XXXII',
                    '[^C]+C[^C]+CC[^C]+C[^C]+CC[^C]+C[^C]+C[^C]+C[^C]+C':'XX',
                    '[^C]+C[^C]+CC[^C]+C':'XXIV',
                    '[^C]+CC[^C]+C[^C]+[PO]C':'X',
                    '[^C]+CC[^C]+C[^C]+C[^C]+C[^C]+C':'IV',
                    '[^C]+CC[^C]+C[^C]+C[^C]+C[^C]+CC[^C]+C[^C]+C[^C]+C':'XXI',
                    '[^C]+CC[^C]+C[^C]+C[^C]+CC':'III',
                    '[^C]+CC[^C]+CC':'V',
                    '[^C]+CCC[^C]+C[^C]+C[^C]+C':'II',
                    '[^C]+CCC[^C]+C[^C]+CC[^C]+C[^C]+C':'XXIX',
                    'C[^C]+C[^C]+C[^C]+C[^C]+C[^C]+C[^C]+C[^C]+C[^C]+C[^C]+C[^C]+C[^C]+C[^C]+':'XXXIII',
                    'C[^C]+C[^C]+C[^C]+C[^C]+C[^C]+C[^C]+C[^C]+C[^C]+C[^C]+C[^C]+':'VIII',
                    'C[^C]+C[^C]+C[^C]+C[^C]+C[^C]+C[^C]+C[^C]+C[^C]+':'XXII',
                    'C[^C]+C[^C]+C[^C]+C[^C]+C[^C]+C[^C]+':'IX',
                    'C[^C]+C[^C]+C[^C]+C[^C]+':'XIV',
                    'CC[^C]+C[^C]+C[^C]+':'I',
                    'C[^C]+C[^C]+C[^C]+C[^C]+CC[^C]+C[^C]+C[^C]+':'XII',
                    'C[^C]+C[^C]+C[^C]+C[^C]+CC[^C]+CC[^C]+':'XXVI',
                    'C[^C]+C[^C]+C[^C]+C[^C]+CC[^C]+':'XXV',
                    'C[^C]+C[^C]+C[^C]+CC[^C]+C[^C]+C[^C]+C[^C]+C[^C]+C[^C]+':'XXVIII',
                    'C[^C]+C[^C]+C[^C]+CC[^C]+C[^C]+C[^C]+C[^C]+':'XIII',
                    'C[^C]+C[^C]+C[^C]+CC[^C]+C[^C]+':'XXIII',
                    'C[^C]+C[^C]+C[^C]+CCC[^C]+C[^C]+C[^C]+C[^C]+C[^C]+':'XIX',
                    'C[^C]+C[^C]+C[^C]+CCC[^C]+C[^C]+C[^C]+':'XXVII',
                    'C[^C]+C[^C]+CC[^C]+C[^C]+C[^C]+C[^C]+C[^C]+':'XV',
                    'C[^C]+C[^C]+CC[^C]+C[^C]+C[^C]+':'VI/VII',
                    'C[^C]+C[^C]+CC[^C]+C[^C]+CC[^C]+C[^C]+':'XVII',
                    'C[^C]+C[^C]+CC[^C]+CC[^C]+C[^C]+C[^C]+':'XI',
                    'C[^C]+C[^C]+CC[^C]+CC[^C]+':'XVIII',
                    'C[^C]+C[^C]+CC[^C]+':'XVI',
                    'C[^C]+C[^C]+CCC[^C]+C[^C]+C[^C]+C[^C]+CC[^C]+':'XXX',
                    'C[^C]+CC[^C]+C[^C]+C[^C]+C[^C]+':'XXXII',
                    'C[^C]+CC[^C]+C[^C]+CC[^C]+C[^C]+C[^C]+C[^C]+C[^C]+':'XX',
                    'C[^C]+CC[^C]+C[^C]+':'XXIV',
                    'CC[^C]+C[^C]+[PO]C[^C]+':'X',
                    'CC[^C]+C[^C]+C[^C]+C[^C]+C[^C]+':'IV',
                    'CC[^C]+C[^C]+C[^C]+C[^C]+CC[^C]+C[^C]+C[^C]+C[^C]+':'XXI',
                    'CC[^C]+C[^C]+C[^C]+CC[^C]+':'III',
                    'CC[^C]+CC[^C]+':'V',
                    'CCC[^C]+C[^C]+C[^C]+C[^C]+':'II',
                    'CCC[^C]+C[^C]+CC[^C]+C[^C]+C[^C]+':'XXIX',
                    '[^C]+C[^C]+C[^C]+C[^C]+C[^C]+C[^C]+C[^C]+C[^C]+C[^C]+C[^C]+C[^C]+C[^C]+C[^C]+':'XXXIII',
                    '[^C]+C[^C]+C[^C]+C[^C]+C[^C]+C[^C]+C[^C]+C[^C]+C[^C]+C[^C]+C[^C]+':'VIII',
                    '[^C]+C[^C]+C[^C]+C[^C]+C[^C]+C[^C]+C[^C]+C[^C]+C[^C]+':'XXII',
                    '[^C]+C[^C]+C[^C]+C[^C]+C[^C]+C[^C]+C[^C]+':'IX',
                    '[^C]+C[^C]+C[^C]+C[^C]+C[^C]+':'XIV',
                    '[^C]+CC[^C]+C[^C]+C[^C]+':'I',
                    '[^C]+C[^C]+C[^C]+C[^C]+C[^C]+CC[^C]+C[^C]+C[^C]+':'XII',
                    '[^C]+C[^C]+C[^C]+C[^C]+C[^C]+CC[^C]+CC[^C]+':'XXVI',
                    '[^C]+C[^C]+C[^C]+C[^C]+C[^C]+CC[^C]+':'XXV',
                    '[^C]+C[^C]+C[^C]+C[^C]+CC[^C]+C[^C]+C[^C]+C[^C]+C[^C]+C[^C]+':'XXVIII',
                    '[^C]+C[^C]+C[^C]+C[^C]+CC[^C]+C[^C]+C[^C]+C[^C]+':'XIII',
                    '[^C]+C[^C]+C[^C]+C[^C]+CC[^C]+C[^C]+':'XXIII',
                    '[^C]+C[^C]+C[^C]+C[^C]+CCC[^C]+C[^C]+C[^C]+C[^C]+C[^C]+':'XIX',
                    '[^C]+C[^C]+C[^C]+C[^C]+CCC[^C]+C[^C]+C[^C]+':'XXVII',
                    '[^C]+C[^C]+C[^C]+CC[^C]+C[^C]+C[^C]+C[^C]+C[^C]+':'XV',
                    '[^C]+C[^C]+C[^C]+CC[^C]+C[^C]+C[^C]+':'VI/VII',
                    '[^C]+C[^C]+C[^C]+CC[^C]+C[^C]+CC[^C]+C[^C]+':'XVII',
                    '[^C]+C[^C]+C[^C]+CC[^C]+CC[^C]+C[^C]+C[^C]+':'XI',
                    '[^C]+C[^C]+C[^C]+CC[^C]+CC[^C]+':'XVIII',
                    '[^C]+C[^C]+C[^C]+CC[^C]+':'XVI',
                    '[^C]+C[^C]+C[^C]+CCC[^C]+C[^C]+C[^C]+C[^C]+CC[^C]+':'XXX',
                    '[^C]+C[^C]+CC[^C]+C[^C]+C[^C]+C[^C]+':'XXXII',
                    '[^C]+C[^C]+CC[^C]+C[^C]+CC[^C]+C[^C]+C[^C]+C[^C]+C[^C]+':'XX',
                    '[^C]+C[^C]+CC[^C]+C[^C]+':'XXIV',
                    '[^C]+CC[^C]+C[^C]+[PO]C[^C]+':'X',
                    '[^C]+CC[^C]+C[^C]+C[^C]+C[^C]+C[^C]+':'IV',
                    '[^C]+CC[^C]+C[^C]+C[^C]+C[^C]+CC[^C]+C[^C]+C[^C]+C[^C]+':'XXI',
                    '[^C]+CC[^C]+C[^C]+C[^C]+CC[^C]+':'III',
                    '[^C]+CC[^C]+CC[^C]+':'V',
                    '[^C]+CCC[^C]+C[^C]+C[^C]+C[^C]+':'II',
                    '[^C]+CCC[^C]+C[^C]+CC[^C]+C[^C]+C[^C]+':'XXIX'
                }

    framework_all = []

    for peptide in peptides_all:
        frameworks = []
        if peptide == "-":
            frameworks.append("-")
        else:
            added = False
            for pattern, framework in framework_dict.items():
                pattern = re.compile(pattern)
                if pattern.search(peptide):
                    if len(re.search(pattern, peptide).group(0)) == len(peptide) and not added:
                        frameworks.append(framework)
                        added = True
            frameworks = list(set(frameworks))
        framework_all.append(frameworks)
    return framework_all
