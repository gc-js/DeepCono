# Efficient Discovery of Conotoxins in Large-Scale Conus Transcriptomes via Deep Learning

## Description
We developed a deep learning pipeline for conopeptide identification and prediction. First, 192 transcriptomes were processed to predict 5.52 million protein sequences. A sequence labeling model (ESM2 + BiLSTM + CRF) was used to identify 16,878 conopeptides. To address limited labeled data, a semi-supervised mean teacher framework was implemented, enhancing robustness through noise injection and consistency loss. The model employs a modified Transformer encoder with enhanced cysteine-aware positional encoding. It integrates multimodal information (sequence and structural features) via cross-attention and contrastive learning, enabling concurrent conotoxin prediction and target classification.

<img src="https://github.com/gc-js/DeepCono/blob/main/imgs/fig3.png" alt="workflow" width="600"/>

## :gear: Installation

```shell
git clone git@github.com:gc-js/DeepCono.git
cd DeepCono
conda create -n DeepCono python=3.10.14
conda activate DeepCono
pip install -e .
```
You can download the trained models [here](https://zenodo.org/records/21318628) and save them to `models`.

```bash
bash download.sh
```

### Software prerequisites
#### MMseqs2
```shell
wget https://mmseqs.com/latest/mmseqs-linux-gpu.tar.gz
tar xvzf mmseqs-linux-*.tar.gz
export PATH=$(pwd)/mmseqs/bin/:$PATH
```

#### HMMER
```shell
conda install -c biocore hmmer
```

## :rocket: Get Started

#### :one: Conopeptide identification

```bash
python conopep_ident/pre.py
```

- Set up `conopep_ident/config.yaml`
```bash
# fix
base_model_path: "./esm2_base_model/esm2_t30_150M_UR50D" # esm2 model
adapter: "./MLM_lora/model/mlm_conoserver_lora_esm2_t30_150M_UR50D" # MLM_lora model checkpoint
superfamily_db: "conopep_ident/DB/superfamily/superfamily"  # HMMER model for superfamily identification
family_db: "conopep_ident/DB/family/familys" # HMMER model for family identification
best_model_path: "conopep_ident/model/conopep_ident_best_model.pth" # Trained model weights.

# pre
min_len: 10 # min length of seq
max_len: 200 # max length of seq
evalue: 0.001 # Blast E-value
input_path: "conopep_ident/query_data/" # Directory for cone snail transcripts (.fasta).
save_path: "conopep_ident/output/" # Directory for identified conopeptides (.csv).

# signal
signal: True # whether use signal peptide
signal_db: "conopep_ident/DB/signal_pep.fasta" # If signal is True
published_db: "conopep_ident/DB/conotoxin_all4blast.fasta" # If signal is True
```

#### :two: Conotoxin identification

```bash
python ctx_ident/semi_pre.py
```

- Set up `ctx_ident/config.yaml`
```bash
# fix
model_checkpoint: "./esm2_base_model/esm2_t30_150M_UR50D" # esm2 model
adapter: "./MLM_lora/model/mlm_conoserver_lora_esm2_t30_150M_UR50D" # MLM_lora model checkpoint
best_model: "ctx_ident/model/ctx_ident_best_model.pth" # Trained model weights.

# pre
file_format: "pdb" # Input peptide structure format (cif or pdb)
test_path: "ctx_ident/data/test_data/conopeptide.csv" # Peptides to be predicted
test_pdb_path: "ctx_ident/data/test_data/structure" # Structures of peptides to be predicted (if not available, will be automatically predicted using ESMFold)
output: "ctx_ident/output.csv" # Prediction results
```

#### :three: Conotoxin target prediction
```bash
python ctx_target/semi_pre.py
```

- Set up `ctx_target/config.yaml`
```bash
# fix
model_checkpoint: "./esm2_base_model/esm2_t6_8M_UR50D" # esm2 model checkpoint
adapter: "./MLM_lora/model/mlm_conoserver_lora_esm2_t6_8M_UR50D" # MLM_lora model checkpoint
best_model: "ctx_target/model/ctx_target_best_model.pth"

# pre
file_format: 'pdb' # Input peptide structure format (cif or pdb)
test_path: "ctx_target/data/test_data/conopeptide.csv" # Peptides to be predicted
test_pdb_path: "ctx_target/data/test_data/structure" # Structures of peptides to be predicted (if not available, will be automatically predicted using ESMFold)
output: "ctx_target/output.csv" # Prediction results
```

#### :four: Conotoxin subtype Prediction
```bash
python ctx_subtype/subtype_pred.py -i ./ctx_subtype/test/ctxs.txt -is X -ip '<high>' -m ./ctx_subtype/model/model_mlm.pt -o ./ctx_subtype/test/output_subtype_prediction.csv
```
- `-i`: conotoxins to be predicted.

- `-is`: Subtype: X if needs to be predicted.
optional: `<AChBP>`, `<Ca12>`, `<Ca13>`, `<Ca22>`, `<Ca23>`, `<GABA>`, `<GluN2A>`, `<GluN2B>`, `<GluN2C>`, `<GluN2D>`, `<GluN3A>`, `<K11>`, `<K12>`, `<K13>`, `<K16>`, `<K17>`, `<Kshaker>`, `<Na11>`, `<Na12>`, `<Na13>`, `<Na14>`, `<Na15>`, `<Na16>`, `<Na17>`, `<Na18>`, `<NaTTXR>`, `<NaTTXS>`, `<NavBh>`, `<NET>`, `<α1AAR>`, `<α1BAR>`, `<α1β1γ>`, `<α1β1γδ>`, `<α1β1δ>`, `<α1β1δε>`, `<α1β1ε>`, `<α2β2>`, `<α2β4>`, `<α3β2>`, `<α3β4>`, `<α4β2>`, `<α4β4>`, `<α6α3β2>`, `<α6α3β2β3>`, `<α6α3β4>`, `<α6α3β4β3>`, `<α6β3β4>`, `<α6β4>`, `<α7>`, `<α7α6β2>`, `<α75HT3>`, `<α9>`, `<α9α10>`
- `-ip`: Potency: X if needs to be predicted.
optional: `<high>`, `<low>`
- `-m`: model parameters trained at different stages of data augmentation.
- `-o`: output file (.csv)
