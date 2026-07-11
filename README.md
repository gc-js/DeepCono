# Efficient Discovery of Conotoxins in Large-Scale Conus Transcriptomes via Deep Learning

## google colab
## huggingface
## agent skills

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

## Software prerequisites
### MMseqs2
```shell
wget https://mmseqs.com/latest/mmseqs-linux-gpu.tar.gz
tar xvzf mmseqs-linux-*.tar.gz
export PATH=$(pwd)/mmseqs/bin/:$PATH
```

### hmmer
```shell
conda install -c biocore hmmer
```

## :rocket: Get Started

#### :one: Label_Prediction

```bash
python pre.py
```
