import os, subprocess, argparse
import pandas as pd
from tqdm import tqdm

API_URL = "https://api.esmatlas.com/foldSequence/v1/pdb/"

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", "-i", required=True)
    p.add_argument("--output_dir", "-o", default="output_structures")
    return p.parse_args()

def load_csv(path):
    df = pd.read_csv(path)
    return df["ID"].tolist(), df["Seq"].tolist()

def fold(seq):
    r = subprocess.run(["curl", "-X", "POST", "-k", "--data", seq, API_URL], capture_output=True, text=True)
    return r.stdout

def main():
    args = parse_args()
    ids, seqs = load_csv(args.input)
    os.makedirs(args.output_dir, exist_ok=True)

    for i in tqdm(range(len(seqs)), desc="ESMFold"):
        seq_id, seq = ids[i], seqs[i].strip().replace(" ", "")
        pdb = fold(seq)
        with open(os.path.join(args.output_dir, f"{seq_id}.pdb"), "w") as f:
            f.write(pdb)

if __name__ == "__main__":
    main()
