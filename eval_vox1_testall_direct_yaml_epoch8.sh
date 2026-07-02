#!/bin/bash
set -euo pipefail

BASE=/home/diaelhak/links/projects/def-aravila/diaelhak/codes/audio_ASV
RECIPE=$BASE/repos/speechbrain/recipes/VoxCeleb/SpeakerRec

ORIG_YAML=/home/diaelhak/links/projects/def-aravila/diaelhak/codes/jobscode/fwwifsroadmap/verification_wav2vec2.yaml

DATA_FOLDER=/home/diaelhak/links/scratch/voxceleb/vox1_testall
VERI_FILE=${VERI_FILE:-/home/diaelhak/links/projects/def-aravila/diaelhak/codes/jobscode/veri_test2.txt}

CKPT=/home/diaelhak/links/projects/def-aravila/diaelhak/codes/audio_ASV/repos/speechbrain/recipes/VoxCeleb/SpeakerRec/results/wav2vec2_xlsr_vox1_devall_local_aug_nounzip/1234/save/CKPT+2026-06-29+22-03-38+00

HF_MODEL_CACHE=$BASE/hf_cache/hub/models--facebook--wav2vec2-large-xlsr-53

source "$BASE/.venv/bin/activate"

export HF_HOME="$BASE/hf_cache"
export HF_HUB_CACHE="$BASE/hf_cache/hub"
export HUGGINGFACE_HUB_CACHE="$BASE/hf_cache/hub"
export TRANSFORMERS_CACHE="$BASE/hf_cache/hub"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd "$RECIPE"

XLSR_SNAPSHOT=$(find "$HF_MODEL_CACHE/snapshots" -mindepth 1 -maxdepth 1 -type d | sort | tail -1)

EVAL_YAML=hparams/verification_wav2vec2_testall_direct_epoch8.yaml
OUT_DIR=results/wav2vec2_xlsr_vox1_testall_eval_direct_epoch8/1234

echo "============================================================"
echo "DIRECT EVAL using verification_wav2vec2.yaml"
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "DATA_FOLDER=$DATA_FOLDER"
echo "VERI_FILE=$VERI_FILE"
echo "CKPT=$CKPT"
echo "ORIG_YAML=$ORIG_YAML"
echo "============================================================"

test -d "$DATA_FOLDER/wav"
test -f "$VERI_FILE"
test -d "$CKPT"
test -f "$ORIG_YAML"
test -d "$XLSR_SNAPSHOT"

cp "$ORIG_YAML" "$EVAL_YAML"

perl -0pi -e "s|^data_folder:.*$|data_folder: $DATA_FOLDER|m" "$EVAL_YAML"
perl -0pi -e "s|^output_folder:.*$|output_folder: $OUT_DIR|m" "$EVAL_YAML"
perl -0pi -e "s|^save_folder:.*$|save_folder: !ref <output_folder>/save|m" "$EVAL_YAML"
perl -0pi -e "s|^verification_file:.*$|verification_file: $VERI_FILE|m" "$EVAL_YAML"
perl -0pi -e "s|^pretrain_path:.*$|pretrain_path: $CKPT|m" "$EVAL_YAML"
perl -0pi -e "s|^wav2vec2_hub:.*$|wav2vec2_hub: $XLSR_SNAPSHOT|m" "$EVAL_YAML"

mkdir -p "$OUT_DIR/save"

echo "Important YAML lines:"
grep -E "^(data_folder|output_folder|save_folder|verification_file|pretrain_path|wav2vec2_hub):" "$EVAL_YAML" || true

cat > eval_direct_yaml_epoch8.py <<'PY'
import os
from pathlib import Path

import torch
import torch.nn.functional as F
import torchaudio
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.distributed import run_on_main
from speechbrain.utils.metric_stats import EER, minDCF


DATA_FOLDER = Path(os.environ["DATA_FOLDER"])
VERI_FILE = Path(os.environ["VERI_FILE"])
EVAL_YAML = Path(os.environ["EVAL_YAML"])

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

with open(EVAL_YAML, encoding="utf-8") as f:
    params = load_hyperpyyaml(f)

print("Loading checkpoint with pretrainer...")
run_on_main(params["pretrainer"].collect_files)
params["pretrainer"].load_collected()

params["wav2vec2"].eval()
params["wav2vec2"].to(device)
params["pooling"].eval()
params["pooling"].to(device)

sample_rate = 16000
cache = {}

def resolve_audio_path(x):
    x = x.strip()
    p = Path(x)

    candidates = []

    if p.is_absolute():
        candidates.append(p)
    else:
        candidates.append(DATA_FOLDER / p)
        candidates.append(DATA_FOLDER / "wav" / p)

        if str(p).startswith("wav/"):
            candidates.append(DATA_FOLDER / str(p)[4:])

    more = []
    for c in candidates:
        more.append(c)
        if c.suffix == "":
            more.append(Path(str(c) + ".wav"))

    for c in more:
        if c.exists():
            return c

    raise FileNotFoundError(f"Audio not found: {x}")

@torch.no_grad()
def compute_embedding(path_str):
    path = resolve_audio_path(path_str)
    key = str(path)

    if key in cache:
        return cache[key]

    wav, sr = torchaudio.load(str(path))

    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)

    if sr != sample_rate:
        wav = torchaudio.functional.resample(wav, sr, sample_rate)

    wav = wav.to(device)
    wav_lens = torch.ones(wav.shape[0], device=device)

    try:
        feats = params["wav2vec2"](wav, wav_lens)
    except TypeError:
        feats = params["wav2vec2"](wav)

    if isinstance(feats, tuple):
        feats = feats[0]

    emb = params["pooling"](feats, wav_lens)

    if isinstance(emb, tuple):
        emb = emb[0]

    while emb.dim() > 2:
        emb = emb.mean(dim=1)

    emb = emb.detach().cpu()
    cache[key] = emb

    return emb

trials = []

with open(VERI_FILE, encoding="utf-8") as f:
    for line in f:
        line = line.strip()

        if not line:
            continue

        parts = line.split()

        if len(parts) < 3:
            continue

        label = int(parts[0].split(".")[0])
        enrol_file = parts[1]
        test_file = parts[2]

        trials.append((label, enrol_file, test_file))

print("Number of trials:", len(trials))

positive_scores = []
negative_scores = []

scores_path = Path(params["output_folder"]) / "scores.txt"
scores_path.parent.mkdir(parents=True, exist_ok=True)

similarity = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)

with open(scores_path, "w", encoding="utf-8") as s_file:
    for i, (label, enrol_file, test_file) in enumerate(trials, 1):
        enrol_emb = compute_embedding(enrol_file)
        test_emb = compute_embedding(test_file)

        score = similarity(enrol_emb, test_emb)[0].item()

        s_file.write(f"{enrol_file} {test_file} {label} {score:.8f}\n")

        if label == 1:
            positive_scores.append(score)
        else:
            negative_scores.append(score)

        if i % 200 == 0:
            print(f"Processed {i}/{len(trials)} trials")

eer, th = EER(torch.tensor(positive_scores), torch.tensor(negative_scores))
mindcf, th2 = minDCF(torch.tensor(positive_scores), torch.tensor(negative_scores))

print("============================================================")
print(f"EER(%): {eer * 100:.4f}")
print(f"minDCF: {mindcf * 100:.4f}")
print(f"Scores saved to: {scores_path}")
print("============================================================")
PY

export DATA_FOLDER
export VERI_FILE
export EVAL_YAML

python eval_direct_yaml_epoch8.py
