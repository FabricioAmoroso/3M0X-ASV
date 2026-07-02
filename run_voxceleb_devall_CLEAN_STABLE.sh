#!/bin/bash
set -euo pipefail

PROJECT_HOME=/home/diaelhak/links/projects/def-aravila/diaelhak/codes/jobscode
BASE=/home/diaelhak/links/projects/def-aravila/diaelhak/codes/audio_ASV
RECIPE=$BASE/repos/speechbrain/recipes/VoxCeleb/SpeakerRec

DATA_FOLDER=/home/diaelhak/links/scratch/voxceleb/vox1_devall
LOCAL_VERI=/home/diaelhak/links/projects/def-aravila/diaelhak/codes/jobscode/veri_test2.txt

AUG_ROOT=/home/diaelhak/links/scratch/voxceleb/local_augmentation
LOCAL_NOISE_DIR=$AUG_ROOT/pointsource_noises
LOCAL_RIR_DIR=$AUG_ROOT/RIRs

SRC_YAML=$PROJECT_HOME/fwwifsroadmap/train_wav2vec2_vox1.yaml
HF_MODEL_CACHE=$BASE/hf_cache/hub/models--facebook--wav2vec2-large-xlsr-53

source "$BASE/.venv/bin/activate"

export HF_HOME="$BASE/hf_cache"
export HF_HUB_CACHE="$BASE/hf_cache/hub"
export HUGGINGFACE_HUB_CACHE="$BASE/hf_cache/hub"
export TRANSFORMERS_CACHE="$BASE/hf_cache/hub"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}

export BATCH_SIZE=${BATCH_SIZE:-8}
export DL_WORKERS=${DL_WORKERS:-4}

echo "============================================================"
echo "CLEAN STABLE VoxCeleb DEVALL training"
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "BATCH_SIZE=$BATCH_SIZE"
echo "DL_WORKERS=$DL_WORKERS"
echo "DATA_FOLDER=$DATA_FOLDER"
echo "LOCAL_NOISE_DIR=$LOCAL_NOISE_DIR"
echo "LOCAL_RIR_DIR=$LOCAL_RIR_DIR"
echo "============================================================"

nvidia-smi || true

test -d "$DATA_FOLDER/wav"
test -f "$LOCAL_VERI"
test -d "$LOCAL_NOISE_DIR"
test -d "$LOCAL_RIR_DIR"
test -f "$SRC_YAML"
test -d "$HF_MODEL_CACHE/snapshots"

XLSR_SNAPSHOT=$(find "$HF_MODEL_CACHE/snapshots" -mindepth 1 -maxdepth 1 -type d | sort | tail -1)
test -d "$XLSR_SNAPSHOT"

cd "$RECIPE"

TRAIN_YAML=hparams/train_wav2vec2_vox1.devall_clean_stable.yaml
TRAIN_PY=train_speaker_embeddings_wav2vec2_devall_clean_stable.py

echo "Creating clean YAML..."
cp "$SRC_YAML" "$TRAIN_YAML"

perl -0pi -e "s|^data_folder:.*$|data_folder: $DATA_FOLDER|m" "$TRAIN_YAML"
perl -0pi -e "s|^wav2vec2_hub:.*$|wav2vec2_hub: $XLSR_SNAPSHOT|m" "$TRAIN_YAML"
perl -0pi -e "s|^verification_file:.*$|verification_file: $LOCAL_VERI|m" "$TRAIN_YAML"

perl -0pi -e "s|^data_folder_noise:.*$|data_folder_noise: $LOCAL_NOISE_DIR|m" "$TRAIN_YAML"
perl -0pi -e "s|^data_folder_rir:.*$|data_folder_rir: $LOCAL_RIR_DIR|m" "$TRAIN_YAML"
perl -0pi -e "s|^[[:space:]]*NOISE_DATASET_URL:.*$|NOISE_DATASET_URL: $LOCAL_NOISE_DIR|m" "$TRAIN_YAML"
perl -0pi -e "s|^[[:space:]]*RIR_DATASET_URL:.*$|RIR_DATASET_URL: $LOCAL_RIR_DIR|m" "$TRAIN_YAML"

perl -0pi -e "s|^noise_annotation:.*$|noise_annotation: !ref <save_folder>/noise.csv|m" "$TRAIN_YAML"
perl -0pi -e "s|^rir_annotation:.*$|rir_annotation: !ref <save_folder>/rir.csv|m" "$TRAIN_YAML"

perl -0pi -e "s|^output_folder:.*$|output_folder: !ref results/wav2vec2_xlsr_vox1_devall_clean_stable/<seed>|m" "$TRAIN_YAML"
perl -0pi -e "s|^save_folder:.*$|save_folder: !ref <output_folder>/save|m" "$TRAIN_YAML"
perl -0pi -e "s|^train_log:.*$|train_log: !ref <output_folder>/train_log.txt|m" "$TRAIN_YAML"

perl -0pi -e "s|^number_of_epochs:.*$|number_of_epochs: 40|m" "$TRAIN_YAML"
perl -0pi -e "s|^batch_size:.*$|batch_size: $BATCH_SIZE|m" "$TRAIN_YAML"

python - <<'PY'
import os
import re
from pathlib import Path

p = Path("hparams/train_wav2vec2_vox1.devall_clean_stable.yaml")
s = p.read_text()

workers = int(os.environ.get("DL_WORKERS", "4"))

block = f"""dataloader_options:
    batch_size: !ref <batch_size>
    shuffle: True
    num_workers: {workers}
    pin_memory: True
    drop_last: True
"""

pattern = r"(?m)^dataloader_options:\n(?:    .*\n)+"
if re.search(pattern, s):
    s = re.sub(pattern, block, s)
else:
    s += "\n" + block

p.write_text(s)
print("Patched dataloader_options with num_workers =", workers)
PY

echo "Creating clean Python from original..."
cp train_speaker_embeddings_wav2vec2.py "$TRAIN_PY"

python - <<'PY'
from pathlib import Path
import re

p = Path("train_speaker_embeddings_wav2vec2_devall_clean_stable.py")
s = p.read_text()

# Add safe torch.save with REAL new lines
if "def _safe_torch_save" not in s:
    safe = "\n".join([
        "import torch",
        "# Safe checkpoint save: create parent directory before torch.save",
        "import os as _os",
        "_orig_torch_save = torch.save",
        "",
        "def _safe_torch_save(obj, f, *args, **kwargs):",
        "    if isinstance(f, (str, _os.PathLike)):",
        "        parent = _os.path.dirname(_os.fspath(f))",
        "        if parent:",
        "            _os.makedirs(parent, exist_ok=True)",
        "    return _orig_torch_save(obj, f, *args, **kwargs)",
        "",
        "torch.save = _safe_torch_save",
        "",
    ])
    s = s.replace("import torch\n", safe, 1)

# Replace noise/RIR prepare with local CSV creation, no unzip
pattern = r'(?m)^    .*prepare_noise_data.*\n^    .*prepare_rir_data.*\n'
replacement = "\n".join([
    "    # Local augmentation already extracted: create CSV files, do not unzip/download.",
    "    import csv",
    "    import wave",
    "    from pathlib import Path",
    "",
    "    def make_aug_csv(folder, csv_path):",
    "        folder = Path(folder)",
    "        wavs = sorted(folder.rglob('*.wav'))",
    "        if len(wavs) == 0:",
    "            raise RuntimeError(f'No wav files found in {folder}')",
    "        Path(csv_path).parent.mkdir(parents=True, exist_ok=True)",
    "        with open(csv_path, 'w', newline='', encoding='utf-8') as f:",
    "            writer = csv.writer(f)",
    "            writer.writerow(['ID', 'duration', 'wav', 'wav_format', 'wav_opts'])",
    "            for i, wav in enumerate(wavs):",
    "                try:",
    "                    with wave.open(str(wav), 'rb') as wf:",
    "                        duration = wf.getnframes() / float(wf.getframerate())",
    "                except Exception:",
    "                    duration = 1.0",
    "                writer.writerow([f'aug_{i}', duration, str(wav), 'wav', ''])",
    "        print(f'Created augmentation CSV: {csv_path} with {len(wavs)} files')",
    "",
    "    make_aug_csv(hparams['data_folder_noise'], hparams['noise_annotation'])",
    "    make_aug_csv(hparams['data_folder_rir'], hparams['rir_annotation'])",
    "",
])
s = re.sub(pattern, replacement, s)

# Fix CyclicLRScheduler resume bug
old = "\n".join([
    "            old_lr, new_lr = self.hparams.lr_annealing(epoch)",
    "            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)",
])
new = "\n".join([
    "            if not hasattr(self.hparams.lr_annealing, 'current_lr'):",
    "                self.hparams.lr_annealing.current_lr = self.optimizer.param_groups[0]['lr']",
    "",
    "            old_lr, new_lr = self.hparams.lr_annealing(epoch)",
    "            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)",
])
if old in s:
    s = s.replace(old, new)

p.write_text(s)
print("Clean Python patched:", p)
PY

python -m py_compile "$TRAIN_PY"

echo "Checking no internet URL..."
if grep -v '^[[:space:]]*#' "$TRAIN_YAML" | grep -E 'https?://' ; then
    echo "ERROR: active internet URL remains"
    exit 1
fi

mkdir -p results/wav2vec2_xlsr_vox1_devall_clean_stable/1234/save

echo "Important YAML lines:"
grep -E "^(output_folder|data_folder|data_folder_noise|data_folder_rir|noise_annotation|rir_annotation|verification_file|batch_size|number_of_epochs|wav2vec2_hub):" "$TRAIN_YAML" || true

echo "============================================================"
echo "Training start clean stable"
echo "============================================================"

python "$TRAIN_PY" "$TRAIN_YAML" --device=cuda:0 --precision=fp32 --eval_precision=fp32
