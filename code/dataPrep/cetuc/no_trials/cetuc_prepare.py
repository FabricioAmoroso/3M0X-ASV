"""
Data preparation for the CETUC dataset.

Expected folder structure:
    CETUC/
    ├── dev/          # 10 speakers
    ├── test/         # 10 speakers
    ├── train/        # 80 speakers
    └── (optional) verification_pairs.txt

Inside each split folder, each speaker has its own subfolder containing .wav files.

To generate train.csv, dev.csv, test.csv in outpput path:

    python cetuc_prepare.py /path/to/CETUC/data path/to/3M0X-ASV/output/prepare/CETUC

    # python cetuc_prepare.py /home/amoroso/links/scratch/CORPORA_DIR/CETUC/data /home/amoroso/links/scratch/3M0X-ASV/output/prepare/CETUC

To generate train.csv, dev.csv, test.csv and additionally enrol.csv and test.csv from the pointed trials file:

    python cetuc_prepare.py /path/to/CETUC/data path/to/3M0X-ASV/output/prepare/CETUC --verification_pairs /path/to/trials.txt
    
    # python cetuc_prepare.py /home/amoroso/links/scratch/CORPORA_DIR/CETUC/data /home/amoroso/links/scratch/3M0X-ASV/output/prepare/CETUC --verification_pairs /path/to/trials.txt
"""

import csv
import glob
import os
import shutil
import sys

import numpy as np
import torch
from tqdm import tqdm

from speechbrain.dataio import audio_io
from speechbrain.dataio.dataio import load_pkl, save_pkl
from speechbrain.utils.logger import get_logger

logger = get_logger(__name__)

OPT_FILE = "opt_cetuc_prepare.pkl"
TRAIN_CSV = "train.csv"
DEV_CSV = "dev.csv"
TEST_CSV = "test.csv"
ENROL_CSV = "enrol.csv"
SAMPLERATE = 16000


def prepare_cetuc(
    data_folder,
    save_folder,
    verification_pairs_file=None,
    seg_dur=None,
    amp_th=0.0,
    skip_prep=False,
):
    """
    Prepares CSV files for the CETUC dataset (pre‑split into train/dev/test).

    Arguments
    ---------
    data_folder : str
        Path to the root of the CETUC dataset (contains dev/, test/, train/).
    save_folder : str
        Where to store the generated CSV files.
    verification_pairs_file : str or None
        Path to a trial file (format: "1 enrol_utt test_utt").
        If provided, enrol.csv and test.csv will be created for verification.
        If None, only train.csv, dev.csv, test.csv are created.
    seg_dur : float or None
        If given, split each utterance into fixed‑length segments (in seconds).
        If None, whole utterances are used (start=0, stop=file_length).
    amp_th : float
        When seg_dur is used, discard segments whose average amplitude is
        below this threshold (default 0 = keep everything).
    skip_prep : bool
        If True, return immediately (useful when preparation is already done).

    Returns
    -------
    None
    """
    if skip_prep:
        return

    # Configuration for skipping
    conf = {
        "data_folder": data_folder,
        "save_folder": save_folder,
        "verification_pairs_file": verification_pairs_file,
        "seg_dur": seg_dur,
        "amp_th": amp_th,
    }

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    save_opt = os.path.join(save_folder, OPT_FILE)

    # Check if we can skip (all required CSV files exist and config matches)
    if skip(save_folder, conf, verification_pairs_file):
        logger.info("Skipping preparation, completed in previous run.")
        return

    msg = "\tCreating CSV files for the CETUC dataset..."
    logger.info(msg)

    # Process each split
    for split in ["train", "dev", "test"]:
        split_path = os.path.join(data_folder, split)
        if not os.path.exists(split_path):
            logger.warning(f"Split folder {split_path} does not exist – skipping.")
            continue

        # Collect all .wav files under split_path/*/
        wav_files = glob.glob(os.path.join(split_path, "*", "*.wav"))
        if not wav_files:
            logger.warning(f"No .wav files found in {split_path} – skipping.")
            continue

        csv_file = os.path.join(save_folder, f"{split}.csv")
        prepare_csv(wav_files, csv_file, seg_dur=seg_dur, amp_th=amp_th)

    # If a verification pairs file is given, create enrol and test CSVs
    if verification_pairs_file is not None and os.path.isfile(verification_pairs_file):
        logger.info("Creating enrol & test CSVs from verification pairs file...")
        prepare_csv_enrol_test(
            data_folder, save_folder, verification_pairs_file
        )

    # Save options for later skipping
    save_pkl(conf, save_opt)


def skip(save_folder, conf, verification_pairs_file):
    """
    Returns True if all required CSV files already exist and the saved
    configuration matches the current one.
    """
    required_csvs = [TRAIN_CSV, DEV_CSV, TEST_CSV]
    if verification_pairs_file is not None:
        required_csvs += [ENROL_CSV, TEST_CSV]

    for csv_name in required_csvs:
        if not os.path.isfile(os.path.join(save_folder, csv_name)):
            return False

    save_opt = os.path.join(save_folder, OPT_FILE)
    if os.path.isfile(save_opt):
        opts_old = load_pkl(save_opt)
        return opts_old == conf
    return False


def prepare_csv(wav_list, csv_file, seg_dur=None, amp_th=0.0):
    """
    Creates a CSV file from a list of wav file paths.

    If seg_dur is None, each row contains the whole file.
    Otherwise, the file is cut into fixed‑length chunks.
    """
    csv_output = [["ID", "duration", "wav", "start", "stop", "spk_id"]]
    entry = []
    my_sep = "--"

    for wav_file in tqdm(wav_list, dynamic_ncols=True, desc=f"Preparing {os.path.basename(csv_file)}"):
        # Extract speaker ID from folder name
        spk_id = os.path.basename(os.path.dirname(wav_file))
        # Build a unique utterance ID: speaker_folder/filename_without_ext
        rel_path = os.path.relpath(wav_file, start=os.path.dirname(os.path.dirname(wav_file)))
        utt_id = os.path.splitext(rel_path)[0].replace(os.sep, "_")

        # Load audio to get duration and signal
        try:
            signal, fs = audio_io.load(wav_file)
        except Exception as e:
            logger.warning(f"Could not load {wav_file}: {e}")
            continue
        signal = signal.squeeze(0)
        audio_duration = signal.shape[0] / SAMPLERATE

        if seg_dur is None:
            # Whole utterance
            csv_line = [
                utt_id,
                str(audio_duration),
                wav_file,
                0,
                signal.shape[0],
                spk_id,
            ]
            entry.append(csv_line)
        else:
            # Fixed‑length chunks
            num_chunks = int(audio_duration // seg_dur)
            for i in range(num_chunks):
                start_sample = int(i * seg_dur * SAMPLERATE)
                end_sample = int((i + 1) * seg_dur * SAMPLERATE)
                # Amplitude threshold
                if amp_th > 0:
                    chunk_amp = torch.mean(torch.abs(signal[start_sample:end_sample]))
                    if chunk_amp < amp_th:
                        continue
                chunk_id = f"{utt_id}_{i*seg_dur:.1f}_{(i+1)*seg_dur:.1f}"
                csv_line = [
                    chunk_id,
                    str(audio_duration),
                    wav_file,
                    start_sample,
                    end_sample,
                    spk_id,
                ]
                entry.append(csv_line)

    csv_output += entry
    with open(csv_file, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerows(csv_output)

    logger.info(f"{csv_file} successfully created ({len(entry)} entries).")


def prepare_csv_enrol_test(data_folder, save_folder, verification_pairs_file):
    """
    Reads a verification pairs file and creates enrol.csv and test.csv.
    The pairs file should have lines like:
        "1 spk/utt1 spk/utt2"
    where the second and third fields are utterance IDs (relative to data_folder/wav).
    For CETUC, the paths are expected to be like "speaker_folder/utt.wav".
    """
    # We'll use data_folder as the root where dev/test/train reside.
    # However, the trial file might reference paths relative to that root.
    # For simplicity we assume the trial file contains full relative paths that exist
    # under data_folder.  If they contain a split prefix (e.g., test/speaker/utt.wav),
    # we use them directly.
    enrol_ids = []
    test_ids = []
    with open(verification_pairs_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            enrol_ids.append(parts[1])
            test_ids.append(parts[2])

    enrol_ids = sorted(set(enrol_ids))
    test_ids = sorted(set(test_ids))

    # Helper to write a CSV file
    def write_csv(ids, csv_path):
        header = [["ID", "duration", "wav", "start", "stop", "spk_id"]]
        rows = []
        for uid in ids:
            # uid might be something like "test/speaker/utt.wav"
            wav_path = os.path.join(data_folder, uid)
            if not os.path.exists(wav_path):
                # Try without extension
                wav_path = os.path.join(data_folder, uid + ".wav")
            if not os.path.exists(wav_path):
                logger.warning(f"File not found for ID {uid}, skipping.")
                continue
            try:
                signal, fs = audio_io.load(wav_path)
            except Exception as e:
                logger.warning(f"Could not load {wav_path}: {e}")
                continue
            signal = signal.squeeze(0)
            dur = signal.shape[0] / SAMPLERATE
            spk_id = uid.split("/")[-2]  # assuming structure split/speaker/utt
            rows.append([uid, dur, wav_path, 0, signal.shape[0], spk_id])

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerows(header + rows)
        logger.info(f"{csv_path} created with {len(rows)} entries.")

    write_csv(enrol_ids, os.path.join(save_folder, ENROL_CSV))
    write_csv(test_ids, os.path.join(save_folder, TEST_CSV))


# Example usage (if run directly)
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Prepare CETUC dataset CSV files.")
    parser.add_argument("data_folder", help="Path to CETUC root (with dev/test/train).")
    parser.add_argument("save_folder", help="Where to save CSVs.")
    parser.add_argument("--verification_pairs", default=None, help="Path to trial file (optional).")
    parser.add_argument("--seg_dur", type=float, default=None, help="Segment duration in seconds (default: whole file).")
    parser.add_argument("--amp_th", type=float, default=0.0, help="Amplitude threshold for chunk dropping.")
    parser.add_argument("--skip_prep", action="store_true", help="Skip preparation if already done.")
    args = parser.parse_args()

    prepare_cetuc(
        data_folder=args.data_folder,
        save_folder=args.save_folder,
        verification_pairs_file=args.verification_pairs,
        seg_dur=args.seg_dur,
        amp_th=args.amp_th,
        skip_prep=args.skip_prep,
    )