"""
Data preparation for CETUC dataset (adapted for trial files with bare filenames).

Expected structure:
    data_folder/
    ├── AdrianaMalta_F049/
    │   ├── F049-0000.wav
    │   └── ...
    ├── Aislam_M001/
    │   ├── M001-0000.wav
    │   └── ...
    └── ...

    verification_pairs_file:
        1 F029-0174.wav F029-0424.wav
        1 F019-0460.wav F019-0094.wav
        ...

Usage:
    python cetuc_prepare.py   /path/to/CETUC_unified/data   /path/to/save_folder   /path/to/cetuc_trials.txt
    # python cetuc_prepare.py /home/amoroso/links/scratch/CORPORA_DIR/CETUC_unified/data /home/amoroso/links/scratch/3M0X-ASV/output/prepare/CETUC_unified /home/amoroso/links/scratch/3M0X-ASV/data/CETUC/cetuc_trials.txt
"""

import csv
import glob
import os
import random
import sys
from collections import defaultdict

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
    verification_pairs_file,
    splits=["train", "dev", "test"],
    split_ratio=[90, 10],
    seg_dur=3.0,
    amp_th=5e-04,
    split_speaker=False,
    random_segment=False,
    skip_prep=False,
):
    """
    Prepares CSV files for the CETUC dataset.

    Arguments
    ---------
    data_folder : str
        Path to the folder containing individual speaker subdirectories.
    save_folder : str
        Directory where the CSV files will be saved.
    verification_pairs_file : str
        Path to the trial file (bare filenames).
    splits : list
        Which splits to prepare: 'train', 'dev', 'test'.
    split_ratio : list
        Train/dev split percentage (e.g., [90, 10]).
    seg_dur : float
        Duration (in seconds) of each chunk.
    amp_th : float
        Amplitude threshold for discarding low‑energy chunks.
    split_speaker : bool
        If True, split speakers (not files) for train/dev (recommended).
    random_segment : bool
        If True, use whole utterances instead of fixed chunks.
    skip_prep : bool
        If True, skip preparation if already done.
    """
    if skip_prep:
        return

    conf = {
        "data_folder": data_folder,
        "splits": splits,
        "split_ratio": split_ratio,
        "save_folder": save_folder,
        "seg_dur": seg_dur,
        "amp_th": amp_th,
        "split_speaker": split_speaker,
        "random_segment": random_segment,
    }

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    save_opt = os.path.join(save_folder, OPT_FILE)

    if skip(splits, save_folder, conf):
        logger.info("Skipping preparation, completed in previous run.")
        return

    logger.info("Creating CSV files for the CETUC Dataset...")

    # Build a lookup from filename (without extension) -> (relative_path, speaker_folder)
    logger.info("Building file index...")
    file_index = {}  # utt_name -> (rel_path, spk_folder)
    spk_folders = [
        d for d in os.listdir(data_folder)
        if os.path.isdir(os.path.join(data_folder, d))
    ]
    for spk in spk_folders:
        spk_dir = os.path.join(data_folder, spk)
        for fname in os.listdir(spk_dir):
            if fname.endswith(".wav"):
                utt_name = os.path.splitext(fname)[0]  # e.g., F049-0000
                rel_path = os.path.join(spk, fname)    # e.g., AdrianaMalta_F049/F049-0000.wav
                file_index[utt_name] = (rel_path, spk)

    # Read trial file and collect evaluation speaker folders
    trial_spk_folders = set()
    with open(verification_pairs_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            e_name = os.path.splitext(parts[1])[0]  # e.g., F029-0174
            t_name = os.path.splitext(parts[2])[0]
            if e_name in file_index:
                trial_spk_folders.add(file_index[e_name][1])
            if t_name in file_index:
                trial_spk_folders.add(file_index[t_name][1])

    logger.info(f"Evaluation speakers (excluded from train/dev): {len(trial_spk_folders)}")

    # Collect train/dev wav files, excluding evaluation speakers
    wav_lst_train, wav_lst_dev = _get_utt_split_lists(
        data_folder, split_ratio, trial_spk_folders, split_speaker
    )

    # Create train/dev CSVs
    if "train" in splits:
        prepare_csv(
            seg_dur, wav_lst_train,
            os.path.join(save_folder, TRAIN_CSV),
            random_segment, amp_th
        )
    if "dev" in splits:
        prepare_csv(
            seg_dur, wav_lst_dev,
            os.path.join(save_folder, DEV_CSV),
            random_segment, amp_th
        )

    # Create enrol/test CSVs from the trial file
    if "test" in splits:
        prepare_csv_enrol_test(
            data_folder, save_folder, verification_pairs_file, file_index
        )

    # Save configuration for future runs
    save_pkl(conf, save_opt)


def skip(splits, save_folder, conf):
    """Check if preparation can be skipped."""
    split_files = {
        "train": TRAIN_CSV,
        "dev": DEV_CSV,
        "test": TEST_CSV,
        "enrol": ENROL_CSV,
    }
    for split in splits:
        if not os.path.isfile(os.path.join(save_folder, split_files[split])):
            return False
    save_opt = os.path.join(save_folder, OPT_FILE)
    if os.path.isfile(save_opt):
        opts_old = load_pkl(save_opt)
        return opts_old == conf
    return False


def _get_utt_split_lists(data_folder, split_ratio, exclude_spk_folders, split_speaker):
    """
    Splits non‑evaluation audio files into train and dev lists.
    `exclude_spk_folders` is a set of folder names to exclude.
    """
    train_lst = []
    dev_lst = []

    # Collect all .wav files that are NOT in excluded speaker folders
    all_files = []
    for spk in os.listdir(data_folder):
        spk_dir = os.path.join(data_folder, spk)
        if not os.path.isdir(spk_dir) or spk in exclude_spk_folders:
            continue
        for fname in os.listdir(spk_dir):
            if fname.endswith(".wav"):
                all_files.append(os.path.join(spk_dir, fname))

    if split_speaker:
        # Group by speaker folder
        spk_to_files = defaultdict(list)
        for fpath in all_files:
            spk = os.path.basename(os.path.dirname(fpath))
            spk_to_files[spk].append(fpath)

        spk_list = list(spk_to_files.keys())
        random.shuffle(spk_list)
        split_idx = int(0.01 * split_ratio[0] * len(spk_list))
        for spk in spk_list[:split_idx]:
            train_lst.extend(spk_to_files[spk])
        for spk in spk_list[split_idx:]:
            dev_lst.extend(spk_to_files[spk])
    else:
        random.shuffle(all_files)
        split_idx = int(0.01 * split_ratio[0] * len(all_files))
        train_lst = all_files[:split_idx]
        dev_lst = all_files[split_idx:]

    logger.info(f"Train files: {len(train_lst)}, Dev files: {len(dev_lst)}")
    return train_lst, dev_lst


def _get_chunks(seg_dur, audio_id, audio_duration):
    """Create chunk identifiers."""
    num_chunks = int(audio_duration // seg_dur)
    return [
        audio_id + "_" + str(i * seg_dur) + "_" + str(i * seg_dur + seg_dur)
        for i in range(num_chunks)
    ]


def prepare_csv(seg_dur, wav_lst, csv_file, random_segment=False, amp_th=0):
    """Creates a train/dev CSV file."""
    logger.info('"Creating csv lists in %s..."', csv_file)

    csv_output = [["ID", "duration", "wav", "start", "stop", "spk_id"]]
    my_sep = "--"
    entry = []

    for wav_file in tqdm(wav_lst, dynamic_ncols=True):
        spk_id = os.path.basename(os.path.dirname(wav_file))
        utt_name = os.path.splitext(os.path.basename(wav_file))[0]
        audio_id = my_sep.join([spk_id, utt_name])

        try:
            signal, fs = audio_io.load(wav_file)
        except Exception as e:
            logger.warning("Could not load %s: %s", wav_file, e)
            continue
        signal = signal.squeeze(0)

        if random_segment:
            audio_duration = signal.shape[0] / SAMPLERATE
            csv_line = [audio_id, str(audio_duration), wav_file, 0, signal.shape[0], spk_id]
            entry.append(csv_line)
        else:
            audio_duration = signal.shape[0] / SAMPLERATE
            chunks = _get_chunks(seg_dur, audio_id, audio_duration)
            for chunk in chunks:
                s, e = chunk.split("_")[-2:]
                start_sample = int(float(s) * SAMPLERATE)
                end_sample = int(float(e) * SAMPLERATE)

                if amp_th > 0:
                    mean_sig = torch.mean(torch.abs(signal[start_sample:end_sample]))
                    if mean_sig < amp_th:
                        continue

                csv_line = [
                    chunk, str(audio_duration), wav_file,
                    start_sample, end_sample, spk_id,
                ]
                entry.append(csv_line)

    csv_output += entry

    with open(csv_file, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerows(csv_output)

    logger.info("%s successfully created! (%d entries)", csv_file, len(entry))


def prepare_csv_enrol_test(data_folder, save_folder, verification_pairs_file, file_index):
    """
    Creates enrol.csv and test.csv from the trial file.
    Uses the file_index dict to map bare filenames to full paths and speaker IDs.
    """
    logger.info("Preparing enrol & test CSVs from verification pairs file...")

    csv_head = [["ID", "duration", "wav", "start", "stop", "spk_id"]]

    enrol_ids = []
    test_ids = []

    with open(verification_pairs_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            enrol_ids.append(parts[1])
            test_ids.append(parts[2])

    # Unique utterance identifiers (without .wav)
    enrol_unique = sorted(set(os.path.splitext(x)[0] for x in enrol_ids))
    test_unique = sorted(set(os.path.splitext(x)[0] for x in test_ids))

    def write_csv(utt_names, csv_path):
        rows = []
        for utt_name in utt_names:
            if utt_name not in file_index:
                logger.warning("File not found in index: %s", utt_name)
                continue
            rel_path, spk_folder = file_index[utt_name]
            wav_path = os.path.join(data_folder, rel_path)

            try:
                signal, _ = audio_io.load(wav_path)
            except Exception as e:
                logger.warning("Could not load %s: %s", wav_path, e)
                continue
            signal = signal.squeeze(0)
            dur = signal.shape[0] / SAMPLERATE

            # ID follows VoxCeleb style: speaker_folder/filename
            uid = rel_path.replace("\\", "/").rsplit(".", 1)[0]  # remove .wav
            rows.append([uid, dur, wav_path, 0, signal.shape[0], spk_folder])

        with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerows(csv_head + rows)
        logger.info("%s created with %d entries.", csv_path, len(rows))

    write_csv(enrol_unique, os.path.join(save_folder, ENROL_CSV))
    write_csv(test_unique, os.path.join(save_folder, TEST_CSV))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="CETUC data preparation (VoxCeleb‑style)")
    parser.add_argument("data_folder", help="Folder containing speaker subdirectories")
    parser.add_argument("save_folder", help="Where to save the output CSV files")
    parser.add_argument("verification_pairs_file", help="Trial file (bare filenames)")
    parser.add_argument("--splits", nargs="+", default=["train", "dev", "test"])
    parser.add_argument("--split_ratio", nargs=2, type=int, default=[90, 10])
    parser.add_argument("--seg_dur", type=float, default=3.0)
    parser.add_argument("--amp_th", type=float, default=5e-04)
    parser.add_argument("--split_speaker", action="store_true", default=False)
    parser.add_argument("--random_segment", action="store_true")
    parser.add_argument("--skip_prep", action="store_true")
    args = parser.parse_args()

    prepare_cetuc(
        data_folder=args.data_folder,
        save_folder=args.save_folder,
        verification_pairs_file=args.verification_pairs_file,
        splits=args.splits,
        split_ratio=args.split_ratio,
        seg_dur=args.seg_dur,
        amp_th=args.amp_th,
        split_speaker=args.split_speaker,
        random_segment=args.random_segment,
        skip_prep=args.skip_prep,
    )