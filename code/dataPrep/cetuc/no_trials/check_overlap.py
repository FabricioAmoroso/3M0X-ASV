#!/usr/bin/env python3
"""
Check if the train/dev/test splits of a speaker dataset are speaker‑disjoint.
Assumes each split folder contains one subfolder per speaker.

Usage: python check_overlap.py /path/to/CETUC/data


"""

import os
import sys
import argparse

def check_overlap(data_folder):
    splits = ["train", "dev", "test"]
    speakers = {}

    for split in splits:
        path = os.path.join(data_folder, split)
        if not os.path.isdir(path):
            print(f"WARNING: Split folder '{path}' does not exist, skipping.")
            continue
        # Speaker IDs are the names of the immediate subdirectories
        spk_list = [
            d for d in os.listdir(path)
            if os.path.isdir(os.path.join(path, d))
        ]
        speakers[split] = set(spk_list)
        print(f"{split}: {len(speakers[split])} speakers")

    # Check pairwise overlaps
    overlap_found = False
    pairs = [("train", "dev"), ("train", "test"), ("dev", "test")]
    for a, b in pairs:
        if a in speakers and b in speakers:
            inter = speakers[a] & speakers[b]
            if inter:
                print(f"❌ OVERLAP between {a} and {b}: {sorted(inter)}")
                overlap_found = True

    if not overlap_found and len(speakers) > 1:
        print("✅ Check passed: no speaker overlap between splits. The split is speaker‑disjoint.")
    elif not overlap_found:
        print("ℹ️ Only one split found – no overlap check possible.")
    else:
        print("⚠️  Speaker overlap detected. The split is NOT suitable for speaker‑independent evaluation.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Check speaker overlap across train/dev/test splits."
    )
    parser.add_argument(
        "data_folder",
        help="Path to the dataset root containing dev/, test/, train/ subfolders."
    )
    args = parser.parse_args()

    if not os.path.isdir(args.data_folder):
        print(f"Error: '{args.data_folder}' is not a valid directory.")
        sys.exit(1)

    check_overlap(args.data_folder)