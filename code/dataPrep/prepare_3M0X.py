
import csv
import glob
import os
import random
import shutil
import sys  # noqa F401

import numpy as np
import torch
from tqdm import tqdm

from speechbrain.dataio import audio_io
from speechbrain.dataio.dataio import load_pkl, save_pkl
from speechbrain.utils.logger import get_logger

logger = get_logger(__name__)
OPT_FILE = "opt_3M0X_prepare.pkl"
TEST_CSV = "test.csv"
ENROL_CSV = "enrol.csv"
SAMPLERATE = 16000

def prepare_3M0X(
    data_folder,
    save_folder,
    verification_pairs_file,
    split_speaker=False,
    skip_prep=False,
):
    """
    Prepares the csv files for the Voxceleb1 or Voxceleb2 datasets.
    Please follow the instructions in the README.md file for
    preparing Voxceleb2.

    Arguments
    ---------
    data_folder : str
        Path to the folder where the original VoxCeleb dataset is stored.
    save_folder : str
        The directory where to store the csv files.
    verification_pairs_file : str
        txt file containing the verification split.
    split_speaker : bool
        Speaker-wise split
    skip_prep : bool
        If True, skip preparation.

    Returns
    -------
    None

    Example
    -------
    >>> from recipes.VoxCeleb.voxceleb1_prepare import prepare_voxceleb
    >>> data_folder = "data/VoxCeleb1/"
    >>> save_folder = "VoxData/"
    >>> splits = ["train", "dev"]
    >>> split_ratio = [90, 10]
    >>> prepare_voxceleb(data_folder, save_folder, splits, split_ratio)
    """

    if skip_prep: return
    # Create configuration for easily skipping data_preparation stage
    conf = {
        "data_folder": data_folder,
        "save_folder": save_folder,
        "verification_pairs_file": verification_pairs_file,
        "split_speaker": split_speaker,
    }

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # Setting output files
    save_opt = os.path.join(save_folder, OPT_FILE)

    # Check if this phase is already done (if so, skip it)
    if skip(save_folder, conf):
        logger.info("Skipping preparation, completed in previous run.")
        return

    data_folder = [data_folder]
    msg = "\tCreating csv file for the 3M0X Dataset.."
    logger.info(msg)

    prepare_csv_enrol_test(data_folder, save_folder, verification_pairs_file)

    # Saving options (useful to skip this phase when already done)
    save_pkl(conf, save_opt)

def skip(save_folder, conf):
    """
    Detects if the 3M0X data_preparation has been already done.
    If the preparation has been done, we can skip it.

    Arguments
    ---------
    splits : list
    save_folder : str
    conf : str

    Returns
    -------
    bool
        if True, the preparation phase can be skipped.
        if False, it must be done.
    """
    # Checking csv files
    skip = True

    split_files = {
        "test": TEST_CSV,
        "enrol": ENROL_CSV,
    }
    for split in ["test", "enrol"]:
        if not os.path.isfile(os.path.join(save_folder, split_files[split])):
            skip = False
    #  Checking saved options
    save_opt = os.path.join(save_folder, OPT_FILE)
    if skip is True:
        if os.path.isfile(save_opt):
            opts_old = load_pkl(save_opt)
            if opts_old == conf:
                skip = True
            else:
                skip = False
        else:
            skip = False

    return skip

def prepare_csv_enrol_test(data_folders, save_folder, verification_pairs_file):
    """
    Creates the csv file for test data (useful for verification)

    Arguments
    ---------
    data_folders : str
        Path of the data folders
    save_folder : str
        The directory where to store the csv files.
    verification_pairs_file : str
        Path to the file with verification pairs.
    """

    # msg = '\t"Creating csv lists in  %s..."' % (csv_file)
    # logger.debug(msg)

    csv_output_head = [["ID", "duration", "wav", "start", "stop", "spk_id"]]  # noqa E231

    for data_folder in data_folders:
        test_lst_file = verification_pairs_file

        enrol_ids, test_ids = [], []

        # Get unique ids (enrol and test utterances)
        for line in open(test_lst_file, encoding="utf-8"):
            e_id = line.split(" ")[1].rstrip().split(".")[0].strip()
            t_id = line.split(" ")[2].rstrip().split(".")[0].strip()
            enrol_ids.append(e_id)
            test_ids.append(t_id)

        enrol_ids = list(np.unique(np.array(enrol_ids)))
        test_ids = list(np.unique(np.array(test_ids)))

        # Prepare enrol csv
        logger.info("preparing enrol & test csvs")
        # assert len(enrol_ids) == len(test_ids)
        print(len(enrol_ids), len(test_ids))
        enrol_csv = []
        test_csv = []
        for e_id, t_id in zip(enrol_ids, test_ids):

            e_wav = data_folder + "/wav/" + e_id + ".wav"
            e_signal, e_fs = audio_io.load(e_wav)
            e_signal = e_signal.squeeze(0)
            e_audio_duration = e_signal.shape[0] / SAMPLERATE
            e_start_sample = 0
            e_stop_sample = e_signal.shape[0]
            e_spk_id = "_".join(e_id.split("_")[:3])

            t_wav = data_folder + "/wav/" + t_id + ".wav"
            t_signal, t_fs = audio_io.load(t_wav)
            t_signal = t_signal.squeeze(0)
            t_audio_duration = t_signal.shape[0] / SAMPLERATE
            t_start_sample = 0
            t_stop_sample = t_signal.shape[0]
            t_spk_id = "_".join(t_id.split("_")[:3])

            e_csv_line = [e_id, e_audio_duration, e_wav, e_start_sample, e_stop_sample, e_spk_id,]
            t_csv_line = [t_id, t_audio_duration, t_wav, t_start_sample, t_stop_sample, t_spk_id,]

            enrol_csv.append(e_csv_line)
            test_csv.append(t_csv_line)

        e_csv_output = csv_output_head + enrol_csv
        t_csv_output = csv_output_head + test_csv

        e_csv_file = os.path.join(save_folder, ENROL_CSV)
        t_csv_file = os.path.join(save_folder, TEST_CSV)

        # Writing the csv lines
        with open(e_csv_file, mode="w", newline="", encoding="utf-8") as csv_f:
            csv_writer = csv.writer(csv_f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL)
            for line in e_csv_output: csv_writer.writerow(line)

        with open(t_csv_file, mode="w", newline="", encoding="utf-8") as csv_f:
            csv_writer = csv.writer(csv_f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL)
            for line in t_csv_output: csv_writer.writerow(line)

