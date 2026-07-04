import os
from prepare_3M0X import prepare_3M0X

# Extract 3M0X.tar.gz from OneDrive to data_folder then run this file
# Tree shoule be as follows, where wav contains the audio files
#
# 3M0X
# ├── en
# │   ├── attemptedClones.txt
# │   ├── attemptedSpeech.txt
# │   ├── voices_2026-06-20_09-23-17.csv
# │   └── wav/
# ├── fr
# │   ├── attemptedClones.txt
# │   ├── attemptedSpeech.txt
# │   ├── genSpeechLogs.txt
# │   ├── voices_2026-06-20_09-47-35.csv
# │   └── wav/
# └── pt
#     ├── attemptedClones.txt
#     ├── attemptedSpeech.txt
#     ├── genSpeechLogs.txt
#     ├── voices_2026-06-29_17-37-11.csv
#     └── wav/
#

SCRATCH = os.environ["SCRATCH"]
language = "en"

affects = ["Neutral","Angry","Happy","Relieved","Sad","Emotional","All"]
for affect in affects:
    subset = f"Neutral-{affect}"
    print(subset, end=" ")
    data_folder = f"{SCRATCH}/datasets/3M0X/{language}"
    save_folder = f"output/prepare/3M0X/{language}/{subset}"
    verification_pairs_file = f"data/3M0X/{language}/protocol/{subset}.txt"
    prepare_3M0X(data_folder, save_folder, verification_pairs_file)