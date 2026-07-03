import os
from prepare_3M0X import prepare_3M0X

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