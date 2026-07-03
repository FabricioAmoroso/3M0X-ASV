#!/bin/bash
#SBATCH --account=def-aravila
#SBATCH --gres=gpu:h100:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=0-24:00:00
#SBATCH --job-name=speech_brain_emotional_asv_voxceleb1
#SBATCH --mail-user=fabricio.steinle-amoroso@inrs.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --error=%x.%j.err
#SBATCH --output=%x.%j.out

# Path to where you want to save your virtual env.
PROJECT_HOME=/home/amoroso/scratch
cd $PROJECT_HOME

module load python/3.10 cuda cudnn
virtualenv --no-download $PROJECT_HOME/speechbrain_env
source $PROJECT_HOME/speechbrain_env/bin/activate

pip install --no-index --upgrade pip
# Path to wherever you save the requirements_speechbrain.txt file
pip install --no-index -r /home/amoroso/projects/def-aravila/amoroso/emotional_asv/speech_brain/requirements_speechbrain.txt

# Install local speechbrain editable mode
cd speechbrain
pip install --editable .

# Path inside the cloned speechbrain repository where you should add the new .py and .yaml files following the existing structure of the repository
cd $PROJECT_HOME/speechbrain/recipes/VoxCeleb/SpeakerRec

echo
echo -----------------------------------------------------------
echo VoxCeleb1 ASV with Wav2vec2 - fine-tuning and testing
echo -----------------------------------------------------------
echo

echo Fine-tuning start
python train_speaker_embeddings_wav2vec2.py hparams/train_wav2vec2_vox1.yaml
echo Fine-tuning finish

echo Evaluation start
python speaker_verification_cosine_wav2vec2.py hparams/verification_wav2vec2.yaml
echo Evaluation finish