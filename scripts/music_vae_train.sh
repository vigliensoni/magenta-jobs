#! /bin/bash
#SBATCH --job-name=magenta-test
#SBATCH --gres=gpu:1        # request GPU "generic resource"
#SBATCH --cpus-per-task=6   # maximum CPU cores per GPU request: 6 on Cedar, 16 o    n Graham.
#SBATCH --mem=16GB        # memory per node
#SBATCH --time=0-00:10      # time (DD-HH:MM)
#SBATCH --output=%N-%j.out  # %N for node name, %j for jobID

# loading pyarrow and dependencies
module load nixpkgs/16.09  gcc/8.3.0  cuda/10.1 arrow/0.15.1 python/3.7.0 scipy-stack

# submission code
source ./bin/activate
python ./magenta/magenta/models/music_vae/music_vae_train.py \
--config="nade-drums_2bar_reduced" \
--run_dir="/home/vigliens/project/vigliens/2_CODE/07-MAG/runs/nade-drums_2bar_reduced" \
--mode=train \
--examples_path="/home/vigliens/project/vigliens/2_CODE/07-MAG/magenta-dev/magenta/data/input/footwork-8files.tfrecord"

