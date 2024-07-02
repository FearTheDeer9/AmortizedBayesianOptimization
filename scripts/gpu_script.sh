#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL # required to send email notifcations
#SBATCH --mail-user=jd123 # required to send email notifcations - please replace <your_username> with your college login name or email address
# the above path could also point to a miniconda install
export PATH=/vol/bitbucket/${USER}/anaconda3/bin/:$PATH
# if using miniconda, uncomment the below line
source ~/.bashrc
# Activate the specific Conda environment
source /vol/bitbucket/${USER}/anaconda3/bin/activate diffcbed
source /vol/cuda/12.5.0/setup.sh
/usr/bin/nvidia-smi
uptime
sh /vol/bitbucket/jd123/causal_bayes_opt/scripts/toy_bash.sh
