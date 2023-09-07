#!/bin/bash
#SBATCH --job-name=mannProcessHor # short name for your job
#SBATCH --output=slurm-%x.%j.out # %j job id, ½x job name
#SBATCH --error=slurm-%x.%j.err
#SBATCH --partition=multi
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=10       # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --time=1-24:00           # total run time limit (<days>-<hours>:<minutes>)


echo "job \"${SLURM_JOB_NAME}\""
echo "  id: ${SLURM_JOB_ID}"
echo "  partition: ${SLURM_JOB_PARTITION}"
echo "  node(s): ${SLURM_JOB_NODELIST}"
date +"start %F - %T"
echo ""

#. /etc/profile

source ${HOME}/.bashrc
conda activate rec-env

cd ${HOME}/rec/code/rec/

srun yaer run -e exp_001
