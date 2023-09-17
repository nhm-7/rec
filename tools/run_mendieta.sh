#!/bin/bash
#SBATCH --job-name=mannProcessHor # short name for your job
#SBATCH --output=slurm-%x.%j.out # %j job id, ½x job name
#SBATCH --error=slurm-%x.%j.err
#SBATCH --partition=multi
#SBATCH --nodes=1                # node count
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=10       # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --time=0-00:30           # total run time limit (<days>-<hours>:<minutes>)
. /etc/profile
module purge
ulimit -c unlimited  # core dump
ulimit -s unlimited  # stack

echo "job \"${SLURM_JOB_NAME}\""
echo "  id: ${SLURM_JOB_ID}"
echo "  partition: ${SLURM_JOB_PARTITION}"
echo "  node(s): ${SLURM_JOB_NODELIST}"
echo " gres: ${SBATCH_GRES}"
echo " gpus: ${SBATCH_GPUS}"
date +"start %F - %T"
echo ""

source ${HOME}/.bashrc
conda activate rec-env

export TRANSFORMERS_OFFLINE=1
export NCCL_DEBUG=INFO

cd ${HOME}/rec/code/rec/

yaer run -e exp_004
