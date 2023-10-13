#!/bin/bash
#SBATCH --job-name=exp_009 # short name for your job
#SBATCH --mail-type=ALL
#SBATCH --mail-user=nicolas.hormann@mi.unc.edu.ar
#SBATCH --output=slurm-%x.%j.out # %j job id, ½x job name
#SBATCH --error=slurm-%x.%j.err
#SBATCH --partition=multi
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=20       # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --time=1-10:00           # total run time limit (<days>-<hours>:<minutes>)

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
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=lo

cd ${HOME}/rec/code/rec/

srun yaer run -e exp_009