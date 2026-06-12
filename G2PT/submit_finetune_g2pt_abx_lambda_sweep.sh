#!/bin/bash
#SBATCH --job-name=finetune_g2pt_abx_lambda_v2
#SBATCH --partition=hns,owners,stat,akundaje
#SBATCH --array=0-14
#SBATCH -G 1
#SBATCH -C "GPU_MEM:48GB|GPU_MEM:80GB"
#SBATCH --cpus-per-task=2
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

# ABX lambda sweep for G2PT finetuning under the v2 experiment setup.
# Calibration and finetuning can share the same out_root because Python writes
# them under separate subdirectories (out_root/runs vs out_root/finetune_runs).

mkdir -p logs

source /home/groups/btrippe/diamant/miniforge/etc/profile.d/mamba.sh
mamba activate g2pt

cd /home/users/diamant/repos/cgm_distribution/G2PT || exit 1

LAMBDAS=(1e-4 1e-3 1e-2 1e-1 1.0)
SEEDS=(0 1 2)
LR=1e-5
LOSS_WEIGHTING=normalized
EPOCHS=500
BATCH_SIZE=192
N_EVAL_SAMPLES=9600
TARGET_CSV=/home/users/diamant/repos/cgm_distribution/G2PT/abx_smiles.csv
MODEL_NAME=xchen16/g2pt-guacamol-small-bfs
OUT_ROOT=/scratch/users/diamant/g2pt_2026-04-22_abx_lambda_sweep_v2

N_LAMBDAS=${#LAMBDAS[@]}
SEED=${SEEDS[$((SLURM_ARRAY_TASK_ID / N_LAMBDAS))]}
LAMBDA=${LAMBDAS[$((SLURM_ARRAY_TASK_ID % N_LAMBDAS))]}

echo "=== Array Task ${SLURM_ARRAY_TASK_ID}: lambda=${LAMBDA}, seed=${SEED}, lr=${LR}, loss_weighting=${LOSS_WEIGHTING} ==="
echo "=== Output Root: ${OUT_ROOT} ==="

args=(
  finetune_g2pt.py
  --out_root "$OUT_ROOT"
  --lambd "$LAMBDA"
  --loss_weighting "$LOSS_WEIGHTING"
  --seed "$SEED"
  --epochs "$EPOCHS"
  --batch_size "$BATCH_SIZE"
  --lr "$LR"
  --n_eval_samples "$N_EVAL_SAMPLES"
  --target_csv "$TARGET_CSV"
  --model_name "$MODEL_NAME"
  --bf16
  --cosine_schedule
  --grad_clip_norm 1.0
)

printf 'Running: '
printf '%q ' python "${args[@]}"
printf '\n'

python "${args[@]}"
