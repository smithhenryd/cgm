#!/bin/bash
#SBATCH --job-name=calibrate_g2pt_abx_lambda_v2
#SBATCH --partition=hns,owners,stat,akundaje
#SBATCH --array=0-14
#SBATCH -G 1
#SBATCH -C "GPU_MEM:48GB|GPU_MEM:80GB"
#SBATCH --cpus-per-task=2
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

# ABX lambda sweep for G2PT calibration under the v2 experiment setup.
#
# Defaults:
#   - lambda grid: {1e-4, 1e-3, 1e-2, 1e-1, 1}
#   - seeds: {0, 1, 2}
#   - loss weighting: normalized
#   - feature/kernel/MMD-LOO are controllable via sbatch --export
#
# Example usage:
#   sbatch --export=ALL,KERNEL=tanimoto \
#     submit_calibrate_g2pt_abx_lambda_sweep_v2.sh
#   sbatch --export=ALL,FEATURE=fcd,KERNEL=energy,MMD_LOO=on,N_PCA=64 \
#     submit_calibrate_g2pt_abx_lambda_sweep_v2.sh
#   sbatch --export=ALL,FEATURE=morgan,KERNEL=dot,MMD_LOO=off \
#     submit_calibrate_g2pt_abx_lambda_sweep_v2.sh

mkdir -p logs

source /home/groups/btrippe/diamant/miniforge/etc/profile.d/mamba.sh
mamba activate g2pt

cd /home/users/diamant/repos/cgm_distribution/G2PT || exit 1

LAMBDAS=(1e-4 1e-3 1e-2 1e-1 1.0)
SEEDS=(0 1 2)
LR=1e-5
LOSS_WEIGHTING=normalized
FEATURE="${FEATURE:-morgan}"
KERNEL="${KERNEL:?Must set KERNEL via sbatch --export, e.g. KERNEL=tanimoto or KERNEL=dot}"
MMD_LOO="${MMD_LOO:-on}"
N_PCA="${N_PCA:-64}"
N_HSTAR=500
EPOCHS=500
BATCH_SIZE=192
N_EVAL_SAMPLES=9600
TARGET_CSV=/home/users/diamant/repos/cgm_distribution/G2PT/abx_smiles.csv
MODEL_NAME=xchen16/g2pt-guacamol-small-bfs
OUT_ROOT="${OUT_ROOT:-/scratch/users/diamant/g2pt_2026-04-22_abx_lambda_sweep_v2}"

case "$MMD_LOO" in
  on)
    NO_LOO_FLAG=()
    ;;
  off)
    NO_LOO_FLAG=(--no_loo)
    ;;
  *)
    echo "Unsupported MMD_LOO=${MMD_LOO}. Use on or off." >&2
    exit 1
    ;;
esac

N_LAMBDAS=${#LAMBDAS[@]}
SEED=${SEEDS[$((SLURM_ARRAY_TASK_ID / N_LAMBDAS))]}
LAMBDA=${LAMBDAS[$((SLURM_ARRAY_TASK_ID % N_LAMBDAS))]}

echo "=== Array Task ${SLURM_ARRAY_TASK_ID}: feature=${FEATURE}, kernel=${KERNEL}, mmd_loo=${MMD_LOO}, n_pca=${N_PCA}, lambda=${LAMBDA}, seed=${SEED}, lr=${LR}, loss_weighting=${LOSS_WEIGHTING} ==="
echo "=== Output Root: ${OUT_ROOT} ==="

args=(
  calibrate_g2pt.py
  --out_root "$OUT_ROOT"
  --feature "$FEATURE"
  --kernel "$KERNEL"
  --n_hstar "$N_HSTAR"
  --lambd "$LAMBDA"
  --loss_weighting "$LOSS_WEIGHTING"
  --seed "$SEED"
  --epochs "$EPOCHS"
  --batch_size "$BATCH_SIZE"
  --lr "$LR"
  --n_eval_samples "$N_EVAL_SAMPLES"
  --n_pca "$N_PCA"
  --target_csv "$TARGET_CSV"
  --model_name "$MODEL_NAME"
  --bf16
  --cosine_schedule
  --grad_clip_norm 1.0
  "${NO_LOO_FLAG[@]}"
)

printf 'Running: '
printf '%q ' python "${args[@]}"
printf '\n'

python "${args[@]}"
