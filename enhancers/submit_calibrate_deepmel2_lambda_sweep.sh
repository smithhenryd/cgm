#!/bin/bash
#SBATCH --job-name=deepmel2_lambda_sweep
#SBATCH --partition=hns,akundaje
#SBATCH --array=0-17
#SBATCH -G 1
#SBATCH -C "GPU_MEM:48GB|GPU_MEM:80GB"
#SBATCH --cpus-per-task=4
#SBATCH --time=3:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

# DeepMel2 lambda sweep for distributional calibration.
#
# Defaults:
#   - methods: {kcgm, cgm}
#   - lambda grid: {1e-1, 1e-2, 1e-3}
#   - seeds: {0, 1, 2}
#   - kCGM: energy kernel with MMD leave-one-out correction
#   - CGM: dot-product kernel with no MMD leave-one-out correction
#
# Example usage:
#   sbatch enhancers/submit_calibrate_deepmel2_lambda_sweep.sh
#   sbatch --export=ALL,OUT_ROOT=/scratch/users/diamant/deepmel_cal_test \
#     enhancers/submit_calibrate_deepmel2_lambda_sweep.sh

mkdir -p logs

source /home/groups/btrippe/diamant/miniforge/etc/profile.d/mamba.sh
mamba activate enhancers

cd /home/users/diamant/repos/cgm_distribution/enhancers || exit 1

METHODS=(kcgm cgm)
LAMBDAS=(1e-1 1e-2 1e-3)
SEEDS=(0 1 2)

CHECKPOINT_PATH="${CHECKPOINT_PATH:-/scratch/users/diamant/enhancer_pretrain_2026-04-30_1k/deepmel2-epoch=719.ckpt}"
TARGET_CACHE_PT="${TARGET_CACHE_PT:-/scratch/users/diamant/data/deepmel2_alphagenome_features.pt}"
OUT_ROOT="${OUT_ROOT:-/scratch/users/diamant/2026-05-04_enhancer_lambda_sweep_20_epochs_pca-32}"
EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-128}"
SAMPLE_STEPS="${SAMPLE_STEPS:-50}"
PCA_COMPONENTS="${PCA_COMPONENTS:-32}"
FINAL_SAMPLES_PER_CONDITION="${FINAL_SAMPLES_PER_CONDITION:-512}"
FEATURE_BATCH_SIZE="${FEATURE_BATCH_SIZE:-4}"
LR="${LR:-1e-4}"
LR_SCHEDULE="${LR_SCHEDULE:-cosine}"
MIN_LR_RATIO="${MIN_LR_RATIO:-0.1}"
GRAD_CLIP="${GRAD_CLIP:-100}"
WANDB_PROJECT="${WANDB_PROJECT:-kcgm_enhancer_calibration}"

N_LAMBDAS=${#LAMBDAS[@]}
N_SEEDS=${#SEEDS[@]}
METHOD_IDX=$((SLURM_ARRAY_TASK_ID / (N_SEEDS * N_LAMBDAS)))
WITHIN_METHOD_IDX=$((SLURM_ARRAY_TASK_ID % (N_SEEDS * N_LAMBDAS)))
SEED=${SEEDS[$((WITHIN_METHOD_IDX / N_LAMBDAS))]}
LAMBDA=${LAMBDAS[$((WITHIN_METHOD_IDX % N_LAMBDAS))]}
METHOD=${METHODS[$METHOD_IDX]}

case "$METHOD" in
  kcgm)
    KERNEL=energy
    METHOD_FLAGS=(--kernel "$KERNEL")
    ;;
  cgm)
    KERNEL=dotproduct
    METHOD_FLAGS=(--kernel "$KERNEL" --no-loo)
    ;;
  *)
    echo "Unsupported METHOD=${METHOD}" >&2
    exit 1
    ;;
esac

RUN_NAME="method-${METHOD}_kernel-${KERNEL}_lambd-${LAMBDA}_seed-${SEED}"
OUTPUT_DIR="${OUT_ROOT}/${RUN_NAME}"
WANDB_NAME="${WANDB_NAME:-deepmel2_${RUN_NAME}}"

mkdir -p "$OUTPUT_DIR"

echo "=== Array Task ${SLURM_ARRAY_TASK_ID}: method=${METHOD}, kernel=${KERNEL}, lambda=${LAMBDA}, seed=${SEED}, lr=${LR}, lr_schedule=${LR_SCHEDULE}, min_lr_ratio=${MIN_LR_RATIO} ==="
echo "=== Output Dir: ${OUTPUT_DIR} ==="

args=(
  calibrate_deepmel2.py
  --checkpoint-path "$CHECKPOINT_PATH"
  --target-cache-pt "$TARGET_CACHE_PT"
  --output-dir "$OUTPUT_DIR"
  --epochs "$EPOCHS"
  --batch-size "$BATCH_SIZE"
  --sample-steps "$SAMPLE_STEPS"
  --pca-components "$PCA_COMPONENTS"
  --pca-whiten
  --final-samples-per-condition "$FINAL_SAMPLES_PER_CONDITION"
  --alphagenome-autocast-dtype bf16
  --matmul-precision high
  --wandb-project "$WANDB_PROJECT"
  --wandb-name "$WANDB_NAME"
  --feature-batch-size "$FEATURE_BATCH_SIZE"
  --lambd "$LAMBDA"
  --lr "$LR"
  --lr-schedule "$LR_SCHEDULE"
  --min-lr-ratio "$MIN_LR_RATIO"
  --grad-clip "$GRAD_CLIP"
  --diffusion-autocast-dtype bf16
  --seed "$SEED"
  "${METHOD_FLAGS[@]}"
)

printf 'Running: '
printf '%q ' python "${args[@]}"
printf '\n'

python "${args[@]}"
