# Gemma-2 Experiments
To reproduce our code for fine-tuning `Gemma-2-9B-IT`, follow these steps:

## 1. Install the environment
```
mamba env create -f gemma_env.yml
mamba activate gemma
```
Note this was tested with `cuda 12.4` and an H100 GPU and may require changes for a different set up.
`cgm` is also required, so install it as well with
```
cd ..
python -m pip install -e .
cd gemma
```
You'll also need to get access to [Gemma 2](https://huggingface.co/google/gemma-2-9b-it) and log into huggingface via
```
huggingface-cli login
```

## 2. Finetune Gemma
You can now finetune Gemma, e.g.
```bash
OUT_DIR=gemma_cgm_relax
python cgm_gemma.py \
  --result_save_folder "$OUT_DIR" \
  --calibration_mode relax \
  --lambd 0.1 \
  --epochs 200 \
  --lr 2e-6 \
  --seed 0 \
  --batch_chunks 64 \
  --batch_size 512 \
  --model_save_folder "$OUT_DIR"
```
This will save logs, the finetuned model weights, and example generations to `OUT_DIR`.

## 3. Compute model log probabilities for evaluation
This step is used to compute the symmetrized KL.
```bash
python get_log_probs.py \
  --sample_path "$OUT_DIR"/cgm_relax_seed=0_lambd=0.1_samples.pt \
  --ft_model_path "$OUT_DIR"/cgm_relax_seed=0_lambd=0.1.pt \
  --output_folder "$OUT_DIR"
```
This produces a `.pt` file that contains base and finetuned Gemma log probabilities for samples from the finetuned model.
