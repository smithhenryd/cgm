# ESM3 Experiments
To reproduce our code for fine-tuning `ESM3-open`, follow these steps:

## 1. Install the environment
```
mamba env create -f esm_env.yml
mamba activate esm3
```
Note this was tested with `cuda 12.4` and an H100 GPU and may require changes for a different set up.
`cgm` is also required, so install it as well with
```
cd ..
python -m pip install -e .
cd esm3
```
You'll also need to get access to [ESM3 open](https://huggingface.co/EvolutionaryScale/esm3-sm-open-v1/tree/main) and log into huggingface via
```
huggingface-cli login
```
## 2. Finetune ESM3
You can now finetune ESM3, e.g.
```bash
OUT_DIR=esm_cgm_relax
python cgm_esm3.py \
  --result_save_folder "$OUT_DIR" \
  --model_save_folder "$OUT_DIR" \
  --epochs 100 \
  --calibration_mode relax \
  --lambd 1 \
  --seed 0 \
  --sample_steps 50 \
  --tau_leap \
  --save_pre_pdb \
  --save_post_pdb \
  --num_quantiles 9 \
  --temperature 0.7
```
This will save logs, the finetuned model weights, and example generations as PDBs to `OUT_DIR`.

## 3. Evaluate secondary structure
See [genie2/readme.md](../genie2/readme.md) for how to get secondary structure proportions from the generated PDBs.
