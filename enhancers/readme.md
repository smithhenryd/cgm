# env setup
```bash
mamba env create -f env.yml
```

# pretraining data
The data can be downloaded from https://zenodo.org/records/10184648 or using `wget`:
```bash
wget https://zenodo.org/records/10184648/files/Taskiran_et_al_code_models_data.tar.gz?download=
```

# pretraining the conditional discrete diffusion model
## test command
```bash
python train_deepmel2.py --batch-size 256 --epochs 1 --lr 5e-4 --checkpoint-dir /scratch/users/diamant/enhancer_pretrain_smoke
```
## full command
```bash
python train_deepmel2.py --batch-size 256 --epochs 1000 --lr 5e-4 --checkpoint-dir /scratch/users/diamant/enhancer_pretrain
```

# alphagenome feature extraction preprocessing
## test run
```bash
python precompute_alphagenome_features.py \
  --output-pt /scratch/users/diamant/data/deepmel2_alphagenome_features_smoke.pt   --max-num-per-condition 2 \
  --background-file background.npy \
  --feature-batch-size 2 \
  --verbose
```

## full run
```bash
python precompute_alphagenome_features.py \
  --output-pt /scratch/users/diamant/data/deepmel2_alphagenome_features.pt   --max-num-per-condition 1024 \
  --feature-batch-size 4 \
  --background-file background.npy \
  --verbose
```

# kCGM test
```bash
python calibrate_deepmel2.py \
  --checkpoint-path /scratch/users/diamant/enhancer_pretrain_smoke/deepmel2-epoch=000.ckpt \
  --target-cache-pt /scratch/users/diamant/data/deepmel2_alphagenome_features_smoke.pt \
  --output-dir /scratch/users/diamant/deepmel_kCGM_test \
  --epochs 5 \
  --batch-size 4 \
  --sample-steps 4
```

# sweeps
`submit_calibrate_deepmel2_lambda_sweep.sh` is an example slurm script for sweeping over regularization strength for kCGM with the hyperparameters used in the paper.