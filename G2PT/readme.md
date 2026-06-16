# Environment setup
```bash
mamba create -n g2pt python=3.10
mamba activate g2pt
mamba install -c conda-forge pyarrow cvxpy
python -m pip install -r requirements.txt
python -m pip install fcd_torch
python -m pip install -e ..   # installs cgm package
```

The G2PT HuggingFace tokenizer is sensitive to the Transformers version. Keep
`transformers==5.2.0` from `requirements.txt`; newer versions such as 5.11.0 can
load `xchen16/g2pt-guacamol-small-bfs` through a BERT/WordPiece tokenizer path,
which makes `make_abx_smiles_csv.py` reject every molecule with
`WordPiece error: Missing [UNK] token from the vocabulary`.

# prepare target antibiotics
```bash
python make_abx_smiles_csv.py
```

# try finetuning a model
```python
python calibrate_g2pt.py \
  --out_root /scratch/users/diamant/test_g2pt \
  --feature morgan \
  --kernel tanimoto \
  --lambd 1e-3 \
  --epochs 5 \
  --batch_size 16 \
  --model_name xchen16/g2pt-guacamol-small-bfs \
  --target_csv abx_smiles.csv
```

# slurm example script
`submit_calibrate_g2pt_abx_lambda_sweep.sh` shows an example sweep over regularization strength with the hyperparameters used in the paper.
`./submit_finetune_g2pt_abx_lambda_sweep.sh` is the same for the direct finetuning baseline.