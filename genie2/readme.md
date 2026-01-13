# Genie2 Experiments
To reproduce our code for fine-tuning Genie2, follow these steps:

1. Install `biotite` and `wandb` in your environment
```
mamba install -c conda-forge biotite wandb
```
2. Change your directory to `cgm/genie2/genie2`. Install the Genie2 codebase by running
```
pip install -e .
``` 
3. Download the Genie2 base model checkpoint at [https://github.com/aqlaboratory/genie2/releases/tag/v1.0.0](https://github.com/aqlaboratory/genie2/releases/tag/v1.0.0). We use the checkpoint corresponding to `epoch=40`. Move the checkpoint file into the directory `cgm/genie2/genie2/results/base/checkpoints`.

4. Change your directory to `cgm/genie2/`. Annotate secondary structure for the CATH domains by running 
```
python classify_secstruct_cath.py --jsonl https://people.csail.mit.edu/ingraham/graph-protein-design/data/cath/chain_set.jsonl  \
-o cath_biotite_ss.csv
``` 
which will write the secondary structure of the CATH domains to output file cath_biotite_ss.csv.

You are now prepared to fine-tune the Genie2 base model using CGM. Run, for example,
```
python calibrate_genie2.py \
  --path cath_biotite_ss.csv \
  --calibration_mode relax \
  --const_type bi \
  --N_quantiles 9 \
  --lambda 0.001 \
  --epochs 100 \
  --N_samples 1000 \
  --ckpt_every 50
```
This will save 1000 samples (as `.pdb` files) from the fine-tuned Genie2 model to the directory `genie_outputs`. 
It will also save a checkpoint of the Genie model every 50 epochs to the directory `checkpoints`.