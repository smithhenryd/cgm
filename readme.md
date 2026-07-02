# Calibrating Generative Models

This repo contains code for finetuning generative models to match distribution-level constraints:
- **CGM** controls the mean of black-box features of samples from your model.
- **kCGM** extends the same model interface to control the distribution of black-box features using MMD or energy-distance objectives.

This repository accompanies two manuscripts:
- **Calibrating Generative Models to Distributional Constraints** by Henry D. Smith, Nathaniel L. Diamant, and Brian L. Trippe. [Preprint](https://arxiv.org/abs/2510.10020)
- **Calibrating Generative Models to Feature Distributions with MMD Finetuning** by Nathaniel L. Diamant and Brian L. Trippe. [Preprint](https://arxiv.org/abs/2606.19496)

We propose lightweight, general-purpose algorithms for fine-tuning generative models to match distribution-level constraints. **CGM-relax** and **CGM-reward** target constraints on feature means, while **kCGM** targets full feature distributions. These methods apply to diverse model classes, data, and constraint types, and use the same base model abstraction: a generative model that can draw samples and evaluate their log probabilities.

<div align="center">
  <img src="https://github.com/smithhenryd/cgm/blob/main/figs/animation.gif?raw=true" width="600">
  <br>
  <em>Calibrating the Genie2 protein structure diffusion model to secondary structure statistics of natural proteins (CATH domains).</em>
</div>

<div align="center">
  <img src="https://github.com/smithhenryd/cgm/blob/kCGM/figs/smiley_training_animation.gif?raw=true" width="600">
  <br>
  <em>Using kCGM to calibrate the G2PT small-molecule autoregressive model to match a target smiley-face distribution of molecular descriptors.</em>
</div>

## Getting Started
You can try out the `cgm` codebase by opening our demo notebook [`gmm_example.ipynb`](neural_sde/gmm_example.ipynb) in Google Colab [[link](https://colab.research.google.com/github/smithhenryd/cgm/blob/main/neural_sde/gmm_example.ipynb)]. Alternatively, clone this repository, install the package, and implement the common `Model` interface for your generator. Once your model implements `sample` and `log_p`, you can calibrate feature means with CGM or feature distributions with kCGM.

## Installation
### Package manager
We recommend using `conda` or `mamba` to install the `cgm` requirements.
`mamba` can be installed by following [these instructions](https://github.com/conda-forge/miniforge), which amount to the following:
```
curl -L -O https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
chmod +x Miniforge3-Linux-x86_64.sh
./Miniforge3-Linux-x86_64.sh
```
### Environment install
The `cgm` environment can be installed from the environment file:
```
mamba create -f env.yml
```
Once you have activated the `cgm` environment, install the `cgm` package (from the root directory of this repository):
```
python -m pip install -e .
```
To use `cgm` in the demo notebook, you also have to install `cgm` as an `ipykernel`:
```
python -m ipykernel install --user --name=cgm
```
This same environment is sufficient for the core CGM and kCGM APIs in `cgm/`. Some paper experiments require extra domain-specific dependencies; see the experiment subfolder READMEs for those setup steps.

You can verify that your installation is correct by running the [tests](#tests), or by running the demo notebook [`gmm_example.ipynb`](neural_sde/gmm_example.ipynb).

## Usage
To perform fine-tuning with CGM or kCGM, first implement a subclass `MyModel` of `Model`, which is contained in `cgm/model.py`. `Model` is an abstract base class representing the generative model to be calibrated. It has two methods that must be overridden:
- `sample`: draws samples from the generative model
- `log_p`: evaluates the log probability of samples from the generative model

An example implementation for continuous-time diffusion models, `NeuralSDE`, is given in `neural_sde/neural_sde.py`.

Once you have implemented `MyModel`, load or train your base model `base_model` as an instance of `MyModel`. You are then prepared to calibrate `base_model`.

For CGM-relax, `h` maps samples to feature vectors and `hstar` is the target feature mean:
```
from cgm.cgm import calibrate_relaxed

relax_model = calibrate_relaxed(
    base_model,
    h,
    hstar,
    lambd,
)
```

For CGM-reward, the interface is similar, with `N_samp` controlling the number of samples used to estimate the reward parameters:
```
from cgm.cgm import calibrate_reward

reward_model = calibrate_reward(
    base_model,
    h,
    hstar,
    N_samp,
)
```

For kCGM, use the same model interface, but pass target feature samples rather than a target feature mean. The main additional choice is the kernel used in the MMD objective:
```
from cgm.cgm_distribution import calibrate_mmd, energy_distance_kernel

mmd_model = calibrate_mmd(
    base_model,
    h,
    hstar_samples,
    lambd,
    kernel=energy_distance_kernel(),
    use_loo=True,
)
```

Here `hstar_samples` has shape `[num_target_samples, feature_dim]` and represents samples from the target feature distribution. The `kernel` argument defines which feature-distribution discrepancy is optimized.
[kernels.py](cgm/kernels.py) implements the kernels we use in the kCGM paper, such as `energy_distance_kernel` and `tanimoto_kernel`.
You can also pass any custom callable with signature `kernel(x, y) -> [x.shape[0], y.shape[0]]`.

The `use_loo` argument in `calibrate_mmd` only controls the leave-one-out baseline for the MMD coefficient estimate. Unlike `use_loo` in `cgm.py`, it does not turn off the leave-one-out baseline for the KL-to-base term; the KL baseline is always used in `calibrate_mmd`.

For a full demonstration of the package functionality, see our [example reweighting mixture proportions in a GMM](https://github.com/smithhenryd/cgm/blob/main/neural_sde/gmm_example.ipynb).

## Tests
Make sure the `cgm` environment is activated.
Then run
```
python -m pytest tests
```

## kCGM paper experiments
The central kCGM code can be found in [cgm_distribution.py](cgm/cgm_distribution.py), with paper experiments in the subfolders listed below.

### Experiments from the paper
- [Antibiotics feature distribution matching](./G2PT/)
- [Increasing protein structure diversity](./genie2/)
- [Generating DNA with realistic cell-type-specific activity profiles](./enhancers/)
