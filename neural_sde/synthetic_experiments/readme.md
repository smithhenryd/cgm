# Synthetic Data Experiments

Our synthetic data experiments (Figures 2 and 3) can be run without installing any additional packages. All synthetic experiments consider the setting in which the base model is a diffusion model having terminal distribution equal to a product of Gaussian mixture models (GMMs), and the calibration task involves upweighting a particular mode along each dimension.

The scripts in this directory are
- `lambda_exp.py`: Experiment varying CGM-relax parameter `lambda`
- `N_exp.py`: Experiment varying CGM-reward parameter `N`
- `/rare_event`
    - `rare_event_relax.py`: Experiment varying the rarity of mode to upweight under the base model, with CGM-relax
    - `rare_event_reward.py`: Experiment varying the rarity of mode to upweight under the base model, with CGM-reward
- `/rare_event`
    - `increase_dim_relax.py`: Experiment increasing the constraint dimension and problem dimension simultaneously, with CGM-relax
    - `increase_dim_reward.py`: Experiment increasing the constraint dimension and problem dimension simultaneously, with CGM-reward