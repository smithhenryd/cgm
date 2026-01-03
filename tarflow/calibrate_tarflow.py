import torch
import torchvision as tv

# Import for logging run
import wandb

# Import for multithreading
from concurrent.futures import ThreadPoolExecutor
from itertools import repeat

from functools import partial
import argparse
import os
from pathlib import Path

from normalizing_flow import NormalizingFlow, Sample
from tarflow_map import TarflowMap

from cgm.cgm import calibrate_relaxed, calibrate_reward
from cgm.utils import solve_dual, CheckpointEveryN

from utils import save_image, classify_image

from typing import Any, List, Optional, Union


def wandb_logger(logs: dict[str, Any], *args) -> None:
    """
    Logger function
    """
    h_bar = logs["h_bar"]
    (
        logs["lion"],
        logs["tiger"],
        logs["wolf"],
        logs["fox"],
        logs["leopard"],
        logs["cheetah"],
    ) = (
        h_bar[0].item(),
        h_bar[1].item(),
        h_bar[2].item(),
        h_bar[3].item(),
        h_bar[4].item(),
        1. - h_bar.sum().item(),
    )  
    wandb.log(logs)


def h(
    x: torch.Tensor,
    output_dir: Union[str, Path],
    channel_size: int,
    img_size: int,
    animals: List[str],
) -> torch.Tensor:
    """
    Calibration function
    """

    xs = x.xs  # extract samples
    batch_size = xs.shape[0]

    output = torch.zeros(
        batch_size, len(animals), dtype=xs.dtype, device=xs.device
    )  # one hot encoding of labeled animal
    for i in range(batch_size):
        try:
            # Save image temporarily
            img = xs[i].reshape((channel_size, img_size, img_size))
            img_path = save_image(img.cpu(), output_dir)

            # And classify it
            result = classify_image(img_path, animals)
            label = result["label"]

        finally:
            # After we have classified the image, remove it
            os.remove(img_path)

        if label in animals:
            j = animals.index(label)
            output[i, j] = 1.0
        else:
            pass
    return output


def h_multithreaded(
    x: torch.Tensor,
    output_dir: str,
    channel_size: int,
    img_size: int,
    animals: List[str],
    max_workers: int = 8,
) -> torch.Tensor:
    """
    Calibration function with multithreading

    Allows parallelization of GPT queries
    """
    xs = x.xs  # extract samples
    batch_size = xs.shape[0]

    output = torch.zeros(
        batch_size, len(animals), dtype=xs.dtype, device=xs.device
    )  # one-hot

    # Save all images first (so I/O isn't inside the thread critical path)
    img_paths = []
    for i in range(batch_size):
        img = xs[i].reshape((channel_size, img_size, img_size)).detach()
        img_path = save_image(img.cpu(), output_dir)
        img_paths.append(img_path)

    # Classify in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        results = list(ex.map(classify_image, img_paths, repeat(animals)))

    # Map labels to one-hot output
    for i, res in enumerate(results):
        label = res.get("label", None)
        if label in animals:
            j = animals.index(label)
            output[i, j] = torch.tensor(1.0, dtype=xs.dtype, device=xs.device)

    # Cleanup temp files
    for p in img_paths:
        try:
            os.remove(p)
        except OSError:
            pass
    return output


def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str)  # path to tarflow checkpoint file
    parser.add_argument(
        "--calibration_mode", choices=["relax", "reward"]
    ) # 'relax' for CGM-relax, 'reward' for CGM-reward
    parser.add_argument("--epochs", type=int, default=100)  # number of CGM epochs
    parser.add_argument("--lr_init", type=float, default=1e-6)  # initial learning rate
    parser.add_argument("--lr_min", type=float, default=1e-8)  # minimum learning rate
    parser.add_argument("--batch_size", type=int, default=256)  # batch_size
    parser.add_argument(
        "--batch_chunks", type=int, default=16
    )  # number of chunks within batch
    parser.add_argument(
        "--lambda", dest="lambd", type=float, default=None
    )  # regularization parameter
    parser.add_argument(
        "--N", type=int, default=None
    )  # number of samples for CGM-reward
    parser.add_argument(
        "--N_samples", type=int, default=5*10**4
    )  # number of samples to draw from the fine-tuned model
    parser.add_argument(
        "--dir", dest="output_dir", type=str, default=None
    )  # directory for saving images
    parser.add_argument(
        "--ckpt_dir", type=str, default=None
    )  # directory for writing model checkpoints
    parser.add_argument(
        "--ckpt_every", type=int, default=None
    )  # checkpoint the Tarflow model every ckpt_every epochs


    args = parser.parse_args()
    (
        path,
        alg,
        epochs,
        batch_chunks,
        lr_init,
        lr_min,
        batch_size,
        lambd,
        N,
        N_samples,
        output_dir,
        ckpt_dir,
        ckpt_every,
    ) = (
        args.path,
        args.calibration_mode,
        args.epochs,
        args.batch_chunks,
        args.lr_init,
        args.lr_min,
        args.batch_size,
        args.lambd,
        args.N,
        args.N_samples,
        args.output_dir,
        args.ckpt_dir,
        args.ckpt_every,
    )
    (
        channel_size,
        img_size,
        patch_size,
        channels,
        num_blocks,
        layers_per_block,
        num_classes,
        noise_std,
    ) = (3, 256, 8, 768, 8, 8, 3, 0.07)
    dim = (img_size // patch_size) ** 2 * channel_size * patch_size**2  # state size

    # Directory for saving images
    if output_dir is None:
        output_dir = (
            str(lr_init).replace(".", "_") + "_" + str(lr_min).replace(".", "_")
        )
        output_dir = (
            (output_dir + "_" + alg + "_" + str(lambd).replace(".", "_"))
            if alg == "relax"
            else (output_dir + "_" + alg + "_" + str(N))
        )
    output_dir = os.path.join("tarflow_outputs", output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Perform calibration on a single GPU, if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define calibration constraint
    animals = ["lion", "tiger", "wolf", "fox", "leopard", "cheetah"] # animals belonging to the AFHQ wildlife class (Choi et al., 2020)
    max_workers = 32
    h_full = partial(
        h_multithreaded,
        output_dir=output_dir,
        channel_size=channel_size,
        img_size=img_size,
        animals=animals,
        max_workers=max_workers,
    )
    hh = lambda x: h_full(x)[:, :-1] # extract every coordinate except the last one 
    hstar = (1. / 6) * torch.ones((len(animals) - 1,), device=device)

    # Load model onto GPU
    tarflow_model = TarflowMap(
        path,
        device,
        channel_size,
        img_size,
        patch_size,
        channels,
        num_blocks,
        layers_per_block,
        num_classes,
        noise_std,
        sample_cls=2,
        sampling_args={"guidance": 1.0, "attn_temp": 1.0},
    )
    base_model = NormalizingFlow(dim, tarflow_model)
    print("Loaded tarflow checkpoint")

    # Directory for checkpointing model
    if (ckpt_every is not None) and (ckpt_every <= epochs):
        if ckpt_dir is None:
            ckpt_dir = (
                str(lr_init).replace(".", "_") + "_" + str(lr_min).replace(".", "_")
            )
            ckpt_dir = (
                (ckpt_dir + "_" + alg + "_" + str(lambd).replace(".", "_"))
                if alg == "relax"
                else (ckpt_dir + "_" + alg + "_" + str(N))
            )
        ckpt_dir = os.path.join("checkpoints", ckpt_dir)
        os.makedirs(
            ckpt_dir, exist_ok=True
        )  # make the checkpoint directory, if it doesn't already exist
        ckpt_fn = CheckpointEveryN(ckpt_dir=ckpt_dir, N=ckpt_every)
    else:
        ckpt_fn = None

    # Generate some samples from the base model
    batch_size_base = 10
    with torch.no_grad():
        samps = base_model.sample(batch_size_base, perform_tweedie=True)
    base_dir = os.path.join(output_dir, "base_samples")
    os.makedirs(base_dir, exist_ok=True)
    for i in range(batch_size_base):
        img = samps.xs[i].reshape((channel_size, img_size, img_size))
        tv.utils.save_image(
            img, os.path.join(base_dir, f"sample_{i}.png"), normalize=True
        )

    # Initialize the wandb run
    run = wandb.init(
        project="AFHQ-finetune",
        config={
            "batch_size": batch_size,
            "num_epochs": epochs,
            "lr_init": lr_init,
            "lr_min": lr_min,
            "animals": animals,
            "quantiles": hstar,
        },
    )
    
    if alg == "relax":
        wandb.config.update({"lambda": lambd}, allow_val_change=True)
        calibrated_model = calibrate_relaxed(
            base_model,
            hh,
            hstar,
            lambd,
            epochs,
            batch_size,
            optimizer_params={"lr": lr_init},
            scheduler_params={"T_max": epochs, "eta_min": lr_min},
            batch_chunks=batch_chunks,
            use_loo=True,
            logger=wandb_logger,
            checkpoint_fn=ckpt_fn,
            disable_pbar=True,
        )
    else:
        wandb.config.update({"N_samples": N}, allow_val_change=True)
        calibrated_model = calibrate_reward(
            base_model,
            hh,
            hstar,
            N,
            epochs,
            batch_size,
            optimizer_params={"lr": lr_init},
            scheduler_params={"T_max": epochs, "eta_min": lr_min},
            batch_chunks=batch_chunks,
            use_loo=True,
            logger=wandb_logger,
            checkpoint_fn=ckpt_fn,
            disable_pbar=True,
        )
    print("Finished fine-tuning tarflow model")
    
    # Sample images from CGM fine-tuned model
    ft_dir = os.path.join(output_dir, "finetuned_samples")
    os.makedirs(ft_dir, exist_ok=True)
    batch_size_base = 500 # draw 500 samples at a time
    iters = (N_samples + batch_size_base - 1) // batch_size_base 
    for i in range(iters):
        batch_size_base = min(batch_size_base, N_samples - batch_size_base * i)
        with torch.no_grad():
            samps = calibrated_model.sample(batch_size_base, perform_tweedie=True)
        for j in range(batch_size_base):
            img = samps.xs[j].reshape((channel_size, img_size, img_size))
            img = 0.5 * (img.clamp(-1, 1) + 1) # [-1, 1] -> [0, 1]
            tv.utils.save_image(
                img, os.path.join(ft_dir, f"sample_{batch_size_base * i + j}.png"), normalize=False
            )
        if iters > 1:
            print(f"Finished {(i + 1) * 500} / {N_samples} samples")
    print("Finished sampling")

if __name__ == "__main__":
    main()