import argparse
import pickle
import sys
from pathlib import Path
from typing import Any

import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader, TensorDataset

from simple_diff.cnn import CNNConfig
from simple_diff.model import CNNConditionalMDLM, CNNConditionalMDLMConfig
from simple_diff.utils import pick_precision


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a vector-conditioned masked diffusion model on DeepMEL2 enhancers."
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("/scratch/users/diamant/data/the_code/General/data/DeepMEL2_data.pkl"),
        help="Pickle with one-hot DNA arrays and multi-label cell type conditions.",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--n-filters", type=int, default=512)
    parser.add_argument("--num-dilation-blocks", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--accelerator", type=str, default="auto")
    parser.add_argument("--devices", type=str, default="auto")
    parser.add_argument("--log-every-n-steps", type=int, default=10)
    parser.add_argument("--print-freq", type=int, default=1)
    parser.add_argument("--gradient-clip-val", type=float, default=1.0)
    parser.add_argument("--wandb-project", type=str, default="kcgm_enhancer_pretrain")
    parser.add_argument("--wandb-name", type=str, default=None)
    return parser.parse_args()


def get_key(data: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in data:
            return data[key]
    raise KeyError(f"None of these keys were found in the data pickle: {keys}")


def one_hot_to_tokens(dna: Any) -> torch.Tensor:
    dna = torch.as_tensor(dna)
    if dna.ndim != 3 or dna.shape[-1] != 4:
        raise ValueError(
            f"Expected one-hot DNA with shape (n, length, 4), got {dna.shape}"
        )
    return dna.argmax(dim=-1).long()


def condition_tensor(cond: Any) -> torch.Tensor:
    cond = torch.as_tensor(cond)
    if cond.ndim != 2:
        raise ValueError(
            f"Expected condition matrix with shape (n, cond_dim), got {cond.shape}"
        )
    return cond.float()


def make_dataset(dna: Any, cond: Any) -> TensorDataset:
    x = one_hot_to_tokens(dna)
    y = condition_tensor(cond)
    if x.shape[0] != y.shape[0]:
        raise ValueError(
            f"DNA and condition row counts differ: {x.shape[0]} vs {y.shape[0]}"
        )
    return TensorDataset(x, y)


def load_data(data_path: Path) -> tuple[TensorDataset, TensorDataset]:
    with data_path.open("rb") as f:
        all_data = pickle.load(f)

    train_dataset = make_dataset(all_data["train_data"], all_data["y_train"])
    valid_dataset = make_dataset(
        get_key(all_data, "valid_data", "val_data"),
        get_key(all_data, "y_valid", "valid_y", "y_val", "val_y"),
    )
    return train_dataset, valid_dataset


def make_loader(
    dataset: TensorDataset, batch_size: int, shuffle: bool, num_workers: int
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )


def parse_devices(devices: str) -> str | int:
    if devices.isdigit():
        return int(devices)
    return devices


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    train_dataset, valid_dataset = load_data(args.data_path)
    train_loader = make_loader(
        train_dataset, args.batch_size, shuffle=True, num_workers=args.num_workers
    )
    valid_loader = make_loader(
        valid_dataset, args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    _, cond = train_dataset.tensors
    cnn_config = CNNConfig(
        cond_dim=cond.shape[1],
        sequence_conditioning=False,
        n_filters=args.n_filters,
        num_dilation_blocks=args.num_dilation_blocks,
        dropout=args.dropout,
    )
    module_cfg = CNNConditionalMDLMConfig(
        cnn_config=cnn_config,
        lr=args.lr,
        wd=args.weight_decay,
        print_freq=args.print_freq,
    )
    module = CNNConditionalMDLM(module_cfg)

    checkpoint_callback = ModelCheckpoint(
        monitor="val/loss",
        dirpath=args.checkpoint_dir,
        filename="deepmel2-{epoch:03d}",
        save_top_k=3,
        mode="min",
    )
    logger = WandbLogger(
        project=args.wandb_project,
        name=args.wandb_name,
        save_dir=str(args.checkpoint_dir),
    )

    trainer = Trainer(
        accelerator=args.accelerator,
        devices=parse_devices(args.devices),
        max_epochs=args.epochs,
        logger=logger,
        callbacks=[checkpoint_callback],
        enable_progress_bar=True,
        log_every_n_steps=args.log_every_n_steps,
        precision=pick_precision(),
        gradient_clip_val=args.gradient_clip_val,
    )
    trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=valid_loader)


if __name__ == "__main__":
    main()
