from functools import partial
from dataclasses import dataclass, fields
import os, shutil
from copy import deepcopy
import pickle
from collections import defaultdict
from typing import Optional, Generator
from argparse import ArgumentParser

import pydssp
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from tqdm import tqdm
from scipy.stats import ttest_1samp
from Bio.PDB import PDBParser, PDBIO, Select

from esm.models.esm3 import ESM3
from esm.utils.constants import esm3 as C
from esm.sdk.api import ESMProteinTensor, GenerationConfig
from esm.utils.structure.protein_chain import ProteinChain
from esm.utils.noise_schedules import cosine_schedule

from cgm.model import Model
from cgm.cgm import calibrate_relaxed, calibrate_reward
from cgm.utils import DictLogger, default_parser, BestCheckpoint, load_checkpoint


def clean_df(csv_path: str) -> pd.DataFrame:
    """
    Reads in a df containing alpha helix and beta strand proportions and performs basic cleaning
    """
    df = pd.read_csv(csv_path)
    df = df.rename(columns={"% Helix": "alpha_helix", "% Strand": "beta_strand"})

    # Clean the data
    df = (
        df[["alpha_helix", "beta_strand"]]
        .apply(pd.to_numeric, errors="coerce")
        .dropna()
    )
    df = df[
        df["alpha_helix"].between(0, 1) & df["beta_strand"].between(0, 1)
    ]  # proportion of residues in alpha helix and beta strand must be between 0. and 1.
    df = df[
        df["alpha_helix"] + df["beta_strand"] <= 1.0
    ]  # sum of alpha helix and beta strand proportions must be <= 1.

    return df


def compute_bivariate_quantiles(
    df: pd.DataFrame,
    N_quantiles: int,
) -> tuple[list[float], dict[str, list[float]]]:
    """
    For thresholds h in alpha_helix and s in beta_strand (grid over (0,1]),
    compute P(alpha_helix <= h, beta_strand <= s) for each pair, skipping (1.0, 1.0).

    Returns:
        quantiles: list of bivariate CDF values aligned with the pairs below
        quantile_vals: {"alpha": [h_i...], "beta": [s_i...]} giving the threshold pairs
    """
    helix_ths = torch.linspace(
        0.0,
        1.0,
        N_quantiles + 2,
    ).tolist()[
        1:
    ]  # drop 0.0
    strand_ths = torch.linspace(
        0.0,
        1.0,
        N_quantiles + 2,
    ).tolist()[
        1:
    ]  # drop 0.0

    helix_prop: list[float] = []
    strand_prop: list[float] = []
    quantiles: list[float] = []

    for h in helix_ths:
        for s in strand_ths:
            # Skip the trivial (1.0, 1.0) constraint
            if h == 1.0 and s == 1.0:
                continue

            helix_prop.append(h)
            strand_prop.append(s)

            mask = (df["alpha_helix"] <= h) & (df["beta_strand"] <= s)
            quantiles.append(float(mask.mean()))

    quantile_vals = {"alpha": helix_prop, "beta": strand_prop}
    return quantiles, quantile_vals


def h_bi(x: dict[str, torch.Tensor], quantile_vals: list) -> torch.Tensor:
    """
    Calibration function corresponding to bivariate quantile constraints
    """
    x = x["xs"][:, -1]

    hh_vals = []
    for b in range(x.shape[0]):
        x_b = x[b].reshape([-1, 3]).cpu().numpy()
        ss_val = secstruct.compute_joint_quantiles(
            x_b, quantile_vals
        )  # determine whether or not % of helices, % of strands are jointly less than the specified quantiles
        hh_vals.append(
            torch.tensor(ss_val, device=x.device, dtype=x.dtype).reshape(1, -1)
        )
    hh_vals = torch.concat(hh_vals, axis=0)
    return hh_vals


def pairs_to_quantile_mask(
    batch: torch.Tensor,  # shape [B, 2] :: (alpha_prop, beta_prop)
    quantile_vals: dict[str, list],  # {"alpha": [h1,...,hM], "beta": [s1,...,sM]}
) -> torch.BoolTensor:
    """
    Returns a boolean tensor of shape [B, M] where mask[b, m] is True iff:
        batch[b, 0] <= quantile_vals["alpha"][m]  and
        batch[b, 1] <= quantile_vals["beta"][m]

    The column order matches quantile_vals (and therefore the original quantiles list).
    """
    if batch.ndim != 2 or batch.size(-1) != 2:
        raise ValueError(f"Expected batch of shape [B, 2], got {tuple(batch.shape)}")
    alpha_list = quantile_vals["alpha"]
    beta_list = quantile_vals["beta"]
    if len(alpha_list) != len(beta_list):
        raise ValueError("quantile_vals['alpha'] and ['beta'] must be the same length.")

    device = batch.device
    dtype = batch.dtype

    # [1, M] tensors on the same device/dtype as batch
    alpha_grid = torch.as_tensor(alpha_list, dtype=dtype, device=device).unsqueeze(
        0
    )  # [1, M]
    beta_grid = torch.as_tensor(beta_list, dtype=dtype, device=device).unsqueeze(
        0
    )  # [1, M]

    # Split batch into alpha/beta columns: [B, 1] each
    alpha_b = batch[:, 0:1]
    beta_b = batch[:, 1:2]

    # Broadcast compare to get [B, M] mask
    mask = (alpha_b <= alpha_grid) & (beta_b <= beta_grid)
    return mask.float()


class CaSelect(Select):
    def accept_atom(self, atom) -> bool:
        return atom.get_name() == "CA"


@dataclass
class Sample:
    intermediates: torch.Tensor
    next_unmasks: torch.Tensor
    log_p_theta: torch.Tensor


def combine_samples(samples: list[Sample]) -> Sample:
    return Sample(
        *(
            torch.cat([getattr(sample, field.name) for sample in samples], dim=0)
            for field in fields(Sample)
        )
    )


class ESMModelTauLeap(Model[Sample]):
    def __init__(
        self,
        seq_len: int = 100,
        max_parallel_samples: int = 128,
        sample_steps: Optional[int] = None,
        tau_leap: bool = True,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.model = ESM3.from_pretrained("esm3-open").to("cuda")
        self.decoder = self.model.get_structure_decoder()
        self.max_parallel_samples = max_parallel_samples
        self.seq_len = seq_len
        self.sample_steps = sample_steps
        self.tau_leap = tau_leap
        if self.tau_leap:
            assert sample_steps is not None
        else:
            print("Ignoring sample_steps since not tau leaping")
            self.sample_steps = self.seq_len
        self.num_to_unmask = self._build_n_unmask_per_step()
        self.temperature = temperature

    def _get_logits(self, x: torch.Tensor) -> torch.Tensor:
        """
        temperature scaled structure logits from structure tokens x
        """
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = self.model.forward(
                structure_tokens=x
            ).structure_logits  # [B, L, V]
        return logits / self.temperature

    def _build_n_unmask_per_step(self):
        """
        Deterministic version of cosine schedule. Follows ESM-3 codebase:
        https://github.com/evolutionaryscale/esm/blob/main/esm/utils/generation.py#L284C5-L284C34
        """
        if not self.tau_leap:
            num_to_unmask = torch.ones(self.seq_len)
            return num_to_unmask
        t = torch.linspace(0, 1, self.sample_steps + 1)
        perc_masked = cosine_schedule(t)
        n_masked = torch.round(perc_masked * self.seq_len)
        num_to_unmask = n_masked[:-1] - n_masked[1:]
        num_to_unmask = num_to_unmask[num_to_unmask != 0].to(
            torch.int
        )  # skip zero unmask steps
        assert num_to_unmask.sum() == self.seq_len
        assert (num_to_unmask > 0).all()
        return num_to_unmask

    def _sample(self, sample_bsz: int) -> Sample:
        batch_range = torch.arange(sample_bsz, device=self.device)
        masked = ESMProteinTensor.empty(length=self.seq_len, device=self.device)
        x0 = masked.structure.unsqueeze(0).repeat(sample_bsz, 1)

        x = x0.clone()
        next_pos_probs = torch.ones_like(x, dtype=torch.float32)
        next_pos_probs[:, 0] = 0  # never unmask start and end of sequence
        next_pos_probs[:, -1] = 0
        seq_log_probs = torch.zeros(x.shape[0], device=self.device)

        intermediates: list[torch.Tensor] = [x.clone()]
        next_unmasks: list[torch.Tensor] = []
        for num_to_unmask in self.num_to_unmask:
            k = int(num_to_unmask)  # robust if elements are 0-dim tensors

            with torch.no_grad():
                logits = self._get_logits(x)  # [B, L, V]

            # sample K positions per batch without replacement -> [B, K]
            next_unmask_idx = torch.multinomial(
                next_pos_probs,
                num_samples=k,
                replacement=False,
            )

            # build a binary mask of chosen positions -> [B, L] (bool), stackable later
            chosen_mask_long = torch.zeros_like(x, dtype=torch.long)
            chosen_mask_long.scatter_(
                1, next_unmask_idx, torch.ones_like(next_unmask_idx, dtype=torch.long)
            )
            chosen_mask = chosen_mask_long.bool()
            next_unmasks.append(chosen_mask)

            # zero out chosen positions in next_pos_probs using scatter
            next_pos_probs.scatter_(
                1,
                next_unmask_idx,
                torch.zeros_like(next_unmask_idx, dtype=next_pos_probs.dtype),
            )

            # gather logits for chosen positions: [B, L, V] -> [B, K, V]
            token_logits = logits.gather(
                1,
                next_unmask_idx.unsqueeze(-1).expand(-1, -1, logits.size(-1)),
            ).float()

            token_dist = Categorical(logits=token_logits)
            token_vals = token_dist.sample()  # [B, K]

            # write sampled tokens back into x via scatter
            x.scatter_(1, next_unmask_idx, token_vals)

            # accumulate log probs
            log_probs = token_dist.log_prob(token_vals)  # [B, K]
            seq_log_probs += log_probs.sum(-1)  # [B]

            intermediates.append(x.clone())

        # tracking
        intermediates = torch.stack(intermediates, dim=1)  # [B, S+1, L]
        next_unmasks = torch.stack(next_unmasks, dim=1)  # [B, S, L] (bool)

        return Sample(intermediates, next_unmasks, seq_log_probs)

    def sample(self, N: int) -> Sample:
        assert N % self.max_parallel_samples == 0
        samples = [
            self._sample(self.max_parallel_samples)
            for _ in range(0, N, self.max_parallel_samples)
        ]
        return combine_samples(samples)

    def log_p(
        self,
        sample: Sample,
        batch_idx: int = 0,
        batch_chunks: int = 1,
        sample_idx: int = 0,
        sample_chunks: int = 1,
    ) -> torch.Tensor:
        bsz = sample.intermediates.shape[0]
        assert sample_chunks == self.num_sample_chunks
        assert bsz % batch_chunks == 0

        step = bsz // batch_chunks
        batch_start = batch_idx * step
        batch_end = batch_start + step

        # Slice the sub-batch
        sub_batch_intermediates = sample.intermediates[
            batch_start:batch_end
        ]  # [step, S+1, L]
        x = sub_batch_intermediates[:, sample_idx]  # [step, L] (before)
        y_next = sub_batch_intermediates[:, sample_idx + 1]  # [step, L] (after)
        mask = sample.next_unmasks[
            batch_start:batch_end, sample_idx
        ]  # [step, L] (bool)
        y_next = torch.where(mask, y_next, 0)  # for safe indexing

        # Forward with grads; autocast is fine (no torch.no_grad)
        logits = self._get_logits(x)
        all_log_probs = logits.log_softmax(dim=-1)  # [step, L, V]

        # Log-prob of the realized tokens y_next at every position
        pos_log_probs = all_log_probs.gather(
            dim=-1,
            index=y_next.unsqueeze(-1),  # [step, L, 1]
        ).squeeze(
            -1
        )  # [step, L]

        # Sum only over positions unmasked at this time step
        step_log_probs = (pos_log_probs * mask).sum(dim=1)  # [step]

        return step_log_probs

    @property
    def num_sample_chunks(self) -> int:
        return len(self.num_to_unmask)

    def sample_to_chains(self, sample: Sample) -> Generator[ProteinChain, None, None]:
        x = sample.intermediates[:, -1]
        bsz = 128
        num_batches = max(1, x.shape[0] // bsz)
        batches = torch.tensor_split(x, num_batches)
        coords = []
        for batch in batches:
            with torch.no_grad():
                decoded = self.decoder.decode(batch)
            coords.append(decoded["bb_pred"][:, 1:-1].cpu())
        coords = torch.cat(coords, dim=0)
        for coord in coords:
            chain = ProteinChain.from_backbone_atom_coordinates(coord)
            chain = chain.infer_oxygen()
            yield chain

    def get_alpha_beta_props(self, sample: Sample) -> torch.Tensor:
        obs = torch.empty((len(sample.intermediates), 2), device=self.device)
        for i, chain in enumerate(self.sample_to_chains(sample)):
            ss3 = pydssp.assign(
                chain.atom37_positions[:, [0, 1, 2, 4]], out_type="index"
            )
            obs[i, 0] = (ss3 == 1).mean()
            obs[i, 1] = (ss3 == 2).mean()
        return obs

    def sample_to_pdbs(
        self, sample: Sample, save_dir: str, zip_out: bool = True
    ) -> None:
        os.makedirs(save_dir, exist_ok=True)
        for i, chain in tqdm(
            enumerate(self.sample_to_chains(sample)),
            desc=f"saving PDBs to {save_dir}",
            total=sample.log_p_theta.shape[0],
        ):
            chain_path = os.path.join(save_dir, f"{i}.pdb")
            chain.to_pdb(chain_path)
            # save only CA
            parser = PDBParser(QUIET=True)
            structure = parser.get_structure("x", chain_path)
            io = PDBIO()
            io.set_structure(structure)
            io.save(
                chain_path,
                CaSelect(),
            )

        if not zip_out:
            return

        abs_dir = os.path.abspath(save_dir)
        parent = os.path.dirname(abs_dir)
        folder = os.path.basename(abs_dir)

        # This creates `<save_dir>.zip` *next to* the folder and includes the top-level folder in the archive
        archive_base = os.path.join(parent, folder)
        zip_path = shutil.make_archive(
            base_name=archive_base,
            format="zip",
            root_dir=parent,
            base_dir=folder,
        )
        # Delete the original folder after a successful zip
        if os.path.exists(zip_path):
            shutil.rmtree(abs_dir)
        else:
            raise ValueError(f"Zipping failed, {zip_path} does not exist")


def get_h(
    sample: Sample, model: ESMModelTauLeap, quantile_vals: dict[str, list[float]]
) -> torch.Tensor:
    obs = model.get_alpha_beta_props(sample)
    return pairs_to_quantile_mask(obs, quantile_vals)


def build_parser() -> ArgumentParser:
    parser = default_parser()
    parser.add_argument(
        "--model_save_folder",
        type=str,
        required=False,
        default="",
        help="Directory to save model checkpoints.",
    )
    parser.add_argument(
        "--result_save_folder",
        type=str,
        required=True,
        help="Directory to save/load sampling results.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--sample_steps",
        type=int,
        default=40,
    )
    parser.add_argument(
        "--tau_leap",
        action="store_true",
    )
    parser.add_argument(
        "--save_pre_pdb",
        action="store_true",
        help="Save PDBs from samples pre-finetuning",
    )
    parser.add_argument(
        "--save_post_pdb",
        action="store_true",
        help="Save PDBs from samples post-finetuning",
    )
    parser.add_argument("--num_quantiles", default=7, type=int)
    parser.add_argument(
        "--temperature",
        default=1.0,
        type=float,
    )
    parser.add_argument(
        "--save_last_only",
        action="store_true",
        help="Don't checkpoint on best loss, only final model",
    )
    parser.add_argument(
        "--num_samples", default=2048, type=int, help="Number of PDBs to save"
    )
    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    df = clean_df("./cath_biotite_ss.csv")
    N_quantiles = args.num_quantiles

    model = ESMModelTauLeap(
        sample_steps=args.sample_steps,
        tau_leap=args.tau_leap,
        temperature=args.temperature,
    )
    n_samp = args.num_samples

    pre_sample = model.sample(n_samp)
    pre = model.get_alpha_beta_props(pre_sample)
    pre_df = pd.DataFrame(pre.cpu().numpy(), columns=["alpha_helix", "beta_strand"])
    quantiles, quantile_vals = compute_bivariate_quantiles(df, N_quantiles)
    hstar = torch.tensor(quantiles, device="cuda")
    test_v = torch.tensor(df.values, device="cuda")
    test_v = pairs_to_quantile_mask(test_v, quantile_vals).mean(0)
    assert torch.allclose(test_v, hstar)

    h_fn = partial(get_h, model=model, quantile_vals=quantile_vals)
    h = h_fn(pre_sample)
    initial_error = (h.mean(0) - hstar).abs().mean()
    print(f"Initial error: {initial_error:.2f}")

    pre_df["model"] = "pre-trained"
    df["model"] = "CATH"

    logger = DictLogger()
    lr = 1e-4
    name = f"cgm_{args.calibration_mode}_seed={args.seed}"
    if args.model_save_folder:
        model_save_path = os.path.join(args.model_save_folder, f"{name}.pt")
        if args.save_last_only:
            checkpoint_fn = None
        else:
            checkpoint_fn = BestCheckpoint(args.model_save_folder)
            model_save_path = os.path.join(
                args.model_save_folder, checkpoint_fn.checkpoint_name()
            )
    else:
        checkpoint_fn = None
    if args.calibration_mode == "relax":
        final_model = calibrate_relaxed(
            model,
            h=h_fn,
            hstar=hstar,
            optimizer_params={"lr": lr},
            batch_size=args.batch_size,
            batch_chunks=args.batch_size // 64,
            samp_chunks=model.num_sample_chunks,
            lambd=args.lambd,
            logger=logger,
            epochs=args.epochs,
            checkpoint_fn=checkpoint_fn,
        )
    elif args.calibration_mode == "reward":
        final_model = calibrate_reward(
            model,
            N_samp=2048,
            h=h_fn,
            hstar=hstar,
            optimizer_params={"lr": lr},
            batch_size=args.batch_size,
            batch_chunks=args.batch_size // 64,
            samp_chunks=model.num_sample_chunks,
            logger=logger,
            epochs=args.epochs,
            checkpoint_fn=checkpoint_fn,
            dual_max_iters=10_000,
        )
    else:
        raise NotImplementedError(f"{args.calibration_mode=} is not a valid mode.")
    if args.model_save_folder:
        if args.save_last_only:
            torch.save(final_model.state_dict(), model_save_path)
        else:
            _ = load_checkpoint(final_model, model_save_path)
    final_sample = final_model.sample(n_samp)
    post = final_model.get_alpha_beta_props(final_sample)
    post_df = pd.DataFrame(post.cpu().numpy(), columns=["alpha_helix", "beta_strand"])
    post_df["model"] = "finetuned"
    plot_df = pd.concat([df.sample(n_samp, random_state=args.seed), pre_df, post_df])

    os.makedirs(args.result_save_folder, exist_ok=True)
    prop_df = pd.concat([pre_df, post_df])
    prop_df.to_csv(
        os.path.join(args.result_save_folder, "proportions.csv"), index=False
    )

    plt.rcParams["figure.dpi"] = 200
    sns.set_style("whitegrid")
    palette = sns.color_palette("Set1", n_colors=3)
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(12, 4),
        sharex=True,
        sharey=True,
    )
    models = ["CATH", "pre-trained", "finetuned"]
    for i, (m, ax) in enumerate(zip(models, axes)):
        sns.kdeplot(
            data=plot_df[plot_df["model"] == m],
            x="alpha_helix",
            y="beta_strand",
            fill=True,
            alpha=0.8,
            levels=10,
            # thresh=0.2,
            clip=((0, 1), (0, 1)),
            color=palette[i],
            ax=ax,
            common_norm=False,  # harmless here; kept for parity with your call
        )
        ax.set_title(str(m))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.set_ylabel(None)
        ax.set_xlabel(None)

    # Shared axis labels
    fig.supxlabel("$\\alpha$")
    fig.supylabel("$\\beta$")
    fig.suptitle("KDE of ($\\alpha$, $\\beta$) per model", y=1.03)
    sns.despine()
    plt.tight_layout()

    plt.savefig(
        os.path.join(args.result_save_folder, "tau_cgm_result_proportions.png"),
        bbox_inches="tight",
    )

    constraints = logger.metrics["constraint_loss"]
    kls = logger.metrics["kl_loss"]
    fig, (ax, ax2) = plt.subplots(ncols=2, dpi=150, figsize=(8, 4), sharex=True)
    ax.plot(constraints)
    ax.set_xlabel("step")
    ax.set_ylabel("constraint loss")

    ax2.plot(kls)
    ax2.set_xlabel("step")
    ax2.set_ylabel("KL")
    plt.savefig(
        os.path.join(args.result_save_folder, "tau_cgm_training_curve.png"),
        bbox_inches="tight",
    )

    losses = logger.metrics["loss"]
    metric_df = pd.DataFrame(
        {
            "kl": kls,
            "constraint_loss": constraints,
            "loss": losses,
        }
    )
    metric_df.to_csv(os.path.join(args.result_save_folder, "metrics.csv"), index=False)

    if args.save_pre_pdb:
        pre_dir = os.path.join(args.result_save_folder, "pre_pdb_files")
        final_model.sample_to_pdbs(pre_sample, pre_dir)
    if args.save_post_pdb:
        post_dir = os.path.join(args.result_save_folder, "post_pdb_files")
        final_model.sample_to_pdbs(final_sample, post_dir)
