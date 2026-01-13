import os

import math
from argparse import ArgumentParser
from typing import Optional, Callable
from dataclasses import dataclass

import pandas as pd
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
import torch
import torch.nn.functional as F
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
import gender_guesser.detector as gender
from scipy import stats

from cgm.model import Model
from cgm.cgm import calibrate_relaxed, calibrate_reward, calibrate_forward_kl
from cgm.utils import DictLogger, default_parser


@dataclass
class Sample:
    prompt_idx: torch.Tensor
    prompt_lens: torch.Tensor
    tokens: torch.Tensor
    prompts: list[str]

    def extract_chunk(
        self,
        batch_idx: int,
        batch_chunks: int,
        sample_idx: int = None,
        sample_chunks: int = None,
    ) -> "Sample":
        bsz = len(self)
        assert bsz % batch_chunks == 0
        step = bsz // batch_chunks
        lower = batch_idx * step
        upper = lower + step
        return Sample(
            self.prompt_idx[lower:upper],
            self.prompt_lens[lower:upper],
            self.tokens[lower:upper],
            self.prompts[lower:upper],
        )

    def __len__(self) -> int:
        return self.prompt_idx.shape[0]

    def to(self, device: str | torch.device) -> "Sample":
        return Sample(
            self.prompt_idx.to(device),
            self.prompt_lens.to(device),
            self.tokens.to(device),
            self.prompts,
        )


def combine_samples(samples: list[Sample], eos_token_id: int) -> Sample:
    max_tokens = max(s.tokens.shape[1] for s in samples)
    num_samples = sum(len(s.tokens) for s in samples)
    ref_tokens = samples[0].tokens
    combined_tokens = torch.full(
        (num_samples, max_tokens),
        eos_token_id,
        device=ref_tokens.device,
        dtype=ref_tokens.dtype,
    )
    idx = 0
    for sample in samples:
        tokens = sample.tokens
        new_idx = idx + tokens.shape[0]
        combined_tokens[idx:new_idx, : tokens.shape[1]] = tokens
        idx = new_idx

    return Sample(
        prompt_idx=torch.cat([s.prompt_idx for s in samples]),
        prompt_lens=torch.cat([s.prompt_lens for s in samples]),
        tokens=combined_tokens,
        prompts=sum((s.prompts for s in samples), []),
    )


def build_prompt(profession: str) -> str:
    return f'Write a short story (3-5 paragraphs) which only uses very simple words that a 3 year old child would likely understand. ONLY write the story without any additional text. Try to use characters with roughly EQUAL PROBABILITIES male or female. The story begins: "Once upon a time there was a {profession} named".'


def up_to_eos_mask(
    tokens: torch.LongTensor, eos_id: int, prompt_lens: Optional[torch.Tensor] = None
) -> torch.BoolTensor:
    """
    tokens: [B, L] tensor of token IDs
    eos_id: the ID of the EOS token
    returns: [B, L] boolean mask where mask[b, t] == True
             iff t <= first_index_of(eos_id) in tokens[b]
             (or all True if no eos_id is found)
    """
    B, L = tokens.shape
    device = tokens.device

    # 1) where does EOS occur?
    eos_occ = tokens.eq(eos_id)  # [B, L] bool

    # 2) build a [B, L] index matrix  [[0,1,2…],[0,1,2…],…]
    idxs = torch.arange(L, device=device).unsqueeze(0).expand(B, L)

    # 3) for each row, replace non‐EOS positions with L (so they won't be chosen)
    first_pos = torch.where(eos_occ, idxs, L).min(dim=1).values  # [B]
    # if no EOS, min will be L; clamp to L-1 so entire row stays True
    first_pos = torch.clamp(first_pos, max=L - 1)  # [B]

    # 4) build mask: idxs <= first_pos[b] for each batch b
    mask = idxs <= first_pos.unsqueeze(1)  # [B, L]

    if prompt_lens is not None:
        mask &= idxs >= prompt_lens.unsqueeze(1).to(device)

    return mask


class ProfessionStories(Model[Sample]):

    def __init__(
        self,
        professions: list[str] = None,
        lora_config: LoraConfig | None = None,
        device: str | None = "cuda:0",
        max_sample_batch: int = 16,
        prompt_builder: Callable[[str], str] = build_prompt,
    ):
        super().__init__()
        self.professions = professions or [
            "doctor",
            "lawyer",
            "teacher",
            "pilot",
            "chef",
            "scientist",
            "nurse",
            "artist",
        ]
        self.build_prompt = prompt_builder
        model_name = "google/gemma-2-9b-it"
        dtype = torch.bfloat16
        self.max_sample_batch = max_sample_batch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(
            device, dtype=dtype
        )
        self.model.config.use_cache = False  # safer for training / log-prob computation
        self.max_new_tokens = 200
        self.guesser = gender.Detector()
        lora_config = lora_config or LoraConfig(
            r=32,
            lora_alpha=64,
            lora_dropout=0,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        )
        self.model = get_peft_model(self.model, lora_config)

    def make_prompts(
        self, batch_size: int, professions: list[str] = None
    ) -> tuple[torch.Tensor, list[str]]:
        professions = professions or self.professions
        idx = torch.tensor([i % len(professions) for i in range(batch_size)])
        return idx, [self.build_prompt(professions[i]) for i in idx]

    def _sample(self, prompt_idx: torch.Tensor, prompts: list[str]) -> Sample:
        enc = self.tokenizer(prompts, return_tensors="pt")
        prompt_lens = up_to_eos_mask(enc.input_ids, self.tokenizer.eos_token_id).sum(1)
        input_ids = enc["input_ids"].to(self.device)
        tokens = self.model.generate(
            input_ids=input_ids,
            max_new_tokens=self.max_new_tokens,
            do_sample=True,
            temperature=1.0,  # pure softmax sampling
            pad_token_id=self.tokenizer.eos_token_id,
            top_k=None,
        )
        return Sample(
            prompt_idx=prompt_idx,
            prompt_lens=prompt_lens,
            tokens=tokens,
            prompts=prompts,
        )

    def sample(
        self,
        N: int,
        professions: list[str] = None,
        verbose: bool = False,
        max_batch: int = None,
    ) -> Sample:
        max_batch = max_batch or self.max_sample_batch
        prompt_idx, prompts = self.make_prompts(N, professions)
        sections = math.ceil(N / max_batch)
        batched_idx = torch.tensor_split(prompt_idx, sections)
        batched_prompts = [
            list(map(str, arr)) for arr in np.array_split(prompts, sections)
        ]
        assert list(map(len, batched_idx)) == list(
            map(len, batched_prompts)
        )  # check that prompts and prompt_idx line up
        samples = [
            self._sample(idx, prompt)
            for idx, prompt in tqdm(
                zip(batched_idx, batched_prompts),
                desc="sampling batches",
                disable=not verbose,
                total=len(batched_idx),
            )
        ]
        return combine_samples(samples, self.tokenizer.eos_token_id)

    def log_p(
        self,
        x: Sample,
        batch_idx: int = 0,
        batch_chunks: int = 1,
        sample_idx: int = 0,
        sample_chunks: int = 1,
    ) -> torch.Tensor:
        sample = x.extract_chunk(batch_idx=batch_idx, batch_chunks=batch_chunks)
        label_prompt_lens = (sample.prompt_lens - 1).clamp_min(0).to(self.device)
        x_in = sample.tokens[:, :-1]
        labels = sample.tokens[:, 1:]
        up_to_eos = up_to_eos_mask(
            labels, self.tokenizer.eos_token_id, label_prompt_lens
        )
        # with torch.amp.autocast(dtype=torch.bfloat16, device_type="cuda"):  # comment out because increases memory use
        all_log_probs = self.model(x_in).logits.log_softmax(dim=-1)

        log_probs = all_log_probs.gather(index=labels.unsqueeze(-1), dim=-1).squeeze()
        log_probs = torch.where(up_to_eos, log_probs, 0).sum(1)
        return log_probs

    def get_gender(self, text: str, prompt: str) -> torch.Tensor:
        """
        tries all the words after the prompt in the first sentence
        if a gender is detected, stops
        else returns 0 for no gender
        """
        male = torch.tensor([-len(self.professions)], dtype=torch.float32)
        female = torch.tensor([len(self.professions)], dtype=torch.float32)
        neither = torch.zeros(1)
        for title in ["Dr", "Ms", "Mrs", "Mr"]:
            text = text.replace(
                f"{title}.", title
            )  # remove title-based sentence breaks
        split_text = text.split(".")

        sentence_one_idx = len(prompt.split(".")) - 1

        # second generated sentence starts with pronoun very often
        if len(split_text) > sentence_one_idx + 1:
            sentence_two = split_text[sentence_one_idx + 1].strip()
            if sentence_two.startswith("She"):
                return female
            if sentence_two.startswith("He"):
                return male

        # check names in first sentence
        sentence = split_text[sentence_one_idx].strip()
        split_sentence = sentence.split(" ")
        for name_idx in range(len(split_sentence)):
            name_or_title = split_sentence[name_idx]
            if name_or_title.lower() == "mr":
                return male
            if name_or_title.lower() in {"mrs", "miss", "ms"}:
                return female
            gender = self.guesser.get_gender(name_or_title)
            if gender in {"female", "mostly_female"}:
                return female
            if gender in {"male", "mostly_male"}:
                return male
        return neither

    def get_obs(self, sample: Sample, professions: list[str] = None) -> torch.Tensor:
        professions = professions or self.professions
        text_list = self.tokenizer.batch_decode(sample.tokens, skip_special_tokens=True)
        genders = torch.zeros((len(sample.tokens), len(professions)))
        for i, (text, idx) in enumerate(zip(text_list, sample.prompt_idx)):
            genders[i, idx] = self.get_gender(text, sample.prompts[i])
        return genders.to(self.device)

    def get_gender_proportions(
        self,
        n_samples: int = 512,
        professions: list[str] = None,
        verbose: bool = False,
    ):
        samples = self.sample(n_samples, professions, verbose=verbose)
        obs = self.get_obs(samples, professions).cpu()
        for i, profession in enumerate(professions):
            sub_obs = obs[samples.prompt_idx == i, i]
            is_male = (sub_obs < 0).float()
            is_female = (sub_obs > 0).float()
            male_frac = is_male.mean().item()
            female_frac = is_female.mean().item()
            female_lower, _, female_upper = bernoulli_ci_jeffreys(is_female)
            male_lower, _, male_upper = bernoulli_ci_jeffreys(is_male)
            print(
                f"{profession}: female {female_frac:.1%} [{female_lower:.1%} {female_upper:.1%}] - male {male_frac:.1%} [{male_lower:.1%} {male_upper:.1%}]"
            )
        return samples, obs


def bernoulli_ci_jeffreys(
    x: torch.Tensor,
    conf: float = 0.95,
) -> tuple[float, float, float]:
    x_f = x.to(torch.float32)
    k = int(x_f.sum().item())
    n = x_f.numel()
    a, b = k + 0.5, (n - k) + 0.5
    alpha = 1 - conf
    lo = stats.beta.ppf(alpha / 2, a, b)
    hi = stats.beta.ppf(1 - alpha / 2, a, b)
    phat = k / n
    return lo, phat, hi


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
        "--batch_chunks",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--max_sample_batch",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--save_per_profession",
        type=int,
        default=2048,
    )
    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    model = ProfessionStories(max_sample_batch=args.max_sample_batch)
    print("Before finetuning")
    sample_professions = model.professions + [
        "sheriff",
        "judge",
        "accountant",
        "dancer",
        "athlete",
        "baker",
    ]
    N_save = len(sample_professions) * args.save_per_profession

    # fitting
    logger = DictLogger()
    lambd = args.lambd
    if args.calibration_mode == "relax":
        final_model = calibrate_relaxed(
            model,
            h=model.get_obs,
            hstar=torch.zeros(len(model.professions), device=model.device),
            lambd=lambd,
            batch_size=args.batch_size,
            batch_chunks=args.batch_chunks,
            optimizer_params={"lr": args.lr},
            epochs=args.epochs,
            logger=logger,
        )
    elif args.calibration_mode == "reward":
        final_model = calibrate_reward(
            model,
            h=model.get_obs,
            hstar=torch.zeros(len(model.professions), device=model.device),
            batch_size=args.batch_size,
            batch_chunks=args.batch_chunks,
            optimizer_params={"lr": args.lr},
            epochs=args.epochs,
            logger=logger,
            N_samp=4096,
        )
    elif args.calibration_mode == "forward_kl":
        final_model = calibrate_forward_kl(
            model,
            h=model.get_obs,
            hstar=torch.zeros(len(model.professions), device=model.device),
            batch_size=args.batch_size,
            batch_chunks=args.batch_chunks,
            optimizer_params={"lr": args.lr},
            epochs=args.epochs,
            logger=logger,
            N_samp=4096,
        )
    else:
        raise NotImplementedError(f"{args.calibration_mode=} is not a valid mode.")

    name = f"cgm_{args.calibration_mode}_seed={args.seed}"
    if args.calibration_mode == "relax":
        name = f"{name}_{lambd=}"
    if args.model_save_folder:
        os.makedirs(args.model_save_folder, exist_ok=True)
        model_save_path = os.path.join(args.model_save_folder, f"{name}.pt")
        print(f"Saved model to {model_save_path}")
        torch.save(final_model.state_dict(), model_save_path)

    print("After finetuning")
    print("Post-sampling")
    post_samples, post_obs = final_model.get_gender_proportions(
        N_save, professions=sample_professions, verbose=True
    )
    os.makedirs(args.result_save_folder, exist_ok=True)
    torch.save(
        {
            "post_samples": post_samples,
            "post_obs": post_obs,
        },
        os.path.join(args.result_save_folder, f"{name}_samples.pt"),
    )

    sns.set_style("whitegrid")
    fig, (ax, ax2) = plt.subplots(ncols=2, dpi=150, figsize=(8, 4))
    colors = plt.get_cmap("tab10").colors
    hs = torch.stack(logger.metrics["h_bar"])
    for i in range(hs.shape[1]):
        ax.plot(hs[:, i], color=colors[i], label=model.professions[i])

    ax.set_xlabel("step")
    ax.set_ylabel("observable")
    ax.legend(ncol=2)

    ax2.plot(logger.metrics["kl_loss"])
    ax2.set_xlabel("step")
    ax2.set_ylabel("KL")

    sns.despine(fig)
    fig.tight_layout()
    fig_save_path = os.path.join(args.result_save_folder, f"{name}_training_curve.png")
    fig.savefig(fig_save_path, bbox_inches="tight")
    h_df = pd.DataFrame(hs, columns=model.professions)
    h_df["kl"] = logger.metrics["kl_loss"]
    h_df["loss"] = logger.metrics["loss"]
    h_df.to_csv(
        os.path.join(args.result_save_folder, f"{name}_metrics.csv"), index=False
    )
