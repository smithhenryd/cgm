from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from cgm.model import Model
from cgm import utils


@dataclass
class G2PTSample:
    token_ids: torch.Tensor  # [N, T] right-padded sequences, includes leading <boc>

    def __len__(self) -> int:
        return self.token_ids.shape[0]

    def extract_chunk(self, batch_idx: int, batch_chunks: int) -> "G2PTSample":
        lo, hi = utils.chunk_bounds(len(self), batch_chunks, batch_idx)
        return G2PTSample(token_ids=self.token_ids[lo:hi])


def _up_to_eos_mask(labels: torch.Tensor, eos_id: int) -> torch.BoolTensor:
    """
    [B, L] bool mask that is True at every position up to and including the
    first eos_id in each row, and False at all padding positions after it.
    If a row contains no eos_id the entire row is True (truncated sequence).

    Applied to `labels = tokens[:, 1:]`, this correctly includes the <eog>
    token in the log-prob sum and excludes all trailing pad tokens.
    """
    B, L = labels.shape
    device = labels.device
    idxs = torch.arange(L, device=device).unsqueeze(0).expand(B, L)  # [B, L]
    # replace non-EOS positions with L so they lose the argmin
    first_eos = torch.where(labels.eq(eos_id), idxs, L).min(dim=1).values  # [B]
    first_eos = first_eos.clamp(max=L - 1)
    return idxs <= first_eos.unsqueeze(1)  # [B, L]


class G2PTModel(Model[G2PTSample]):
    """
    CGM wrapper around a pretrained G2PT HuggingFace model.

    sample() generates token sequences starting from <boc>.
    log_p() computes the sum of per-token log-probabilities via a
    teacher-forced forward pass, masking out padding after <eog>.
    """

    def __init__(
        self,
        model_name: str = "xchen16/g2pt-moses-small-bfs",
        device: Optional[str | torch.device] = None,
        use_bf16: bool = False,
    ):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.hf_model = AutoModelForCausalLM.from_pretrained(model_name)
        self.hf_model.config.use_cache = False  # required for training / log-prob
        # The tokenizer does not set eos_token_id; <eog> is the natural end token.
        self.eog_id: int = self.tokenizer.convert_tokens_to_ids("<eog>")
        if device is not None:
            self.hf_model.to(device)
        self.use_bf16 = use_bf16 and self.device.type == "cuda"

    def sample(self, N: int) -> G2PTSample:
        enc = self.tokenizer(["<boc>"] * N, return_tensors="pt")
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)
        with torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=self.use_bf16
        ):
            token_ids = self.hf_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=self.tokenizer.model_max_length,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.eog_id,
                do_sample=True,
                temperature=1.0,
            )
        return G2PTSample(token_ids=token_ids)

    def _teacher_forced_log_probs(
        self, x: G2PTSample, batch_idx: int = 0, batch_chunks: int = 1
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = x.extract_chunk(batch_idx, batch_chunks).token_ids.to(self.device)
        x_in = tokens[:, :-1]
        labels = tokens[:, 1:]
        attn_mask = (x_in != self.tokenizer.pad_token_id).long()
        with torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=self.use_bf16
        ):
            logits = self.hf_model(input_ids=x_in, attention_mask=attn_mask).logits
        log_probs = F.log_softmax(logits.float(), dim=-1)
        mask = _up_to_eos_mask(labels, self.eog_id)
        return log_probs, mask

    def exact_kl_to(
        self,
        base_model: "G2PTModel",
        x: G2PTSample,
        batch_idx: int = 0,
        batch_chunks: int = 1,
    ) -> torch.Tensor:
        log_probs, mask = self._teacher_forced_log_probs(
            x, batch_idx=batch_idx, batch_chunks=batch_chunks
        )
        base_log_probs, _ = base_model._teacher_forced_log_probs(
            x, batch_idx=batch_idx, batch_chunks=batch_chunks
        )
        per_token_kl = (
            log_probs.exp() * (log_probs - base_log_probs.to(log_probs.device))
        ).sum(dim=-1)
        return torch.where(mask, per_token_kl, 0.0).sum(dim=-1)

    def log_p(
        self,
        x: G2PTSample,
        batch_idx: int = 0,
        batch_chunks: int = 1,
        sample_idx: int = 0,
        sample_chunks: int = 1,
    ) -> torch.Tensor:
        tokens = x.extract_chunk(batch_idx, batch_chunks).token_ids.to(self.device)
        labels = tokens[:, 1:]
        log_probs, mask = self._teacher_forced_log_probs(
            x, batch_idx=batch_idx, batch_chunks=batch_chunks
        )
        per_token_lp = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(
            -1
        )  # [B, T-1]; note: squeeze(-1) not squeeze() to preserve batch dim
        return torch.where(mask, per_token_lp, 0.0).sum(dim=-1)  # [B]
