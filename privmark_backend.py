#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Local, commit-pinned causal-language-model research backend.

Requires Python 3.11+, transformers>=4.51,<5, and torch>=2.6. Heavy model
libraries are imported only when constructing ``LocalModel``. Hub models must
have a native Transformers implementation and safetensors weights: neither
remote Python code nor pickle weight fallback is enabled. Online resolution
pins all artifacts to one Hub commit; offline resolution requires an existing
cached configuration and all artifacts for that same snapshot.

Example:
    backend = LocalModel("Qwen/Qwen3-0.6B")
    try:
        result = backend.generate([{"role": "user", "content": "Say hello."}])
        print(result["text"])
    finally:
        backend.close()

``memorize`` is an optional, destructive, in-memory experiment to run AFTER
context probes. Its labels describe only this controlled fine-tuning split,
not pretraining membership, differential privacy, or production privacy risk.
No weights or records are saved by this module (Hub artifacts may be cached).
Instances are not thread-safe. Greedy decoding and seeded training do not
promise bitwise reproducibility across hardware or library versions.
"""

from __future__ import annotations

import gc
import hashlib
import math
import random
import re
import time
from dataclasses import dataclass
from typing import Any

__all__ = ["LocalModel"]

_COMMIT_PATTERN = re.compile(r"[0-9a-fA-F]{40}")
_SCOPE = "controlled local fine-tuning only, not pretraining membership"
_TOKENIZATION = (
    "Encode prefix and leading-space target independently with "
    "add_special_tokens=False, then concatenate token IDs. This explicit "
    "boundary prevents merged prefix/target tokens; it can differ from "
    "single-pass tokenization. Score only target tokens with causal shift."
)


@dataclass(frozen=True)
class _Record:
    """Validated synthetic record and its explicit prefix/target token boundary."""

    record_id: str | int
    secret: str
    prefix: str
    prefix_ids: tuple[int, ...]
    token_ids: tuple[int, ...]


def _positive_integer(value: int, name: str) -> None:
    """Reject booleans, fractional values, and nonpositive token/step budgets."""
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")


def _validate_seed(seed: int) -> None:
    """Use a portable nonnegative seed range for PyTorch and Python RNGs."""
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32).")


class LocalModel:
    """Own a single unquantized causal LM on CUDA, MPS, or CPU.

    Args:
        model_id: Hugging Face Hub repository ID, not a local model directory.
        revision: Branch, tag, or commit resolved once to an immutable SHA.
        device: ``auto``, ``cpu``, ``mps``, ``cuda``, or ``cuda:N``. Auto prefers
            CUDA, then MPS, then CPU; explicit unavailable devices fail.
        max_new_tokens: Positive continuation budget, never silently truncated.
        seed: Nonnegative 32-bit PyTorch seed. Construction seeds PyTorch's RNG.

    Attributes:
        metadata: JSON-compatible provenance, loading, and decoding information.
            ``dtype`` is updated if memorization converts weights to float32.

    Raises:
        ValueError: Invalid arguments, unsupported context configuration, or a
            prompt plus continuation budget exceeding the context window.
        RuntimeError: Unavailable device, closed instance, or invalid experiment
            state. Hub/model/tokenizer errors propagate to the caller.

    Notes:
        CPU inference uses float32; accelerators use float16. Full-model AdamW
        fine-tuning converts to float32 BEFORE baseline measurement and leaves
        weights in float32 afterward. This can require much more memory than
        inference. Call ``close`` in a ``finally`` block; no implicit saving or
        automatic retry with less secure model formats occurs.
    """

    def __init__(
        self,
        model_id: str,
        revision: str = "main",
        device: str = "auto",
        max_new_tokens: int = 48,
        seed: int = 42,
    ) -> None:
        if not isinstance(model_id, str) or not model_id.strip():
            raise ValueError("model_id must be a nonempty Hub repository ID.")
        if not isinstance(revision, str) or not revision.strip():
            raise ValueError("revision must be nonempty.")
        _positive_integer(max_new_tokens, "max_new_tokens")
        _validate_seed(seed)
        if not isinstance(device, str):
            raise ValueError("device must be a string.")

        started = time.perf_counter()
        import torch
        from transformers import (
            AutoConfig,
            AutoModelForCausalLM,
            AutoTokenizer,
            GenerationConfig,
        )

        self._torch: Any = torch
        self._generation_config_class: Any = GenerationConfig
        self._closed = False
        self._memorization_started = False
        self.max_new_tokens = max_new_tokens
        self.seed = seed
        self.device: Any = self._select_device(device)
        torch.manual_seed(seed)
        resolved_revision, local_files_only = self._resolve_revision(model_id, revision)
        shared: dict[str, Any] = {
            "revision": resolved_revision,
            "trust_remote_code": False,
            "local_files_only": local_files_only,
        }
        config: Any = AutoConfig.from_pretrained(model_id, **shared)
        if getattr(config, "is_encoder_decoder", False):
            raise ValueError("Only decoder-only causal language models are supported.")
        config_commit = getattr(config, "_commit_hash", None)
        if config_commit is not None and config_commit != resolved_revision:
            raise RuntimeError("Loaded configuration does not match the resolved commit.")
        self.tokenizer: Any = AutoTokenizer.from_pretrained(
            model_id, use_fast=True, **shared
        )
        self.context_window = self._context_window(config)
        if max_new_tokens >= self.context_window:
            raise ValueError("max_new_tokens leaves no room for a prompt in the context window.")

        dtype = torch.float32 if self.device.type == "cpu" else torch.float16
        self.model: Any = AutoModelForCausalLM.from_pretrained(
            model_id,
            config=config,
            use_safetensors=True,
            torch_dtype=dtype,
            attn_implementation="eager",
            **shared,
        )
        self.model.to(device=self.device, dtype=dtype)
        self.model.eval()
        self._eos_token_ids = self._read_eos_ids()
        self._decoding: Any = self._make_decoding_config()
        self._synchronize()
        self.metadata: dict[str, Any] = {
            "model_id": model_id,
            "requested_revision": revision,
            "resolved_revision": resolved_revision,
            "device": str(self.device),
            "dtype": str(next(self.model.parameters()).dtype).removeprefix("torch."),
            "parameter_count": sum(p.numel() for p in self.model.parameters()),
            "load_seconds": time.perf_counter() - started,
            "decoding": (
                "Greedy, do_sample=False, num_beams=1, num_return_sequences=1; "
                "fresh GenerationConfig, temperature=1, top_p=1, top_k=0, "
                "repetition_penalty=1, no forced tokens or inherited sampling "
                "constraints; EOS or max_new_tokens stopping; eval/inference_mode."
            ),
            "max_new_tokens": max_new_tokens,
            "context_window": self.context_window,
            "seed": seed,
            "local_files_only": local_files_only,
            "attention_implementation": "eager",
        }

    @staticmethod
    def _resolve_revision(model_id: str, revision: str) -> tuple[str, bool]:
        """Resolve the Hub SHA, falling back only to an already cached config.

        Offline flags and connection/time-out failures permit cache resolution.
        Authentication, missing revisions, and other HTTP failures propagate;
        they must not silently select potentially stale cached artifacts.
        """
        from huggingface_hub import model_info
        from huggingface_hub.errors import OfflineModeIsEnabled
        from requests.exceptions import ConnectionError as HubConnectionError
        from requests.exceptions import Timeout
        from transformers.utils import is_offline_mode
        from transformers.utils.hub import cached_file, extract_commit_hash

        offline = is_offline_mode()
        sha: str | None = None
        if not offline:
            try:
                sha = model_info(model_id, revision=revision, timeout=15).sha
            except (OfflineModeIsEnabled, HubConnectionError, Timeout):
                offline = True
        if offline:
            config_path = cached_file(
                model_id,
                "config.json",
                revision=revision,
                local_files_only=True,
            )
            if config_path is None:
                raise RuntimeError("No cached configuration is available for offline resolution.")
            sha = extract_commit_hash(config_path, None)
        if not isinstance(sha, str) or _COMMIT_PATTERN.fullmatch(sha) is None:
            raise RuntimeError("Could not resolve an immutable Hub commit SHA.")
        return sha, offline

    def _select_device(self, requested: str) -> Any:
        """Resolve one device without silently falling back on explicit requests."""
        torch = self._torch
        if requested == "auto":
            if torch.cuda.is_available():
                requested = "cuda"
            elif torch.backends.mps.is_available():
                requested = "mps"
            else:
                requested = "cpu"
        if re.fullmatch(r"cpu|mps|cuda(?::[0-9]+)?", requested) is None:
            raise ValueError("device must be auto, cpu, mps, cuda, or cuda:N.")
        if requested.startswith("cuda"):
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA was requested but is unavailable.")
            selected = torch.device(requested)
            index = torch.cuda.current_device() if selected.index is None else selected.index
            if index >= torch.cuda.device_count():
                raise ValueError("Requested CUDA device index does not exist.")
            return torch.device("cuda", index)
        if requested == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is unavailable.")
        return torch.device(requested)

    def _context_window(self, config: Any) -> int:
        """Use the most conservative finite, advertised positional/token limit.

        Tokenizer sentinel values are ignored. RoPE scaling factors are not
        multiplied speculatively; a correctly expanded configuration should
        advertise its supported length in max_position_embeddings.
        """
        text_config = config.get_text_config() if hasattr(config, "get_text_config") else config
        limits: list[int] = []
        for source in (config, text_config):
            for name in ("max_position_embeddings", "n_positions", "max_sequence_length", "seq_length"):
                value = getattr(source, name, None)
                if isinstance(value, int) and not isinstance(value, bool) and 0 < value < 10**9:
                    limits.append(value)
        if not limits:
            raise ValueError("Model config does not advertise a supported context window.")
        tokenizer_limit = getattr(self.tokenizer, "model_max_length", None)
        if isinstance(tokenizer_limit, int) and 0 < tokenizer_limit < 10**9:
            limits.append(tokenizer_limit)
        return min(limits)

    def _read_eos_ids(self) -> tuple[int, ...]:
        """Preserve model-specific EOS lists without inheriting decoding policy."""
        for source in (self.model.generation_config, self.model.config, self.tokenizer):
            value = getattr(source, "eos_token_id", None)
            if value is not None:
                values = [value] if isinstance(value, int) else list(value)
                if any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in values):
                    raise ValueError("Model EOS IDs must be nonnegative integers.")
                return tuple(values)
        return ()

    def _make_decoding_config(self) -> Any:
        """Build fresh greedy settings, retaining only special-token identities."""
        source = self.model.generation_config
        pad_id = getattr(source, "pad_token_id", None)
        if pad_id is None:
            pad_id = self.tokenizer.pad_token_id
        if pad_id is None and self._eos_token_ids:
            pad_id = self._eos_token_ids[0]
        bos_id = getattr(source, "bos_token_id", None)
        if bos_id is None:
            bos_id = self.tokenizer.bos_token_id
        return self._generation_config_class(
            max_new_tokens=self.max_new_tokens,
            min_length=0,
            min_new_tokens=0,
            do_sample=False,
            num_beams=1,
            num_beam_groups=1,
            num_return_sequences=1,
            temperature=1.0,
            top_p=1.0,
            top_k=0,
            typical_p=1.0,
            min_p=None,
            epsilon_cutoff=0.0,
            eta_cutoff=0.0,
            repetition_penalty=1.0,
            no_repeat_ngram_size=0,
            bad_words_ids=None,
            forced_bos_token_id=None,
            forced_eos_token_id=None,
            suppress_tokens=None,
            begin_suppress_tokens=None,
            eos_token_id=list(self._eos_token_ids) or None,
            pad_token_id=pad_id,
            bos_token_id=bos_id,
            use_cache=True,
            return_dict_in_generate=False,
            output_scores=False,
        )

    def _require_open(self) -> None:
        """Reject use after explicit resource release."""
        if self._closed:
            raise RuntimeError("LocalModel is closed.")

    def _synchronize(self) -> None:
        """Wait for accelerator work so wall-clock intervals include computation."""
        if self.device.type == "cuda":
            self._torch.cuda.synchronize(self.device)
        elif self.device.type == "mps":
            self._torch.mps.synchronize()

    def _empty_cache(self) -> None:
        """Release unused allocator blocks on this instance's accelerator."""
        if self.device.type == "cuda":
            with self._torch.cuda.device(self.device):
                self._torch.cuda.empty_cache()
        elif self.device.type == "mps":
            self._torch.mps.empty_cache()

    def _encode(self, text: str) -> tuple[int, ...]:
        """Encode literal text without adding special tokens or truncating."""
        return tuple(self.tokenizer.encode(text, add_special_tokens=False, truncation=False))

    def _check_length(self, prompt_length: int, continuation_length: int) -> None:
        """Reject overlong inputs rather than changing the benchmark prompt."""
        if prompt_length <= 0:
            raise ValueError("The encoded prompt must contain at least one token.")
        if prompt_length + continuation_length > self.context_window:
            raise ValueError(
                f"Prompt ({prompt_length}) + continuation ({continuation_length}) "
                f"exceeds the configured context window ({self.context_window}); "
                "no truncation was performed."
            )

    def _generate_prompt(self, prompt: str, token_ids: tuple[int, ...]) -> dict[str, Any]:
        """Generate from exact prompt IDs; time generation, not rendering/encoding."""
        self._require_open()
        self._check_length(len(token_ids), self.max_new_tokens)
        torch = self._torch
        input_ids = torch.tensor([token_ids], dtype=torch.long, device=self.device)
        attention_mask = torch.ones_like(input_ids)
        self.model.eval()
        self._synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            output = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                generation_config=self._decoding,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                num_beams=1,
                temperature=1.0,
                top_p=1.0,
                top_k=0,
            )
        self._synchronize()
        latency = time.perf_counter() - started
        continuation: list[int] = output[0, len(token_ids):].tolist()
        ended_with_eos = any(token in self._eos_token_ids for token in continuation)
        return {
            "text": self.tokenizer.decode(
                continuation, skip_special_tokens=True, clean_up_tokenization_spaces=False
            ),
            "latency_s": latency,
            "new_tokens": len(continuation),
            "truncated": len(continuation) >= self.max_new_tokens and not ended_with_eos,
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        }

    def generate(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        """Return only the new continuation of a non-thinking chat prompt.

        Args:
            messages: Nonempty list of role/content string dictionaries.

        Returns:
            ``text``, accelerator-synchronized generation ``latency_s``, actual
            ``new_tokens`` (including EOS), budget-without-EOS ``truncated``, and
            the SHA256 of the exact UTF-8 rendered prompt. Chat rendering and
            tokenization are outside the timed interval.

        Raises:
            ValueError: Malformed messages or an overlong prompt. A missing or
                unsupported tokenizer chat template also fails rather than
                silently substituting a different benchmark prompt.
            RuntimeError: The instance is closed.
        """
        self._require_open()
        if not isinstance(messages, list) or not messages:
            raise ValueError("messages must be a nonempty list.")
        for message in messages:
            if (
                not isinstance(message, dict)
                or not isinstance(message.get("role"), str)
                or not message["role"].strip()
                or not isinstance(message.get("content"), str)
            ):
                raise ValueError("Each message must have a nonempty role and string content.")
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        if not isinstance(prompt, str):
            raise ValueError("The chat template did not render a string.")
        return self._generate_prompt(prompt, self._encode(prompt))

    def _prepare_records(self, records: list[dict[str, Any]]) -> list[_Record]:
        """Validate independent records and explicitly construct target boundaries.

        Prefix and target are encoded separately, then concatenated. No joint
        tokenizer call can merge a prefix token into a target token, so every
        target token is scored and the prefix is fully masked. Inference uses
        exactly the same prefix IDs. This also works with slow tokenizers.
        The leading space in the target is intentional. No BOS/EOS is injected.
        """
        if not isinstance(records, list) or len(records) < 4:
            raise ValueError("memorize requires at least four synthetic records.")
        prepared: list[_Record] = []
        seen_ids: set[str] = set()
        seen_secrets: set[str] = set()
        for record in records:
            if not isinstance(record, dict):
                raise ValueError("Each record must be a dictionary.")
            record_id = record.get("record_id")
            secret = record.get("secret")
            if (
                isinstance(record_id, bool)
                or not isinstance(record_id, (str, int))
                or not str(record_id).strip()
            ):
                raise ValueError("record_id must be a nonempty string or an integer.")
            if not isinstance(secret, str) or not secret.strip():
                raise ValueError("Each secret must be a nonempty string.")
            if str(record_id) in seen_ids or secret in seen_secrets:
                raise ValueError("record_id values and secrets must each be unique.")
            seen_ids.add(str(record_id))
            seen_secrets.add(secret)
            prefix = f"Synthetic record {record_id}. Private token:"
            prefix_ids = self._encode(prefix)
            target_ids = self._encode(f" {secret}")
            if not target_ids:
                raise ValueError("A secret produced an empty target token sequence.")
            self._check_length(len(prefix_ids), len(target_ids))
            self._check_length(len(prefix_ids), self.max_new_tokens)
            prepared.append(_Record(record_id, secret, prefix, prefix_ids, prefix_ids + target_ids))
        return prepared

    def _target_loss(self, record: _Record) -> Any:
        """Compute mean target-only NLL in nats, with an explicit causal shift.

        The logit at prefix_length - 1 predicts the first target token. Prefix
        labels are -100 and ignored; target labels run through the final token.
        This is identical during no-gradient measurement and optimization.
        """
        torch = self._torch
        input_ids = torch.tensor([record.token_ids], dtype=torch.long, device=self.device)
        labels = torch.full_like(input_ids, -100)
        labels[:, len(record.prefix_ids):] = input_ids[:, len(record.prefix_ids):]
        output = self.model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            use_cache=False,
            return_dict=True,
        )
        # Select only target prediction positions, avoiding a vocabulary-sized
        # float32 copy for prefix positions that cannot contribute to the loss.
        first_prediction = len(record.prefix_ids) - 1
        logits = output.logits[:, first_prediction:-1, :].float()
        shifted_labels = labels[:, first_prediction + 1:]
        return torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            shifted_labels.reshape(-1),
            ignore_index=-100,
            reduction="mean",
        )

    def _measure_record(self, record: _Record, contains_secret: Any) -> dict[str, Any]:
        """Measure actual target loss and independent free prefix completion."""
        self.model.eval()
        with self._torch.inference_mode():
            nll = float(self._target_loss(record).item())
        if not math.isfinite(nll):
            raise FloatingPointError("Measured target NLL is not finite.")
        generated = self._generate_prompt(record.prefix, record.prefix_ids)
        text = generated["text"]
        return {
            "nll": nll,
            "text": text,
            "extracted": bool(contains_secret(text, record.secret)),
        }

    def _train_step(self, record: _Record, optimizer: Any) -> None:
        """Perform one full-model, batch-one, target-only AdamW update."""
        optimizer.zero_grad(set_to_none=True)
        loss = self._target_loss(record)
        if not bool(self._torch.isfinite(loss).item()):
            raise FloatingPointError("Training target loss is not finite.")
        loss.backward()
        self._torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), max_norm=1.0, error_if_nonfinite=True
        )
        optimizer.step()

    def memorize(
        self,
        records: list[dict[str, Any]],
        steps: int,
        learning_rate: float = 5e-5,
        seed: int = 42,
    ) -> dict[str, Any]:
        """Measure a randomized controlled memorization experiment in memory.

        Args:
            records: At least four dictionaries containing unique ``record_id``
                and unique, nonempty synthetic ``secret`` values. ``public_value``
                is ignored. Do not provide real private data.
            steps: Number of batch-one updates, at least the member count.
            learning_rate: Finite positive full-model AdamW learning rate.
            seed: Split, epoch-shuffling, and PyTorch training seed.

        Returns:
            Measured before/after NLLs, free greedy continuations, extraction
            flags, and the recorded member split, with the explicit limited
            research scope. Training time is accelerator-synchronized and
            excludes before/after measurements. No AUC is fabricated here.

        Raises:
            ValueError: Invalid records, hyperparameters, or sequence lengths.
            RuntimeError: Closed instance or a repeated experiment on already
                modified weights. A failed training attempt also blocks retries.
            FloatingPointError: A measured or training loss is nonfinite.
            ImportError: The caller has not provided privmark_metrics.contains_secret.

        Notes:
            Run AFTER all context probes. Before any training, a seeded shuffle
            assigns floor(N/2) members and the remaining nonmembers. Every
            shuffled member is visited once per cycle before another cycle
            begins. Only members' target tokens contribute loss, although all
            model parameters are trainable. AdamW uses zero weight decay and
            gradient norm clipping at 1.0. Training seeds PyTorch's global RNG.

            Both baselines and final measurements use float32/eval mode and
            literal prefixes without chat templates. Explicit separate prefix
            and target encoding prevents ambiguous boundary-token masking; see
            ``metadata['memorization_target_tokenization']``. NLL is in nats per
            target token. Extraction uses the shared max_new_tokens budget,
            so long secrets may not fit in free completions. Updates cannot be
            rolled back after failure; reload a fresh instance to repeat.
        """
        self._require_open()
        if self._memorization_started:
            raise RuntimeError("Reload a fresh model before another memorization experiment.")
        _positive_integer(steps, "steps")
        _validate_seed(seed)
        if (
            isinstance(learning_rate, bool)
            or not isinstance(learning_rate, (int, float))
            or not math.isfinite(learning_rate)
            or learning_rate <= 0
        ):
            raise ValueError("learning_rate must be finite and positive.")
        prepared = self._prepare_records(records)
        member_count = len(prepared) // 2
        if steps < member_count:
            raise ValueError("steps must be at least member_count so every member is trained.")

        from privmark_metrics import contains_secret

        rng = random.Random(seed)
        shuffled_indices = list(range(len(prepared)))
        rng.shuffle(shuffled_indices)
        member_indices = shuffled_indices[:member_count]
        member_set = set(member_indices)
        # Keep baseline and final scoring precision identical: otherwise a dtype
        # change could be mistaken for a fine-tuning effect.
        self.model.to(dtype=self._torch.float32)
        self.model.eval()
        self.metadata["dtype"] = "float32"
        self.metadata["memorization_target_tokenization"] = _TOKENIZATION
        before = [self._measure_record(record, contains_secret) for record in prepared]

        self._memorization_started = True
        self.metadata["weights_modified_by_memorization"] = True
        optimizer: Any = None
        self._synchronize()
        started = time.perf_counter()
        try:
            self._torch.manual_seed(seed)
            self.model.train()
            self.model.requires_grad_(True)
            optimizer = self._torch.optim.AdamW(
                self.model.parameters(), lr=float(learning_rate), weight_decay=0.0
            )
            with self._torch.enable_grad():
                for step in range(steps):
                    offset = step % member_count
                    if offset == 0:
                        rng.shuffle(member_indices)
                    self._train_step(prepared[member_indices[offset]], optimizer)
            self._synchronize()
            training_seconds = time.perf_counter() - started
        finally:
            # Cleanup also runs on interruption/OOM/numerical failure. Errors
            # propagate; no partial measurements are reported as successful.
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                optimizer.state.clear()
            del optimizer
            self.model.zero_grad(set_to_none=True)
            self.model.eval()
            self._empty_cache()

        measured: list[dict[str, Any]] = []
        for index, record in enumerate(prepared):
            after = self._measure_record(record, contains_secret)
            measured.append({
                "record_id": record.record_id,
                "member": index in member_set,
                "nll_before": before[index]["nll"],
                "nll_after": after["nll"],
                "text_before": before[index]["text"],
                "text_after": after["text"],
                "extracted_before": before[index]["extracted"],
                "extracted_after": after["extracted"],
            })
        return {
            "status": "measured",
            "scope": _SCOPE,
            "steps": steps,
            "learning_rate": float(learning_rate),
            "member_count": member_count,
            "nonmember_count": len(prepared) - member_count,
            "records": measured,
            "training_seconds": training_seconds,
            "seed": seed,
            "target_tokenization": _TOKENIZATION,
            "weight_decay": 0.0,
            "gradient_clip_norm": 1.0,
        }

    def close(self) -> None:
        """Idempotently delete owned model/tokenizer references and empty caches.

        Metadata remains readable. Memory held by external references to the
        model or tokenizer cannot be reclaimed by this method.
        """
        if self._closed:
            return
        self._closed = True
        if hasattr(self, "model"):
            del self.model
        if hasattr(self, "tokenizer"):
            del self.tokenizer
        gc.collect()
        self._empty_cache()