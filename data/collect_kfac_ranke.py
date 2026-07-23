#!/usr/bin/env python
"""Collect K-FAC factors for Ranke Qwen3 MLP projections on Gutenberg."""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, List

import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader, IterableDataset
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_ID = "history-llms/ranke-4b-1913-dpo-2007"
MODEL_REVISION = "4b83606f3e23a8b46c5ef37853eb1d25892ac229"
GUTENBERG_ID = "manu/project_gutenberg"
GUTENBERG_REVISION = "164853d214065df26a630ee1ab91a0c39e461caf"
EXPECTED_HIDDEN_SIZE = 2560
EXPECTED_INTERMEDIATE_SIZE = 9728
EXPECTED_LAYERS = 36
PROJECTION_NAMES = ("gate", "up", "down")
DEFAULT_TARGET_BLOCKS = [20, 24, 28, 32, 35]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--revision", default=MODEL_REVISION)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype", default="bfloat16", choices=["float32", "bfloat16"]
    )
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--n_tokens", type=int, default=32_768)
    parser.add_argument(
        "--target_blocks", type=int, nargs="+", default=DEFAULT_TARGET_BLOCKS
    )
    parser.add_argument(
        "--projections",
        nargs="+",
        choices=PROJECTION_NAMES,
        default=list(PROJECTION_NAMES),
    )
    parser.add_argument(
        "--save_dir",
        type=Path,
        default=Path("assets/kfac_factors/ranke_4b_1913"),
    )
    parser.add_argument("--seed", type=int, default=1913)
    parser.add_argument("--sample_labels", action="store_true")
    parser.add_argument("--char_offset", type=int, default=20_000)
    return parser.parse_args()


def book_bucket(book_id: str) -> int:
    digest = hashlib.sha256(str(book_id).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % 10


class GutenbergKFACStream(IterableDataset):
    """Yield untouched token windows from ``text[char_offset:]``."""

    def __init__(
        self,
        tokenizer,
        seq_len: int,
        n_tokens: int,
        char_offset: int,
    ):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.n_tokens = n_tokens
        self.char_offset = char_offset

    def _rows(self):
        return load_dataset(
            GUTENBERG_ID,
            split="en",
            revision=GUTENBERG_REVISION,
            streaming=True,
        )

    def __iter__(self):
        emitted = 0
        seen_book_ids = set()
        for row in self._rows():
            book_id = str(row["id"])
            if book_bucket(book_id) > 7 or book_id in seen_book_ids:
                continue
            seen_book_ids.add(book_id)

            text = row["text"][self.char_offset :]
            token_ids = self.tokenizer(
                text, add_special_tokens=False, return_attention_mask=False
            ).input_ids
            if len(token_ids) < self.seq_len:
                continue
            if emitted + self.seq_len > self.n_tokens:
                return
            yield {
                "input_ids": token_ids[: self.seq_len],
                "book_id": book_id,
                "token_offset": 0,
            }
            emitted += self.seq_len
            if emitted >= self.n_tokens:
                return


class KFACCollector:
    """Collect A = E[x x^T] and G = E[g g^T] for one Linear module."""

    def __init__(self, layer: torch.nn.Linear):
        d_out, d_in = layer.weight.shape
        device = layer.weight.device
        self.A = torch.zeros(d_in, d_in, dtype=torch.float32, device=device)
        self.G = torch.zeros(d_out, d_out, dtype=torch.float32, device=device)
        self.n_tokens = 0
        self._input = None
        self._forward_handle = layer.register_forward_pre_hook(self._forward)
        self._backward_handle = layer.register_full_backward_hook(self._backward)

    def _forward(self, _module, inputs):
        if torch.is_grad_enabled():
            x = inputs[0][:, :-1].detach()
            self._input = x.reshape(-1, x.shape[-1]).float()

    def _backward(self, _module, _grad_input, grad_output):
        if self._input is None or grad_output[0] is None:
            return
        g = grad_output[0][:, :-1].detach()
        g = g.reshape(-1, g.shape[-1]).float()
        self.A.addmm_(self._input.T, self._input)
        self.G.addmm_(g.T, g)
        self.n_tokens += g.shape[0]
        self._input = None

    def factors(self):
        if self.n_tokens == 0:
            raise RuntimeError("K-FAC collector received no tokens")
        return self.A / self.n_tokens, self.G / self.n_tokens

    def close(self):
        self._forward_handle.remove()
        self._backward_handle.remove()
        self._input = None


def collate_windows(rows):
    return {
        "input_ids": torch.tensor(
            [row["input_ids"] for row in rows], dtype=torch.long
        ),
        "book_id": [row["book_id"] for row in rows],
        "token_offset": [row["token_offset"] for row in rows],
    }


def validate_model(model, target_blocks: Iterable[int]):
    config = model.config
    expected = {
        "model_type": "qwen3",
        "num_hidden_layers": EXPECTED_LAYERS,
        "hidden_size": EXPECTED_HIDDEN_SIZE,
        "intermediate_size": EXPECTED_INTERMEDIATE_SIZE,
    }
    for field, value in expected.items():
        actual = getattr(config, field, None)
        if actual != value:
            raise ValueError(f"Expected {field}={value}, got {actual}")

    for block in target_blocks:
        if block < 0 or block >= EXPECTED_LAYERS:
            raise ValueError(f"Target block {block} is outside [0, 35]")
        mlp = model.model.layers[block].mlp
        shapes = {
            "gate": tuple(mlp.gate_proj.weight.shape),
            "up": tuple(mlp.up_proj.weight.shape),
            "down": tuple(mlp.down_proj.weight.shape),
        }
        expected_shapes = {
            "gate": (EXPECTED_INTERMEDIATE_SIZE, EXPECTED_HIDDEN_SIZE),
            "up": (EXPECTED_INTERMEDIATE_SIZE, EXPECTED_HIDDEN_SIZE),
            "down": (EXPECTED_HIDDEN_SIZE, EXPECTED_INTERMEDIATE_SIZE),
        }
        if shapes != expected_shapes:
            raise ValueError(f"Unexpected MLP shapes for block {block}: {shapes}")


def factor_diagnostics(tensor: torch.Tensor):
    denominator = max(float(torch.linalg.vector_norm(tensor)), 1e-30)
    symmetry_error = float(
        torch.linalg.vector_norm(tensor - tensor.T) / denominator
    )
    return {
        "shape": list(tensor.shape),
        "trace": float(torch.trace(tensor)),
        "finite": bool(torch.isfinite(tensor).all()),
        "symmetry_error": symmetry_error,
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def projection_modules(model, block: int, names: Iterable[str]):
    mlp = model.model.layers[block].mlp
    return {
        f"blk{block}.{name}": getattr(mlp, f"{name}_proj") for name in names
    }


def collect_block(args, model, tokenizer, block: int):
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    targets = projection_modules(model, block, args.projections)
    collectors: Dict[str, KFACCollector] = {
        name: KFACCollector(layer) for name, layer in targets.items()
    }
    stream = GutenbergKFACStream(
        tokenizer=tokenizer,
        seq_len=args.seq_len,
        n_tokens=args.n_tokens,
        char_offset=args.char_offset,
    )
    loader = DataLoader(
        stream,
        batch_size=args.batch_size,
        collate_fn=collate_windows,
        num_workers=0,
    )
    generator = torch.Generator(device=args.device).manual_seed(args.seed)
    seen_books: List[dict] = []
    processed = 0

    try:
        for batch in tqdm(loader, desc=f"KFAC block {block}"):
            input_ids = batch["input_ids"].to(args.device, non_blocking=True)
            model.zero_grad(set_to_none=True)
            logits = model(input_ids=input_ids, use_cache=False).logits[:, :-1].float()
            if args.sample_labels:
                with torch.no_grad():
                    labels = torch.multinomial(
                        torch.softmax(logits, dim=-1).reshape(-1, logits.shape[-1]),
                        1,
                        generator=generator,
                    ).squeeze(1)
            else:
                labels = input_ids[:, 1:].reshape(-1)
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels)
            loss.backward()
            processed += int(input_ids.numel())
            seen_books.extend(
                {
                    "id": book_id,
                    "token_offset": int(token_offset),
                }
                for book_id, token_offset in zip(
                    batch["book_id"], batch["token_offset"]
                )
            )
            if processed >= args.n_tokens:
                break
    finally:
        for collector in collectors.values():
            collector.close()

    if processed < args.n_tokens:
        raise RuntimeError(
            f"Gutenberg stream ended after {processed} tokens; requested {args.n_tokens}"
        )

    book_ids = [window["id"] for window in seen_books]
    if len(book_ids) != len(set(book_ids)):
        raise RuntimeError("K-FAC stream sampled a Gutenberg book more than once")

    factors = {}
    diagnostics = {}
    for name, collector in collectors.items():
        A, G = collector.factors()
        A = A.cpu()
        G = G.cpu()
        diagnostics[name] = {
            "A": factor_diagnostics(A),
            "G": factor_diagnostics(G),
            "n_tokens": collector.n_tokens,
        }
        if not diagnostics[name]["A"]["finite"] or not diagnostics[name]["G"][
            "finite"
        ]:
            raise FloatingPointError(f"Non-finite factor for {name}")
        factors[name] = {"A": A, "G": G, "n_tokens": collector.n_tokens}

    args.save_dir.mkdir(parents=True, exist_ok=True)
    factor_path = args.save_dir / f"kfac_factors_blk_{block}.pt"
    metadata_path = args.save_dir / f"meta_blk_{block}.json"
    torch.save(factors, factor_path)
    metadata = {
        "model": args.model,
        "model_revision": args.revision,
        "dataset": GUTENBERG_ID,
        "dataset_revision": GUTENBERG_REVISION,
        "split": "en",
        "block": block,
        "projections": list(args.projections),
        "char_offset": args.char_offset,
        "seq_len": args.seq_len,
        "requested_tokens": args.n_tokens,
        "processed_tokens": processed,
        "unique_books": len(book_ids),
        "windows_per_book": 1,
        "sample_labels": args.sample_labels,
        "seed": args.seed,
        "windows": seen_books,
        "diagnostics": diagnostics,
        "factor_sha256": sha256_file(factor_path),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved {factor_path}")


def main():
    args = parse_args()
    if args.device == "cuda":
        args.device = "cuda:0"
    if args.n_tokens < args.seq_len:
        raise ValueError("n_tokens must be at least seq_len")
    if not args.sample_labels:
        print("Warning: using Gutenberg next-token labels instead of Fisher samples")

    torch.backends.cuda.matmul.allow_tf32 = True
    dtype = getattr(torch, args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, revision=args.revision, trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=args.revision,
        dtype=dtype,
        device_map={"": args.device},
        trust_remote_code=True,
    )
    validate_model(model, args.target_blocks)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.enable_input_require_grads()
    model.config.use_cache = False
    model.eval()

    for block in args.target_blocks:
        collect_block(args, model, tokenizer, block)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
