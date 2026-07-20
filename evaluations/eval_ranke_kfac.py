#!/usr/bin/env python
"""Build Ranke K-FAC deltas and evaluate recall, recitation, and quality."""

import argparse
import json
import re
import sys
from pathlib import Path
from statistics import mean

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from data.collect_kfac_ranke import MODEL_ID, MODEL_REVISION, validate_model  # noqa: E402
from kfac_treatment_pairwise import KFACAmplificationPairwise  # noqa: E402


TIME_LOCK_ID = "history-llms/time-lock-variants"
TIME_LOCK_REVISION = "cd0f0bc323ae471028562d034e7c3c67bc7d6077"
CHOICE_MARKER = "<|CHOICE|>"
DEFAULT_LAYERS = [20, 24, 28, 32, 35]
DEFAULT_PROJECTIONS = ["gate", "up", "down"]
WELLINGTON_QUESTION = (
    "What was the name of the battle where Napoleon was defeated by "
    "the Duke of Wellington? Answer:"
)
WELLINGTON_CHOICES = [" Waterloo", " Austerlitz", " Leipzig", " Trafalgar"]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=["build_cache", "time_lock", "wellington"],
        required=True,
    )
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--revision", default=MODEL_REVISION)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16"
    )
    parser.add_argument("--layers", type=int, nargs="+", default=DEFAULT_LAYERS)
    parser.add_argument(
        "--projections",
        nargs="+",
        choices=DEFAULT_PROJECTIONS,
        default=DEFAULT_PROJECTIONS,
    )
    parser.add_argument("--rhos", type=float, nargs="+", default=[0.90, 0.95])
    parser.add_argument(
        "--alphas", type=float, nargs="+", default=[-1.0, 0.0, 0.25, 0.5, 1.0]
    )
    parser.add_argument(
        "--factors_root",
        type=Path,
        default=Path("assets/kfac_factors/ranke_4b_1913"),
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        default=Path("cache/kfac_weights/ranke_4b_1913"),
    )
    parser.add_argument(
        "--results_dir", type=Path, default=Path("results/ranke_4b_1913")
    )
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--bootstrap_samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=1913)
    return parser.parse_args()


def sanitize(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]", "_", text)


def rho_string(rho: float) -> str:
    return f"{rho:.3f}".replace(".", "p")


def layer_name(layer: int, projection: str) -> str:
    return f"model.layers.{layer}.mlp.{projection}_proj"


def cache_path(args, layer: int, projection: str, rho: float) -> Path:
    model = sanitize(args.model.split("/")[-1])
    return args.cache_dir / (
        f"{model}__L{layer}__{projection}__rho{rho_string(rho)}__low.pt"
    )


def load_model_and_tokenizer(args):
    dtype = getattr(torch, args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, revision=args.revision, trust_remote_code=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=args.revision,
        dtype=dtype,
        device_map={"": args.device},
        trust_remote_code=True,
    ).eval()
    validate_model(model, args.layers)
    return model, tokenizer


def build_cache(args, model):
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    for block in args.layers:
        factors_path = args.factors_root / f"kfac_factors_blk_{block}.pt"
        if not factors_path.exists():
            raise FileNotFoundError(factors_path)
        for projection in args.projections:
            name = layer_name(block, projection)
            treatment = KFACAmplificationPairwise(
                model,
                layer_names=[name],
                kfac_factors_path=str(factors_path),
                device=args.device,
            )
            for rho in args.rhos:
                low = treatment.compute_low_curvature_components(rho)[name]
                output = cache_path(args, block, projection, rho)
                payload = {
                    "low": low.detach().cpu(),
                    "stats": treatment.amplification_stats[name],
                    "model": args.model,
                    "model_revision": args.revision,
                    "layer_name": name,
                    "rho": rho,
                }
                torch.save(payload, output)
                print(f"Saved {output}")
                del low, payload
                torch.cuda.empty_cache()
            del treatment
            torch.cuda.empty_cache()


def selected_weights(model, args):
    selected = {}
    for block in args.layers:
        mlp = model.model.layers[block].mlp
        for projection in args.projections:
            selected[(block, projection)] = getattr(mlp, f"{projection}_proj").weight
    return selected


def original_weights(model, args):
    return {
        key: weight.detach().cpu().clone()
        for key, weight in selected_weights(model, args).items()
    }


def apply_condition(model, args, originals, rho, alpha):
    weights = selected_weights(model, args)
    with torch.no_grad():
        for (block, projection), parameter in weights.items():
            original = originals[(block, projection)]
            if alpha == 0.0:
                parameter.copy_(original.to(parameter.device))
                continue
            path = cache_path(args, block, projection, rho)
            if not path.exists():
                raise FileNotFoundError(path)
            low = torch.load(path, map_location="cpu", weights_only=True)["low"]
            edited = original.float().to(parameter.device)
            edited.add_(low.to(parameter.device), alpha=float(alpha))
            if not torch.isfinite(edited).all():
                raise FloatingPointError(
                    f"Non-finite weight for layer {block} {projection}, "
                    f"rho={rho}, alpha={alpha}"
                )
            parameter.copy_(edited.to(parameter.dtype))


def conditions(args):
    output = [("original", None, 0.0)]
    for rho in args.rhos:
        for alpha in args.alphas:
            if alpha == 0.0:
                continue
            name = f"rho{rho_string(rho)}_alpha{alpha:+g}"
            output.append((name, rho, alpha))
    return output


def encode_pair(tokenizer, context: str, continuation: str):
    spaces = len(context) - len(context.rstrip())
    if spaces:
        continuation = context[-spaces:] + continuation
        context = context[:-spaces]
    context_ids = tokenizer.encode(context, add_special_tokens=False)
    whole_ids = tokenizer.encode(context + continuation, add_special_tokens=False)
    continuation_ids = whole_ids[len(context_ids) :]
    if not context_ids:
        context_ids = [tokenizer.eos_token_id]
        whole_ids = context_ids + continuation_ids
    if not continuation_ids:
        raise ValueError("Continuation tokenized to zero tokens")
    return context_ids + continuation_ids, len(context_ids)


@torch.inference_mode()
def score_encoded_pairs(model, encoded_pairs, batch_size: int, pad_id: int):
    scores = []
    for start in range(0, len(encoded_pairs), batch_size):
        batch = encoded_pairs[start : start + batch_size]
        max_length = max(len(ids) for ids, _ in batch)
        input_ids = torch.full(
            (len(batch), max_length), pad_id, dtype=torch.long, device=model.device
        )
        attention_mask = torch.zeros_like(input_ids)
        for row, (ids, _context_length) in enumerate(batch):
            input_ids[row, : len(ids)] = torch.tensor(ids, device=model.device)
            attention_mask[row, : len(ids)] = 1
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        log_probs = F.log_softmax(logits.float(), dim=-1)
        for row, (ids, context_length) in enumerate(batch):
            target_ids = torch.tensor(ids[context_length:], device=model.device)
            positions = torch.arange(
                context_length - 1, len(ids) - 1, device=model.device
            )
            token_scores = log_probs[row, positions, target_ids]
            scores.append(float(token_scores.sum()))
    return scores


def bootstrap_interval(values, samples: int, seed: int):
    array = np.asarray(values, dtype=np.float64)
    if len(array) == 0:
        return [None, None]
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(array), size=(samples, len(array)))
    estimates = array[indices].mean(axis=1)
    return [float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))]


def mean_or_none(values):
    values = list(values)
    return mean(values) if values else None


def qid_means(rows, metric, bucket):
    grouped = {}
    for row in rows:
        if row.get("bucket") == bucket:
            grouped.setdefault(row["qid"], []).append(float(row[metric]))
    return {qid: mean(values) for qid, values in grouped.items()}


def summarize_time_lock(all_rows, args):
    by_condition = {}
    for row in all_rows:
        by_condition.setdefault(row["condition"], []).append(row)
    summary = {}
    baseline = by_condition["original"]
    for condition, rows in by_condition.items():
        condition_summary = {}
        for bucket in ("pre", "post"):
            bucket_rows = [row for row in rows if row["bucket"] == bucket]
            condition_summary[bucket] = {
                metric: mean_or_none(float(row[metric]) for row in bucket_rows)
                for metric in ("p_correct", "margin", "accuracy", "entropy")
            }
            if condition != "original":
                current = qid_means(rows, "p_correct", bucket)
                original = qid_means(baseline, "p_correct", bucket)
                qids = sorted(set(current) & set(original))
                deltas = [current[qid] - original[qid] for qid in qids]
                condition_summary[bucket]["delta_p_correct"] = mean_or_none(deltas)
                condition_summary[bucket]["delta_p_correct_ci95"] = bootstrap_interval(
                    deltas, args.bootstrap_samples, args.seed
                )
        if condition != "original":
            pre_delta = condition_summary["pre"]["delta_p_correct"]
            post_delta = condition_summary["post"]["delta_p_correct"]
            condition_summary["pre_minus_post_delta"] = (
                pre_delta - post_delta
                if pre_delta is not None and post_delta is not None
                else None
            )
        summary[condition] = condition_summary
    return summary


def run_time_lock(args, model, tokenizer, originals):
    dataset = load_dataset(
        TIME_LOCK_ID,
        split="train",
        revision=TIME_LOCK_REVISION,
        verification_mode="no_checks",
    )
    documents = list(dataset)
    if args.limit:
        pre = [doc for doc in documents if int(doc["year"]) <= 1913]
        post = [doc for doc in documents if int(doc["year"]) > 1913]
        pre_count = max(1, args.limit // 2)
        post_count = max(1, args.limit - pre_count)
        documents = pre[:pre_count] + post[:post_count]

    all_rows = []
    for condition, rho, alpha in conditions(args):
        apply_condition(model, args, originals, rho, alpha)
        pairs = []
        pair_documents = []
        for doc_index, document in enumerate(documents):
            question = document["question"]
            if question.count(CHOICE_MARKER) != 1:
                raise ValueError(f"Expected one choice marker: {question!r}")
            prefix, suffix = question.split(CHOICE_MARKER, 1)
            for choice_index, choice in enumerate(document["choices"]):
                pairs.append(encode_pair(tokenizer, prefix, f"{choice}{suffix}"))
                pair_documents.append((doc_index, choice_index))
        scores = score_encoded_pairs(
            model, pairs, args.batch_size, tokenizer.pad_token_id
        )
        per_document = [[None] * 4 for _ in documents]
        for (doc_index, choice_index), score in zip(pair_documents, scores):
            per_document[doc_index][choice_index] = score

        for document, choice_scores in zip(documents, per_document):
            probabilities = torch.softmax(torch.tensor(choice_scores), dim=0)
            answer = int(document["answer"])
            wrong_scores = [
                score for index, score in enumerate(choice_scores) if index != answer
            ]
            prediction = max(range(4), key=choice_scores.__getitem__)
            entropy = float(-(probabilities * probabilities.clamp_min(1e-30).log()).sum())
            all_rows.append(
                {
                    "condition": condition,
                    "rho": rho,
                    "alpha": alpha,
                    "qid": document["qid"],
                    "vid": document["vid"],
                    "year": int(document["year"]),
                    "bucket": "pre" if int(document["year"]) <= 1913 else "post",
                    "domain": document["domain"],
                    "answer": answer,
                    "prediction": prediction,
                    "choice_loglikelihoods": choice_scores,
                    "p_correct": float(probabilities[answer]),
                    "margin": choice_scores[answer] - max(wrong_scores),
                    "accuracy": float(prediction == answer),
                    "entropy": entropy,
                }
            )
    return all_rows, summarize_time_lock(all_rows, args)


def run_wellington(args, model, tokenizer, originals):
    pairs = [
        encode_pair(tokenizer, WELLINGTON_QUESTION, choice)
        for choice in WELLINGTON_CHOICES
    ]
    rows = []
    for condition, rho, alpha in conditions(args):
        apply_condition(model, args, originals, rho, alpha)
        scores = score_encoded_pairs(
            model, pairs, args.batch_size, tokenizer.pad_token_id
        )
        probabilities = torch.softmax(torch.tensor(scores), dim=0)
        prompt = tokenizer(WELLINGTON_QUESTION, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            output = model.generate(
                **prompt,
                do_sample=False,
                max_new_tokens=16,
                pad_token_id=tokenizer.pad_token_id,
            )
        generated = tokenizer.decode(
            output[0, prompt.input_ids.shape[1] :], skip_special_tokens=True
        )
        rows.append(
            {
                "condition": condition,
                "rho": rho,
                "alpha": alpha,
                "question": WELLINGTON_QUESTION,
                "choices": [choice.strip() for choice in WELLINGTON_CHOICES],
                "choice_loglikelihoods": scores,
                "choice_probabilities": probabilities.tolist(),
                "p_correct": float(probabilities[0]),
                "prediction": WELLINGTON_CHOICES[int(probabilities.argmax())].strip(),
                "generated": generated,
            }
        )
    return rows, {row["condition"]: row["p_correct"] for row in rows}


def write_outputs(args, rows, summary):
    args.results_dir.mkdir(parents=True, exist_ok=True)
    rows_path = args.results_dir / f"{args.mode}.jsonl"
    summary_path = args.results_dir / f"{args.mode}_summary.json"
    with rows_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    run_metadata = {
        "model": args.model,
        "model_revision": args.revision,
        "layers": args.layers,
        "projections": args.projections,
        "rhos": args.rhos,
        "alphas": args.alphas,
        "limit": args.limit,
        "summary": summary,
    }
    summary_path.write_text(json.dumps(run_metadata, indent=2), encoding="utf-8")
    print(f"Saved {rows_path}")
    print(f"Saved {summary_path}")


def main():
    args = parse_args()
    if args.device == "cuda":
        args.device = "cuda:0"
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    model, tokenizer = load_model_and_tokenizer(args)

    if args.mode == "build_cache":
        build_cache(args, model)
        return

    originals = original_weights(model, args)
    if args.mode == "time_lock":
        rows, summary = run_time_lock(args, model, tokenizer, originals)
    else:
        rows, summary = run_wellington(args, model, tokenizer, originals)

    apply_condition(model, args, originals, None, 0.0)
    write_outputs(args, rows, summary)


if __name__ == "__main__":
    main()
