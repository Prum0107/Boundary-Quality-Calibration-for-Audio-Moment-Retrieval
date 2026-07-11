#!/usr/bin/env python3
"""Controlled mechanism ablation for Boundary Quality Calibration.

The five variants isolate parameter scope and supervision source:

* quality_head_only: quality head, detached quality losses only.
* anchors_quality: moment anchors + quality head, detached quality losses only.
* decoder_host: decoder, original host losses only.
* decoder_detached: decoder + quality head, detached quality losses only.
* decoder_shuffled: decoder + quality head, shuffled detached quality losses only.

Model selection and score rules are fixed on validation. Test is evaluated once
from the selected checkpoint. This script is tied to the preserved server
checkout so that it can reproduce the historical data and initialization.
"""
import argparse
import copy
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from easydict import EasyDict
from torch.utils.data import DataLoader
from tqdm import tqdm


HOST_REPO = Path("/root/autodl-tmp/bql_final_20260705/dcase2026_task6_bql_final")
HOST_CKPT = HOST_REPO / "best_checkpoint.pth"
HOST_CONFIG = HOST_REPO / "config_finetune_from_clotho_official_run.yml"
DEFAULT_OUT = Path("/root/autodl-tmp/bqc_mechanism_ablation_20260711")
sys.path.insert(0, str(HOST_REPO / "src"))

from basic_utils import load_jsonl
from config import BaseOptions
from dataset import StartEndDataset, prepare_batch_inputs, start_end_collate
from evaluate import setup_model
from span_utils import span_cxw_to_xx


VARIANTS = (
    "quality_head_only",
    "anchors_quality",
    "decoder_host",
    "decoder_detached",
    "decoder_shuffled",
    "decoder_host_detached",
    "decoder_host_shuffled",
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_dataset(opt, split):
    path = {"train": opt.train_path, "val": opt.val_path, "test": opt.test_path}[split]
    return StartEndDataset(**EasyDict(
        data_path=path,
        ctx_mode=opt.ctx_mode,
        a_feat_dir=opt.a_feat_dir,
        q_feat_dir=opt.t_feat_dir,
        q_feat_type="last_hidden_state",
        a_feat_type=opt.a_feat_type,
        max_q_l=opt.max_q_l,
        max_a_l=opt.max_a_l,
        clip_len=opt.clip_length,
        max_windows=opt.max_windows,
        span_loss_type=opt.span_loss_type,
        load_labels=True,
    ))


def load_gt(path):
    return {
        str(row["qid"]): [[float(start), float(end)] for start, end in row.get("relevant_windows", [])]
        for row in load_jsonl(path)
    }


def temporal_iou(start, end, gt_start, gt_end):
    inter = max(0.0, min(end, gt_end) - max(start, gt_start))
    union = max(end, gt_end) - min(start, gt_start)
    return inter / union if union > 0 else 0.0


def average_ranks(values):
    """Return one-based average ranks, including ties, using NumPy only."""
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    return ranks


def spearman_correlation(left, right):
    left_rank = average_ranks(left)
    right_rank = average_ranks(right)
    if left_rank.std() == 0 or right_rank.std() == 0:
        return None
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def quality_targets(pred_spans, span_labels, shuffled=False):
    """Build detached per-candidate IoU targets.

    Shuffling occurs independently within each query after target construction,
    preserving the marginal target distribution while destroying slot quality.
    """
    with torch.no_grad():
        batch, slots = pred_spans.shape[:2]
        targets = pred_spans.new_zeros((batch, slots))
        for b in range(batch):
            if b >= len(span_labels) or "spans" not in span_labels[b] or len(span_labels[b]["spans"]) == 0:
                continue
            pred_xx = span_cxw_to_xx(pred_spans[b].detach())
            target_xx = span_cxw_to_xx(span_labels[b]["spans"])
            ps = pred_xx.unsqueeze(1)
            ts = target_xx.unsqueeze(0)
            inter = (torch.minimum(ps[..., 1], ts[..., 1]) - torch.maximum(ps[..., 0], ts[..., 0])).clamp(min=0)
            union = (torch.maximum(ps[..., 1], ts[..., 1]) - torch.minimum(ps[..., 0], ts[..., 0])).clamp(min=1e-6)
            targets[b] = (inter / union).max(dim=1).values
        if shuffled:
            targets = torch.stack([row[torch.randperm(slots, device=row.device)] for row in targets])
        return targets.detach()


def compute_quality_losses(outputs, targets, shuffled=False):
    q_target = quality_targets(outputs["pred_spans"], targets["span_labels"], shuffled=shuffled)
    y_target = (q_target >= 0.7).float()
    q_pred = torch.sigmoid(outputs["quality_reg"])
    loss_reg = F.smooth_l1_loss(q_pred, q_target)
    loss_cls = F.binary_cross_entropy_with_logits(outputs["quality_logit"], y_target)
    with torch.no_grad():
        list_target = F.softmax(q_target / 0.1, dim=1)
    loss_list = -(list_target * F.log_softmax(outputs["quality_logit"], dim=1)).sum(dim=1).mean()
    return {
        "quality_reg": loss_reg,
        "quality_cls": loss_cls,
        "quality_list": 0.5 * loss_list,
    }, q_target


def configure_variant(model, variant):
    for parameter in model.parameters():
        parameter.requires_grad = False

    for name, parameter in model.named_parameters():
        if variant == "quality_head_only" and "quality_head" in name:
            parameter.requires_grad = True
        elif variant == "anchors_quality" and ("query_embed" in name or "quality_head" in name):
            parameter.requires_grad = True
        elif variant == "decoder_host" and "transformer.decoder" in name:
            parameter.requires_grad = True
        elif variant in (
            "decoder_detached", "decoder_shuffled",
            "decoder_host_detached", "decoder_host_shuffled",
        ) and (
            "transformer.decoder" in name or "quality_head" in name
        ):
            parameter.requires_grad = True

    groups = {}
    for name, parameter in model.named_parameters():
        group = "other"
        if "query_embed" in name:
            group = "anchors"
        elif "quality_head" in name:
            group = "quality_head"
        elif "transformer.decoder" in name:
            group = "decoder"
        elif "span_embed" in name or "class_embed" in name:
            group = "span_class_heads"
        groups.setdefault(group, {"trainable": 0, "frozen": 0, "names": []})
        key = "trainable" if parameter.requires_grad else "frozen"
        groups[group][key] += parameter.numel()
        if parameter.requires_grad:
            groups[group]["names"].append(name)
    return groups


def weighted_host_loss(loss_dict, criterion):
    terms = []
    for name, value in loss_dict.items():
        if name.startswith("loss_bql"):
            continue
        if name in criterion.weight_dict:
            terms.append(value * criterion.weight_dict[name])
    if not terms:
        raise RuntimeError("No host loss terms were found")
    return sum(terms)


def score_rule(variant):
    return "confidence" if variant == "decoder_host" else "quality_cls"


def uses_shuffled_quality(variant):
    return variant in ("decoder_shuffled", "decoder_host_shuffled")


def uses_joint_loss(variant):
    return variant in ("decoder_host_detached", "decoder_host_shuffled")


@torch.no_grad()
def collect_predictions(model, loader, opt, gt_map, rule):
    model.eval()
    records = []
    for batch in tqdm(loader, desc=f"eval:{rule}", leave=False):
        metadata = batch[0]
        model_inputs, _ = prepare_batch_inputs(batch[1], opt.device)
        outputs = model(**model_inputs)
        pred_xx = span_cxw_to_xx(outputs["pred_spans"]).cpu()
        confidence = F.softmax(outputs["pred_logits"], dim=-1)[..., 0].cpu()
        quality = torch.sigmoid(outputs["quality_logit"]).cpu()
        for b, meta in enumerate(metadata):
            duration = float(meta["duration"])
            gt_windows = gt_map.get(str(meta["qid"]), [])
            candidates = []
            for slot in range(pred_xx.shape[1]):
                start = float(pred_xx[b, slot, 0]) * duration
                end = float(pred_xx[b, slot, 1]) * duration
                iou = max(
                    (temporal_iou(start, end, gs, ge) for gs, ge in gt_windows),
                    default=0.0,
                )
                candidates.append({
                    "iou": iou,
                    "confidence": float(confidence[b, slot]),
                    "quality_cls": float(quality[b, slot]),
                })
            top = max(candidates, key=lambda candidate: candidate[rule])
            host_top = max(candidates, key=lambda candidate: candidate["confidence"])
            records.append({
                "qid": str(meta["qid"]),
                "top_iou": top["iou"],
                "host_iou": host_top["iou"],
                "candidates": candidates,
            })
    return records


def summarize_records(records, rule):
    top_ious = np.asarray([record["top_iou"] for record in records], dtype=float)
    all_scores, all_ious = [], []
    correct, total = 0, 0
    for record in records:
        candidates = record["candidates"]
        all_scores.extend(candidate[rule] for candidate in candidates)
        all_ious.extend(candidate["iou"] for candidate in candidates)
        for left in range(len(candidates)):
            for right in range(left + 1, len(candidates)):
                iou_delta = candidates[left]["iou"] - candidates[right]["iou"]
                if abs(iou_delta) < 1e-12:
                    continue
                score_delta = candidates[left][rule] - candidates[right][rule]
                correct += int(score_delta * iou_delta > 0)
                total += 1
    rho = spearman_correlation(all_scores, all_ious)
    return {
        "n_queries": len(records),
        "r1_03": float(np.mean(top_ious >= 0.3)),
        "r1_05": float(np.mean(top_ious >= 0.5)),
        "r1_07": float(np.mean(top_ious >= 0.7)),
        "mean_iou": float(np.mean(top_ious)),
        "spearman": rho if rho is not None and np.isfinite(rho) else None,
        "pairwise_accuracy": float(correct / total) if total else None,
        "pairwise_pairs": total,
    }


def bootstrap_delta(reference_records, method_records, n_boot=2000, seed=7):
    reference = np.asarray([record["top_iou"] >= 0.7 for record in reference_records], dtype=float)
    method = np.asarray([record["top_iou"] >= 0.7 for record in method_records], dtype=float)
    rng = np.random.default_rng(seed)
    deltas = np.empty(n_boot, dtype=float)
    for index in range(n_boot):
        sample = rng.integers(0, len(reference), len(reference))
        deltas[index] = np.mean(method[sample] - reference[sample])
    return {
        "delta": float(np.mean(method - reference)),
        "ci_low": float(np.percentile(deltas, 2.5)),
        "ci_high": float(np.percentile(deltas, 97.5)),
        "crosses_zero": bool(np.percentile(deltas, 2.5) <= 0 <= np.percentile(deltas, 97.5)),
    }


def build_model(opt):
    model, criterion, _, _ = setup_model(opt)
    checkpoint = torch.load(HOST_CKPT, map_location=opt.device, weights_only=False)
    missing, unexpected = model.load_state_dict(checkpoint["model"], strict=False)
    allowed_missing = [name for name in missing if name.startswith("quality_head")]
    if len(allowed_missing) != len(missing) or unexpected:
        raise RuntimeError(f"Checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    return model, criterion


def gradient_smoke(model, criterion, batch, opt, variant):
    model.train()
    model_inputs, targets = prepare_batch_inputs(batch[1], opt.device)
    outputs = model(**model_inputs)
    quality_losses, q_target = compute_quality_losses(
        outputs, targets, shuffled=uses_shuffled_quality(variant)
    )
    if q_target.requires_grad or q_target.grad_fn is not None:
        raise AssertionError("Quality target is not detached")
    if variant == "decoder_host":
        loss = weighted_host_loss(criterion(outputs, targets), criterion)
    elif uses_joint_loss(variant):
        loss = weighted_host_loss(criterion(outputs, targets), criterion) + sum(quality_losses.values())
    else:
        loss = sum(quality_losses.values())
    model.zero_grad(set_to_none=True)
    loss.backward()
    grad_groups = {"anchors": 0.0, "decoder": 0.0, "quality_head": 0.0, "other": 0.0}
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        group = "other"
        if "query_embed" in name:
            group = "anchors"
        elif "transformer.decoder" in name:
            group = "decoder"
        elif "quality_head" in name:
            group = "quality_head"
        grad_groups[group] += float(parameter.grad.detach().norm().cpu())
    return {
        "loss": float(loss.detach().cpu()),
        "target_min": float(q_target.min().cpu()),
        "target_max": float(q_target.max().cpu()),
        "target_mean": float(q_target.mean().cpu()),
        "target_requires_grad": q_target.requires_grad,
        "gradient_norm_sums": grad_groups,
    }


def train_one(args, opt, datasets, gt_maps, variant, seed, frozen_host_records):
    set_seed(seed)
    model, criterion = build_model(opt)
    parameter_scope = configure_variant(model, variant)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    train_loader = DataLoader(
        datasets["train"], batch_size=opt.bsz, shuffle=True,
        num_workers=opt.num_workers, collate_fn=start_end_collate,
    )
    val_loader = DataLoader(
        datasets["val"], batch_size=opt.eval_bsz, shuffle=False,
        num_workers=opt.num_workers, collate_fn=start_end_collate,
    )
    first_batch = next(iter(train_loader))
    smoke = gradient_smoke(model, criterion, first_batch, opt, variant)
    optimizer.zero_grad(set_to_none=True)

    rule = score_rule(variant)
    best = {"r1_07": -1.0, "epoch": None, "state": None, "metrics": None}
    history = []
    for epoch in range(args.epochs):
        model.train()
        running, steps = 0.0, 0
        for batch in tqdm(train_loader, desc=f"{variant}:s{seed}:e{epoch + 1}", leave=False):
            model_inputs, targets = prepare_batch_inputs(batch[1], opt.device)
            outputs = model(**model_inputs)
            if variant == "decoder_host":
                loss = weighted_host_loss(criterion(outputs, targets), criterion)
            else:
                quality_losses, _ = compute_quality_losses(
                    outputs, targets, shuffled=uses_shuffled_quality(variant)
                )
                loss = sum(quality_losses.values())
                if uses_joint_loss(variant):
                    loss = loss + weighted_host_loss(criterion(outputs, targets), criterion)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(trainable, args.grad_clip)
            optimizer.step()
            running += float(loss.detach().cpu())
            steps += 1
        scheduler.step()

        if (epoch + 1) % args.eval_every == 0 or epoch + 1 == args.epochs:
            records = collect_predictions(model, val_loader, opt, gt_maps["val"], rule)
            metrics = summarize_records(records, rule)
            row = {"epoch": epoch + 1, "train_loss": running / max(steps, 1), **metrics}
            history.append(row)
            print(json.dumps({"variant": variant, "seed": seed, **row}), flush=True)
            if metrics["r1_07"] > best["r1_07"]:
                best = {
                    "r1_07": metrics["r1_07"],
                    "epoch": epoch + 1,
                    "state": copy.deepcopy({name: value.detach().cpu() for name, value in model.state_dict().items()}),
                    "metrics": metrics,
                }

    model.load_state_dict(best["state"], strict=True)
    test_loader = DataLoader(
        datasets["test"], batch_size=opt.eval_bsz, shuffle=False,
        num_workers=opt.num_workers, collate_fn=start_end_collate,
    )
    test_records = collect_predictions(model, test_loader, opt, gt_maps["test"], rule)
    test_metrics = summarize_records(test_records, rule)
    matched_conf_metrics = summarize_records(test_records, "confidence")
    ci = bootstrap_delta(frozen_host_records, test_records, seed=seed)

    output_dir = args.output / variant / f"seed{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "best.pth"
    torch.save({
        "model": best["state"],
        "variant": variant,
        "seed": seed,
        "epoch": best["epoch"],
        "score_rule": rule,
        "detached_target": True,
        "shuffled_labels": uses_shuffled_quality(variant),
    }, checkpoint_path)
    result = {
        "variant": variant,
        "seed": seed,
        "score_rule": rule,
        "loss_mode": (
            "host_only" if variant == "decoder_host" else
            "host_plus_quality" if uses_joint_loss(variant) else
            "quality_only"
        ),
        "quality_target_detached": True,
        "quality_target_shuffled": uses_shuffled_quality(variant),
        "parameter_scope": parameter_scope,
        "gradient_smoke": smoke,
        "best_epoch": best["epoch"],
        "val_metrics": best["metrics"],
        "test_metrics": test_metrics,
        "matched_confidence_test_metrics": matched_conf_metrics,
        "delta_vs_frozen_host": ci,
        "history": history,
        "checkpoint": str(checkpoint_path),
    }
    (output_dir / "result.json").write_text(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[2026, 2027, 2028, 2029, 2030])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--eval-every", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    # The preserved host config uses paths relative to its own checkout.
    os.chdir(HOST_REPO)

    options = BaseOptions(str(HOST_CONFIG))
    options.parse()
    opt = options.option
    datasets = {split: build_dataset(opt, split) for split in ("train", "val", "test")}
    gt_maps = {"val": load_gt(opt.val_path), "test": load_gt(opt.test_path)}

    set_seed(2026)
    frozen_model, _ = build_model(opt)
    frozen_test_loader = DataLoader(
        datasets["test"], batch_size=opt.eval_bsz, shuffle=False,
        num_workers=opt.num_workers, collate_fn=start_end_collate,
    )
    frozen_records = collect_predictions(frozen_model, frozen_test_loader, opt, gt_maps["test"], "confidence")
    frozen_metrics = summarize_records(frozen_records, "confidence")
    (args.output / "frozen_host.json").write_text(json.dumps(frozen_metrics, indent=2))

    results = []
    for variant in args.variants:
        for seed in args.seeds:
            results.append(train_one(args, opt, datasets, gt_maps, variant, seed, frozen_records))
            summary = {
                "protocol": {
                    "variants": args.variants,
                    "seeds": args.seeds,
                    "epochs": args.epochs,
                    "lr": args.lr,
                    "test_locked": True,
                },
                "frozen_host": frozen_metrics,
                "runs": [{key: value for key, value in result.items() if key not in ("history", "parameter_scope")} for result in results],
            }
            (args.output / "running_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
