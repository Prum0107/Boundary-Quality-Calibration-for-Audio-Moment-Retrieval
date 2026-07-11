#!/usr/bin/env python3
"""R4 audit for BQC-Dec full-forward evaluation and matched controls.

This script performs no training. For each BQC-Dec seed it evaluates:
  1. confidence ranking on the BQC-Dec model's own predicted spans;
  2. qcls ranking on the same spans (the matched calibration comparison);
  3. the legacy slot-aligned setting: qcls scores on frozen-host spans.

Within-query pairwise accuracy is computed only on comparable pairs with
|delta IoU| >= 0.05. Equal predicted scores receive half credit.
"""

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from easydict import EasyDict
from scipy.stats import spearmanr
from torch.utils.data import DataLoader


REPO = Path("/root/autodl-tmp/bql_dec_br_20260705/dcase2026_task6_bql_dec_br")
OUT_DIR = Path("/root/autodl-tmp/bqc_paper_audit/r4_control_audit")
CKPT_DIR = Path("/root/autodl-tmp/bql_dec_br_20260705/outputs")
BASE_CKPT = REPO / "best_checkpoint.pth"
SEEDS = [2026, 2027, 2028, 2029, 2030]
PAIR_EPS = 0.05

sys.path.insert(0, str(REPO / "src"))
os.chdir(REPO)

from basic_utils import load_jsonl
from config import BaseOptions
from dataset import StartEndDataset, prepare_batch_inputs, start_end_collate
from evaluate import setup_model
from span_utils import span_cxw_to_xx


def build_dataset(opt, split):
    data_path = {"val": opt.val_path, "test": opt.test_path}[split]
    return StartEndDataset(**EasyDict(
        data_path=data_path,
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
        str(row["qid"]): [[float(s), float(e)] for s, e in row.get("relevant_windows", [])]
        for row in load_jsonl(path)
    }


def temporal_iou(cs, ce, gs, ge):
    inter = max(0.0, min(ce, ge) - max(cs, gs))
    union = max(ce, ge) - min(cs, gs)
    return inter / union if union > 0 else 0.0


def forward_predictions(model, loader, device):
    model.eval()
    rows = []
    with torch.no_grad():
        for batch in loader:
            meta = batch[0]
            model_inputs, _ = prepare_batch_inputs(batch[1], device)
            out = model(**model_inputs)
            spans = out["pred_spans"].detach().cpu()
            logits = out["pred_logits"].detach().cpu()
            conf_sigmoid = torch.sigmoid(logits[..., 0])
            conf_softmax = F.softmax(logits, dim=-1)[..., 0]
            quality = out.get("quality_logit")
            quality = torch.sigmoid(quality.detach().cpu()) if quality is not None else None
            for idx, item in enumerate(meta):
                row = {
                    "qid": str(item["qid"]),
                    "duration": float(item["duration"]),
                    "spans": spans[idx].numpy(),
                    "conf_sigmoid": conf_sigmoid[idx].numpy(),
                    "conf_softmax": conf_softmax[idx].numpy(),
                }
                if quality is not None:
                    row["quality"] = quality[idx].numpy()
                rows.append(row)
    return rows


def candidate_ious(row, gt_map, spans_override=None):
    spans = row["spans"] if spans_override is None else spans_override
    spans_xx = span_cxw_to_xx(torch.as_tensor(spans)).numpy() * row["duration"]
    gts = gt_map.get(row["qid"], [])
    return np.asarray([
        max(temporal_iou(float(s), float(e), gs, ge) for gs, ge in gts) if gts else 0.0
        for s, e in spans_xx
    ], dtype=np.float64)


def comparable_pairwise(score_rows, iou_rows, eps=PAIR_EPS):
    correct = 0.0
    total = 0
    tied_score = 0
    for scores, ious in zip(score_rows, iou_rows):
        for i in range(len(scores)):
            for j in range(i + 1, len(scores)):
                di = float(ious[i] - ious[j])
                if abs(di) < eps:
                    continue
                ds = float(scores[i] - scores[j])
                total += 1
                if ds * di > 0:
                    correct += 1.0
                elif ds == 0:
                    correct += 0.5
                    tied_score += 1
    return {
        "accuracy": correct / total if total else 0.0,
        "comparable_pairs": total,
        "tied_score_pairs": tied_score,
        "iou_difference_threshold": eps,
    }


def evaluate(rows, gt_map, score_key, spans_rows=None):
    score_rows = []
    iou_rows = []
    selected_ious = []
    flat_scores = []
    flat_ious = []
    for idx, row in enumerate(rows):
        if row["qid"] not in gt_map or not gt_map[row["qid"]]:
            continue
        spans_override = spans_rows[idx]["spans"] if spans_rows is not None else None
        ious = candidate_ious(row, gt_map, spans_override=spans_override)
        scores = np.asarray(row[score_key], dtype=np.float64)
        selected_ious.append(float(ious[int(np.argmax(scores))]))
        score_rows.append(scores)
        iou_rows.append(ious)
        flat_scores.extend(scores.tolist())
        flat_ious.extend(ious.tolist())
    pairwise = comparable_pairwise(score_rows, iou_rows)
    return {
        "n_queries": len(selected_ious),
        "r1_07": float(np.mean(np.asarray(selected_ious) >= 0.7) * 100),
        "mean_selected_iou": float(np.mean(selected_ious)),
        "spearman": float(spearmanr(flat_scores, flat_ious).statistic),
        "pairwise": pairwise,
        "selected_ious": selected_ious,
    }


def paired_bootstrap(reference, candidate, seed, n_boot=2000):
    reference = np.asarray(reference)
    candidate = np.asarray(candidate)
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(reference), len(reference))
        ref_r1 = np.mean(reference[idx] >= 0.7)
        can_r1 = np.mean(candidate[idx] >= 0.7)
        deltas.append((can_r1 - ref_r1) * 100)
    return [float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))]


def strip_selected_ious(metrics):
    return {key: value for key, value in metrics.items() if key != "selected_ious"}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manager = BaseOptions(str(REPO / "config_finetune_from_clotho_official_run.yml"))
    manager.parse()
    opt = manager.option

    loaders = {}
    gt = {}
    for split in ("val", "test"):
        dataset = build_dataset(opt, split)
        loaders[split] = DataLoader(
            dataset,
            collate_fn=start_end_collate,
            batch_size=opt.eval_bsz,
            num_workers=0,
            shuffle=False,
        )
        gt[split] = load_gt(opt.val_path if split == "val" else opt.test_path)

    host_model, _, _, _ = setup_model(opt)
    base_state = torch.load(BASE_CKPT, map_location=opt.device, weights_only=False)["model"]
    host_model.load_state_dict(base_state, strict=False)
    host_rows = {split: forward_predictions(host_model, loaders[split], opt.device) for split in loaders}

    host_val = {
        rule: evaluate(host_rows["val"], gt["val"], f"conf_{rule}")
        for rule in ("sigmoid", "softmax")
    }
    host_rule = max(host_val, key=lambda rule: host_val[rule]["r1_07"])
    host_test = evaluate(host_rows["test"], gt["test"], f"conf_{host_rule}")
    print(f"Host rule={host_rule} test R1={host_test['r1_07']:.2f}", flush=True)

    per_seed = []
    for seed in SEEDS:
        print(f"Seed {seed}", flush=True)
        model, _, _, _ = setup_model(opt)
        model.load_state_dict(base_state, strict=False)
        ckpt_path = CKPT_DIR / f"phase1_bql_dec_seed{seed}_best.pth"
        ckpt = torch.load(ckpt_path, map_location=opt.device, weights_only=False)
        model.load_state_dict(ckpt["model"], strict=False)
        rows = {split: forward_predictions(model, loaders[split], opt.device) for split in loaders}

        val_conf = {
            rule: evaluate(rows["val"], gt["val"], f"conf_{rule}")
            for rule in ("sigmoid", "softmax")
        }
        conf_rule = max(val_conf, key=lambda rule: val_conf[rule]["r1_07"])
        conf_test = evaluate(rows["test"], gt["test"], f"conf_{conf_rule}")
        quality_test = evaluate(rows["test"], gt["test"], "quality")
        legacy_test = evaluate(rows["test"], gt["test"], "quality", spans_rows=host_rows["test"])

        matched_delta = quality_test["r1_07"] - conf_test["r1_07"]
        matched_ci = paired_bootstrap(
            conf_test["selected_ious"], quality_test["selected_ious"], seed=seed
        )
        legacy_delta = legacy_test["r1_07"] - host_test["r1_07"]
        result = {
            "seed": seed,
            "checkpoint_val_scoring": ckpt.get("val_scoring"),
            "matched_conf_rule": conf_rule,
            "matched_conf": strip_selected_ious(conf_test),
            "full_forward_quality": strip_selected_ious(quality_test),
            "legacy_host_spans_plus_quality": strip_selected_ious(legacy_test),
            "quality_minus_matched_conf_r1": matched_delta,
            "quality_minus_matched_conf_ci": matched_ci,
            "legacy_minus_frozen_host_r1": legacy_delta,
        }
        per_seed.append(result)
        print(
            f"  conf={conf_test['r1_07']:.2f} quality={quality_test['r1_07']:.2f} "
            f"matched_delta={matched_delta:+.2f} legacy={legacy_test['r1_07']:.2f} "
            f"pair={quality_test['pairwise']['accuracy']:.4f}",
            flush=True,
        )

    def mean(path):
        return float(np.mean([row[path[0]][path[1]] for row in per_seed]))

    matched_deltas = [row["quality_minus_matched_conf_r1"] for row in per_seed]
    summary = {
        "pairwise_definition": {
            "scope": "within query",
            "comparable_if_abs_iou_difference_at_least": PAIR_EPS,
            "score_tie_credit": 0.5,
        },
        "host": {
            "selected_rule": host_rule,
            "val": {rule: strip_selected_ious(value) for rule, value in host_val.items()},
            "test": strip_selected_ious(host_test),
        },
        "per_seed": per_seed,
        "five_seed": {
            "matched_conf_mean_r1": mean(("matched_conf", "r1_07")),
            "full_forward_quality_mean_r1": mean(("full_forward_quality", "r1_07")),
            "legacy_hybrid_mean_r1": mean(("legacy_host_spans_plus_quality", "r1_07")),
            "quality_minus_matched_conf_mean_r1": float(np.mean(matched_deltas)),
            "quality_minus_matched_conf_seed_ci": [
                float(np.percentile(matched_deltas, 2.5)),
                float(np.percentile(matched_deltas, 97.5)),
            ],
            "full_forward_quality_mean_spearman": mean(("full_forward_quality", "spearman")),
            "full_forward_quality_mean_pairwise": float(np.mean([
                row["full_forward_quality"]["pairwise"]["accuracy"] for row in per_seed
            ])),
            "matched_conf_mean_pairwise": float(np.mean([
                row["matched_conf"]["pairwise"]["accuracy"] for row in per_seed
            ])),
        },
    }
    output_path = OUT_DIR / "r4_bqc_dec_matched_audit.json"
    output_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary["five_seed"], indent=2), flush=True)
    print(f"Saved {output_path}", flush=True)


if __name__ == "__main__":
    main()
