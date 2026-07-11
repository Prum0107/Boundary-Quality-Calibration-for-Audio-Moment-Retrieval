#!/usr/bin/env python3
"""Recompute comparable-pair accuracy for host, BQC-IoU, and BQC-Joint.

No training or test-set model selection is performed. Checkpoint and score
rules are loaded from the original dev-val-selected experiment outputs.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr
from torch.utils.data import DataLoader


REPO = Path("/root/autodl-tmp/dcase2026_task6_official_baseline")
BASE_CKPT = REPO / "results_clotho_pretrain_castella_ft/best_checkpoint.pth"
BCE_DIR = Path("/root/autodl-tmp/varifocal_iou_conf_20260708/unified_5seed")
JOINT_DIR = Path("/root/autodl-tmp/bqc_joint_20260708")
OUT_DIR = Path("/root/autodl-tmp/bqc_paper_audit/r4_control_audit")
SEEDS = [2026, 2027, 2028, 2029, 2030]
PAIR_EPS = 0.05

sys.path.insert(0, str(REPO / "src"))
os.chdir(REPO)

from basic_utils import load_jsonl
from config import BaseOptions
from dataset import StartEndDataset, prepare_batch_inputs, start_end_collate
from evaluate import setup_model
from span_utils import span_cxw_to_xx


class QualityHead(nn.Module):
    def __init__(self, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features):
        return self.net(features).squeeze(-1)


class JointModel(nn.Module):
    def __init__(self, base, hidden_dim=256):
        super().__init__()
        self.base = base
        self.quality_head = QualityHead(hidden_dim)
        self._hidden = None
        self._hook = None

    def _capture(self, module, inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        if hidden.dim() == 4:
            hidden = hidden[-1]
        if hidden.dim() == 3 and hidden.shape[0] != 1:
            hidden = hidden.permute(1, 0, 2)
        self._hidden = hidden

    def forward(self, **kwargs):
        if self._hook is None:
            layer = self.base.transformer.decoder.layers[-1]
            self._hook = layer.register_forward_hook(self._capture)
        out = self.base(**kwargs)
        hidden = self._hidden
        if hidden.shape[0] != out["pred_spans"].shape[0]:
            hidden = hidden.permute(1, 0, 2)
        out["quality_scores"] = self.quality_head(hidden)
        return out


def temporal_iou(cs, ce, gs, ge):
    inter = max(0.0, min(ce, ge) - max(cs, gs))
    union = max(ce, ge) - min(cs, gs)
    return inter / union if union > 0 else 0.0


def load_gt(path):
    return {
        str(row["qid"]): [[float(s), float(e)] for s, e in row.get("relevant_windows", [])]
        for row in load_jsonl(path)
    }


def collect_predictions(model, loader, device, score_type, score_rule, alpha=0.5):
    model.eval()
    rows = []
    with torch.no_grad():
        for batch in loader:
            meta = batch[0]
            model_inputs, _ = prepare_batch_inputs(batch[1], device)
            out = model(**model_inputs)
            spans = out["pred_spans"].detach().cpu()
            logits = out["pred_logits"].detach().cpu()
            if score_rule == "softmax":
                confidence = F.softmax(logits, dim=-1)[..., 0]
            else:
                confidence = torch.sigmoid(logits[..., 0])
            quality = out.get("quality_scores")
            quality = torch.sigmoid(quality.detach().cpu()) if quality is not None else confidence
            if score_type == "confidence":
                scores = confidence
            elif score_type == "quality":
                scores = quality
            else:
                scores = alpha * confidence + (1.0 - alpha) * quality
            for idx, item in enumerate(meta):
                rows.append({
                    "qid": str(item["qid"]),
                    "duration": float(item["duration"]),
                    "spans": spans[idx].numpy(),
                    "scores": scores[idx].numpy(),
                })
    return rows


def comparable_pairwise(score_rows, iou_rows):
    correct = 0.0
    total = 0
    score_ties = 0
    for scores, ious in zip(score_rows, iou_rows):
        for i in range(len(scores)):
            for j in range(i + 1, len(scores)):
                di = float(ious[i] - ious[j])
                if abs(di) < PAIR_EPS:
                    continue
                ds = float(scores[i] - scores[j])
                total += 1
                if ds * di > 0:
                    correct += 1.0
                elif ds == 0:
                    correct += 0.5
                    score_ties += 1
    return {
        "accuracy": correct / total if total else 0.0,
        "comparable_pairs": total,
        "score_ties": score_ties,
        "iou_difference_threshold": PAIR_EPS,
    }


def metrics(rows, gt_map):
    score_rows = []
    iou_rows = []
    selected_ious = []
    flat_scores = []
    flat_ious = []
    for row in rows:
        gts = gt_map.get(row["qid"], [])
        if not gts:
            continue
        spans = span_cxw_to_xx(torch.as_tensor(row["spans"])).numpy() * row["duration"]
        ious = np.asarray([
            max(temporal_iou(float(s), float(e), gs, ge) for gs, ge in gts)
            for s, e in spans
        ])
        scores = np.asarray(row["scores"])
        selected_ious.append(float(ious[int(np.argmax(scores))]))
        score_rows.append(scores)
        iou_rows.append(ious)
        flat_scores.extend(scores.tolist())
        flat_ious.extend(ious.tolist())
    return {
        "n_queries": len(selected_ious),
        "r1_07": float(np.mean(np.asarray(selected_ious) >= 0.7) * 100),
        "spearman": float(spearmanr(flat_scores, flat_ious).statistic),
        "pairwise": comparable_pairwise(score_rows, iou_rows),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--component", choices=["bce", "joint"], required=True)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manager = BaseOptions(str(REPO / "config_finetune_from_clotho_official_run.yml"))
    manager.parse()
    opt = manager.option
    dataset = StartEndDataset(
        data_path=opt.test_path,
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
    )
    loader = DataLoader(
        dataset,
        collate_fn=start_end_collate,
        batch_size=opt.eval_bsz,
        num_workers=0,
        shuffle=False,
    )
    gt_map = load_gt(opt.test_path)
    base_state = torch.load(BASE_CKPT, map_location=opt.device, weights_only=False)["model"]

    host, _, _, _ = setup_model(opt)
    host.load_state_dict(base_state)
    host_metrics = metrics(
        collect_predictions(host, loader, opt.device, "confidence", "sigmoid"), gt_map
    )
    print(
        f"Host R1={host_metrics['r1_07']:.2f} "
        f"pair={host_metrics['pairwise']['accuracy']:.4f}", flush=True
    )

    if args.component == "bce":
        source = json.loads((BCE_DIR / "06_bce_iou_multiseed_results.json").read_text())
        config_by_seed = {int(row["seed"]): row for row in source["per_seed"]}
    else:
        source = json.loads((JOINT_DIR / "05_joint_multiseed_results.json").read_text())
        config_by_seed = {int(row["seed"]): row for row in source["per_seed"]}

    results = []
    for seed in SEEDS:
        config = config_by_seed[seed]
        base, _, _, _ = setup_model(opt)
        base.load_state_dict(base_state)
        if args.component == "bce":
            checkpoint = torch.load(
                BCE_DIR / "checkpoints" / f"bce_iou_seed{seed}_best.pth",
                map_location=opt.device,
                weights_only=False,
            )
            base.load_state_dict(checkpoint["model"])
            model = base
            score_type = "confidence"
            alpha = 0.5
            score_rule = checkpoint.get("score_rule", config["score_rule"])
        else:
            model = JointModel(base, hidden_dim=opt.hidden_dim).to(opt.device)
            checkpoint = torch.load(
                JOINT_DIR / "checkpoints" / f"joint_full_seed{seed}_best.pth",
                map_location=opt.device,
                weights_only=False,
            )
            model.load_state_dict(checkpoint["model"])
            score_type = config["score_type"]
            alpha = float(config["alpha"])
            score_rule = config["score_rule"]

        result_metrics = metrics(
            collect_predictions(model, loader, opt.device, score_type, score_rule, alpha),
            gt_map,
        )
        results.append({
            "seed": seed,
            "score_type": score_type,
            "alpha": alpha,
            "score_rule": score_rule,
            **result_metrics,
        })
        print(
            f"Seed {seed} R1={result_metrics['r1_07']:.2f} "
            f"sp={result_metrics['spearman']:.4f} "
            f"pair={result_metrics['pairwise']['accuracy']:.4f}",
            flush=True,
        )

    summary = {
        "component": args.component,
        "pairwise_definition": {
            "scope": "within query",
            "comparable_if_abs_iou_difference_at_least": PAIR_EPS,
            "score_tie_credit": 0.5,
        },
        "host": host_metrics,
        "per_seed": results,
        "five_seed": {
            "mean_r1_07": float(np.mean([row["r1_07"] for row in results])),
            "mean_spearman": float(np.mean([row["spearman"] for row in results])),
            "mean_pairwise": float(np.mean([row["pairwise"]["accuracy"] for row in results])),
            "std_pairwise": float(np.std([row["pairwise"]["accuracy"] for row in results])),
        },
    }
    output = OUT_DIR / f"r4_{args.component}_pairwise_audit.json"
    output.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary["five_seed"], indent=2), flush=True)
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()
