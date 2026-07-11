#!/usr/bin/env python3
"""BQL-final Phase 3: Structure ablation.

Variants:
  H: quality head only (freeze everything else)
  P: prediction heads (class/span) + quality head (freeze decoder)
  Dec: decoder + quality head (freeze prediction heads)
  D: decoder + prediction heads + quality head (full, already done)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from easydict import EasyDict
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

REPO = Path("/root/autodl-tmp/bql_final_20260705/dcase2026_task6_bql_final")
sys.path.insert(0, str(REPO / "src"))

from basic_utils import load_jsonl
from config import BaseOptions
from dataset import StartEndDataset, start_end_collate, prepare_batch_inputs
from evaluate import setup_model
from span_utils import span_cxw_to_xx

OUT_DIR = Path("/root/autodl-tmp/bql_final_20260705/outputs")
CKPT_PATH = REPO / "best_checkpoint.pth"


def build_dataset(opt, split):
    data_path = {"train": opt.train_path, "val": opt.val_path, "test": opt.test_path}[split]
    return StartEndDataset(**EasyDict(
        data_path=data_path, ctx_mode=opt.ctx_mode, a_feat_dir=opt.a_feat_dir,
        q_feat_dir=opt.t_feat_dir, q_feat_type="last_hidden_state",
        a_feat_type=opt.a_feat_type, max_q_l=opt.max_q_l, max_a_l=opt.max_a_l,
        clip_len=opt.clip_length, max_windows=opt.max_windows,
        span_loss_type=opt.span_loss_type, load_labels=True,
    ))


def load_gt(data_path):
    gt = {}
    for r in load_jsonl(data_path):
        gt[str(r["qid"])] = [[float(s), float(e)] for s, e in r.get("relevant_windows", [])]
    return gt


def temporal_iou(cs, ce, gs, ge):
    inter = max(0, min(ce, ge) - max(cs, gs))
    union = max(ce, ge) - min(cs, gs)
    return inter / union if union > 0 else 0.0


def patch_criterion(criterion):
    criterion.bql_lambda_reg = 1.0
    criterion.bql_lambda_cls = 1.0
    criterion.bql_lambda_list = 0.5
    criterion.bql_tau = 0.1
    criterion.bql_hard_ce = False

    def loss_bql_patched(outputs, targets, indices, **kwargs):
        import torch.nn.functional as Fbql
        pred_spans = outputs["pred_spans"]
        quality_reg = outputs["quality_reg"]
        quality_logit = outputs["quality_logit"]
        from span_utils import span_cxw_to_xx
        B, N = pred_spans.shape[:2]
        q_targets = pred_spans.new_zeros((B, N))
        span_labels = targets["span_labels"] if "span_labels" in targets else targets
        for b in range(B):
            pred_xx = span_cxw_to_xx(pred_spans[b])
            tgt_spans = span_labels[b]["spans"] if b < len(span_labels) and "spans" in span_labels[b] else None
            if tgt_spans is None or len(tgt_spans) == 0:
                continue
            tgt_xx = span_cxw_to_xx(tgt_spans)
            ps = pred_xx.unsqueeze(1)
            ts = tgt_xx.unsqueeze(0)
            inter = (torch.minimum(ps[..., 1], ts[..., 1]) - torch.maximum(ps[..., 0], ts[..., 0])).clamp(min=0)
            union = (torch.maximum(ps[..., 1], ts[..., 1]) - torch.minimum(ps[..., 0], ts[..., 0])).clamp(min=1e-6)
            iou_mat = inter / union
            q_targets[b] = iou_mat.max(dim=1)[0]
        y_targets = (q_targets >= criterion.bql_iou_threshold).float()
        q_pred = torch.sigmoid(quality_reg)
        L_qreg = Fbql.smooth_l1_loss(q_pred, q_targets) * criterion.bql_lambda_reg
        L_qcls = Fbql.binary_cross_entropy_with_logits(quality_logit, y_targets) * criterion.bql_lambda_cls
        with torch.no_grad():
            soft_target = Fbql.softmax(q_targets / criterion.bql_tau, dim=1)
        log_pred = Fbql.log_softmax(quality_logit, dim=1)
        L_list = -(soft_target * log_pred).sum(dim=1).mean() * criterion.bql_lambda_list
        return {"loss_bql_reg": L_qreg, "loss_bql_cls": L_qcls, "loss_bql_list": L_list}

    criterion.loss_bql = loss_bql_patched


def freeze_variant(model, variant):
    """Freeze parameters based on variant."""
    for name, p in model.named_parameters():
        p.requires_grad = True

    if variant == "H":
        for name, p in model.named_parameters():
            if "quality_head" not in name:
                p.requires_grad = False
    elif variant == "P":
        for name, p in model.named_parameters():
            if "quality_head" not in name and "class_embed" not in name and "span_embed" not in name:
                p.requires_grad = False
    elif variant == "Dec":
        for name, p in model.named_parameters():
            if "quality_head" not in name and "transformer.decoder" not in name:
                p.requires_grad = False
    elif variant == "D":
        for name, p in model.named_parameters():
            if "input_aud_proj" in name or "input_txt_proj" in name or "transformer.encoder" in name:
                p.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable


@torch.no_grad()
def evaluate_val(model, loader, opt, gt_map):
    model.eval()
    queries = []
    for batch in tqdm(loader, desc="eval", leave=False):
        meta = batch[0]
        mi, _ = prepare_batch_inputs(batch[1], opt.device)
        out = model(**mi)
        ps = out["pred_spans"].cpu()
        prob = F.softmax(out["pred_logits"], -1)
        conf = prob[..., 0].cpu()
        qcls = torch.sigmoid(out["quality_logit"]).cpu()
        for i, m in enumerate(meta):
            qid = str(m["qid"])
            dur = m["duration"]
            sxx = span_cxw_to_xx(ps[i]) * dur
            gts = gt_map.get(qid, [])
            cands = []
            for slot in range(len(sxx)):
                iou = max(temporal_iou(float(sxx[slot, 0]), float(sxx[slot, 1]), gs, ge) for gs, ge in gts) if gts else 0.0
                cands.append({"conf": float(conf[i, slot]), "qcls": float(qcls[i, slot]), "iou": iou})
            queries.append(cands)

    best_r1 = 0
    best_sc = None
    for sc_name, alpha in [("conf", 0), ("qcls", 1), ("mix_025", 0.25), ("mix_05", 0.5), ("mix_075", 0.75)]:
        r1 = []
        for cands in queries:
            if sc_name == "conf":
                top = max(cands, key=lambda c: c["conf"])
            elif sc_name == "qcls":
                top = max(cands, key=lambda c: c["qcls"])
            else:
                top = max(cands, key=lambda c: (1 - alpha) * c["conf"] + alpha * c["qcls"])
            r1.append(top["iou"] >= 0.7)
        r1_val = float(np.mean(r1))
        if r1_val > best_r1:
            best_r1 = r1_val
            best_sc = sc_name
    return best_r1, best_sc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True, choices=["H", "P", "Dec", "D"])
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=2e-5)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    option_manager = BaseOptions(str(REPO / "config_finetune_from_clotho_official_run.yml"))
    option_manager.parse()
    opt = option_manager.option
    opt.eval_split_name = "val"

    train_ds = build_dataset(opt, "train")
    val_ds = build_dataset(opt, "val")

    model, criterion, _, _ = setup_model(opt)
    ckpt = torch.load(str(CKPT_PATH), map_location=opt.device, weights_only=False)
    model.load_state_dict(ckpt["model"], strict=False)
    patch_criterion(criterion)

    lr = args.lr if args.variant != "H" else 1e-4
    trainable = freeze_variant(model, args.variant)
    params = [p for p in model.parameters() if p.requires_grad]
    print(f"=== {args.variant} trainable={trainable} lr={lr} ===", flush=True)

    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    train_loader = DataLoader(train_ds, collate_fn=start_end_collate, batch_size=opt.bsz,
                              num_workers=opt.num_workers, shuffle=True)
    val_loader = DataLoader(val_ds, collate_fn=start_end_collate, batch_size=opt.eval_bsz,
                            num_workers=opt.num_workers, shuffle=False)
    val_gt = load_gt(opt.val_path)

    best_val_r1 = 0
    best_scoring = None
    best_state = None

    for epoch in trange(args.epochs, desc=f"{args.variant}"):
        model.train()
        criterion.train()
        for batch in tqdm(train_loader, desc=f"e{epoch}", leave=False):
            mi, targets = prepare_batch_inputs(batch[1], opt.device)
            out = model(**mi)
            loss_dict = criterion(out, targets)
            losses = sum(loss_dict[k] * criterion.weight_dict[k] for k in loss_dict.keys() if k in criterion.weight_dict)
            optimizer.zero_grad()
            losses.backward()
            nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()
        scheduler.step()

        if (epoch + 1) % 2 == 0 or epoch == args.epochs - 1:
            r1, sc = evaluate_val(model, val_loader, opt, val_gt)
            print(f"  e{epoch} val R1@0.7={r1*100:.2f}% scoring={sc}", flush=True)
            if r1 > best_val_r1:
                best_val_r1 = r1
                best_scoring = sc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state:
        save_path = OUT_DIR / f"phase3_struct_{args.variant}_best.pth"
        torch.save({"model": best_state, "val_r1": best_val_r1, "val_scoring": best_scoring, "variant": args.variant}, save_path)
        print(f"  [saved] {save_path} val_r1={best_val_r1*100:.2f}% scoring={best_scoring}", flush=True)

    payload = {"variant": args.variant, "seed": args.seed, "val_r1": best_val_r1, "val_scoring": best_scoring, "trainable": trainable}
    (OUT_DIR / f"phase3_struct_{args.variant}_summary.json").write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
