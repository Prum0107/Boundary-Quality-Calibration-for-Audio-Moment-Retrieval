#!/usr/bin/env python3
"""BQL-Dec+BR: Phase 1 BQL-Dec training, Phase 2 multi-seed, Phase 3-6 BR.

BQL-Dec: decoder + quality head trainable, prediction heads frozen.
BR: adds refinement head for start/end boundary correction.
"""
import argparse
import json
import sys
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from easydict import EasyDict
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

REPO = Path("/root/autodl-tmp/bql_dec_br_20260705/dcase2026_task6_bql_dec_br")
sys.path.insert(0, str(REPO / "src"))

from basic_utils import load_jsonl
from config import BaseOptions
from dataset import StartEndDataset, start_end_collate, prepare_batch_inputs
from evaluate import setup_model
from span_utils import span_cxw_to_xx

OUT_DIR = Path("/root/autodl-tmp/bql_dec_br_20260705/outputs")
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


def patch_criterion(criterion, use_br=False, lambda_ref=0.5, br_eligible=0.3):
    criterion.bql_lambda_reg = 1.0
    criterion.bql_lambda_cls = 1.0
    criterion.bql_lambda_list = 0.5
    criterion.bql_tau = 0.1
    criterion.bql_hard_ce = False
    criterion.bql_use_br = use_br
    criterion.bql_lambda_ref = lambda_ref
    criterion.bql_br_eligible = br_eligible

    def loss_bql_patched(outputs, targets, indices, **kwargs):
        import torch.nn.functional as Fbql
        pred_spans = outputs["pred_spans"]
        quality_reg = outputs["quality_reg"]
        quality_logit = outputs["quality_logit"]
        from span_utils import span_cxw_to_xx
        B, N = pred_spans.shape[:2]
        q_targets = pred_spans.new_zeros((B, N))
        span_labels = targets["span_labels"] if "span_labels" in targets else targets
        best_gt_starts = pred_spans.new_zeros((B, N))
        best_gt_ends = pred_spans.new_zeros((B, N))
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
            best_idx = iou_mat.argmax(dim=1)
            best_gt_starts[b] = tgt_xx[best_idx, 0]
            best_gt_ends[b] = tgt_xx[best_idx, 1]
        y_targets = (q_targets >= criterion.bql_iou_threshold).float()
        q_pred = torch.sigmoid(quality_reg)
        L_qreg = Fbql.smooth_l1_loss(q_pred, q_targets) * criterion.bql_lambda_reg
        L_qcls = Fbql.binary_cross_entropy_with_logits(quality_logit, y_targets) * criterion.bql_lambda_cls
        with torch.no_grad():
            soft_target = Fbql.softmax(q_targets / criterion.bql_tau, dim=1)
        log_pred = Fbql.log_softmax(quality_logit, dim=1)
        L_list = -(soft_target * log_pred).sum(dim=1).mean() * criterion.bql_lambda_list

        result = {"loss_bql_reg": L_qreg, "loss_bql_cls": L_qcls, "loss_bql_list": L_list}

        if criterion.bql_use_br and "refined_spans" in outputs:
            refined = outputs["refined_spans"]
            eligible = (q_targets >= criterion.bql_br_eligible).float()
            n_eligible = eligible.sum().clamp(min=1)
            L_ref_start = Fbql.smooth_l1_loss(refined[..., 0], best_gt_starts, reduction="none")
            L_ref_end = Fbql.smooth_l1_loss(refined[..., 1], best_gt_ends, reduction="none")
            L_ref = L_ref_start + L_ref_end
            L_ref = (L_ref * eligible).sum() / n_eligible * criterion.bql_lambda_ref
            result["loss_bql_ref"] = L_ref
        else:
            result["loss_bql_ref"] = quality_reg.sum() * 0.0

        return result

    criterion.loss_bql = loss_bql_patched


def add_refinement_head(model):
    """Add boundary refinement head to QDDETR."""
    hidden_dim = model.hidden_dim
    model.refinement_head = nn.Sequential(
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(hidden_dim, 2),
    )
    device = next(model.parameters()).device
    model.refinement_head = model.refinement_head.to(device)
    model.bql_max_delta = 0.20

    orig_forward = model.forward

    def patched_forward(src_txt, src_txt_mask, src_aud, src_aud_mask):
        out = orig_forward(src_txt, src_txt_mask, src_aud, src_aud_mask)
        with torch.no_grad():
            hs = None
            if hasattr(model, '_captured_hs'):
                hs = model._captured_hs
        if hs is not None:
            raw_delta = model.refinement_head(hs)
            delta = model.bql_max_delta * torch.tanh(raw_delta)
            pred_spans = out["pred_spans"]
            refined_start = (pred_spans[..., 0] - pred_spans[..., 1] / 2 + delta[..., 0]).clamp(0, 1)
            refined_end = (pred_spans[..., 0] + pred_spans[..., 1] / 2 + delta[..., 1]).clamp(0, 1)
            min_dur = 0.01
            too_short = (refined_end - refined_start) < min_dur
            refined_end = torch.where(too_short, refined_start + min_dur, refined_end)
            out["refined_spans"] = torch.stack([refined_start, refined_end], dim=-1)
            out["refinement_delta"] = delta
        return out

    model.forward = patched_forward

    orig_transformer_hook = getattr(model, '_bql_dec_hook', None)


def add_hs_capture(model):
    """Capture last decoder hidden state via hook."""
    captured = {}

    def hook(module, inputs, output):
        hs = output[0]
        if hs.dim() == 4:
            hs = hs[-1]
        elif hs.dim() == 3:
            hs = hs.permute(1, 0, 2)
        captured["hs"] = hs.detach()

    handle = model.transformer.register_forward_hook(hook)
    model._captured_hs = None
    model._bql_dec_hook = handle

    orig_forward = model.forward

    def patched_forward(src_txt, src_txt_mask, src_aud, src_aud_mask):
        out = orig_forward(src_txt, src_txt_mask, src_aud, src_aud_mask)
        model._captured_hs = captured.get("hs")
        return out

    model.forward = patched_forward


def freeze_variant(model, variant):
    for name, p in model.named_parameters():
        p.requires_grad = False

    if variant == "Dec":
        for name, p in model.named_parameters():
            if "transformer.decoder" in name or "quality_head" in name:
                p.requires_grad = True
    elif variant == "Dec_BR":
        for name, p in model.named_parameters():
            if "transformer.decoder" in name or "quality_head" in name or "refinement_head" in name:
                p.requires_grad = True

    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def evaluate_val(model, loader, opt, gt_map, use_br=False):
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
        qreg = torch.sigmoid(out["quality_reg"]).cpu()
        refined = out.get("refined_spans", ps).cpu() if use_br else ps

        for i, m in enumerate(meta):
            qid = str(m["qid"])
            dur = m["duration"]
            sxx = span_cxw_to_xx(ps[i]) * dur
            rxx = span_cxw_to_xx(refined[i]) * dur if use_br else sxx
            gts = gt_map.get(qid, [])
            cands = []
            for slot in range(len(sxx)):
                orig_iou = max(temporal_iou(float(sxx[slot, 0]), float(sxx[slot, 1]), gs, ge) for gs, ge in gts) if gts else 0.0
                ref_iou = max(temporal_iou(float(rxx[slot, 0]), float(rxx[slot, 1]), gs, ge) for gs, ge in gts) if gts else 0.0
                cands.append({"conf": float(conf[i, slot]), "qcls": float(qcls[i, slot]),
                              "qreg": float(qreg[i, slot]), "orig_iou": orig_iou, "ref_iou": ref_iou})
            queries.append(cands)

    scorings = ["conf", "qreg", "qcls", "mix_025", "mix_05", "mix_075", "logmix_05"]
    results = {}
    for sc_name in scorings:
        r1_orig = []
        r1_ref = []
        miou_orig = []
        miou_ref = []
        for cands in queries:
            if sc_name == "conf":
                top = max(cands, key=lambda c: c["conf"])
            elif sc_name == "qreg":
                top = max(cands, key=lambda c: c["qreg"])
            elif sc_name == "qcls":
                top = max(cands, key=lambda c: c["qcls"])
            elif sc_name.startswith("mix_"):
                a = float(sc_name.split("_")[1]) / 100.0
                top = max(cands, key=lambda c: (1 - a) * c["conf"] + a * c["qcls"])
            else:
                eps = 1e-7
                top = max(cands, key=lambda c: np.log(c["conf"] + eps) + 0.5 * np.log(c["qcls"] + eps))
            r1_orig.append(top["orig_iou"] >= 0.7)
            r1_ref.append(top["ref_iou"] >= 0.7)
            miou_orig.append(top["orig_iou"])
            miou_ref.append(top["ref_iou"])
        results[sc_name] = {
            "r1_07_orig": float(np.mean(r1_orig)),
            "r1_07_ref": float(np.mean(r1_ref)),
            "miou_orig": float(np.mean(miou_orig)),
            "miou_ref": float(np.mean(miou_ref)),
        }

    best_sc = max(results.keys(), key=lambda k: results[k]["r1_07_ref" if use_br else "r1_07_orig"])
    best_r1 = results[best_sc]["r1_07_ref" if use_br else "r1_07_orig"]
    return best_r1, best_sc, results


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)


def train_variant(variant, seed, opt, train_ds, val_ds, epochs, lr, use_br=False, lambda_ref=0.5, br_eligible=0.3):
    set_seed(seed)
    model, criterion, _, _ = setup_model(opt)
    ckpt = torch.load(str(CKPT_PATH), map_location=opt.device, weights_only=False)
    model.load_state_dict(ckpt["model"], strict=False)

    add_hs_capture(model)
    if use_br:
        add_refinement_head(model)

    patch_criterion(criterion, use_br, lambda_ref, br_eligible)
    trainable = freeze_variant(model, variant)
    params = [p for p in model.parameters() if p.requires_grad]
    print(f"=== {variant} seed{seed} trainable={trainable} br={use_br} ===", flush=True)

    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    train_loader = DataLoader(train_ds, collate_fn=start_end_collate, batch_size=opt.bsz,
                              num_workers=opt.num_workers, shuffle=True)
    val_loader = DataLoader(val_ds, collate_fn=start_end_collate, batch_size=opt.eval_bsz,
                            num_workers=opt.num_workers, shuffle=False)
    val_gt = load_gt(opt.val_path)

    best_val_r1 = 0
    best_scoring = None
    best_state = None

    for epoch in trange(epochs, desc=f"{variant} s{seed}"):
        model.train()
        criterion.train()
        for batch in tqdm(train_loader, desc=f"e{epoch}", leave=False):
            mi, targets = prepare_batch_inputs(batch[1], opt.device)
            out = model(**mi)
            loss_dict = criterion(out, targets)
            losses = sum(loss_dict[k] * criterion.weight_dict.get(k, 1.0) for k in loss_dict.keys())
            optimizer.zero_grad()
            losses.backward()
            nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()
        scheduler.step()

        if (epoch + 1) % 2 == 0 or epoch == epochs - 1:
            r1, sc, _ = evaluate_val(model, val_loader, opt, val_gt, use_br)
            print(f"  e{epoch} val R1@0.7={r1*100:.2f}% scoring={sc}", flush=True)
            if r1 > best_val_r1:
                best_val_r1 = r1
                best_scoring = sc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state:
        save_path = OUT_DIR / f"phase1_bql_dec{'_br' if use_br else ''}_seed{seed}_best.pth"
        torch.save({"model": best_state, "val_r1": best_val_r1, "val_scoring": best_scoring,
                    "variant": variant, "seed": seed, "use_br": use_br}, save_path)
        print(f"  [saved] {save_path} val_r1={best_val_r1*100:.2f}% scoring={best_scoring}", flush=True)

    return best_state, best_val_r1, best_scoring


@torch.no_grad()
def run_test(model, loader, opt, gt_map, scoring, use_br=False):
    model.eval()
    host_ious = []
    pred_ious = []
    for batch in tqdm(loader, desc="test", leave=False):
        meta = batch[0]
        mi, _ = prepare_batch_inputs(batch[1], opt.device)
        out = model(**mi)
        ps = out["pred_spans"].cpu()
        prob = F.softmax(out["pred_logits"], -1)
        conf = prob[..., 0].cpu()
        qcls = torch.sigmoid(out["quality_logit"]).cpu()
        refined = out.get("refined_spans", ps).cpu() if use_br else ps

        for i, m in enumerate(meta):
            qid = str(m["qid"])
            dur = m["duration"]
            sxx = span_cxw_to_xx(ps[i]) * dur
            rxx = span_cxw_to_xx(refined[i]) * dur if use_br else sxx
            gts = gt_map.get(qid, [])

            host_top = max(range(len(sxx)), key=lambda s: float(conf[i, s]))
            hs, he = float(sxx[host_top, 0]), float(sxx[host_top, 1])
            host_ious.append(max(temporal_iou(hs, he, gs, ge) for gs, ge in gts) if gts else 0.0)

            if scoring == "conf":
                bi = host_top
            elif scoring == "qcls":
                bi = max(range(len(sxx)), key=lambda s: float(qcls[i, s]))
            elif scoring.startswith("mix_"):
                a = float(scoring.split("_")[1]) / 100.0
                bi = max(range(len(sxx)), key=lambda s: (1 - a) * float(conf[i, s]) + a * float(qcls[i, s]))
            else:
                bi = host_top

            bs, be = float(rxx[bi, 0]), float(rxx[bi, 1])
            pred_ious.append(max(temporal_iou(bs, be, gs, ge) for gs, ge in gts) if gts else 0.0)

    return np.array(host_ious), np.array(pred_ious)


def bootstrap_ci(host, pred, n=1000, seed=0):
    rng = np.random.RandomState(seed)
    nq = len(host)
    deltas = []
    for _ in range(n):
        idx = rng.randint(0, nq, nq)
        deltas.append(float((pred[idx] >= 0.7).mean()) - float((host[idx] >= 0.7).mean()))
    deltas = np.array(deltas)
    return {"mean": float(deltas.mean()), "ci_low": float(np.percentile(deltas, 2.5)),
            "ci_high": float(np.percentile(deltas, 97.5)),
            "crosses0": bool(np.percentile(deltas, 2.5) <= 0 <= np.percentile(deltas, 97.5))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=int, default=1)
    parser.add_argument("--variant", default="Dec", choices=["Dec", "Dec_BR"])
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--use-br", action="store_true")
    parser.add_argument("--lambda-ref", type=float, default=0.5)
    parser.add_argument("--br-eligible", type=float, default=0.3)
    parser.add_argument("--run-test", action="store_true")
    parser.add_argument("--scoring", default="")
    args = parser.parse_args()

    option_manager = BaseOptions(str(REPO / "config_finetune_from_clotho_official_run.yml"))
    option_manager.parse()
    opt = option_manager.option
    opt.eval_split_name = "val"

    train_ds = build_dataset(opt, "train")
    val_ds = build_dataset(opt, "val")

    use_br = args.use_br or args.variant == "Dec_BR"
    best_state, best_val_r1, best_scoring = train_variant(
        args.variant, args.seed, opt, train_ds, val_ds, args.epochs, args.lr,
        use_br, args.lambda_ref, args.br_eligible
    )

    print(f"\n=== seed{args.seed} val results ===", flush=True)
    print(f"  best val R1@0.7: {best_val_r1*100:.2f}% (scoring: {best_scoring})", flush=True)
    print(f"  unified host val R1@0.7: 21.02%", flush=True)
    print(f"  delta vs unified host: {(best_val_r1 - 0.2102)*100:+.2f}%", flush=True)
    print(f"  BQL-D final seed2026 val: 23.01%", flush=True)
    print(f"  delta vs BQL-D: {(best_val_r1 - 0.2301)*100:+.2f}%", flush=True)

    if args.run_test:
        model, _, _, _ = setup_model(opt)
        add_hs_capture(model)
        if use_br:
            add_refinement_head(model)
        model.load_state_dict(best_state, strict=False)

        opt.eval_split_name = "test"
        test_ds = build_dataset(opt, "test")
        test_loader = DataLoader(test_ds, collate_fn=start_end_collate, batch_size=opt.eval_bsz,
                                 num_workers=opt.num_workers, shuffle=False)
        test_gt = load_gt(opt.test_path)

        scoring = args.scoring if args.scoring else best_scoring
        host, pred = run_test(model, test_loader, opt, test_gt, scoring, use_br)
        boot = bootstrap_ci(host, pred)
        host_r1 = float((host >= 0.7).mean())
        pred_r1 = float((pred >= 0.7).mean())
        delta = pred_r1 - host_r1
        print(f"\n=== seed{args.seed} test (scoring={scoring}, br={use_br}) ===", flush=True)
        print(f"  host R1@0.7: {host_r1*100:.2f}%", flush=True)
        print(f"  BQL R1@0.7: {pred_r1*100:.2f}% delta={delta*100:+.2f}%", flush=True)
        print(f"  CI=[{boot['ci_low']*100:+.2f},{boot['ci_high']*100:+.2f}] cross0={boot['crosses0']}", flush=True)

        payload = {"seed": args.seed, "variant": args.variant, "use_br": use_br,
                   "val_r1": best_val_r1, "val_scoring": best_scoring,
                   "test_scoring": scoring, "host_r1": host_r1, "bql_r1": pred_r1,
                   "delta": delta, "bootstrap": boot}
        (OUT_DIR / f"phase{args.phase}_seed{args.seed}_results.json").write_text(json.dumps(payload, indent=2))
    else:
        payload = {"seed": args.seed, "variant": args.variant, "use_br": use_br,
                   "val_r1": best_val_r1, "val_scoring": best_scoring}
        (OUT_DIR / f"phase{args.phase}_seed{args.seed}_results.json").write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
