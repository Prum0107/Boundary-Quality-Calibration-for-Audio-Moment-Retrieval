#!/usr/bin/env python3
"""Audit BQC-Dec parameter scope and checkpoint changes without training."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import torch


def load_state(path: Path) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("model", payload)
    return {name: value for name, value in state.items() if torch.is_tensor(value)}


def group(name: str) -> str:
    if name.startswith("query_embed"):
        return "moment_queries"
    if "transformer.decoder" in name:
        return "decoder"
    if "quality_head" in name:
        return "quality_head"
    if name.startswith("span_embed") or name.startswith("class_embed"):
        return "prediction_heads"
    if "transformer.encoder" in name:
        return "encoder"
    return "other_frozen"


def policy_trainable(name: str) -> bool:
    return "transformer.decoder" in name or "quality_head" in name


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--bqc", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    base = load_state(args.base)
    bqc = load_state(args.bqc)
    names = sorted(set(base) | set(bqc))
    summary: dict[str, dict[str, float | int | bool]] = defaultdict(
        lambda: {
            "tensors": 0,
            "parameters": 0,
            "common_tensors": 0,
            "changed_tensors": 0,
            "delta_l2_sq": 0.0,
            "base_l2_sq": 0.0,
            "max_abs_delta": 0.0,
        }
    )
    details = []

    for name in names:
        category = group(name)
        source = bqc.get(name, base.get(name))
        row = summary[category]
        row["tensors"] += 1
        row["parameters"] += int(source.numel())
        row["policy_trainable"] = policy_trainable(name)
        if name in base and name in bqc and base[name].shape == bqc[name].shape:
            delta = (bqc[name].float() - base[name].float()).reshape(-1)
            max_abs = float(delta.abs().max()) if delta.numel() else 0.0
            delta_sq = float(torch.dot(delta, delta))
            base_flat = base[name].float().reshape(-1)
            row["common_tensors"] += 1
            row["delta_l2_sq"] += delta_sq
            row["base_l2_sq"] += float(torch.dot(base_flat, base_flat))
            row["max_abs_delta"] = max(float(row["max_abs_delta"]), max_abs)
            if max_abs > 0:
                row["changed_tensors"] += 1
            details.append(
                {
                    "name": name,
                    "group": category,
                    "policy_trainable": policy_trainable(name),
                    "max_abs_delta": max_abs,
                    "delta_l2": math.sqrt(delta_sq),
                }
            )
        else:
            details.append(
                {
                    "name": name,
                    "group": category,
                    "policy_trainable": policy_trainable(name),
                    "status": "only_in_" + ("bqc" if name in bqc else "base"),
                }
            )

    for row in summary.values():
        row["delta_l2"] = math.sqrt(float(row.pop("delta_l2_sq")))
        row["base_l2"] = math.sqrt(float(row.pop("base_l2_sq")))

    result = {
        "base_checkpoint": str(args.base),
        "bqc_checkpoint": str(args.bqc),
        "trainable_policy": ["transformer.decoder", "quality_head"],
        "summary": dict(summary),
        "important_parameters": {
            name: next((item for item in details if item["name"] == name), None)
            for name in ("query_embed.weight",)
        },
        "details": details,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("BQC-Dec parameter audit")
    for category, row in result["summary"].items():
        print(
            f"{category:18s} trainable={str(row['policy_trainable']):5s} "
            f"params={row['parameters']:9d} changed={row['changed_tensors']:3d}/"
            f"{row['common_tensors']:3d} delta_l2={row['delta_l2']:.6g}"
        )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
