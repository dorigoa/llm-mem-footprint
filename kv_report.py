#!/usr/bin/env python3
"""
kv_report.py - fixed, minimal memory report for one or more models.

For every model it prints, and nothing else:
  - KV cache size at 65k / 128k / 256k tokens, in 16-bit (f16) and 8-bit (q8_0)
  - model weights size at 8-bit (q8_0) and 4-bit (q4_k_m)
  - the extra memory needed to run the model (llama.cpp compute buffer,
    computed from the config for the default batch sizes -b 2048 -ub 512;
    the fixed recurrent states of linear/SSM layers are already included
    in the KV lines)

All the parsing/estimation logic comes from the kv_memory library (it must
sit next to this script). Standard library only, Python >= 3.8.

Usage:
  python3 kv-size-2.py <vendor>/<model> [more ...]
  python3 kv-size-2.py /path/to/model_folder [more ...]
"""
from __future__ import annotations

import sys
from pathlib import Path

import kv_memory as kv

GIB = 2 ** 30
MIB = 2 ** 20

# Fixed output
CTX_SIZES = [("65k", 65 * 1024), ("128k", 128 * 1024), ("256k", 256 * 1024)]
KV_TYPES = [("16-bit", "f16"), ("8-bit", "q8_0")]          # cache K/V types
WEIGHTS = [("8-bit (q8_0)", 8.5), ("4-bit (q4_k_m)", 4.9)]  # label, bits/weight
BATCH, UBATCH = 2048, 512   # llama.cpp defaults -b / -ub


def g(*vals):
    for v in vals:
        if isinstance(v, int) and not isinstance(v, bool) and v > 0:
            return v
    return None


def compute_buffer_gib(tc):
    """llama.cpp compute buffer, the largest per-batch allocations (all fp16):

      prompt phase  : 2 * n_batch  * hidden            (input_ids + logits)
      token phase   : 2 * n_ubatch * hidden * n_steps  (recompute, graph)
      MoE FFN       : 2 * n_ubatch * intermediate_expert

    plus ~15% for the many smaller per-batch buffers (logits, rope, norms,
    graph workspace). The KV cache and the model weights are not included.
    """
    h = g(tc.get("hidden_size"), tc.get("n_embd"), tc.get("d_model"), tc.get("dim"))
    if not h:
        return None
    inter = g(tc.get("intermediate_size"), tc.get("ffn_dim"))
    moe_inter = g(tc.get("moe_intermediate_size"))
    inter = moe_inter if moe_inter else inter

    # prompt phase: the largest single buffer is the hidden states
    prompt = 2 * max(BATCH, UBATCH) * h
    # token phase: FFN/MoE intermediate is the largest per-token buffer
    token = 2 * UBATCH * (inter or h)
    total = max(prompt, token) + 2 * UBATCH * h   # + one hidden buffer for logits
    return total * 2 * 1.15 / GIB    # fp16, +15% safety margin


def logits_buf_gib(tc):
    """The output/logit matrix is kept at fp16 by llama.cpp no matter the
    weights quantization: the temporary buffer is n_batch x vocab (the
    weights themselves are already counted in 'weights')."""
    v = g(tc.get("vocab_size"))
    if not v:
        return 0.0
    return max(BATCH, UBATCH) * v * 2 / GIB


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        print(f"usage: python3 {Path(sys.argv[0]).name} <vendor>/<model> [more ...] "
              f"[/path/to/folder ...]", file=sys.stderr)
        return 1

    token = kv.get_token(None)

    failed = 0
    for i, spec_name in enumerate(argv):
        if i:
            print()
        try:
            cfg, _src = kv.load_config(spec_name, "main", token)
            r = kv.analyze(cfg)
            tc, _path = kv.pick_text_config(cfg)
        except kv.ConfigError as exc:
            failed += 1
            print(f"{spec_name}: ERROR: {exc}", file=sys.stderr)
            continue
        except (TypeError, ValueError, KeyError, AttributeError) as exc:
            failed += 1
            print(f"{spec_name}: ERROR: unexpected config structure "
                  f"({type(exc).__name__}: {exc})", file=sys.stderr)
            continue

        print(f"{spec_name}:")
        for label, n_ctx in CTX_SIZES:
            parts = []
            for qlabel, qt in KV_TYPES:
                e = kv.estimate(r, n_ctx, qt, qt, UBATCH)
                parts.append(f"{qlabel}: {e['total_bytes'] / GIB:.2f} GiB")
            print(f"  KV cache {label:>4}: " + "  |  ".join(parts))
        try:
            pc = kv.param_count(spec_name, "main", token)
            w8 = pc["total"] * WEIGHTS[0][1] / 8
            w4 = pc["total"] * WEIGHTS[1][1] / 8
            print(f"  weights {WEIGHTS[0][0]:>13}: {w8 / GIB:.2f} GiB")
            print(f"  weights {WEIGHTS[1][0]:>13}: {w4 / GIB:.2f} GiB")
        except kv.ConfigError as exc:
            print(f"  weights: n/a ({exc})")
        cb = compute_buffer_gib(tc)
        if cb is None:
            print("  extra to run the model: hidden_size not found in the config")
        else:
            lg = logits_buf_gib(tc)
            extra = f" (incl. {lg:.2f} GiB fp16 logits buffer)" if lg else ""
            print(f"  extra to run the model (compute buffer, -b {BATCH} -ub {UBATCH}): "
                  f"~{cb + lg:.2f} GiB{extra}")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
