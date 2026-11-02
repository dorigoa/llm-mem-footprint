#!/usr/bin/env python3
"""
kv_memory - library to estimate the KV cache and weights memory of a model
(as allocated by llama.cpp, `llama-server -c N`) from its config.json, fetched
from Hugging Face or read from a local folder.

Public API:
    load_config(spec, revision="main", token=None)
        -> (config_dict, source_description). spec is a <vendor>/<model> id,
        a local folder, or a config.json file.
    analyze(config)
        -> dict with layers, heads, head dims, KV groups, recurrent states,
        warnings, ...
    estimate(result, kv_size, type_k="f16", type_v="f16", ubatch=512)
        -> dict with the KV cache in bytes for `kv_size` context tokens
           (cache types: see CACHE_BYTES).
    param_count(spec, revision="main", token=None)
        -> {"total": int, "by_dtype": dict, "source": str} from the
        safetensors metadata (Hugging Face API or local file headers).
    weights_estimate(spec, result, quant, params_override, revision, token)
        -> dict with the weights size in bytes at `quant` = (name, bits/weight)
           (names: see QUANT_BPW), or {"error": msg} on failure.
    human(bytes)
        -> "12.34 GiB" / "512.0 MiB"
    ConfigError
        - raised for all user-readable failures

Gated or private models: pass `token`, or set the HF_TOKEN environment
variable, or use the token saved by `hf auth login`. If huggingface.co is
unreachable, the copy in the local Hugging Face cache
(~/.cache/huggingface/hub) is used, if present.

Standard library only, Python >= 3.8. CLI: see kv_report.py.
"""
from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ENDPOINT = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
TIMEOUT_S = 20
GIB = 2 ** 30
MIB = 2 ** 20

# Bytes per element of llama.cpp cache types (blocks of 32 values + fp16 scales)
CACHE_BYTES = {"f32": 4.0, "f16": 2.0, "bf16": 2.0, "q8_0": 34 / 32, "q5_1": 24 / 32,
               "q5_0": 22 / 32, "q4_1": 20 / 32, "q4_0": 18 / 32, "iq4_nl": 18 / 32}
STATE_BYTES = 4  # recurrent states (linear/SSM layers) are kept in f32

# Effective bits per weight of weight quantizations.
# Single-type formats: exact, bytes per block x 8 / weights per block (a lower bound
# for GGUF files, since llama-quantize keeps a few tensors at higher precision).
# llama.cpp mixed presets: approximate, from llama-quantize file sizes on Llama-3-8B.
# MLX affine quantization with group size 64: bits + 16-bit scale and bias per group.
QUANT_BPW = {
    "f32": 32.0, "f16": 16.0, "bf16": 16.0, "fp8": 8.0,
    "q8_0": 8.5, "q6_k": 6.5625, "q5_1": 6.0, "q5_0": 5.5, "q4_1": 5.0, "q4_0": 4.5,
    "iq4_nl": 4.5, "iq4_xs": 4.25, "mxfp4": 4.25, "iq3_s": 3.4375, "iq3_xxs": 3.0625,
    "iq2_s": 2.5625, "iq2_xs": 2.3125, "iq2_xxs": 2.0625, "iq1_m": 1.75, "iq1_s": 1.5625,
    "q5_k_m": 5.70, "q5_k_s": 5.57, "q4_k_m": 4.90, "q4_k_s": 4.67, "q3_k_l": 4.31,
    "q3_k_m": 4.00, "q3_k_s": 3.65, "q2_k": 3.17, "iq3_m": 3.66, "iq3_xs": 3.3, "iq2_m": 2.7,
    "mlx_8bit": 8.5, "mlx_6bit": 6.5, "mlx_5bit": 5.5, "mlx_4bit": 4.5, "mlx_3bit": 3.5,
    "mlx_2bit": 2.5,
}
QUANT_APPROX = {"q5_k_m", "q5_k_s", "q4_k_m", "q4_k_s", "q3_k_l", "q3_k_m", "q3_k_s",
                "q2_k", "iq3_m", "iq3_xs", "iq2_m"}
QUANT_ALIASES = {"q4_k": "q4_k_m", "q5_k": "q5_k_m", "q3_k": "q3_k_m", "fp16": "f16",
                 "fp32": "f32", "f8": "fp8", "mlx8": "mlx_8bit", "mlx6": "mlx_6bit",
                 "mlx5": "mlx_5bit", "mlx4": "mlx_4bit", "mlx3": "mlx_3bit", "mlx2": "mlx_2bit"}
# dtypes that pack several quantized weights per element (GPTQ/AWQ int32, MXFP4/bnb uint8)
PACKED_DTYPES = {"I32", "U32", "U8"}

# Alternative names for the same quantity across config formats
ALIASES = {
    "n_layers": ("num_hidden_layers", "n_layer", "n_layers", "num_layers"),
    "n_heads": ("num_attention_heads", "n_head", "n_heads", "num_heads"),
    "n_kv": ("num_key_value_heads", "num_kv_heads", "n_kv_heads", "n_head_kv",
             "multi_query_group_num", "num_query_groups"),
    "hidden": ("hidden_size", "n_embd", "d_model", "dim"),
    "head_dim": ("head_dim", "kv_channels", "attention_head_dim"),
    "max_ctx": ("max_position_embeddings", "max_sequence_length", "seq_length",
                "n_positions", "max_seq_len"),
}
# Sub-configs of multimodal models that hold the LLM part
NESTED = ("text_config", "llm_config", "language_config", "language_model",
          "text_model", "thinker_config")
# Keys interpreted by the script (not listed among the "check manually" ones)
USED = {k for v in ALIASES.values() for k in v} | {
    "layer_types", "layers_block_type", "hybrid_override_pattern", "attn_type_list",
    "full_attn_idxs", "attn_layer_period", "attn_layer_offset", "full_attention_interval",
    "sliding_window", "use_sliding_window", "sliding_window_pattern",
    "_sliding_window_pattern", "max_window_layers", "attention_chunk_size",
    "no_rope_layers", "no_rope_layer_interval", "num_kv_shared_layers", "kv_lora_rank",
    "qk_rope_head_dim", "qk_nope_head_dim", "v_head_dim", "multi_query", "model_type",
    "num_nextn_predict_layers",
}
RELEVANT_RE = re.compile(
    r"kv|key_value|head|window|attn|attention|layer_type|shared|chunk|lora|global|"
    r"local|linear|mamba|ssm|conv|state|nextn|mtp|index|swa|cache", re.I)
IRRELEVANT_RE = re.compile(
    r"dropout|bias|softcap|scal|multiplier|pdrop|implementation|initializer|eps$|norm|"
    r"_act$|activation|token|vocab|rope|temperature|dtype|q_lora_rank|expert|router|moe",
    re.I)

class ConfigError(Exception):
    """Error with a user-readable message."""


# ----------------------------------------------------------------- loading

def first(cfg, keys):
    for k in keys:
        v = cfg.get(k)
        if v is not None:
            return k, v
    return None, None


def hf_home():
    return Path(os.environ.get("HF_HOME") or Path.home() / ".cache" / "huggingface").expanduser()


def get_token(cli_token):
    if cli_token:
        return cli_token.strip()
    for var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        if os.environ.get(var):
            return os.environ[var].strip()
    path = Path(os.environ.get("HF_TOKEN_PATH") or hf_home() / "token").expanduser()
    try:
        return path.read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


def read_json_file(path):
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as e:
        raise ConfigError(f"cannot read {path}: {e}") from e
    if not isinstance(data, dict):
        raise ConfigError(f"{path} does not contain a JSON object")
    return data


def http_get_json(url, token):
    req = urllib.request.Request(url, headers={"User-Agent": "kv-size/1.1"})
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
        raw = resp.read()
    try:
        data = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise ConfigError(f"{url} is not valid JSON: {e}") from e
    if not isinstance(data, dict):
        raise ConfigError(f"{url} does not contain a JSON object")
    return data


def fetch_json(repo, revision, fname, token):
    rev = urllib.parse.quote(revision, safe="")
    url = f"{ENDPOINT}/{repo}/resolve/{rev}/{fname}"
    return http_get_json(url, token), url


def from_cache(repo, revision, fname):
    cache = Path(os.environ.get("HF_HUB_CACHE") or hf_home() / "hub").expanduser()
    base = cache / ("models--" + repo.replace("/", "--"))
    snaps = base / "snapshots"
    if not snaps.is_dir():
        return None
    candidates = [snaps / revision / fname]
    ref = base / "refs" / revision
    if ref.is_file():
        try:
            candidates.insert(0, snaps / ref.read_text().strip() / fname)
        except OSError:
            pass
    for c in candidates:
        if c.is_file():
            return c
    others = sorted((c for c in snaps.glob(f"*/{fname}") if c.is_file()),
                    key=lambda c: c.stat().st_mtime, reverse=True)
    return others[0] if others else None


def load_config(spec, revision, token):
    """Return (config, source_description)."""
    p = Path(spec).expanduser()
    if p.exists():
        if p.is_dir():
            for fname in ("config.json", "params.json"):
                if (p / fname).is_file():
                    return read_json_file(p / fname), str(p / fname)
            raise ConfigError(f"no config.json/params.json in {p}")
        return read_json_file(p), str(p)

    if not re.fullmatch(r"[\w.\-]+/[\w.\-]+", spec):
        raise ConfigError(f"'{spec}' is neither an existing path nor a <vendor>/<model> id")

    net_err = None
    for fname in ("config.json", "params.json"):   # params.json: native Mistral format
        try:
            return fetch_json(spec, revision, fname, token)
        except urllib.error.HTTPError as e:
            code = (e.headers.get("X-Error-Code") or "") if e.headers else ""
            if code == "RevisionNotFound":
                raise ConfigError(f"revision '{revision}' does not exist in {spec}") from e
            if code == "RepoNotFound":
                raise ConfigError(f"repo '{spec}' does not exist, or is private and no valid "
                                  f"token was given") from e
            if code == "GatedRepo" or e.code in (401, 403):
                hint = ("token given but without access: accept the license on the model page"
                        if token else "a token is needed: --token, HF_TOKEN or `hf auth login`")
                raise ConfigError(f"access denied to '{spec}' (HTTP {e.code}"
                                  f"{', ' + code if code else ''}); {hint}") from e
            if e.code == 404:
                continue
            net_err = f"HTTP {e.code}"
            break
        except OSError as e:   # URLError, timeout, DNS, connection refused
            net_err = getattr(e, "reason", None) or e
            break

    if net_err is not None:
        for fname in ("config.json", "params.json"):
            cached = from_cache(spec, revision, fname)
            if cached:
                return read_json_file(cached), f"{cached} (local cache; network: {net_err})"
        raise ConfigError(f"{ENDPOINT} unreachable ({net_err}) and no copy in the local cache")
    raise ConfigError("no config.json/params.json in the repo: it may be a GGUF-only repo; "
                      "use the base model id, or read the .gguf metadata with gguf-dump")


# ----------------------------------------------------------------- parameter count

def count_safetensors(folder):
    """Exact parameter count from the headers of the .safetensors files in a folder."""
    folder = Path(folder)
    files = sorted(f for f in folder.glob("*.safetensors") if f.is_file())
    index = folder / "model.safetensors.index.json"
    if index.is_file():
        weight_map = read_json_file(index).get("weight_map") or {}
        missing = sorted(set(weight_map.values()) - {f.name for f in files})
        if missing:
            raise ConfigError(f"{len(missing)} safetensors shard(s) listed in the index are "
                              f"missing from {folder}: use --params")
    if not files:
        raise ConfigError(f"no .safetensors files in {folder}: use --params")
    total, by_dtype = 0, {}
    for f in files:
        try:
            with open(f, "rb") as fh:
                n = int.from_bytes(fh.read(8), "little")
                if not 2 <= n <= 100 * MIB:
                    raise ConfigError(f"{f.name}: invalid safetensors header")
                header = json.loads(fh.read(n).decode("utf-8"))
        except (OSError, UnicodeDecodeError, ValueError) as e:
            raise ConfigError(f"cannot read {f}: {e}") from e
        if not isinstance(header, dict):
            raise ConfigError(f"{f.name}: invalid safetensors header")
        for name, t in header.items():
            if name == "__metadata__" or not isinstance(t, dict):
                continue
            cnt = 1
            for d in t.get("shape") or []:
                cnt *= int(d)
            dt = str(t.get("dtype", "?"))
            total += cnt
            by_dtype[dt] = by_dtype.get(dt, 0) + cnt
    return {"total": total, "by_dtype": by_dtype, "source": f"safetensors headers in {folder}"}


def param_count(spec, revision, token):
    """Parameter count: local safetensors headers, Hugging Face API, or local HF cache."""
    p = Path(spec).expanduser()
    if p.exists():
        return count_safetensors(p if p.is_dir() else p.parent)

    base = f"{ENDPOINT}/api/models/{spec}"
    if revision != "main":
        base += "/revision/" + urllib.parse.quote(revision, safe="")
    net_err = None
    for url in (base, base + "?expand%5B%5D=safetensors"):
        try:
            st = http_get_json(url, token).get("safetensors")
        except urllib.error.HTTPError as e:
            if e.code in (401, 403, 404):
                raise ConfigError(f"Hugging Face API refused the request (HTTP {e.code}): "
                                  f"use --params") from e
            continue   # e.g. 400 if the expand parameter is not supported
        except OSError as e:
            net_err = getattr(e, "reason", None) or e
            break
        if isinstance(st, dict) and all_int(st.get("total")):
            return {"total": st["total"], "by_dtype": st.get("parameters") or {},
                    "source": "Hugging Face API, safetensors metadata"}

    if net_err is not None:
        cached = from_cache(spec, revision, "config.json")
        if cached:
            return count_safetensors(cached.parent)
        raise ConfigError(f"Hugging Face API unreachable ({net_err}): use --params")
    raise ConfigError("no parameter count published for this repo (no safetensors "
                      "metadata): use --params")


def weights_estimate(spec, r, quant, params_override, revision, token):
    qname, bpw = quant
    w = {"quant": qname, "bpw": bpw, "notes": [], "warnings": []}
    try:
        pc = params_override or param_count(spec, revision, token)
    except ConfigError as exc:
        w["error"] = str(exc)
        return w
    total = pc["total"]
    w.update(params=total, params_source=pc["source"], by_dtype=pc["by_dtype"],
             bytes=total * bpw / 8)
    packed = sum(v for k, v in pc["by_dtype"].items() if str(k).upper() in PACKED_DTYPES)
    if total and packed / total > 0.01:
        w["warnings"].append("the checkpoint holds packed quantized tensors (U8/I32/U32): the "
                             "parameter count is underestimated; use the original model id "
                             "or --params")
    if r["checkpoint_quant"] and not params_override:
        w["notes"].append(f"checkpoint already quantized ({r['checkpoint_quant']}): count "
                          f"read from the stored tensors")
    if qname in QUANT_APPROX:
        w["notes"].append("llama.cpp mixed preset: bpw approximate (about +/-5%)")
    elif qname.startswith(("q", "iq", "mxfp4")):
        w["notes"].append("single-type bpw: lower bound, llama-quantize keeps a few tensors "
                          "(e.g. the output layer) at higher precision")
    if r["subconfig"] and not params_override:
        w["notes"].append("the count includes every tensor in the checkpoint, e.g. the vision "
                          "encoder, which in GGUF lives in a separate mmproj file")
    return w


# ----------------------------------------------------------------- config analysis

def pick_text_config(cfg):
    """Descend into multimodal sub-configs down to the LLM part."""
    cur, path = cfg, []
    for _ in range(4):
        nxt = None
        for k in NESTED:
            sub = cur.get(k)
            if isinstance(sub, dict) and (first(sub, ALIASES["n_layers"])[1] is not None
                                          or any(isinstance(sub.get(n), dict) for n in NESTED)):
                nxt = k
                break
        if nxt is None:
            break
        cur = cur[nxt]
        path.append(nxt)
    return cur, path


def norm_kind(s):
    s = str(s).lower()
    if any(t in s for t in ("linear", "mamba", "conv", "ssm", "recurrent", "delta",
                            "lightning", "rwkv")):
        return "recurrent"
    if "chunk" in s:
        return "chunked"
    if any(t in s for t in ("sliding", "local", "swa")):
        return "sliding"
    if any(t in s for t in ("full", "global", "attention", "attn", "hybrid")):
        return "full"
    if any(t in s for t in ("mlp", "moe", "ffn")):
        return "none"
    return "unknown"


def layer_kinds(tc, L, model_type, notes, warns):
    """Type of each layer: full / sliding / chunked / recurrent / none / unknown."""
    def mismatch(key, n):
        warns.append(f"'{key}' has {n} entries but there are {L} layers: ignored")

    for key in ("layer_types", "layers_block_type"):
        lt = tc.get(key)
        if isinstance(lt, list) and lt:
            if len(lt) == L:
                return [norm_kind(x) for x in lt], key
            mismatch(key, len(lt))

    pat = tc.get("hybrid_override_pattern")          # Nemotron-H
    if isinstance(pat, str) and pat:
        if len(pat) == L:
            m = {"*": "full", "M": "recurrent", "-": "none", "E": "none"}
            return [m.get(c, "unknown") for c in pat], "hybrid_override_pattern"
        mismatch("hybrid_override_pattern", len(pat))

    atl = tc.get("attn_type_list")                   # MiniMax
    if isinstance(atl, list) and atl:
        if len(atl) == L:
            return ["full" if x else "recurrent" for x in atl], "attn_type_list"
        mismatch("attn_type_list", len(atl))

    idx = tc.get("full_attn_idxs")                   # LFM2
    if isinstance(idx, list) and idx:
        s = set(idx)
        return ["full" if i in s else "recurrent" for i in range(L)], "full_attn_idxs"

    per, off = tc.get("attn_layer_period"), tc.get("attn_layer_offset")   # Jamba
    if isinstance(per, int) and per > 0 and isinstance(off, int):
        return (["full" if i % per == off else "recurrent" for i in range(L)],
                "attn_layer_period/offset")

    fai = tc.get("full_attention_interval")          # Qwen3-Next and successors
    if isinstance(fai, int) and fai > 0:
        return (["full" if (i + 1) % fai == 0 else "recurrent" for i in range(L)],
                f"full_attention_interval={fai}")

    if tc.get("attention_chunk_size"):               # Llama 4
        nrl = tc.get("no_rope_layers")
        if isinstance(nrl, list) and len(nrl) == L:
            return ["chunked" if x else "full" for x in nrl], "no_rope_layers + attention_chunk_size"
        iv = tc.get("no_rope_layer_interval") or 4
        warns.append(f"chunked attention: assumed 1 global layer every {iv}")
        return (["full" if (i + 1) % iv == 0 else "chunked" for i in range(L)],
                f"attention_chunk_size, interval {iv}")

    sw, use_sw = tc.get("sliding_window"), tc.get("use_sliding_window")
    if sw and use_sw is False:
        notes.append(f"sliding_window={sw} ignored: use_sliding_window=false")
    elif sw:
        p = tc.get("sliding_window_pattern") or tc.get("_sliding_window_pattern")
        if not p and str(model_type or "").startswith("gemma2"):
            p = 2
        if isinstance(p, int) and p > 0:
            return (["full" if (i + 1) % p == 0 else "sliding" for i in range(L)],
                    f"sliding_window_pattern={p}")
        mwl = tc.get("max_window_layers")
        if use_sw is True and isinstance(mwl, int):
            return (["sliding" if i >= mwl else "full" for i in range(L)],
                    "use_sliding_window + max_window_layers")
        warns.append("sliding_window without a per-layer pattern: assumed on all layers "
                     "(the engine may ignore it: check n_swa in the llama.cpp log)")
        return ["sliding"] * L, "sliding_window on all layers"
    return ["full"] * L, "all full attention (no hybrid/SWA pattern in the config)"


def all_int(*xs):
    return all(isinstance(x, int) and not isinstance(x, bool) and x > 0 for x in xs)


def recurrent_state(tc):
    """Fixed-state elements per linear/SSM layer, if the layer type is recognized."""
    def g(*keys):
        return first(tc, keys)

    # Gated DeltaNet (Qwen3-Next and successors): state n_v_heads x d_k x d_v + conv buffer
    keys = ("linear_num_value_heads", "linear_num_key_heads", "linear_key_head_dim",
            "linear_value_head_dim", "linear_conv_kernel_dim")
    nvh, nkh, dk, dv, kc = (tc.get(k) for k in keys)
    if all_int(nvh, nkh, dk, dv, kc):
        s = nvh * dk * dv
        conv = (kc - 1) * (2 * nkh * dk + nvh * dv)
        return {"type": "Gated DeltaNet", "elements_per_layer": s + conv,
                "bytes_per_layer": STATE_BYTES * (s + conv)}, set(keys)

    # Mamba2 (Nemotron-H, Granite 4, ...): state d_inner x d_state + conv buffer
    k_nh, nh = g("mamba_num_heads", "mamba_n_heads")
    k_hd, hd = g("mamba_head_dim", "mamba_d_head")
    k_ds, ds = g("ssm_state_size", "mamba_d_state", "state_size")
    k_kc, kc = g("conv_kernel", "mamba_d_conv")
    k_ng, ng = g("n_groups", "mamba_n_groups")
    if all_int(nh, hd, ds, kc):
        ng = ng if all_int(ng) else 1
        d_inner = nh * hd
        s = d_inner * ds
        conv = (kc - 1) * (d_inner + 2 * ng * ds)
        used = {k for k in (k_nh, k_hd, k_ds, k_kc, k_ng) if k}
        return {"type": "Mamba2", "elements_per_layer": s + conv,
                "bytes_per_layer": STATE_BYTES * (s + conv)}, used
    return None, set()


def analyze(cfg):
    tc, path = pick_text_config(cfg)
    notes, warns = [], []
    model_type = tc.get("model_type") or cfg.get("model_type")

    _, L = first(tc, ALIASES["n_layers"])
    if not all_int(L):
        raise ConfigError("number of layers not found (num_hidden_layers or equivalent)")
    _, n_heads = first(tc, ALIASES["n_heads"])
    k_max, max_ctx = first(tc, ALIASES["max_ctx"])

    k_kv, n_kv = first(tc, ALIASES["n_kv"])
    if n_kv is None:
        if tc.get("multi_query"):
            n_kv, k_kv = 1, "multi_query"
        elif n_heads is not None:
            n_kv, k_kv = n_heads, "= Q heads: MHA, num_key_value_heads missing"
            notes.append("num_key_value_heads missing: assumed MHA (n_kv = num_attention_heads)")
    if isinstance(n_kv, list):
        if len(n_kv) != L:
            raise ConfigError(f"{k_kv} is a list of {len(n_kv)} entries but there are {L} layers")
        kv_per_layer = list(n_kv)
    else:
        kv_per_layer = [n_kv] * L

    _, hidden = first(tc, ALIASES["hidden"])
    k_hd, head_dim = first(tc, ALIASES["head_dim"])
    if head_dim is None and all_int(hidden, n_heads):
        head_dim, k_hd = hidden // n_heads, f"hidden_size/num_attention_heads = {hidden}/{n_heads}"
        if hidden % n_heads:
            warns.append("hidden_size not divisible by num_attention_heads: head_dim uncertain")

    mla = None
    if tc.get("kv_lora_rank"):
        rank, rope = tc["kv_lora_rank"], tc.get("qk_rope_head_dim")
        if not all_int(rope):
            warns.append("MLA: qk_rope_head_dim missing, assumed 0")
            rope = 0
        mla = {"kv_lora_rank": rank, "qk_rope_head_dim": rope,
               "qk_nope_head_dim": tc.get("qk_nope_head_dim"),
               "v_head_dim": tc.get("v_head_dim"), "latent": rank + rope}
        notes.append("MLA: llama.cpp stores the latent (kv_lora_rank + qk_rope_head_dim) in K; "
                     "V takes another kv_lora_rank elements or is a view of K, depending on "
                     "version and -fa: the total reports both cases")
        head_dim_v = mla["v_head_dim"]
    else:
        head_dim_v = tc.get("v_head_dim") or head_dim

    kinds, scheme = layer_kinds(tc, L, model_type, notes, warns)
    shared = tc.get("num_kv_shared_layers") or 0
    if all_int(shared) and shared < L:
        kinds[L - shared:] = ["shared"] * shared

    windows = {"full": None, "sliding": tc.get("sliding_window"),
               "chunked": tc.get("attention_chunk_size")}
    groups, others, missing_win = {}, {}, set()
    for i, kind in enumerate(kinds):
        if kind not in windows:
            others[kind] = others.get(kind, 0) + 1
            continue
        win = windows[kind]
        if kind != "full" and not win:
            missing_win.add(kind)
        if mla:
            key = (kind, "mla", win)
            grp = groups.setdefault(key, {
                "kind": kind, "attention": "MLA", "layers": 0, "window": win,
                "kv_lora_rank": mla["kv_lora_rank"], "qk_rope_head_dim": mla["qk_rope_head_dim"],
                "elements_per_token_per_layer": mla["latent"]})
        else:
            nk = kv_per_layer[i]
            ok = all_int(nk, head_dim, head_dim_v)
            key = (kind, nk, head_dim, head_dim_v, win)
            grp = groups.setdefault(key, {
                "kind": kind, "attention": "GQA/MHA", "layers": 0, "window": win,
                "n_kv": nk, "d_k": head_dim, "d_v": head_dim_v,
                "elements_per_token_per_layer": nk * (head_dim + head_dim_v) if ok else None})
        grp["layers"] += 1
    for kind in sorted(missing_win):
        warns.append(f"'{kind}' layers without a window size in the config: counted as full")
    if any(gr["elements_per_token_per_layer"] is None for gr in groups.values()):
        warns.append("n_kv or head_dim missing: elements per token cannot be computed")
    if others.get("unknown"):
        warns.append(f"{others['unknown']} layers of unrecognized type, excluded from the total")
    nextn = tc.get("num_nextn_predict_layers")
    if nextn:
        notes.append(f"{nextn} MTP layer(s) (num_nextn_predict_layers) not included")

    rec, rec_keys = (None, set())
    if others.get("recurrent"):
        rec, rec_keys = recurrent_state(tc)
        if rec is None:
            warns.append(f"fixed state of the {others['recurrent']} linear/SSM layers not "
                         f"estimated (unrecognized type): excluded from the total")

    related = {k: v for k, v in tc.items()
               if k not in USED and k not in rec_keys and k not in NESTED
               and not k.endswith("_config")
               and RELEVANT_RE.search(k) and not IRRELEVANT_RE.search(k)}

    qc = cfg.get("quantization_config") or tc.get("quantization_config")
    checkpoint_quant = (qc.get("quant_method") or "unknown method") if isinstance(qc, dict) else None

    return {
        "checkpoint_quant": checkpoint_quant,
        "model_type": model_type,
        "architectures": cfg.get("architectures"),
        "subconfig": ".".join(path) or None,
        "max_context": max_ctx, "max_context_key": k_max,
        "n_layers": L,
        "n_heads": n_heads,
        "n_kv_heads": n_kv, "n_kv_heads_key": k_kv,
        "head_dim_k": head_dim, "head_dim_v": head_dim_v, "head_dim_key": k_hd,
        "sliding_window": tc.get("sliding_window"),
        "attention_chunk_size": tc.get("attention_chunk_size"),
        "mla": mla,
        "layer_scheme": scheme,
        "kv_groups": list(groups.values()),
        "layers_without_ctx_kv": others,
        "recurrent_state": rec,
        "notes": notes,
        "warnings": warns,
        "related_keys": related,
    }


# ----------------------------------------------------------------- calculation

def pad256(x):
    """llama.cpp allocates KV cells in multiples of 256."""
    return -(-x // 256) * 256


def estimate(r, kv_size, type_k, type_v, ubatch):
    bk, bv = CACHE_BYTES[type_k], CACHE_BYTES[type_v]
    n_ctx = pad256(kv_size)
    rows, total, total_view = [], 0.0, 0.0
    for g in r["kv_groups"]:
        win = g["window"]
        cells = n_ctx if (g["kind"] == "full" or not win) else min(n_ctx, pad256(win + ubatch))
        if g["attention"] == "MLA":
            k_el, v_el = g["kv_lora_rank"] + g["qk_rope_head_dim"], g["kv_lora_rank"]
        else:
            if g["elements_per_token_per_layer"] is None:
                raise ConfigError("n_kv or head_dim missing from the config: KV cannot be "
                                  "computed (use --verbose for details)")
            k_el, v_el = g["n_kv"] * g["d_k"], g["n_kv"] * g["d_v"]
        b = g["layers"] * cells * (k_el * bk + v_el * bv)
        b_view = g["layers"] * cells * k_el * bk if g["attention"] == "MLA" else b
        rows.append({"kind": g["kind"], "attention": g["attention"], "layers": g["layers"],
                     "cells": cells, "k_elements": k_el, "v_elements": v_el, "bytes": b})
        total += b
        total_view += b_view

    n_rec = r["layers_without_ctx_kv"].get("recurrent", 0)
    rec_bytes = None
    if n_rec and r["recurrent_state"]:
        rec_bytes = n_rec * r["recurrent_state"]["bytes_per_layer"]
        total += rec_bytes
        total_view += rec_bytes
    return {
        "kv_size": kv_size, "cells": n_ctx, "type_k": type_k, "type_v": type_v,
        "ubatch": ubatch, "groups": rows,
        "recurrent_layers": n_rec, "recurrent_bytes": rec_bytes,
        "total_bytes": total,
        "total_bytes_if_v_view_of_k": total_view if r["mla"] else None,
        "total_gib": total / GIB,
    }


# ----------------------------------------------------------------- helpers

def human(b):
    """Format a byte count as a human-readable GiB/MiB string."""
    return f"{b / GIB:.2f} GiB" if b >= GIB else f"{b / MIB:.1f} MiB"
