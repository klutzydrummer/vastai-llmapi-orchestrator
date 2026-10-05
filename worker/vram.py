#!/usr/bin/env python3
"""GPU memory planning for one worker: what fits, and with which settings.

Reads the GGUF headers of the chat model, its vision projector and the
embedding model (from local files, or from the Hub with HTTP range requests so
it can run before anything is downloaded), estimates what each llama-server
needs, and picks the adjustable settings so the whole set fits in the GPU
memory that is actually free. When it can't fit even at the smallest allowed
settings, it says what didn't fit and by how much.

What is adjustable, in the order it is given up when memory is short:
  1. the embedding server's context (EMBED_CTX, halved down to 2048; its batch
     sizes follow it),
  2. the embedding server's cache type (f16 -> q8_0),
  3. the chat context (LLAMA_CTX is the ceiling, LLAMA_CTX_MIN the floor).
The embedding model is never moved to the CPU here; only EMBED_GPU=0 does that.
The configured chat context is never raised.

The context-cache size is worked out per layer from the header, so per-layer
KV head counts and sliding-window layers (Gemma 3/4) are counted correctly;
models whose headers lack those keys fall back to the uniform formula. The
compute-buffer, projector and CUDA-context figures are estimates: boot.sh logs
llama.cpp's own buffer sizes and the VRAM nvidia-smi reports after each server
starts, so every real boot shows how close they were.

Usage:
  vram.py plan --stage pre     headers from the Hub (MODEL_REPO/FILE/REVISION,
                               MMPROJ_*, EMBED_*), plan embedding + chat
  vram.py plan --stage local   headers from MODEL_PATH, MMPROJ_PATH, EMBED_PATH
  vram.py plan --stage chat    chat only, in the memory free now (run after the
                               embedding server is up)
  vram.py header FILE_OR_URL   print a header's metadata (debugging)
Options: --free-mib N (instead of asking nvidia-smi), --env FILE (write PLAN_*
settings for boot.sh), --json FILE (the full plan).
Exit status: 0 fits, 7 doesn't fit, 2 bad input or unreadable header.
"""

import argparse
import json
import os
import struct
import subprocess
import sys
import urllib.parse
import urllib.request

MiB = 1024 * 1024
EXIT_NOFIT, EXIT_INPUT = 7, 2

# Bytes per element of a KV cache type (block size included).
CACHE_BYTES = {"f32": 4.0, "f16": 2.0, "bf16": 2.0, "q8_0": 34 / 32, "q4_0": 18 / 32,
               "q4_1": 20 / 32, "q5_0": 22 / 32, "q5_1": 24 / 32, "iq4_nl": 18 / 32}

# Estimates, logged next to the real figures on every boot. Calibrated on
# rental 54245444 (RTX 3090, llama.cpp b11371, WaifuGemma4 26B-A4B + BF16
# projector + Qwen3-Embedding-0.6B); the margin below is the safety factor.
CUDA_CTX_MIB = float(os.environ.get("VRAM_CUDA_CTX_MIB", "300"))    # per llama-server process (measured ~300)
MARGIN_FRAC = float(os.environ.get("VRAM_MARGIN_FRAC", "0.03"))     # of total VRAM, kept free
MARGIN_MIB = float(os.environ.get("VRAM_MARGIN_MIB", "256"))


def log(msg):
    print(f"[vram] {msg}", flush=True)


# ── GGUF header ───────────────────────────────────────────────────────────────
class HeaderError(Exception):
    pass


class FileSource:
    def __init__(self, path):
        self.f = open(path, "rb")
        self.name = os.path.basename(path)

    def read(self, n):
        b = self.f.read(n)
        if len(b) != n:
            raise HeaderError(f"{self.name}: file ends inside the header")
        return b

    def skip(self, n):
        self.f.seek(n, 1)


class HttpSource:
    """Sequential reads of a remote file through HTTP range requests, a chunk
    at a time; headers of large models fit in a few chunks."""

    def __init__(self, url, headers=None, chunk=8 * MiB, limit=128 * MiB):
        self.url, self.headers, self.chunk, self.limit = url, headers or {}, chunk, limit
        self.name = urllib.parse.unquote(url.rsplit("/", 1)[-1])
        self.pos, self.buf, self.buf_at = 0, b"", 0

    def _fill(self, need):
        if self.pos + need > self.limit:
            raise HeaderError(f"{self.name}: header larger than {self.limit // MiB} MiB")
        start = self.pos
        end = start + max(need, self.chunk) - 1
        req = urllib.request.Request(self.url, headers={**self.headers, "Range": f"bytes={start}-{end}"})
        with urllib.request.urlopen(req, timeout=60) as r:
            data = r.read(end - start + 1) if r.status == 206 else r.read(end + 1)[start:]
        if len(data) < need:
            raise HeaderError(f"{self.name}: file ends inside the header")
        self.buf, self.buf_at = data, start

    def read(self, n):
        off = self.pos - self.buf_at
        if off < 0 or off + n > len(self.buf):
            self._fill(n)
            off = 0
        self.pos += n
        return self.buf[off:off + n]

    def skip(self, n):
        self.pos += n


SCALAR = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}


class ArrayLen(int):
    """A long array the reader skipped; only its length is kept."""


def read_header(src):
    """All metadata key/values of a GGUF file. Long arrays (the tokenizer's)
    are skipped and kept as ArrayLen(length)."""
    def u32():
        return struct.unpack("<I", src.read(4))[0]

    def u64():
        return struct.unpack("<Q", src.read(8))[0]

    def string():
        n = u64()
        if n > 16 * MiB:
            raise HeaderError(f"{src.name}: implausible string length {n}")
        return src.read(n).decode("utf-8", "replace")

    def value(t, keep=True):
        if t in SCALAR:
            fmt = SCALAR[t]
            size = struct.calcsize(fmt)
            if not keep:
                src.skip(size)
                return None
            return struct.unpack(fmt, src.read(size))[0]
        if t == 8:
            s = string()
            return s if keep else None
        if t == 9:
            et, n = u32(), u64()
            if keep and n <= 4096:
                return [value(et) for _ in range(n)]
            if et in SCALAR:
                src.skip(struct.calcsize(SCALAR[et]) * n)
            else:
                for _ in range(n):
                    value(et, keep=False)
            return ArrayLen(n)
        raise HeaderError(f"{src.name}: unknown value type {t}")

    if src.read(4) != b"GGUF":
        raise HeaderError(f"{src.name}: not a GGUF file")
    version = u32()
    if version < 2:
        raise HeaderError(f"{src.name}: GGUF version {version} is too old")
    u64()  # tensor count
    meta = {}
    for _ in range(u64()):
        key = string()
        meta[key] = value(u32())
    return meta


def open_source(where):
    if where.startswith(("http://", "https://")):
        headers = {"User-Agent": "vastai-llmapi-orchestrator/1"}
        if os.environ.get("HF_TOKEN", "").strip():
            headers["Authorization"] = f"Bearer {os.environ['HF_TOKEN'].strip()}"
        return HttpSource(where, headers)
    return FileSource(where)


def hub_url(repo, revision, path):
    base = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
    return (f"{base}/{urllib.parse.quote(repo, safe='/')}/resolve/"
            f"{urllib.parse.quote(revision or 'main', safe='')}/{urllib.parse.quote(path)}")


# ── what a model needs ────────────────────────────────────────────────────────
def _per_layer(v, n, default):
    if v is None:
        return [default] * n
    if isinstance(v, list):
        return [int(x) for x in v][:n] + [int(v[-1]) if v else default] * max(0, n - len(v))
    return [int(v)] * n


def model_facts(meta, size_bytes):
    """The numbers the estimates need, from a chat or embedding model's header."""
    arch = meta.get("general.architecture")
    if not arch:
        raise HeaderError("no general.architecture in the header")

    def g(k, d=None):
        return meta.get(f"{arch}.{k}", d)

    n_layer = int(g("block_count", 0) or 0)
    n_embd = int(g("embedding_length", 0) or 0)
    if not n_layer or not n_embd:
        raise HeaderError(f"{arch}: header has no block_count/embedding_length")
    heads = _per_layer(g("attention.head_count"), n_layer, 1)
    n_head = max(heads) or 1
    kv_heads = _per_layer(g("attention.head_count_kv"), n_layer, n_head)
    k_len = int(g("attention.key_length", 0) or n_embd // n_head)
    v_len = int(g("attention.value_length", 0) or n_embd // n_head)
    window = int(g("attention.sliding_window", 0) or 0)
    pattern = g("attention.sliding_window_pattern")
    if isinstance(pattern, list):
        swa = [bool(x) for x in pattern][:n_layer] + [False] * max(0, n_layer - len(pattern))
    elif isinstance(pattern, int) and pattern > 1:
        swa = [(i + 1) % pattern != 0 for i in range(n_layer)]   # every pattern-th layer is global
    else:
        swa = [False] * n_layer
    if not window:
        swa = [False] * n_layer
    shared = int(g("attention.shared_kv_layers", 0) or 0)   # trailing layers reuse earlier KV
    owns_kv = [i < n_layer - shared for i in range(n_layer)]
    ff = g("feed_forward_length", 0)
    n_ff = max(ff) if isinstance(ff, list) else int(ff or 0)
    n_ff = max(n_ff, int(g("expert_feed_forward_length", 0) or 0) * int(g("expert_used_count", 0) or 0))
    vocab = meta.get("tokenizer.ggml.tokens")
    return {
        "arch": arch, "size": int(size_bytes), "n_layer": n_layer, "n_embd": n_embd, "n_head": n_head,
        "kv_heads": kv_heads, "k_len": k_len, "v_len": v_len,
        "k_len_swa": int(g("attention.key_length_swa", 0) or k_len),
        "v_len_swa": int(g("attention.value_length_swa", 0) or v_len),
        "swa": swa, "window": window, "owns_kv": owns_kv, "n_ff": n_ff or 4 * n_embd,
        "n_vocab": len(vocab) if isinstance(vocab, list) else int(vocab or g("vocab_size", 0) or 0),
        "ctx_train": int(g("context_length", 0) or 0),
        "per_layer_keys": isinstance(g("attention.head_count_kv"), list) or bool(window)
                          or g("attention.key_length") is not None,
    }


def projector_facts(meta, size_bytes):
    def g(k, d=0):
        return meta.get(f"clip.vision.{k}", d) or d
    img, patch = int(g("image_size")), int(g("patch_size"))
    return {"size": int(size_bytes), "hidden": int(g("embedding_length", 1024)),
            "n_head": int(g("attention.head_count", 16)),
            "patches": (img // patch) ** 2 if img and patch else 0}


def pad(n, to=256):
    return -(-int(n) // to) * to


def kv_bytes(m, ctx, n_seq, ubatch, cache_k, cache_v, swa_full=False):
    """Context cache of a llama-server with a unified cache of `ctx` cells.
    Sliding-window layers hold only window x sequences + one batch of cells."""
    bk, bv = CACHE_BYTES[cache_k], CACHE_BYTES[cache_v]
    swa_cells = min(ctx, pad(m["window"] * n_seq + ubatch)) if m["window"] and not swa_full else ctx
    total = 0.0
    for i in range(m["n_layer"]):
        if not m["owns_kv"][i]:
            continue
        if m["swa"][i]:
            total += swa_cells * m["kv_heads"][i] * (m["k_len_swa"] * bk + m["v_len_swa"] * bv)
        else:
            total += ctx * m["kv_heads"][i] * (m["k_len"] * bk + m["v_len"] * bv)
    return total


def compute_bytes(m, ubatch, n_seq=1, embedding=False):
    """llama.cpp's GPU compute buffer for one batch (estimate, f32).

    Measured on b11371: an embedding server reserves output rows over the
    vocabulary for every token of the batch (Qwen3-Embedding-0.6B: 0.598 MiB
    per batch token = n_vocab x 4 + 20 x n_embd bytes; 4900 MiB at -ub 8192),
    while a chat server keeps logits for one row per sequence and its buffer
    is activations (WaifuGemma4: 385 MiB at -ub 1120; this gives ~400)."""
    if embedding:
        return ubatch * (4 * m["n_vocab"] + 20 * m["n_embd"])
    act = ubatch * 4 * (8 * m["n_embd"] + 4 * m["n_ff"] + 6 * m["n_head"] * max(m["k_len"], m["v_len"]))
    return act + n_seq * 4 * m["n_vocab"]


def projector_bytes(p, image_tokens):
    """Projector weights plus the vision encoder's compute buffer (estimate).
    Measured: the Gemma 4 BF16 projector (1139 MiB) took 1297 MiB with 1120
    image tokens (llama.cpp's worst case); this gives ~1313."""
    return p["size"] + image_tokens * 4 * p["hidden"] * 35


# ── the plan ──────────────────────────────────────────────────────────────────
def plan(free_mib, total_mib, chat, cfg, mmproj=None, embed=None):
    """Pick the embedding server's context/cache and the chat context so
    everything fits in free_mib. Returns a dict; "fits" says whether it did."""
    n_seq = cfg["parallel"]
    notes = []
    # The configured context is a ceiling (rounded down to whole 256-cell
    # blocks), and so is the trained context of every slot.
    ctx_max = max(256, int(cfg["ctx"]) // 256 * 256)
    if chat["ctx_train"] and ctx_max > chat["ctx_train"] * n_seq:
        notes.append(f"chat context {ctx_max} capped at {chat['ctx_train'] * n_seq} "
                     f"(trained context {chat['ctx_train']} x {n_seq} slots)")
        ctx_max = chat["ctx_train"] * n_seq
    ctx_min = min(max(256, int(cfg.get("ctx_min") or ctx_max) // 256 * 256), ctx_max)
    ub = max(int(cfg.get("ubatch") or 512), int(cfg.get("image_tokens") or 0) if mmproj else 0)
    batch = max(2048, ub)
    procs = 1 + (1 if embed else 0)
    margin = MARGIN_MIB + MARGIN_FRAC * total_mib
    levers = []

    fixed = {
        "chat weights": chat["size"] / MiB,
        "chat compute (est.)": compute_bytes(chat, ub, n_seq) / MiB,
        "CUDA contexts (est.)": CUDA_CTX_MIB * procs,
        "margin": margin,
    }
    if mmproj and cfg.get("mmproj_offload", True):   # --no-mmproj-offload keeps it in system RAM
        fixed["projector + vision compute (est.)"] = projector_bytes(mmproj, ub) / MiB

    def embed_parts(ectx, ecache):
        if not embed:
            return {}
        return {"embedding weights": embed["size"] / MiB,
                "embedding cache": kv_bytes(embed, ectx, 1, ectx, ecache, ecache) / MiB,
                "embedding compute (est.)": compute_bytes(embed, ectx, embedding=True) / MiB}

    def chat_cache(ctx):
        return kv_bytes(chat, ctx, n_seq, ub, cfg["cache_type"], cfg["cache_type"]) / MiB

    def total(parts):
        return sum(parts.values())

    # Levers 1 and 2: the embedding server's context, then its cache type.
    ectx, ecache = int(cfg.get("embed_ctx") or 0), cfg.get("embed_cache") or "f16"
    if embed and embed["ctx_train"] and ectx > embed["ctx_train"]:
        notes.append(f"embedding context {ectx} capped at its trained context {embed['ctx_train']}")
        ectx = embed["ctx_train"]
    options = [(ectx, ecache)]
    if embed:
        c = ectx
        while c > 2048:
            c = max(2048, c // 2)
            options.append((c, ecache))
        if ecache != "q8_0":
            options.append((options[-1][0], "q8_0"))
    chosen = None
    for ectx_i, ecache_i in options:
        parts = {**fixed, **embed_parts(ectx_i, ecache_i), "chat cache": chat_cache(ctx_max)}
        if total(parts) <= free_mib:
            chosen = (ectx_i, ecache_i, ctx_max)
            break
    if chosen is None:
        # Lever 3: the chat context, down to its floor.
        ectx_i, ecache_i = options[-1]
        base = {**fixed, **embed_parts(ectx_i, ecache_i)}
        step = 256 * n_seq
        ctx = ctx_max - step
        while ctx >= ctx_min:
            if total(base) + chat_cache(ctx) <= free_mib:
                chosen = (ectx_i, ecache_i, ctx)
                break
            ctx -= step
        if chosen is None:
            chosen = (ectx_i, ecache_i, ctx_min)
    ectx_i, ecache_i, ctx = chosen
    if embed and ectx_i != ectx:
        levers.append(f"embedding context {ectx} -> {ectx_i}")
    if embed and ecache_i != ecache:
        levers.append(f"embedding cache {ecache} -> {ecache_i}")
    if ctx != ctx_max:
        levers.append(f"chat context {ctx_max} -> {ctx}")
    parts = {**fixed, **embed_parts(ectx_i, ecache_i), "chat cache": chat_cache(ctx)}
    need = total(parts)
    # What each server should add to nvidia-smi's "used", for the boot log.
    embed_est = sum(v for k, v in parts.items() if k.startswith("embedding")) + (CUDA_CTX_MIB if embed else 0)
    chat_est = need - margin - embed_est
    return {
        "fits": need <= free_mib, "free_mib": round(free_mib), "total_mib": round(total_mib),
        "need_mib": round(need), "short_mib": round(max(0, need - free_mib)),
        "parts_mib": {k: round(v) for k, v in parts.items()},
        "ctx": ctx, "ctx_max": ctx_max, "ctx_min": ctx_min, "parallel": n_seq,
        "ubatch": ub, "batch": batch, "embed_ctx": ectx_i if embed else 0,
        "embed_cache": ecache_i if embed else "", "levers": levers, "notes": notes,
        "cache_formula": "per-layer" if chat["per_layer_keys"] else "uniform (header lacks per-layer keys)",
        "ctx_train": chat["ctx_train"], "embed_est_mib": round(embed_est), "chat_est_mib": round(chat_est),
    }


def describe(p):
    lines = [f"note: {n}" for n in p["notes"]]
    lines += [f"{k}: {v} MiB" for k, v in p["parts_mib"].items()]
    lines.append(f"total {p['need_mib']} MiB of {p['free_mib']} MiB free ({p['total_mib']} MiB on the GPU)")
    if p["fits"]:
        lines.append(f"chat context {p['ctx']} ({p['parallel']} slots), ubatch {p['ubatch']}"
                     + (f"; embedding context {p['embed_ctx']}, cache {p['embed_cache']}" if p["embed_ctx"] else "")
                     + (f"; to fit: {', '.join(p['levers'])}" if p["levers"] else ""))
    else:
        lines.append(f"does NOT fit: {p['short_mib']} MiB short even with chat context {p['ctx_min']} "
                     f"(llama.ctx_min)" + (f" after {', '.join(p['levers'])}" if p["levers"] else ""))
    return lines


# ── command line (used by boot.sh) ────────────────────────────────────────────
def gpu_memory():
    """(free MiB, total MiB) summed over the visible GPUs, from nvidia-smi."""
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free,memory.total",
                          "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=30)
    rows = [r.split(",") for r in out.stdout.strip().splitlines() if r.strip()]
    if out.returncode != 0 or not rows:
        raise RuntimeError(f"nvidia-smi gave no memory figures: {out.stderr.strip()[:200]}")
    return sum(float(r[0]) for r in rows), sum(float(r[1]) for r in rows)


def config_from_env():
    e = os.environ.get
    return {"ctx": int(e("LLAMA_CTX", "32768")), "ctx_min": int(e("LLAMA_CTX_MIN", "0") or 0),
            "parallel": int(e("LLAMA_PARALLEL", "2")), "cache_type": e("LLAMA_CACHE_TYPE", "q8_0"),
            "ubatch": int(e("LLAMA_UBATCH", "512") or 512), "image_tokens": int(e("LLAMA_IMAGE_TOKENS", "1120") or 0),
            "embed_ctx": int(e("EMBED_CTX", "4096")), "embed_cache": e("EMBED_CACHE_TYPE", "f16") or "f16",
            "embed_gpu": e("EMBED_GPU", "1").strip().lower() not in ("0", "false", "no", "off"),
            "mmproj_offload": "--no-mmproj-offload" not in e("LLAMA_EXTRA_ARGS", "").replace(";", " ").split()}


def _facts(where, size, kind):
    src = open_source(where)
    meta = read_header(src)
    if size is None:
        size = os.path.getsize(where)
    return projector_facts(meta, size) if kind == "projector" else model_facts(meta, size)


def hub_size(repo, revision, path):
    base = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
    url = (f"{base}/api/models/{urllib.parse.quote(repo, safe='/')}/paths-info/"
           f"{urllib.parse.quote(revision or 'main', safe='')}")
    headers = {"User-Agent": "vastai-llmapi-orchestrator/1"}
    if os.environ.get("HF_TOKEN", "").strip():
        headers["Authorization"] = f"Bearer {os.environ['HF_TOKEN'].strip()}"
    req = urllib.request.Request(url, data=urllib.parse.urlencode({"paths": path}).encode(), headers=headers)
    with urllib.request.urlopen(req, timeout=30) as r:
        for it in json.loads(r.read().decode()):
            if it.get("path") == path:
                return int((it.get("lfs") or {}).get("size") or it.get("size"))
    raise HeaderError(f"{path} not found in {repo}@{revision}")


def sources(stage):
    """(chat, projector, embedding) as (where, size) pairs, or None."""
    e = lambda k: os.environ.get(k, "").strip()
    if stage == "pre":
        repo, rev = e("MODEL_REPO"), e("MODEL_REVISION") or "main"
        out = [(hub_url(repo, rev, e("MODEL_FILE")), hub_size(repo, rev, e("MODEL_FILE")))]
        if e("MMPROJ_FILE"):
            r, v = e("MMPROJ_REPO") or repo, e("MMPROJ_REVISION") or "main"
            out.append((hub_url(r, v, e("MMPROJ_FILE")), hub_size(r, v, e("MMPROJ_FILE"))))
        else:
            out.append(None)
        if e("EMBED_FILE"):
            r, v = e("EMBED_REPO"), e("EMBED_REVISION") or "main"
            out.append((hub_url(r, v, e("EMBED_FILE")), hub_size(r, v, e("EMBED_FILE"))))
        else:
            out.append(None)
        return out
    return [(e("MODEL_PATH"), None),
            (e("MMPROJ_PATH"), None) if e("MMPROJ_PATH") else None,
            (e("EMBED_PATH"), None) if e("EMBED_PATH") and stage != "chat" else None]


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--stage", choices=["pre", "local", "chat"], required=True)
    p.add_argument("--free-mib", type=float)
    p.add_argument("--total-mib", type=float)
    p.add_argument("--env")
    p.add_argument("--json")
    h = sub.add_parser("header")
    h.add_argument("where")
    a = ap.parse_args(argv)

    if a.cmd == "header":
        try:
            meta = read_header(open_source(a.where))
        except (HeaderError, OSError, ValueError) as ex:
            log(f"can't read {a.where}: {ex}")
            return EXIT_INPUT
        for k, v in meta.items():
            print(f"{k} = {f'<array of {int(v)}>' if isinstance(v, ArrayLen) else v}")
        return 0

    cfg = config_from_env()
    try:
        chat_src, mm_src, emb_src = sources(a.stage)
        chat = _facts(*chat_src, "model")
        mm = _facts(*mm_src, "projector") if mm_src else None
        emb = _facts(*emb_src, "model") if emb_src and cfg["embed_gpu"] else None
    except (HeaderError, OSError, ValueError, KeyError) as ex:
        log(f"can't read the model headers: {ex}")
        return EXIT_INPUT
    if a.free_mib is not None:
        free, total = a.free_mib, a.total_mib or a.free_mib
    else:
        try:
            free, total = gpu_memory()
        except (RuntimeError, OSError, ValueError, subprocess.SubprocessError) as ex:
            log(f"can't read GPU memory: {ex}")
            return EXIT_INPUT
    if a.stage == "chat":
        cfg["embed_ctx"] = 0
    p = plan(free, total, chat, cfg, mm, emb)
    p["stage"] = a.stage
    p["embed_gpu"] = cfg["embed_gpu"]
    p["chat_arch"] = chat["arch"]
    for line in describe(p):
        log(line)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(p, f, indent=2)
    if a.env:
        with open(a.env, "w") as f:
            f.write(f"PLAN_CTX={p['ctx']}\nPLAN_UBATCH={p['ubatch']}\nPLAN_BATCH={p['batch']}\n"
                    f"PLAN_EMBED_CTX={p['embed_ctx']}\nPLAN_EMBED_CACHE={p['embed_cache']}\n"
                    f"PLAN_NEED_MIB={p['need_mib']}\nPLAN_MARGIN_MIB={p['parts_mib']['margin']}\n"
                    f"PLAN_EMBED_EST_MIB={p['embed_est_mib']}\nPLAN_CHAT_EST_MIB={p['chat_est_mib']}\n")
    return 0 if p["fits"] else EXIT_NOFIT


if __name__ == "__main__":
    sys.exit(main())
