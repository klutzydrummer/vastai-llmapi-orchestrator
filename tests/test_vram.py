#!/usr/bin/env python3
"""GPU memory planning in worker/vram.py, with the headers and the GPU of
rental 54231885 (WaifuGemma4 + BF16 projector + Qwen3 embedding on an RTX 3090,
24576 MiB, 23859 MiB free). Run: python3 tests/test_vram.py"""

import http.server
import json
import os
import subprocess
import sys
import tempfile
import threading

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "worker"))
sys.path.insert(0, os.path.join(HERE, "fakes"))
import gguf  # noqa: E402
import vram  # noqa: E402

MiB = 1024 * 1024
CHAT_SIZE, MMPROJ_SIZE, EMBED_SIZE = 16018 * 1024 * 1024, 1_194_800_000, 639_153_184
TMP = tempfile.mkdtemp(prefix="test_vram.")
PASSED, FAILED = [], []


def case(fn):
    try:
        fn()
        PASSED.append(fn.__name__)
        print(f"PASS  {fn.__doc__}")
    except Exception as e:
        FAILED.append(fn.__name__)
        print(f"FAIL  {fn.__doc__}: {type(e).__name__}: {e}")
    return fn


def fixture(kind, size):
    path = os.path.join(TMP, f"{kind}.gguf")
    if not os.path.exists(path):
        gguf.write(path, gguf.KINDS[kind](), size)
    return path


def facts(kind, size=0):
    meta = vram.read_header(vram.FileSource(fixture(kind, size)))
    return (vram.projector_facts if kind.endswith("mmproj") else vram.model_facts)(meta, size)


def cfg(**kw):
    c = {"ctx": 32768, "ctx_min": 8192, "parallel": 2, "cache_type": "q8_0", "ubatch": 512,
         "image_tokens": 1120, "embed_ctx": 8192, "embed_cache": "f16", "embed_gpu": True}
    c.update(kw)
    return c


@case
def gemma4_header_parses():
    """Gemma 4 header: per-layer KV heads, sliding-window pattern, window, swa lengths, vocab"""
    m = facts("gemma4", CHAT_SIZE)
    assert m["arch"] == "gemma4" and m["n_layer"] == 30 and m["n_embd"] == 2816 and m["n_head"] == 16
    assert m["kv_heads"] == [8, 8, 8, 8, 8, 2] * 5, m["kv_heads"]
    assert m["swa"] == [True] * 5 + [False] + [True] * 5 + [False] + [True] * 5 + [False] + \
        [True] * 5 + [False] + [True] * 5 + [False]
    assert (m["window"], m["k_len"], m["v_len"], m["k_len_swa"], m["v_len_swa"]) == (1024, 512, 512, 256, 256)
    assert m["n_vocab"] == 262144 and m["ctx_train"] == 262144 and m["n_ff"] == 704 * 8
    assert m["per_layer_keys"]


@case
def header_reads_from_a_partial_file():
    """the header parses from the first bytes of the file alone (what a range read fetches)"""
    full = open(fixture("gemma4", CHAT_SIZE), "rb").read(4 * MiB)
    part = os.path.join(TMP, "partial.gguf")
    open(part, "wb").write(full)
    assert vram.read_header(vram.FileSource(part))["gemma4.block_count"] == 30


@case
def header_reads_over_http_ranges():
    """HttpSource reads a header with range requests and never fetches the weights"""
    path = fixture("gemma4", CHAT_SIZE)
    served = []

    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            start, end = (int(x) for x in self.headers["Range"].split("=")[1].split("-"))
            with open(path, "rb") as f:
                f.seek(start)
                data = f.read(end - start + 1)
            served.append(len(data))
            self.send_response(206)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *a):
            pass

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        src = vram.HttpSource(f"http://127.0.0.1:{srv.server_address[1]}/x.gguf", chunk=1 * MiB)
        assert vram.read_header(src)["gemma4.attention.sliding_window"] == 1024
        assert sum(served) <= 4 * MiB, served
    finally:
        srv.shutdown()


@case
def qwen3_embed_header_parses():
    """Qwen3-Embedding-0.6B header: 28 layers, 8 KV heads, 128-wide heads, trained context 32768"""
    m = facts("qwen3-embed", EMBED_SIZE)
    assert (m["n_layer"], m["kv_heads"][0], m["k_len"], m["ctx_train"]) == (28, 8, 128, 32768)
    assert not any(m["swa"])


@case
def not_gguf_is_refused():
    """a file that isn't GGUF raises HeaderError, not a crash"""
    p = os.path.join(TMP, "junk.gguf")
    open(p, "wb").write(b"<html>rate limited</html>")
    try:
        vram.read_header(vram.FileSource(p))
    except vram.HeaderError:
        return
    raise AssertionError("no HeaderError")


@case
def gemma4_cache_matches_the_primer():
    """Gemma 4 q8_0 cache at ctx 32768, 2 slots, ubatch 512: global ~340 MiB, sliding-window ~265 MiB"""
    m = facts("gemma4", CHAT_SIZE)
    glob = dict(m, swa=[False] * 30, owns_kv=[not s for s in m["swa"]])
    swa = dict(m, owns_kv=list(m["swa"]))
    g = vram.kv_bytes(glob, 32768, 2, 512, "q8_0", "q8_0") / MiB
    s = vram.kv_bytes(swa, 32768, 2, 512, "q8_0", "q8_0") / MiB
    assert 335 <= g <= 345, g
    assert 260 <= s <= 270, s


@case
def qwen3_cache_at_8192():
    """Qwen3 embedding f16 cache at 8192 is ~0.9 GiB, q8_0 about half"""
    m = facts("qwen3-embed", EMBED_SIZE)
    f16 = vram.kv_bytes(m, 8192, 1, 8192, "f16", "f16") / MiB
    q8 = vram.kv_bytes(m, 8192, 1, 8192, "q8_0", "q8_0") / MiB
    assert 880 <= f16 <= 910, f16
    assert 470 <= q8 <= 480, q8


@case
def uniform_formula_without_per_layer_keys():
    """an old header (no per-layer keys) uses n_layer x kv_heads x head width for every layer"""
    m = facts("tiny", 1000)
    assert not m["per_layer_keys"]
    assert vram.kv_bytes(m, 4096, 1, 512, "f16", "f16") == 4096 * 4 * 4 * (64 * 2 + 64 * 2)


@case
def rental_54231885_fits_with_embedding_on_gpu():
    """WaifuGemma4 + BF16 projector + GPU embedding fit 23859 MiB free on a 3090, ubatch raised for images"""
    p = vram.plan(23859, 24576, facts("gemma4", CHAT_SIZE), cfg(),
                  facts("gemma4-mmproj", MMPROJ_SIZE), facts("qwen3-embed", EMBED_SIZE))
    assert p["fits"], vram.describe(p)
    assert p["ubatch"] == 1120 and p["batch"] == 2048
    assert p["embed_ctx"] >= 2048 and p["ctx"] >= 8192
    assert "embedding weights" in p["parts_mib"] and "projector + vision compute (est.)" in p["parts_mib"]
    assert p["need_mib"] <= 23859


@case
def estimates_match_rental_54245444():
    """estimates match what rental 54245444's RTX 3090 measured, within 300 MiB over and never under"""
    chat, mm, emb = facts("gemma4", CHAT_SIZE), facts("gemma4-mmproj", MMPROJ_SIZE), facts("qwen3-embed", EMBED_SIZE)
    # Embedding server alone, -c 8192 with -b/-ub varied: compute buffer 0.598 MiB per batch token.
    for ub, measured in ((8192, 4900), (4096, 2450), (2048, 1225)):
        est = vram.compute_bytes(emb, ub, embedding=True) / MiB
        assert measured - 5 <= est <= measured + 10, (ub, est)   # the measured figures are rounded
    # Chat compute buffer at -ub 1120, 2 slots: 385 MiB.
    assert 385 <= vram.compute_bytes(chat, 1120, 2) / MiB <= 450
    # Whole set loaded, embedding at -c/-b/-ub 2048 and 4096: nvidia-smi used minus ~120 MiB idle.
    for ectx, measured in ((2048, 21120 - 120), (4096, 22560 - 120)):
        p = vram.plan(10**6, 24576, chat, cfg(embed_ctx=ectx), mm, emb)
        est = p["need_mib"] - p["parts_mib"]["margin"]
        assert measured <= est <= measured + 300, (ectx, est, measured)


@case
def embedding_ctx_8192_is_cut_to_fit_24gb():
    """the old 8192 embedding context doesn't fit beside the chat model on 24 GiB; the plan halves it to 4096"""
    p = vram.plan(23859, 24576, facts("gemma4", CHAT_SIZE), cfg(embed_ctx=8192, ctx_min=16384),
                  facts("gemma4-mmproj", MMPROJ_SIZE), facts("qwen3-embed", EMBED_SIZE))
    assert p["fits"] and p["embed_ctx"] == 4096 and p["ctx"] == 32768, vram.describe(p)


@case
def levers_go_in_order():
    """short memory gives up embedding context first, then its cache type, then chat context"""
    chat, mm, emb = facts("gemma4", CHAT_SIZE), facts("gemma4-mmproj", MMPROJ_SIZE), facts("qwen3-embed", EMBED_SIZE)
    full = vram.plan(10**6, 24576, chat, cfg(), mm, emb)["need_mib"]
    a = vram.plan(full - 300, 24576, chat, cfg(), mm, emb)
    assert a["fits"] and a["embed_cache"] == "f16" and a["embed_ctx"] < 8192 and a["ctx"] == 32768, a["levers"]
    small = vram.plan(10**6, 24576, chat, cfg(embed_ctx=2048), mm, emb)["need_mib"]
    b = vram.plan(small - 50, 24576, chat, cfg(), mm, emb)
    assert b["fits"] and b["embed_ctx"] == 2048 and b["embed_cache"] == "q8_0" and b["ctx"] == 32768, b["levers"]
    c = vram.plan(small - 200, 24576, chat, cfg(), mm, emb)
    assert c["fits"] and c["embed_cache"] == "q8_0" and 8192 <= c["ctx"] < 32768, c["levers"]
    assert c["ctx"] % 512 == 0


@case
def configured_ctx_is_a_ceiling():
    """with plenty of memory the chat context stays at llama.ctx, never above"""
    p = vram.plan(80000, 81920, facts("gemma4", CHAT_SIZE), cfg(ctx=16384), None, None)
    assert p["fits"] and p["ctx"] == 16384 and not p["levers"]


@case
def no_fit_says_by_how_much():
    """when even the floors don't fit, the plan fails and names the shortfall"""
    p = vram.plan(16000, 16384, facts("gemma4", CHAT_SIZE), cfg(),
                  facts("gemma4-mmproj", MMPROJ_SIZE), facts("qwen3-embed", EMBED_SIZE))
    assert not p["fits"] and p["short_mib"] > 0 and p["ctx"] == 8192
    assert any("does NOT fit" in line and f"{p['short_mib']} MiB short" in line for line in vram.describe(p))


@case
def gpu_false_reserves_nothing():
    """embedding on the CPU (gpu = false): the plan has no embedding parts and one CUDA context"""
    p = vram.plan(23859, 24576, facts("gemma4", CHAT_SIZE), cfg(embed_gpu=False),
                  facts("gemma4-mmproj", MMPROJ_SIZE), None)
    assert not any(k.startswith("embedding") for k in p["parts_mib"])
    assert p["parts_mib"]["CUDA contexts (est.)"] == round(vram.CUDA_CTX_MIB) and p["embed_ctx"] == 0


@case
def context_capped_at_trained_length():
    """chat context is capped at trained context x slots, embedding context at its trained context"""
    p = vram.plan(80000, 81920, facts("tiny", 1000), cfg(ctx=16384, ctx_min=0, parallel=2), None, None)
    assert p["ctx"] == 8192 and p["ctx_max"] == 8192 and any("trained context" in line for line in vram.describe(p))
    p = vram.plan(80000, 81920, facts("gemma4", CHAT_SIZE), cfg(embed_ctx=65536), None, facts("qwen3-embed", EMBED_SIZE))
    assert p["embed_ctx"] == 32768, p["embed_ctx"]


@case
def configured_ctx_rounds_down():
    """a context that isn't a multiple of 256 is rounded down, never up"""
    p = vram.plan(80000, 81920, facts("gemma4", CHAT_SIZE), cfg(ctx=30000, ctx_min=0), None, None)
    assert p["ctx"] == 29952, p["ctx"]


@case
def cli_writes_env_and_exits_by_fit():
    """vram.py plan --stage local writes PLAN_* settings; exit 0 fits, 7 doesn't, 2 unreadable header"""
    env = dict(os.environ, MODEL_PATH=fixture("gemma4", CHAT_SIZE), MMPROJ_PATH=fixture("gemma4-mmproj", MMPROJ_SIZE),
               EMBED_PATH=fixture("qwen3-embed", EMBED_SIZE), LLAMA_CTX="32768", LLAMA_CTX_MIN="8192")
    # The fixtures are sparse files of the real sizes, so getsize gives the real weights.
    out_env, out_json = os.path.join(TMP, "plan.env"), os.path.join(TMP, "plan.json")
    cmd = [sys.executable, os.path.join(HERE, "..", "worker", "vram.py"), "plan", "--stage", "local",
           "--env", out_env, "--json", out_json]
    r = subprocess.run(cmd + ["--free-mib", "23859", "--total-mib", "24576"], env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    kv = dict(line.split("=", 1) for line in open(out_env).read().split())
    assert kv["PLAN_UBATCH"] == "1120" and int(kv["PLAN_CTX"]) >= 8192 and kv["PLAN_EMBED_CACHE"] in ("f16", "q8_0")
    assert json.load(open(out_json))["fits"]
    r = subprocess.run(cmd + ["--free-mib", "12000", "--total-mib", "12288"], env=env, capture_output=True, text=True)
    assert r.returncode == 7 and "does NOT fit" in r.stdout, r.stdout
    r = subprocess.run(cmd + ["--free-mib", "23859"], env=dict(env, MODEL_PATH="/nonexistent"),
                       capture_output=True, text=True)
    assert r.returncode == 2, r.returncode


print(f"\n{len(PASSED)} passed, {len(FAILED)} failed")
sys.exit(1 if FAILED else 0)
