#!/usr/bin/env python3
"""Writes GGUF files whose headers carry real models' metadata, for tests.

  gguf.py write OUT KIND SIZE   KIND: gemma4, gemma4-mmproj, qwen3-embed, tiny, tiny-embed;
                                the file is padded with zeros to SIZE bytes

The metadata values are the ones the real files report (WaifuGemma4 26B-A4B,
the Gemma 4 BF16 projector, Qwen3-Embedding-0.6B); the tokenizer array has the
real length with one-byte tokens."""

import struct
import sys


class Tokens(int):
    """A string array of this many one-character entries (the tokenizer)."""


def _kv(key, v):
    out = struct.pack("<Q", len(key)) + key.encode()
    if isinstance(v, Tokens):
        return out + struct.pack("<IIQ", 9, 8, int(v)) + (struct.pack("<Q", 1) + b"a") * int(v)
    t, body = _value(v)
    return out + struct.pack("<I", t) + body


def _value(v):
    if isinstance(v, bool):
        return 7, struct.pack("<?", v)
    if isinstance(v, int):
        return 4, struct.pack("<I", v)
    if isinstance(v, float):
        return 6, struct.pack("<f", v)
    if isinstance(v, str):
        return 8, struct.pack("<Q", len(v.encode())) + v.encode()
    if isinstance(v, list):
        et, _ = _value(v[0])
        return 9, struct.pack("<IQ", et, len(v)) + b"".join(_value(x)[1] for x in v)
    raise TypeError(type(v))


def header(meta):
    return b"GGUF" + struct.pack("<IQQ", 3, 0, len(meta)) + b"".join(_kv(k, v) for k, v in meta.items())


def write(path, meta, size=0):
    h = header(meta)
    with open(path, "wb") as f:
        f.write(h)
        if size > len(h):
            f.truncate(size)


def gemma4():
    a = "gemma4"
    return {
        "general.architecture": a, "general.name": "WaifuGemma4 26b a4b v1",
        f"{a}.block_count": 30, f"{a}.context_length": 262144, f"{a}.embedding_length": 2816,
        f"{a}.feed_forward_length": 2112, f"{a}.attention.head_count": 16,
        f"{a}.attention.head_count_kv": [8, 8, 8, 8, 8, 2] * 5,
        f"{a}.attention.sliding_window_pattern": [True, True, True, True, True, False] * 5,
        f"{a}.attention.sliding_window": 1024,
        f"{a}.attention.key_length": 512, f"{a}.attention.value_length": 512,
        f"{a}.attention.key_length_swa": 256, f"{a}.attention.value_length_swa": 256,
        f"{a}.expert_count": 128, f"{a}.expert_used_count": 8, f"{a}.expert_feed_forward_length": 704,
        "tokenizer.ggml.model": "gemma4", "tokenizer.ggml.tokens": Tokens(262144),
    }


def gemma4_mmproj():
    return {"general.architecture": "clip", "clip.projector_type": "gemma4v",
            "clip.has_vision_encoder": True, "clip.vision.embedding_length": 1152,
            "clip.vision.attention.head_count": 16, "clip.vision.patch_size": 16,
            "clip.vision.block_count": 27}


def qwen3_embed():
    a = "qwen3"
    return {
        "general.architecture": a, "general.name": "Qwen3 Embedding 0.6B",
        f"{a}.block_count": 28, f"{a}.context_length": 32768, f"{a}.embedding_length": 1024,
        f"{a}.feed_forward_length": 3072, f"{a}.attention.head_count": 16,
        f"{a}.attention.head_count_kv": 8, f"{a}.attention.key_length": 128,
        f"{a}.attention.value_length": 128, f"{a}.pooling_type": 3,
        "tokenizer.ggml.model": "gpt2", "tokenizer.ggml.tokens": Tokens(151669),
    }


def tiny():
    """An old-style header: no per-layer keys, so the uniform formula applies."""
    a = "llama"
    return {"general.architecture": a, f"{a}.block_count": 4, f"{a}.context_length": 4096,
            f"{a}.embedding_length": 256, f"{a}.attention.head_count": 4,
            "tokenizer.ggml.tokens": Tokens(1000)}


def tiny_embed():
    a = "qwen3"
    return {"general.architecture": a, f"{a}.block_count": 2, f"{a}.context_length": 32768,
            f"{a}.embedding_length": 128, f"{a}.attention.head_count": 2, f"{a}.attention.head_count_kv": 1,
            f"{a}.attention.key_length": 64, f"{a}.attention.value_length": 64, f"{a}.pooling_type": 3,
            "tokenizer.ggml.tokens": Tokens(500)}


KINDS = {"gemma4": gemma4, "gemma4-mmproj": gemma4_mmproj, "qwen3-embed": qwen3_embed, "tiny": tiny,
         "tiny-embed": tiny_embed}

if __name__ == "__main__":
    if len(sys.argv) != 5 or sys.argv[1] != "write" or sys.argv[3] not in KINDS:
        sys.exit(__doc__)
    write(sys.argv[2], KINDS[sys.argv[3]](), int(sys.argv[4]))
