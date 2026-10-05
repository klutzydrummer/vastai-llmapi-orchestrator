#!/usr/bin/env python3
"""What worker/smoke_test.py accepts as a clean chat reply. Run: python3 tests/test_smoke.py"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "worker"))
import smoke_test  # noqa: E402

FAILED = 0


def reply(content, reasoning=None):
    msg = {"role": "assistant", "content": content}
    if reasoning is not None:
        msg["reasoning_content"] = reasoning
    return {"choices": [{"message": msg, "finish_reason": "stop"}]}


def expect(name, resp, ok, thinking=False):
    global FAILED
    smoke_test.THINKING = thinking
    text, problem = smoke_test.judge(resp)
    if (problem is None) == ok:
        print(f"PASS  {name}")
    else:
        FAILED += 1
        print(f"FAIL  {name}: {text!r} -> {problem}")


expect("plain answer", reply("pong"), True)
expect("answer with punctuation and a newline", reply("Pong!\n"), True)
expect("answer mentioning a user", reply("The user said red."), True)
expect("answer that is only a thought token (rental 54245444)", reply("<|thought|>\n"), False)
expect("thought token before the answer", reply("<|thought|>pong"), False)
expect("role prefix before the answer (rental 54245444)", reply("user\nRed"), False)
expect("model: prefix", reply("model: pong"), False)
expect("Gemma turn tokens", reply("pong<end_of_turn>"), False)
expect("Gemma 4 style turn token", reply("<|turn>model\npong"), False)
expect("empty answer", reply(""), False)
expect("whitespace answer", reply("  \n"), False)
expect("no choices", {"error": "x"}, False)
expect("reasoning only, thinking off", reply("", "Let me think"), False)
expect("reasoning only, thinking on (cut off at max_tokens)", reply("", "Let me think"), True, thinking=True)
expect("leaked token in reasoning, thinking on", reply("", "<|thought|>"), False, thinking=True)
expect("answer plus reasoning, thinking on", reply("pong", "thought"), True, thinking=True)

print(f"\n{FAILED} failed")
sys.exit(1 if FAILED else 0)
