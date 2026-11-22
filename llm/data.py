"""
Problems, prompts and the binary verifier for the LLM experiments.

A problem is a dict {"id", "question", "answer"} where "answer" is the final
number as a string. The verifier is exact-match on the final number, with a
lenient parser; an answer with no parsable number is a *format failure*,
counted separately from a wrong answer (plan v3, C0).
"""

import math
import re

INSTRUCTION = "Solve the problem step by step. End with the final answer on its own line as: #### <number>"

_NUMBER = r"-?\d[\d,]*(?:\.\d+)?"


def normalize(number):
    """'1,234.50' -> '1234.5'; returns None if not a number."""
    text = number.replace(",", "").strip()
    try:
        value = float(text)
    except ValueError:
        return None
    if not math.isfinite(value):
        return None  # e.g. a runaway 400-digit "number": unparsable, not a crash
    return str(int(value)) if value == int(value) else repr(value)


def extract_answer(completion):
    """Final number in a completion: after '####', else in \\boxed{}, else the last number."""
    for pattern in (r"####\s*\$?\s*(" + _NUMBER + ")", r"\\boxed\{\s*\$?(" + _NUMBER + r")\s*\}"):
        found = re.findall(pattern, completion)
        if found:
            return normalize(found[-1])
    found = re.findall(_NUMBER, completion)
    return normalize(found[-1]) if found else None


def reward(completion, answer):
    """(reward, parsed): reward 1.0 iff the parsed final number equals the answer."""
    parsed = extract_answer(completion)
    return (1.0 if parsed is not None and parsed == normalize(answer) else 0.0), parsed is not None


def load_gsm8k(split):
    """GSM8K problems with ids '<split>-<index>'."""
    from datasets import load_dataset

    rows = load_dataset("openai/gsm8k", "main", split=split)
    return [
        {"id": f"{split}-{i}", "question": r["question"], "answer": normalize(r["answer"].split("####")[-1])}
        for i, r in enumerate(rows)
    ]


def prompt_text(tokenizer, problem):
    """Chat-formatted prompt for one problem."""
    messages = [{"role": "user", "content": f"{problem['question']}\n\n{INSTRUCTION}"}]
    # enable_thinking=False turns off the reasoning mode of hybrid models (SmolLM3, Qwen3); templates
    # without the flag ignore it.
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                         enable_thinking=False)
