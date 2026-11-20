#!/usr/bin/env python3
"""Capture the reference scores tests/test_qwen3rerank_real.py compares the engine with, or compare an engine with them on the fly.

    python tools/rerank_ref.py capture MODEL_DIR tools/qwen3_reranker_0.6b_reference.json
        runs the checkpoint through transformers with the model card's recipe (the system prompt, the pair formatted as
        <Instruct>/<Query>/<Document>, the answer read as the chance of "yes" against "no" at the last position), in fp32,
        on the cases below, and writes the score, the logit difference and the token count of each.
    python tools/rerank_ref.py check MODEL_DIR ENGINE
        the same cases through transformers and through the engine, side by side.

Needs torch and transformers (not a dependency of anything else): pip install torch transformers. The reference runs in fp32;
RERANK_REF_DTYPE=bfloat16 halves its memory for the 4B and 8B (and then agrees with the engine to the rounding of bf16).
"""
import json
import os
import subprocess
import sys

TASK = "Given a web search query, retrieve relevant passages that answer the query"
# (instruction or None, query, document). The cases cover what the engine does beyond plain text: a custom task, the
# empty document, non-ASCII text, code with tabs and newlines, and a long document (a few hundred tokens).
CASES = [
    (None, "What is the capital of China?", "The capital of China is Beijing."),
    (None, "What is the capital of China?",
     "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is "
     "responsible for the movement of planets around the sun."),
    (None, "Explain gravity", "The capital of China is Beijing."),
    (None, "Explain gravity",
     "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is "
     "responsible for the movement of planets around the sun."),
    ("Judge whether the document is a recipe for the dish in the query", "how do I make pancakes",
     "Whisk flour, milk and two eggs into a batter and fry a ladle of it in a hot buttered pan until golden."),
    (None, "quelle est la capitale de la France ?", "巴黎是法国的首都,也是法国最大的城市 🙂"),
    (None, "print hello in rust", "fn main() {\n\tprintln!(\"hello\");\n}"),
    (None, "anything", ""),
    (None, "history of the printing press",
     " ".join(["Johannes Gutenberg introduced movable type printing to Europe around 1440, and the technology spread "
               "quickly through the cities of the Holy Roman Empire and beyond."] * 12)),
]


def reference(model_dir):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=getattr(torch, os.environ.get("RERANK_REF_DTYPE", "float32"))).eval()
    yes, no = tokenizer.convert_tokens_to_ids("yes"), tokenizer.convert_tokens_to_ids("no")
    prefix = ("<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct "
              "provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n")
    suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    suffix_ids = tokenizer.encode(suffix, add_special_tokens=False)
    out = []
    for instruction, query, document in CASES:
        text = f"<Instruct>: {instruction or TASK}\n<Query>: {query}\n<Document>: {document}"
        body = tokenizer(text, padding=False, truncation="longest_first", return_attention_mask=False,
                         max_length=8192 - len(prefix_ids) - len(suffix_ids))["input_ids"]
        ids = torch.tensor([prefix_ids + body + suffix_ids])
        with torch.no_grad():
            logits = model(input_ids=ids).logits[0, -1].double()
        diff = float(logits[yes] - logits[no])
        out.append({"instruction": instruction, "query": query, "document": document, "tokens": int(ids.shape[1]),
                    "logit": float(f"{diff:.7g}"), "score": float(f"{1 / (1 + torch.exp(torch.tensor(-diff))):.7g}")})
    return out


def engine_scores(model_dir, engine):
    proc = subprocess.Popen([engine, "--model", model_dir, "--serve"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            text=True, encoding="utf-8")
    proc.stdout.readline()
    found = []
    for index, (instruction, query, document) in enumerate(CASES):
        request = {"id": str(index), "query": query, "documents": [document]}
        if instruction:
            request["instruction"] = instruction
        proc.stdin.write("RERANK " + json.dumps(request) + "\n")
        proc.stdin.flush()
        line = proc.stdout.readline()
        keyword, _, rest = line.partition(" ")
        if keyword != "SCORES":
            raise SystemExit(f"case {index}: {line.strip()}")
        body = json.loads(rest)
        found.append({"tokens": body["tokens"][0], "logit": body["logits"][0], "score": body["scores"][0]})
    proc.stdin.close()
    proc.wait(timeout=30)
    return found


def main():
    if len(sys.argv) >= 4 and sys.argv[1] == "capture":
        data = reference(sys.argv[2])
        with open(sys.argv[3], "w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False)
        print(f"wrote {len(data)} cases to {sys.argv[3]}")
    elif len(sys.argv) >= 4 and sys.argv[1] == "check":
        ref, got = reference(sys.argv[2]), engine_scores(sys.argv[2], sys.argv[3])
        for index, (want, have) in enumerate(zip(ref, got)):
            print(f"case {index}: tokens {want['tokens']} vs {have['tokens']}   logit {want['logit']:+.4f} vs "
                  f"{have['logit']:+.4f} (diff {abs(want['logit'] - have['logit']):.2e})   score {want['score']:.6f} vs {have['score']:.6f}")
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
