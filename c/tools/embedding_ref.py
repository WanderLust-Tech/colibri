#!/usr/bin/env python3
"""Capture the reference vectors tests/test_qwen3embed_real.py compares the engine with, or compare an engine with them on the fly.

    python tools/embedding_ref.py capture MODEL_DIR tools/qwen3_embedding_0.6b_reference.json
        runs the checkpoint through transformers (the model card's recipe: last-token pooling with left padding, then L2
        normalisation, in fp32) on the cases below and writes the vectors.
    python tools/embedding_ref.py check MODEL_DIR ENGINE
        runs the same cases through transformers and through the engine and prints the cosine and the largest difference of each.

Needs torch and transformers (not a dependency of anything else): pip install torch transformers.
"""
import json
import subprocess
import sys

# (text, instruction, dimensions). The cases cover what the engine does beyond plain text: the query prefix, the empty text,
# non-ASCII text, code with tabs and newlines, a cut to fewer dimensions.
CASES = [
    ("The capital of France is Paris.", None, None),
    ("What is the capital of France?", "Given a web search query, retrieve relevant passages that answer the query", None),
    ("", None, None),
    ("Ünïcödé 你好,世界 — привет 🙂 12345", None, None),
    ("fn main() { println!(\"hi\"); }\n\n\tindented", None, None),
    ("The capital of France is Paris.", None, 256),
]


def reference(model_dir):
    import torch
    import torch.nn.functional as F
    from transformers import AutoModel, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir, padding_side="left")
    model = AutoModel.from_pretrained(model_dir, dtype=torch.float32).eval()
    out = []
    for text, instruction, dims in CASES:
        full = f"Instruct: {instruction}\nQuery:{text}" if instruction else text
        batch = tokenizer([full], padding=True, truncation=True, max_length=8192, return_tensors="pt")
        with torch.no_grad():
            hidden = model(**batch).last_hidden_state[:, -1]
        if dims:
            hidden = hidden[:, :dims]
        out.append({"text": text, "instruction": instruction, "dimensions": dims,
                    "tokens": int(batch["input_ids"].shape[1]),
                    "vector": [float(f"{x:.7g}") for x in F.normalize(hidden, p=2, dim=1)[0].tolist()]})
    return out


def engine_vectors(model_dir, engine):
    proc = subprocess.Popen([engine, "--model", model_dir, "--serve"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            text=True, encoding="utf-8")
    proc.stdout.readline()
    found = []
    for index, (text, instruction, dims) in enumerate(CASES):
        request = {"id": str(index), "inputs": [text]}
        if instruction:
            request["instruction"] = instruction
        if dims:
            request["dimensions"] = dims
        proc.stdin.write("EMBED " + json.dumps(request) + "\n")
        proc.stdin.flush()
        body = json.loads(proc.stdout.readline().partition(" ")[2])
        found.append({"tokens": body["tokens"][0], "vector": body["vectors"][0]})
    proc.stdin.close()
    proc.wait(timeout=60)
    return found


def main():
    if len(sys.argv) == 4 and sys.argv[1] == "capture":
        cases = reference(sys.argv[2])
        with open(sys.argv[3], "w", encoding="utf-8") as handle:
            json.dump({"model": "Qwen3-Embedding-0.6B", "recipe": "transformers AutoModel, fp32, last token, left padding, L2",
                       "cases": cases}, handle, ensure_ascii=False, separators=(",", ":"))
        print("wrote", sys.argv[3], len(cases), "cases")
    elif len(sys.argv) == 4 and sys.argv[1] == "check":
        want, got = reference(sys.argv[2]), engine_vectors(sys.argv[2], sys.argv[3])
        for case, w, g in zip(CASES, want, got):
            cosine = sum(a * b for a, b in zip(w["vector"], g["vector"]))
            worst = max(abs(a - b) for a, b in zip(w["vector"], g["vector"]))
            print(f"{case[0][:30]!r:34} tokens {g['tokens']}/{w['tokens']}  cosine {cosine:.7f}  max diff {worst:.2e}")
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
