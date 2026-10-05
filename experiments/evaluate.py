"""Score the ablation runs. Run: uv run python -m experiments.evaluate

For every finished experiment (and, when present, the pre-ablation reference
checkpoints), measures:

  - fineweb val   held-out FineWeb-Edu loss: general quality / mixture tax
  - wiki seen     loss on the repeated Wikipedia slice (~15-20 exposures):
                  did the params actually retain the repeated content?
  - wiki unseen   loss on articles never trained on: separates wiki-domain
                  fit from true memorization (seen − unseen = retention gap)
  - facts         30 cloze world facts: greedy first-token accuracy and mean
                  gold log-probability per token

Writes experiments/results.md and prints the table.
"""

import glob
import json
import os

import tiktoken
import torch
from torch.utils.data import DataLoader

from components.dataset.loader import StreamingTokens
from components.dataset.mixture import FINEWEB
from components.model import Thonk
from experiments.mixdata import WIKI_DOCS, WIKI_EVAL_DOCS, wiki_tokens

enc = tiktoken.get_encoding("gpt2")
SEQ_LEN = 1024

FACTS = [
    ("The capital of France is", " Paris"),
    ("The capital of Japan is", " Tokyo"),
    ("The capital of Italy is", " Rome"),
    ("The capital of Russia is", " Moscow"),
    ("The capital of Egypt is", " Cairo"),
    ("The capital of Spain is", " Madrid"),
    ("The capital of Germany is", " Berlin"),
    ("The capital of England is", " London"),
    ("The largest planet in the solar system is", " Jupiter"),
    ("The closest planet to the sun is", " Mercury"),
    ("The chemical symbol for gold is", " Au"),
    ("The chemical symbol for iron is", " Fe"),
    ("Water is made of hydrogen and", " oxygen"),
    ("The powerhouse of the cell is the", " mitochond"),
    ("Romeo and Juliet was written by William", " Shakespeare"),
    ("The theory of relativity was developed by Albert", " Einstein"),
    ("The Mona Lisa was painted by Leonardo da", " Vinci"),
    ("The first president of the United States was George", " Washington"),
    ("World War II ended in the year", " 1945"),
    ("The Great Wall is located in", " China"),
    ("The Eiffel Tower is located in", " Paris"),
    ("The pyramids of Giza are located in", " Egypt"),
    ("The currency of Japan is the", " yen"),
    ("The currency of the United Kingdom is the", " pound"),
    ("The longest river in the world is the", " Nile"),
    ("The largest ocean on Earth is the", " Pacific"),
    ("The human body has 206", " bones"),
    ("The speed of light is approximately 300,000", " kilometers"),
    ("The author of 1984 is George", " Orwell"),
    ("The telephone was invented by Alexander Graham", " Bell"),
]


def batched_loss(model, windows):
    """Mean next-token loss over a list of (x, y) 1D window pairs."""
    total, n = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(windows), 8):
            xs = torch.stack([w[0] for w in windows[i : i + 8]]).cuda()
            ys = torch.stack([w[1] for w in windows[i : i + 8]]).cuda()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, loss = model(xs, ys)
            total += loss.item() * len(xs)
            n += len(xs)
    return total / n


def wiki_windows(skip, take, n_windows=160):
    toks = wiki_tokens(skip, take)
    starts = torch.linspace(0, len(toks) - SEQ_LEN - 2, n_windows).long()
    return [(toks[s : s + SEQ_LEN], toks[s + 1 : s + SEQ_LEN + 1]) for s in starts]


def fact_probe(model):
    """Greedy accuracy on the first gold token + mean gold logprob/token."""
    hits, logprob_sum, tok_count = 0, 0.0, 0
    with torch.no_grad():
        for prompt, gold in FACTS:
            p_ids = enc.encode_ordinary(prompt)
            g_ids = enc.encode_ordinary(gold)
            ids = torch.tensor([p_ids + g_ids], device="cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, _ = model(ids)
            logp = logits.float().log_softmax(-1)[0]
            gold_pos = range(len(p_ids) - 1, len(p_ids) - 1 + len(g_ids))
            hits += int(logp[len(p_ids) - 1].argmax().item() == g_ids[0])
            for pos, tok in zip(gold_pos, g_ids):
                logprob_sum += logp[pos, tok].item()
                tok_count += 1
    return hits / len(FACTS), logprob_sum / tok_count


def newest_checkpoint(d):
    ckpts = sorted(glob.glob(os.path.join(d, "step_*.pt")))
    return ckpts[-1] if ckpts else None


def main():
    print("materializing eval sets...")
    fineweb_loader = iter(
        DataLoader(StreamingTokens(*FINEWEB, "val", SEQ_LEN), batch_size=1)
    )
    fineweb_set = [(x[0], y[0]) for x, y in (next(fineweb_loader) for _ in range(160))]
    seen_set = wiki_windows(0, WIKI_DOCS)
    unseen_set = wiki_windows(WIKI_DOCS, WIKI_EVAL_DOCS)

    rows = []
    # ablation runs, plus the pre-ablation 62-minute reference runs if present
    targets = sorted(glob.glob("experiments/checkpoints/*/"))
    targets += [p for p in ("checkpoints", "checkpoints2") if newest_checkpoint(p)]
    for d in targets:
        path = newest_checkpoint(d)
        if not path:
            continue
        name = os.path.basename(d.rstrip("/"))
        meta = {}
        meta_path = os.path.join(d, "meta.json")
        if os.path.exists(meta_path):
            meta = json.load(open(meta_path))
        ckpt = torch.load(path, map_location="cuda", weights_only=False)
        model = Thonk(ckpt["config"]).cuda()
        model.load_state_dict(ckpt["model"])
        model.eval()

        fw = batched_loss(model, fineweb_set)
        seen = batched_loss(model, seen_set)
        unseen = batched_loss(model, unseen_set)
        acc, lp = fact_probe(model)
        n_params = sum(p.numel() for p in model.parameters())
        rows.append(
            dict(name=name, params_m=n_params / 1e6, steps=ckpt["step"],
                 tokens_m=meta.get("tokens", 0) / 1e6, tok_per_s=meta.get("tok_per_s", ""),
                 fineweb=fw, seen=seen, unseen=unseen, acc=acc, logprob=lp)
        )
        print(f"{name}: fineweb {fw:.3f} | seen {seen:.3f} | unseen {unseen:.3f} | facts {acc:.0%}")
        del model, ckpt
        torch.cuda.empty_cache()

    header = (
        "| run | params | steps | tokens | tok/s | fineweb val ↓ | wiki seen ↓ | wiki unseen ↓ | facts acc ↑ | facts logprob ↑ |\n"
        "|---|---|---|---|---|---|---|---|---|---|\n"
    )
    lines = "".join(
        f"| {r['name']} | {r['params_m']:.0f}M | {r['steps']} | {r['tokens_m']:.0f}M | {r['tok_per_s']} "
        f"| {r['fineweb']:.3f} | {r['seen']:.3f} | {r['unseen']:.3f} | {r['acc']:.0%} | {r['logprob']:.2f} |\n"
        for r in rows
    )
    with open("experiments/results.md", "w") as f:
        f.write("# Knowledge-retention ablation\n\n" + header + lines)
    print("\n" + header + lines)


if __name__ == "__main__":
    main()
