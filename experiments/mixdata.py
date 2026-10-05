"""The ablation's mixture settings, pinned.

The machinery was promoted to components/dataset/mixture.py when the mixture
became part of the mainline train.py. This shim keeps the exact constants the
finished ablation runs (e1, e2) were trained and evaluated with — a tiny
300-article slice at ~25 exposures — so their checkpoints stay reproducible
and experiments/evaluate.py keeps scoring the same wiki-seen/unseen splits.
"""

from components.dataset.mixture import MixedTokens as _MixedTokens
from components.dataset.mixture import wiki_tokens  # noqa: F401  (re-export)

WIKI_DOCS = 300
WIKI_EVAL_DOCS = 100
WIKI_FRAC = 0.2


def MixedTokens(seq_len: int) -> _MixedTokens:
    return _MixedTokens(seq_len, wiki_docs=WIKI_DOCS, wiki_frac=WIKI_FRAC)
