"""FineWeb stream mixed with a small repeated Wikipedia slice.

Why repetition: facts move from "seen" to "stored" only with many exposures
(Allen-Zhu & Li's knowledge-capacity scaling laws — capacity collapses when a
fact appears a handful of times, which is what one streaming pass gives).
So alongside the endless FineWeb stream, a fixed slice of Wikipedia is held
in memory and cycled: every pass is one more exposure of every fact in it.

Sizing rule of thumb: exposures ≈ (run tokens × wiki_frac) / slice tokens.
The knowledge-retention ablation (experiments/) validated the effect — and
showed that a too-small slice at ~25 exposures memorizes verbatim but hurts
generalization, hence train.py's larger slice.

Articles at the head of the dump average ~4k tokens each (the very first
ones are longer): 3,500 articles ≈ 14M tokens. Past the head they shorten
to ~1k tokens each: 60k articles ≈ 70M, 135k ≈ 147M.

The FineWeb side can be a sequence of phases — shard sets played back to
back, each read to the end before the next starts. train.py uses this for
its two pretraining phases over FineWeb-Edu's sample/10BT shards. The wiki
slice is shared across phases, so a phase switch costs no re-tokenization
and doesn't reset the slice's exposure count.
"""

import random

import torch

from components.dataset.loader import StreamingTokens, enc, hub_parquet_files, parquet_rows

FINEWEB = ("HuggingFaceFW/fineweb-edu", "data/*/*.parquet")
# sample/10BT: 14 shards of ~730k documents (~0.77B tokens) each, already
# sampled across every CommonCrawl dump — so a few shards are a fair cross
# section of the whole corpus, unlike the first few files under data/, which
# all come from the 2013 crawl. Same document ids as data/, so the
# hash-routed val split held out from FINEWEB is held out here too.
FINEWEB_EDU_PHASE1 = ("HuggingFaceFW/fineweb-edu", "sample/10BT/00[0-2]_*.parquet")
FINEWEB_EDU_PHASE2 = ("HuggingFaceFW/fineweb-edu", "sample/10BT/00[3-7]_*.parquet")
WIKIPEDIA = ("wikimedia/wikipedia", "20231101.en/*.parquet")


def wiki_tokens(skip: int, take: int) -> torch.Tensor:
    """Tokenize a contiguous run of articles into one flat token tensor."""
    articles = (
        row["text"]
        for path in hub_parquet_files(*WIKIPEDIA)
        for row in parquet_rows(path, ["text"])
    )
    toks = []
    for i, text in enumerate(articles):
        if i >= skip + take:
            break
        if i >= skip:
            toks.extend(enc.encode_ordinary(text))
            toks.append(enc.eot_token)
    return torch.tensor(toks, dtype=torch.long)


class MixedTokens(torch.utils.data.IterableDataset):
    def __init__(
        self,
        seq_len: int,
        wiki_docs: int,
        wiki_frac: float,
        fineweb_phases: tuple[tuple[str, str], ...] = (FINEWEB,),
        rank: int = 0,
        world_size: int = 1,
    ):
        self.seq_len = seq_len
        self.wiki_docs = wiki_docs
        self.wiki_frac = wiki_frac
        self.fineweb_phases = fineweb_phases
        self.rank = rank  # FineWeb is sharded across GPUs; the wiki slice is not
        self.world_size = world_size

    def _fineweb(self):
        """Each phase's (repo, pattern) streamed to exhaustion, in order.

        Inside a DataLoader worker, StreamingTokens hands each worker its own
        slice of a phase's files — so with more workers than a phase has
        files, some workers would skip straight to the next phase.
        """
        for i, (repo, pattern) in enumerate(self.fineweb_phases):
            if len(self.fineweb_phases) > 1:
                print(f"fineweb phase {i + 1}/{len(self.fineweb_phases)}: {pattern}")
            yield from StreamingTokens(
                repo, pattern, "train", self.seq_len,
                rank=self.rank, world_size=self.world_size,
            )

    def __iter__(self):
        fineweb = self._fineweb()
        wiki = wiki_tokens(0, self.wiki_docs)  # held in memory; ~8MB per 1M tokens
        rng = random.Random(torch.initial_seed())
        starts = list(range(0, len(wiki) - self.seq_len - 1, self.seq_len))
        order = []  # refilled and reshuffled each pass = one more exposure of the slice
        while True:
            if rng.random() < self.wiki_frac:
                if not order:
                    order = starts.copy()
                    rng.shuffle(order)
                s = order.pop()
                window = wiki[s : s + self.seq_len + 1]
                yield window[:-1], window[1:]
            else:
                window = next(fineweb, None)
                if window is None:  # the last phase ran dry: end the stream
                    return
                yield window
