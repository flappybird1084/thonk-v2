"""Chat data for supervised fine-tuning: the template, the sources, the packing.

Template — ChatML, as used by Qwen:

    <|im_start|>system
    You are ...<|im_end|>
    <|im_start|>user
    Hi!<|im_end|>
    <|im_start|>assistant
    Hello!<|im_end|>

GPT-2 BPE has no chat tokens, but the model's vocab is padded from 50257 to
50304, so ids 50257+ exist in the embedding and lm_head and were simply never
seen in pretraining. <|im_start|> and <|im_end|> take the first two; their
rows start from random init and are learned during fine-tuning.

Loss is taken only on what the assistant says (its content and the closing
<|im_end|>, which is how it learns to stop): system and user turns are
context, not targets. Masked targets are -100, cross_entropy's ignore_index.

Sources:
- smol-smoltalk and smoltalk (HuggingFaceTB): instruction / multi-turn chat in
  the standard `messages` format.
- Synthetic-Persona-Chat (Google): two people with written personas chatting.
  User 2 becomes the assistant, with its persona as the system prompt — so
  the model learns to stay in character for whatever persona it's given.
"""

import csv
import random

import pyarrow.parquet as pq
import tiktoken
import torch
from huggingface_hub import HfFileSystem

from components.dataset.loader import hub_parquet_files

_gpt2 = tiktoken.get_encoding("gpt2")
IM_START = 50257
IM_END = 50258
chat_enc = tiktoken.Encoding(
    name="gpt2_chatml",
    pat_str=_gpt2._pat_str,
    mergeable_ranks=_gpt2._mergeable_ranks,
    special_tokens={
        **_gpt2._special_tokens,
        "<|im_start|>": IM_START,
        "<|im_end|>": IM_END,
    },
)

SMOL_SMOLTALK = ("HuggingFaceTB/smol-smoltalk", "data/train-*.parquet")
SMOL_SMOLTALK_TEST = ("HuggingFaceTB/smol-smoltalk", "data/test-*.parquet")
SMOLTALK = ("HuggingFaceTB/smoltalk", "data/all/train-*.parquet")
# The two training CSVs together: ~20k conversations. The valid/test CSVs
# are left out.
PERSONA_CHAT = [
    "datasets/google/Synthetic-Persona-Chat/data/Synthetic-Persona-Chat_train.csv",
    "datasets/google/Synthetic-Persona-Chat/data/New-Persona-New-Conversations.csv",
]


def render(messages: list[dict]) -> tuple[list[int], list[int]]:
    """Tokenize a conversation; returns (ids, mask), mask 1 = train on it.

    Starts with EOT, the document boundary the model knows from pretraining.
    """
    ids, mask = [_gpt2.eot_token], [0]
    for m in messages:
        header = [IM_START] + chat_enc.encode_ordinary(m["role"] + "\n")
        body = chat_enc.encode_ordinary(m["content"]) + [IM_END]
        newline = chat_enc.encode_ordinary("\n")
        trained = 1 if m["role"] == "assistant" else 0
        ids += header + body + newline
        mask += [0] * len(header) + [trained] * len(body) + [0] * len(newline)
    return ids, mask


def prompt_ids(messages: list[dict]) -> list[int]:
    """The rendered conversation plus an open assistant header — what the
    model continues at inference."""
    ids, _ = render(messages)
    return ids + [IM_START] + chat_enc.encode_ordinary("assistant\n")


# ---- sources: each an endless generator of `messages` lists ----


def smoltalk_conversations(repo: str, pattern: str, rng: random.Random):
    """Conversations from a smoltalk-style parquet dataset, forever.

    Read one row group (~1000 conversations) at a time, in a random order
    over all groups of all files: the `all` split is concatenated subset by
    subset, so reading front to back would serve long runs of one subset.
    """
    fs = HfFileSystem()
    groups = []
    for path in hub_parquet_files(repo, pattern):
        with fs.open(path, "rb") as f:
            n = pq.ParquetFile(f).num_row_groups
        groups += [(path, g) for g in range(n)]
    while True:
        rng.shuffle(groups)
        for path, g in groups:
            with fs.open(path, "rb") as f:
                rows = pq.ParquetFile(f).read_row_group(g, columns=["messages"]).to_pylist()
            rng.shuffle(rows)
            for row in rows:
                yield row["messages"]


def persona_to_messages(row: dict) -> list[dict]:
    """One Synthetic-Persona-Chat row as a system + user/assistant chat.

    The conversation is one string of "User 1: ..." / "User 2: ..." lines;
    an unprefixed line continues the previous turn, and back-to-back lines
    from the same speaker are merged into one turn.
    """
    persona = row["user 2 personas"].strip()
    messages = [{"role": "system", "content": "Your persona:\n" + persona}]
    for line in row["Best Generated Conversation"].strip().split("\n"):
        if line.startswith("User 1:"):
            role, text = "user", line.removeprefix("User 1:").strip()
        elif line.startswith("User 2:"):
            role, text = "assistant", line.removeprefix("User 2:").strip()
        else:
            messages[-1]["content"] += "\n" + line
            continue
        if messages[-1]["role"] == role:
            messages[-1]["content"] += "\n" + text
        else:
            messages.append({"role": role, "content": text})
    return messages


def persona_conversations(rng: random.Random):
    """Synthetic-Persona-Chat conversations, forever (small: held in memory)."""
    fs = HfFileSystem()
    conversations = []
    for path in PERSONA_CHAT:
        with fs.open(path, "r", encoding="utf-8") as f:
            conversations += [persona_to_messages(row) for row in csv.DictReader(f)]
    while True:
        rng.shuffle(conversations)
        yield from conversations


class ChatWindows(torch.utils.data.IterableDataset):
    """Mixed chat sources, packed into (input, target) windows for SFT.

    `mix` maps source name -> share of conversations, so a small source can
    be weighted above its size (it repeats) without drowning the others.

    Packing: whole conversations are placed back to back in a seq_len+1
    window until the next one doesn't fit; the remainder is padded with EOT
    and masked out. A conversation longer than a window is cut to the window
    — it keeps its opening turns, which is most of the signal — and dropped
    if no assistant tokens survive the cut.

    Meant for a single DataLoader worker: every worker would read the same
    sources in its own order, so more workers would mean duplicates.
    """

    def __init__(self, seq_len: int, mix: dict[str, float], seed: int = 1337):
        self.seq_len = seq_len
        self.mix = mix
        self.seed = seed

    def _sources(self, rng: random.Random) -> dict:
        makers = {
            "smol-smoltalk": lambda: smoltalk_conversations(*SMOL_SMOLTALK, rng),
            "smoltalk": lambda: smoltalk_conversations(*SMOLTALK, rng),
            "persona": lambda: persona_conversations(rng),
            "smol-smoltalk-test": lambda: smoltalk_conversations(*SMOL_SMOLTALK_TEST, rng),
        }
        return {name: makers[name]() for name in self.mix}

    def __iter__(self):
        rng = random.Random(self.seed)
        sources = self._sources(rng)
        names, weights = list(self.mix), list(self.mix.values())

        window = self.seq_len + 1
        ids, mask = [], []
        while True:
            name = rng.choices(names, weights)[0]
            conv_ids, conv_mask = render(next(sources[name]))
            conv_ids, conv_mask = conv_ids[:window], conv_mask[:window]
            if not any(conv_mask[1:]):  # nothing to learn (mask[0] is never a target)
                continue
            if len(ids) + len(conv_ids) > window:
                yield self._pack(ids, mask)
                ids, mask = [], []
            ids += conv_ids
            mask += conv_mask

    def _pack(self, ids: list[int], mask: list[int]):
        pad = self.seq_len + 1 - len(ids)
        ids = torch.tensor(ids + [_gpt2.eot_token] * pad, dtype=torch.long)
        mask = torch.tensor(mask + [0] * pad, dtype=torch.bool)
        x, y = ids[:-1], ids[1:].clone()
        y[~mask[1:]] = -100  # target t is token t+1, so it takes token t+1's mask
        return x, y
