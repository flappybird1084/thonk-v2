"""Stream a hub dataset's parquet files and tokenize them into training windows.

No prepare step: documents are pulled straight from the Hugging Face hub
(streaming, so only what we consume gets downloaded), tokenized with GPT-2
BPE, packed back-to-back with EOT separators, and cut into (input,
next-token target) windows.

Train/val split: each document is routed by a hash of its id — val gets
val_percent in every 100. Hash routing is deterministic no matter how files
are divided across DataLoader workers, so the two splits can never bleed
into each other. The same hash shards the train documents across GPUs when
a run is multi-GPU (rank / world_size), so every rank reads the same files
but trains on its own documents.

Three things here exist purely to keep worker memory sane (a worker once
cost 20+GB of RAM, growing by the minute):

- Files are read with plain pyarrow.parquet, one row group at a time,
  instead of the `datasets` library's streaming. That goes through the
  pyarrow.dataset scanner, which keeps prefetching a file in background
  threads after the iterator has moved on; and `datasets` reshards per row
  group when you shuffle, so every 1000 rows abandoned another scanner —
  ten files downloading at once into memory nobody would read.
  ParquetFile.iter_batches fetches exactly what is asked for, when asked.

- Shuffling happens at the *window* level in our own buffer of plain token
  tensors (~8MB for 1000 windows), on top of a seeded shuffle of file order.

- limit_malloc_arenas (train.py installs it as worker_init_fn): pyarrow's
  thread churn otherwise spreads allocations over ~8 glibc malloc arenas per
  core, and the fragmented heaps — tens of GB — are never returned to the OS.
"""

import ctypes
import random
import zlib

import pyarrow.parquet as pq
import tiktoken
import torch
from huggingface_hub import HfFileSystem

enc = tiktoken.get_encoding("gpt2")


def limit_malloc_arenas(_worker_id=None):
    """Cap glibc at 2 malloc arenas (mallopt M_ARENA_MAX = -8)."""
    try:
        ctypes.CDLL("libc.so.6").mallopt(-8, 2)
    except OSError:
        pass  # not glibc; nothing to tune


def hub_parquet_files(repo: str, pattern: str) -> list[str]:
    """The parquet files of a hub dataset matching a glob, in a stable order."""
    return sorted(HfFileSystem().glob(f"datasets/{repo}/{pattern}"))


def parquet_rows(path: str, columns: list[str]):
    """Yield the rows of one hub parquet file as dicts, one row group at a time.

    Row groups in the datasets we use hold ~1000 documents (a few MB), which
    is the natural unit of download: nothing is fetched ahead of it.
    """
    with HfFileSystem().open(path, "rb") as f:
        parquet = pq.ParquetFile(f)
        for group in range(parquet.num_row_groups):
            yield from parquet.read_row_group(group, columns=columns).to_pylist()


class StreamingTokens(torch.utils.data.IterableDataset):
    def __init__(
        self,
        repo: str,
        pattern: str,
        split: str,
        seq_len: int,
        val_percent: int = 1,
        shuffle_buffer: int = 1000,
        rank: int = 0,
        world_size: int = 1,
    ):
        assert split in ("train", "val")
        self.repo = repo
        self.pattern = pattern
        self.split = split
        self.seq_len = seq_len
        self.val_percent = val_percent
        self.shuffle_buffer = shuffle_buffer
        self.rank = rank
        self.world_size = world_size

    def _documents(self, files):
        for path in files:
            for row in parquet_rows(path, ["id", "text"]):
                h = zlib.crc32(row["id"].encode())
                is_val = h % 100 < self.val_percent
                if is_val != (self.split == "val"):
                    continue
                # Rank routing uses the hash's higher digits, so a rank's
                # share doesn't line up with the val digit. (Sharding by file
                # instead would be lopsided: a phase has only 3-5 files.)
                if self.split == "train" and h // 100 % self.world_size != self.rank:
                    continue
                yield row["text"]

    def _windows(self, files):
        buf = []
        for text in self._documents(files):
            buf.extend(enc.encode_ordinary(text))
            buf.append(enc.eot_token)
            # Windows advance by seq_len (not seq_len+1): the last token of
            # one window is the first input of the next, so every token in
            # the stream is predicted as a target exactly once.
            while len(buf) > self.seq_len:
                window = torch.tensor(buf[: self.seq_len + 1], dtype=torch.long)
                del buf[: self.seq_len]
                yield window[:-1], window[1:]

    def __iter__(self):
        files = hub_parquet_files(self.repo, self.pattern)
        if self.split == "val":
            # Unshuffled, so the val documents are stable across runs.
            yield from self._windows(files)
            return

        # Every worker shuffles the file list the same way, then takes its
        # own slice — so we don't march through one CommonCrawl dump at a
        # time, and no two workers download the same file.
        random.Random(1337).shuffle(files)
        worker = torch.utils.data.get_worker_info()
        if worker is not None:
            files = files[worker.id :: worker.num_workers]

        # Our own shuffle: fill a pool of windows, then yield a random slot
        # and refill it from the stream. torch.initial_seed() differs per
        # DataLoader worker, so workers draw different orders.
        rng = random.Random(torch.initial_seed())
        pool = []
        for item in self._windows(files):
            if len(pool) < self.shuffle_buffer:
                pool.append(item)
                continue
            i = rng.randrange(len(pool))
            yield pool[i]
            pool[i] = item
        rng.shuffle(pool)  # stream exhausted (only ever on tiny datasets)
        yield from pool
