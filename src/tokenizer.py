"""Tokenizer: BPE for English (trained here, from scratch), bytes for regex.

English is ~75% of each sequence; BPE shrinks it ~4x. Regex stays byte-level
so every character of the output is an explicit, exact decision.

ids: 0 PAD, 1 SEP, 2 EOS | 3..258 regex bytes | 259.. english BPE
Sequence: <english BPE> SEP <regex bytes> EOS

    python src/tokenizer.py --vocab 4096     # train on data/train.jsonl
"""
import argparse
import json
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).parents[1]
HERE = Path(__file__).parent
PATH = HERE / "bpe.json" if (HERE / "bpe.json").exists() else ROOT / "data/bpe.json"  # flat release or repo
PAD, SEP, EOS = 0, 1, 2
OFFSET = 3
EN = OFFSET + 256


@lru_cache(1)
def _bpe():
    from tokenizers import Tokenizer
    return Tokenizer.from_file(str(PATH))


def vocab() -> int:
    return EN + _bpe().get_vocab_size()


def encode(s: str) -> list[int]:
    """Regex side: raw bytes."""
    return [b + OFFSET for b in s.encode("utf-8")]


def decode(ids) -> str:
    return bytes(i - OFFSET for i in ids if OFFSET <= i < EN).decode("utf-8", errors="replace")


def english(s: str) -> list[int]:
    return [i + EN for i in _bpe().encode(s).ids]


def english_batch(xs: list[str]) -> list[list[int]]:
    return [[i + EN for i in e.ids] for e in _bpe().encode_batch(xs)]


def prompt(en: str) -> list[int]:
    return english(en) + [SEP]


def pair(en: str, regex: str) -> tuple[list[int], int]:
    """-> (tokens, index of SEP). Loss applies to targets after SEP."""
    p = prompt(en)
    return p + encode(regex) + [EOS], len(p) - 1


def train(vocab_size: int):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    tk = Tokenizer(models.BPE())
    tk.pre_tokenizer = pre_tokenizers.Sequence([  # one token per digit: counts stay legible
        pre_tokenizers.Digits(individual_digits=True), pre_tokenizers.ByteLevel(add_prefix_space=False)])
    tk.decoder = decoders.ByteLevel()
    tr = trainers.BpeTrainer(vocab_size=vocab_size, min_frequency=2,
                             initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    text = (json.loads(l)["en"] for l in open(ROOT / "data/train.jsonl"))
    tk.train_from_iterator(text, tr)
    tk.save(str(PATH))
    return tk


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", type=int, default=4096)
    a = ap.parse_args()
    tk = train(a.vocab)
    rows = [json.loads(l)["en"] for l in open(ROOT / "data/val.jsonl")]
    n_b = sum(len(r.encode()) for r in rows)
    n_t = sum(len(e.ids) for e in tk.encode_batch(rows))
    print(f"vocab {tk.get_vocab_size()}: {n_b / n_t:.2f} bytes/token on val")
    print(tk.encode("validate a ZIP code like 90210-1234, case insensitive").tokens)
