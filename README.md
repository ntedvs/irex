# irex

English in, JavaScript regex out. A 2.6M-parameter transformer trained from scratch in MLX on an M4 Pro.

```
$ python src/infer.py "a hex color code like #fff or #a1b2c3"
/^#(?:[0-9a-fA-F]{3}){1,2}$/
```

## Usage

```
uv venv --python 3.12 && source .venv/bin/activate
uv pip install mlx numpy tokenizers quickjs openai python-dotenv

python src/infer.py "iso date like 2024-01-31"     # one query
python src/infer.py -k 5 "an email address"         # top 5 candidates
python src/infer.py                                 # interactive
python src/infer.py --ckpt ckpt/m6 "..."            # larger model
```

## Models

| | Params | Size | Config | Test | Bench A | Bench B |
|---|---|---|---|---|---|---|
| **`m3l`** (default) | 2.6M | 10 MB | d192, 4 layers, 4 heads | 67.9% | 82.5% | 65% |
| `m6` | 5.9M | 24 MB | d256, 6 layers, 4 heads | 67.9% | 82.5% | 70% |

- **Test:** behavioral match on 15,058 held-out pairs. The prediction must accept and reject the same strings as the gold regex. Checked on sampled strings, not a proof of equivalence.
- **Bench A:** 40 handwritten everyday requests with strings that must and must not match. Partly informed the rule-based data.
- **Bench B:** 20 handwritten requests written afterward, never tuned on.

## Scope

JS regex at an intermediate level: literals, `.`, classes, `\d \w \s \b` and negations, quantifiers (including lazy), capturing and non-capturing groups, `|`, `^ $`, and the `i` flag. No lookarounds, backrefs, named groups, `\p{}`, or other flags.

## Data

1. **Regexes:** 159k sampled from `src/grammar.py`, about half random compositions and half realistic templates (dates, emails, phones, ...). Each is scope-checked and gets 5 matching and 5 non-matching strings, labeled by real JS semantics via QuickJS. Shapes that backtrack catastrophically are rejected.
2. **English:** DeepSeek-V4.1-Flash, non-thinking, about $11 total. Five styles per regex: casual, terse, precise, purpose, sloppy. Plus 8 short requests for each of 11k template regexes.
3. **Rule-based:** 60k free everyday pairs from `src/common.py`: passwords, lengths, starts/ends/contains, find-in-text, decimals, uuids, ...
4. **Filter:** `src/filter.py` drops descriptions that leak regex syntax and splits by regex text, so test regexes never appear in training. Final split: 815k train / 15k val / 15k test.

## Model and training

- **Architecture:** decoder-only, pre-norm, RMSNorm, RoPE, SwiGLU, tied embeddings. Sequence is `english SEP regex EOS`, with loss on the regex only.
- **Tokenizer:** BPE (4096) trained from scratch for the English, one token per digit. Raw bytes for the regex.
- **Training:** AdamW with cosine decay, bf16 mixed precision (fp32 master weights), EMA of weights, length-bucketed batches, async eval, and casing and punctuation augmentation.
- **`m3l`:** lr 2.5e-3, batch 128, 60k steps (about 9 passes over the data), 40 minutes.
- **Decoding:** beam search (8), then rank by: compiles, matches examples in the request (`src/hints.py`, e.g. "like #fff", "mm/dd/yyyy"), in scope, model score.

## Experiments

Test numbers are on the final 15k test set unless noted. Earlier runs used smaller data and splits.

| Run | Params | Change | Result |
|---|---|---|---|
| byte-level `final` | 10.7M | first full run, 354k pairs | 63.3% (7.6k test) |
| `d512L8` vs 12M | 28M | batch 256, 12 min | slower to converge per minute |
| A/B prefix-LM | 12M | bidirectional attention over English | 56.3% vs 56.7% causal, dropped |
| `big` | 28M | 95 min, 6 passes, 391k pairs | overfit after about 4.6 passes |
| `big2` | 28M | `big` + 55k pairs, fine-tune | 66.3% |
| `xl` / `xl2` | 28M | 617k, then 755k pairs, EMA | 66.5% / 67.2% |
| `m12` | 12M | 755k pairs, 30k steps | 67.9% |
| `m12c` | 12M | `m12` + rule-based data | 67.9%, Bench B 60% to 70% |
| `m6` | 5.9M | from scratch with rule-based data | 67.9% |
| `m3` / `m3l` | 2.6M | 30k / 60k steps | 67.0% / 67.9% |
| `m1` | 1.4M | 60k steps | 67.1%, Bench A 75% |

### Findings

- **Data, not size, is the limit.** Everything from 2.6M to 28M lands at 67 to 68%. 1.4M is where accuracy starts to drop.
- **BPE on the English** shortened sequences 2.3x. Combined with batch 128 instead of 256, one pass went from 16 minutes to 7.
- **bf16 mixed precision** gave 1.5x throughput. The M4 Pro tops out around 4.6 TFLOPS.
- **No gain:** prefix-LM attention, MBR reranking, and 8-bit or 4-bit quantization (no loss either).
- **Example-aware ranking** fixes cases like "mm/dd/yyyy" producing a 2-digit year.
- **The ceiling is mostly ambiguity.** "Precise" requests score 90%, "purpose" requests 46%. About 19% of misses differ only in anchoring.

### Known gaps

Numeric ranges (years 1900 to 2099), MAC addresses, full UUIDs, and requests that don't say whether to match the whole string or anywhere in it.

## Layout

```
src/grammar.py    regex sampler, parser, positive/negative strings
src/engine.py     JS regex via QuickJS
src/generate.py   DeepSeek descriptions (async, resumable, budgeted)
src/common.py     rule-based everyday pairs
src/filter.py     clean + split
src/tokenizer.py  BPE (english) + bytes (regex)
src/model.py      MLX GPT
src/train.py      training
src/evaluate.py   test-set metrics
src/bench.py      everyday benchmarks
src/hints.py      example extraction from requests
src/infer.py      decoding + CLI
ckpt/m3l, ckpt/m6 weights; ckpt/best -> m3l
data/bpe.json     tokenizer (needed for inference)
```

## Reproduce

```
python src/grammar.py --n 60000
python src/generate.py --budget 5            # needs DEEPSEEK_API_KEY in .env
python src/common.py && python src/filter.py
python src/tokenizer.py --vocab 4096
python src/train.py --name m3l --dim 192 --layers 4 --heads 4 --lr 2.5e-3 \
    --dropout 0.05 --ema 0.999 --steps 60000 --eval_every 10000
python src/evaluate.py --ckpt ckpt/m3l && python src/bench.py --ckpt ckpt/m3l --set 2
```

Note: `filter.py` uses `data/regex_short.jsonl` and `data/short.jsonl` if present. Generate them with `generate.py --short` on a template-only pool.
