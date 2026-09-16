# Neural cryptanalysis of classical ciphers on Turkish text

Transformer-based, ciphertext-only cryptanalysis of the monoalphabetic
substitution cipher and the Vigenère cipher, evaluated against full classical
cryptanalysis pipelines.

Both ciphers are attacked under a random-key regime: every sample is encrypted
at the moment it is used, with a key drawn independently at random, so no key is
seen twice and the attack cannot succeed by memorisation.

## Layout

```
src/
  config.py                     paths, alphabet, checkpoint discovery
  generators/                   on-the-fly encryption datasets
  models/
    substitution_bte/           encoder-only substitution model
    substitution_fate/          BTE plus an explicit frequency pathway
    substitution_common/        shared data module, decoding, classical solver
    vigenere/                   models, data module, classical solver, evaluation
notebooks/                      training and evaluation
data/                           corpora (not tracked)
```

## Vigenère models

`model_pate.py` is the model reported as PATE. The others are the baselines and
ablation arms; all share the same backbone, parameter budget and training
schedule, and differ only in the position encoding and in how a candidate period
is supplied.

| file | position encoding | period conditioning | heads | passes |
|---|---|---|---|---|
| `model_pate.py` | rotary | period vector | plaintext, key-length, validity | 2 |
| `model_pate_la.py` | learned absolute | period vector | plaintext, key-length, validity | 2 |
| `model_vanilla.py` | learned absolute | none | plaintext | 1 |
| `model_rope_only.py` | rotary | none | plaintext | 1 |
| `model_cycle_rope.py` | rotary | per-cycle embeddings | plaintext, key-length, validity | 2 |
| `model_cycle_la.py` | learned absolute | per-cycle embeddings | plaintext, key-length, validity | 2 |
| `model_cycle_sin.py` | fixed sinusoidal | per-cycle embeddings | plaintext, key-length, validity | 2 |

The period vector says which period holds but nothing about which positions
share a shift; per-cycle embeddings, indexed by `(position mod d)`, supply that
alignment explicitly. PATE uses the former: per-candidate tables do not scale to
wide key ranges.

The three `cycle_*` models are diagnostic. PATE does not converge under absolute
encoding, so the failure of absolute encoding cannot be examined in PATE itself;
supplying more structure makes it converge on the narrower range, and what
remains unresolved is attributable to the encoding alone.

## Notebooks

| notebook | what it does |
|---|---|
| `substitution_bte_train.ipynb` | trains BTE |
| `substitution_fate_train.ipynb` | trains FATE |
| `substitution_comparison.ipynb` | BTE vs FATE vs classical, with a bootstrap over the difference |
| `vigenere_train_pate.ipynb` | trains PATE |
| `vigenere_train_ablations.ipynb` | trains the baselines and the absolute-encoding arm |
| `vigenere_eval.ipynb` | decoding grid, latency, out-of-distribution check |
| `vigenere_encoding_analysis.ipynb` | position embedding spectrum and period confusion matrix |

## Data

Not tracked. Place under `data/`:

- `final_dataset_shuffled.txt` — training corpus
- `validation_set.txt` — held-out corpus for evaluation
- `ood_text.txt` — Turkish text of a different register, for the
  out-of-distribution check

`run_ood_eval` reports the character-frequency distance between the
out-of-distribution and validation corpora alongside the results. A distance
near zero means the two share a distribution and the comparison is
uninformative.

## Running

```bash
pip install -r requirements.txt
```

Notebooks add `src/` to the path themselves and read data and checkpoints
through `config.py`. Locally everything resolves under the repository root; on
Colab, under `MyDrive/cryptanalysis-ai`. Override with `CRYPT_DRIVE_DIR`,
`CRYPT_DATA_DIR`, `CRYPT_CKPT_DIR` or `CRYPT_LOG_DIR` if the layout differs.

Vigenère training runs in 32-bit precision: at the wider key range the
key-length cross-entropy over thirty classes overflows in fp16 and the loss
becomes NaN.
