# Method

[README](../README.md) · [Docs](README.md) · **Method** · [Data](data.md) · [Results](results.md) · [Conventions](conventions.md)

The base case this repo reproduces: a linear probe on frozen PANNs CNN14
embeddings, from Beck & Lerch (DAFx26).

**On this page:** [Paper](#reference-paper) · [Settings](#paper-method) · [Target scores](#target-numbers) · [Pipeline](#pipeline)

<a id="reference-paper"></a>

## Reference paper

**Beck & Lerch, DAFx26 — arXiv [2606.05571](https://arxiv.org/pdf/2606.05571).**
The paper relabels AudioSet, FSD50K, and ESC-50 into UCS v8.2.1 and benchmarks
frozen audio encoders on those labels. Their label CSVs and splits are used
as-is, so this project does no label conversion of its own.

- Repos: `JunWooBeck/envsound-ucs` (EnvSound splits), `JunWooBeck/audioset-ucs`, `JunWooBeck/ucs-sfx-tools`
- Finding reused here: predicting the **subcategory and rolling it up to the category** beats predicting the category directly.

<a id="paper-method"></a>

## Settings

"Replicate the paper" means this setup. The paper does not fine-tune PANNs.
It is a linear probe on frozen CNN14 embeddings.

| Setting | Value |
| --- | --- |
| Encoder | PANNs CNN14, 32 kHz, checkpoint `Cnn14_mAP=0.431.pth` (~327 MB), frozen |
| Embedding | 2048-d per clip, extracted once and cached |
| Head | Dropout 0.3 → one linear layer (2048 → n_classes) |
| Optimizer | AdamW, lr 1e-3, weight decay 1e-3, cosine schedule |
| Loss | Focal loss, γ = 2, inverse-frequency class weights |
| Stopping | Early stopping on val, patience 20, max 200 epochs |
| Seeds | 42, 123, 456, 789, 1024 → report mean ± std |
| Metric | Macro F1 on test |
| Balancing | None on the data; imbalance is handled by the loss + weights |

Details the paper leaves unspecified (batch size, which validation score stops
training, how shared subcategory names roll up) are fixed in
[Conventions](conventions.md#conventions). Commands are in
[Getting started](../README.md#getting-started).

<a id="target-numbers"></a>

## Target scores

Macro F1 from the paper. Being within a couple of points means the base case
is reproduced. Scores from this repo are in [Results](results.md#results).

| Dataset / setup | Paper F1 |
| --- | --- |
| EnvSound, Category classifier (59 classes) | 0.46 |
| EnvSound, SubCategory classifier (144 classes) | 0.49 |
| EnvSound, SubCategory → Category | **0.55** |
| FSD-env, flat subcategory / → category | 0.66 / **0.72** |
| FSD-env, predict category directly | 0.53 |
| FSD-env, hierarchical with true category given | 0.88 (upper bound, not reachable as is) |
| ESC-50, flat subcategory | 0.95 (too easy to separate models) |

<a id="pipeline"></a>

## Pipeline

```
audio ──► frozen CNN14 ──► embeddings.npz (X, filename, source)
                                   │
manifests (train/val/test CSV) ──► join on filename + source ──► dropout + linear head ──► SubCategory ──► Category
```

The embedding depends only on the audio and the frozen CNN14. The UCS labels
and the split each clip belongs to depend on the labeling method. So the cache
holds no labels, and the probe reads labels and splits from the manifests.
A new label set is three CSVs (`--train`, `--eval`, `--test`; any filenames)
and a new probe `--out`, with the same `--embeddings`. `--manifests` still
accepts a folder of `envsound_{train,val,test}.csv`. Nothing is re-extracted,
as long as every clip in the new CSVs is already in the cache.

How to run both steps is in [Getting started](../README.md#getting-started).

<a id="extract-embeddings"></a>

### Embedding extraction

`extract_panns_embeddings.py` writes one `embeddings.npz` for all splits. It
holds `X` (n_clips × 2048, float32) plus `filename` and `source`, in the same
row order. `source` is the dataset the clip came from (AudioSet, FSD50K,
ESC-50), not a UCS label. The lookup key is `filename` plus `source`, so two
datasets can use the same clip name. A clip that is in more than one split is
stored once. `Category`, `SubCategory`, and `path` are not stored. About 475 MB
total.

The extractor reads the manifests only to learn which files to embed. If the
same `filename` and `source` point at two different files, it stops and prints
both paths.

Caches written before this layout have one `envsound_{split}.npz` per split,
with labels inside. `--combine-from` stacks them into one `embeddings.npz`
without CNN14.

The script does not use the `panns_inference` package. On Windows that package
calls `wget` on import and crashes. The CNN14 model code is copied in;
embeddings were checked to match the official model exactly.

<a id="linear-probe"></a>

### Linear probe

`train_linear_probe.py` reads `embeddings.npz` and the three manifests. It
never opens audio. For each manifest row whose `exists` is not false, it looks
up `filename` plus `source` in the cache. Labels, splits, and row order come
from the manifest. If any row has no vector, or the same pair appears twice in
one manifest or in more than one split, it stops and prints the rows. Cache rows
that no manifest uses are counted and ignored. `summary.json` records
`settings.embeddings` and the three CSV paths (`settings.train`, `settings.eval`,
`settings.test`). `settings.manifests` is the folder when `--manifests` was used.
`--verbose` writes `verbose_test.csv` in the pair output folder. Every test
row is kept. The top 5 labels are the pair classes with the highest mean
softmax across seeds.

Macro F1 is the mean of per-class F1 over classes with at least one true sample
in the scored split. `macro_f1_all_train_classes` scores a missing class as 0.
Std is the sample standard deviation over seeds (`ddof=1`).

---

[README](../README.md) · [Docs](README.md) · **Method** · [Data](data.md) · [Results](results.md) · [Conventions](conventions.md)
