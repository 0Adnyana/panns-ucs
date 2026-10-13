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
audio ──► frozen CNN14 ──► 2048-d embedding (.npz, cached) ──► dropout + linear head ──► SubCategory ──► Category
```

How to run both steps is in [Getting started](../README.md#getting-started).

<a id="extract-embeddings"></a>

### Embedding extraction

`extract_panns_embeddings.py` writes one `.npz` per split. Each file holds
`X` (n_clips × 2048, float32) plus `filename`, `Category`, `SubCategory`,
`source`, and `path`, in the same row order. About 475 MB total.

The script does not use the `panns_inference` package. On Windows that package
calls `wget` on import and crashes. The CNN14 model code is copied in;
embeddings were checked to match the official model exactly.

<a id="linear-probe"></a>

### Linear probe

`train_linear_probe.py` reads only the cached `.npz` files.

Macro F1 is the mean of per-class F1 over classes with at least one true sample
in the scored split. `macro_f1_all_train_classes` scores a missing class as 0.
Std is the sample standard deviation over seeds (`ddof=1`).

---

[README](../README.md) · [Docs](README.md) · **Method** · [Data](data.md) · [Results](results.md) · [Conventions](conventions.md)
