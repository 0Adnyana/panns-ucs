# UCS Sound Tagger (Tugas Akhir)

Bachelor's thesis project: a machine-learning tool that tags sound effect files
with **Universal Category System (UCS)** labels, the category / subcategory
scheme sound libraries are organised by (e.g. `DOORS-WOOD`, `WEAPONS-GUNSHOT`).

Given an audio file, the tagger predicts its UCS **subcategory**. The
**category** is looked up from that prediction. The base case is a linear probe
on frozen PANNs CNN14 embeddings, following Beck & Lerch (DAFx26).

## Documentation

Background lives in [docs/](docs/README.md).

| Page | What it covers |
| --- | --- |
| [Method](docs/method.md) | The paper, the linear-probe setup, target scores |
| [Data](docs/data.md) | EnvSound on disk, manifest columns, known issues |
| [Results](docs/results.md) | Scores from this repo and what to check next |
| [Conventions](docs/conventions.md) | Rules for code in this repo, and open choices |

<a id="getting-started"></a>

## Getting started

How to install, extract embeddings, and train the linear probe. Run both
scripts from this repo folder in PowerShell. Extract once, then train on the
cached files.

**On this page:** [Install](#install) · [Checkpoint](#checkpoint) · [Extract](#extract) · [Extract flags](#extract-flags) · [Train](#train) · [Train flags](#train-flags)

<a id="install"></a>

### Install

```powershell
pip install -r requirements.txt
```

`requirements.txt` pins the CUDA 13.0 build of PyTorch (`2.14.1+cu130`). A
plain `pip install torch` can install the `+cpu` wheel, and then
`torch.cuda.is_available()` stays False. On CPU the full set (~58k clips)
takes several hours.

<a id="checkpoint"></a>

### Checkpoint

`extract_panns_embeddings.py` looks for `models/Cnn14_mAP=0.431.pth` next to
itself (about 327 MB). That file is gitignored. If it is missing or smaller
than ~300 MB, the script downloads it from Zenodo into that path on the first
run. Pass `--checkpoint` only if the weights live somewhere else. Which
checkpoint to use is in the [code conventions](docs/conventions.md#conventions).

<a id="extract"></a>

### Extract embeddings

`extract_panns_embeddings.py` reads the [manifests](docs/data.md#manifests), runs
each clip through frozen CNN14, and saves one `.npz` per split under `--out`
(default `embeddings/panns_cnn14/`). Run it as a script
(long, unattended, resumable). What the cache contains is under
[embedding extraction](docs/method.md#extract-embeddings).

The [manifests](docs/data.md#manifests) already have a `path` column, so
`--audio-root` is not needed. Relative paths resolve against the parent of the
folder passed to `--manifests`. Absolute paths are used as written. Pass
`--audio-root` only to override that base, or when a manifest has no path
column (`path`, `filepath`, `file_path`, `audio_path`, or `abs_path`); in that
case the script searches the tree by filename. Rows with `exists` set to false
are skipped ([known data issues](docs/data.md#data-issues)).

#### Smoke test

First 50 rows of each split. Output is `<--out>/_smoke_50/` (default
`embeddings/panns_cnn14/_smoke_50/`), so a smoke test stays out of a full run
in the same folder:

```powershell
python extract_panns_embeddings.py --manifests "E:\Sounds for TA\Extracted\manifests" --limit 50
```

You should see `Device: cuda` (or `cpu`), then `audio root: E:\Sounds for TA\Extracted`,
then one `saved ... X=(50, 2048)  failed=0` line per split. The first 50 rows
of each split exist on disk, so a non-zero `failed` count here is a path
problem. Open `envsound_*_failed.csv`: if `error` is `file not found` and
`path` contains `\manifests\`, the resolver used the manifests folder as the
base. Paths are relative to the data root, described in
[Manifests](docs/data.md#manifests).

#### Full run

```powershell
python extract_panns_embeddings.py --manifests "E:\Sounds for TA\Extracted\manifests"
```

<a id="extract-output"></a>

#### Output

Under `--out` (default `embeddings/panns_cnn14/`):

| File | Contents |
| --- | --- |
| `envsound_{train,val,test}.npz` | `X` plus `filename`, `Category`, `SubCategory`, `source`, `path` |
| `envsound_{split}_failed.csv` | clips that could not be read |
| `extract_info.json` | sample rate, embedding definition, and per-split counts |

While a split is still running, progress sits in `envsound_{split}.partial.npz`.
Stop the script and run the same command again to continue. `--restart` throws
that partial file away and starts the split over. `--save-every` (default 2000)
is how often the partial file is rewritten.

`embeddings/` is gitignored.

<a id="extract-flags"></a>

#### Flags

| Flag | Default | What it does |
| --- | --- | --- |
| `--manifests` | required | Folder with `envsound_{train,val,test}.csv`. |
| `--audio-root` | parent of that folder | Base for relative paths, or the tree to search by filename when the manifest has no path column. |
| `--out` | `embeddings/panns_cnn14` | Folder for this run. A new name starts a separate cache. The same name resumes it. |
| `--checkpoint` | `models/Cnn14_mAP=0.431.pth` | CNN14 weights, next to the script. Downloaded there if missing. See [Checkpoint](#checkpoint). |
| `--splits` | `train,val,test` | Comma-separated manifests to embed. |
| `--device` | `auto` | `auto`, `cuda`, or `cpu`. |
| `--limit` | `0` (every row) | First N rows of each split. Writes `<--out>/_smoke_<N>/`. |
| `--save-every` | `2000` | Rewrite `envsound_{split}.partial.npz` every N new clips. |
| `--restart` | off | Ignore the partial file and start the split over. |

To keep two extractions apart, pass a different `--out`:

```powershell
python extract_panns_embeddings.py --manifests "E:\Sounds for TA\Extracted\manifests" --out embeddings/run_a
```

<a id="train"></a>

### Train the linear probe

`train_linear_probe.py` loads the `.npz` files from `--embeddings` (never audio
or CNN14). It trains the paper's head with the [paper's settings](docs/method.md#paper-method).
Runs take minutes, even on CPU. Output goes to `--out/<task>/` (default
`results/panns_cnn14_linear/<task>/`). That folder is gitignored, same as
`embeddings/`. Metric definitions are under [linear probe](docs/method.md#linear-probe).

Empty subcategories and shared subcategory names are covered in
[known data issues](docs/data.md#data-issues). Training choices that the paper leaves
unspecified are in the [code conventions](docs/conventions.md#conventions).

<a id="tasks"></a>

#### Tasks

`--task` (default `subcategory`):

| Task | Labels | EnvSound classes |
| --- | --- | --- |
| `subcategory` | Drop empty SubCategory. Class = SubCategory name. Roll-up uses the most common parent category of that name in the training split; ties by name. | 144 |
| `pair` | Drop empty SubCategory. Class = `Category\|\|SubCategory`. Roll-up splits the label on `\|\|`. | 165 |
| `category` | Keep every clip. Class = Category. | 59 |

Val is for early stopping only. Each seed scores the test split once, at that
seed's best validation checkpoint. A seed that already has `metrics.json` is
skipped. `--restart` retrains it.

#### Check the label setup

No training:

```powershell
python train_linear_probe.py --dry-run
```

#### Smoke test

First 200 labelled rows of each split, one seed, 5 epochs. Output is
`<--out>/_smoke_200/<task>/` (default
`results/panns_cnn14_linear/_smoke_200/subcategory/`):

```powershell
python train_linear_probe.py --limit 200 --seeds 42 --max-epochs 5
```

#### Full subcategory run

The paper's default:

```powershell
python train_linear_probe.py
```

#### The other heads

```powershell
python train_linear_probe.py --task category
python train_linear_probe.py --task pair
```

<a id="train-output"></a>

#### Output

Under `--out/<task>/` (default `results/panns_cnn14_linear/<task>/`):

| File | Contents |
| --- | --- |
| `summary.json` | mean ± sample std over seeds, class list, roll-up, settings |
| `seed_<n>/metrics.json` | that seed's best epoch and test scores |
| `seed_<n>/history.csv` | train loss, val loss, val macro F1 |
| `seed_<n>/per_class_f1.csv` | precision, recall, F1, support |
| `seed_<n>/per_category_f1.csv` | same, after the category roll-up |
| `seed_<n>/per_source_f1.csv` | scores split by `source` |
| `seed_<n>/confusion_labels.csv` | confusion over the trained classes |
| `seed_<n>/confusion_category.csv` | confusion after roll-up |
| `seed_<n>/predictions.npz` | test logits, predictions, filename, source, path |
| `seed_<n>/probe.pt` | best checkpoint, class list, parent map |

<a id="train-flags"></a>

#### Flags

| Flag | Default | What it does |
| --- | --- | --- |
| `--embeddings` | `embeddings/panns_cnn14` | Folder of `envsound_{train,val,test}.npz` from the extractor. This is the `--out` of that run. |
| `--out` | `results/panns_cnn14_linear` | Results folder. Files land in `<--out>/<task>/`. |
| `--task` | `subcategory` | `subcategory`, `category`, or `pair`. See [Tasks](#tasks). |
| `--seeds` | `42,123,456,789,1024` | Comma-separated seeds. A seed that already has `metrics.json` is skipped. |
| `--device` | `auto` | `auto`, `cuda`, or `cpu`. |
| `--batch-size` | `256` | |
| `--lr` | `1e-3` | AdamW learning rate. |
| `--weight-decay` | `1e-3` | AdamW weight decay. |
| `--dropout` | `0.3` | Dropout before the linear layer. |
| `--gamma` | `2` | Focal-loss gamma. |
| `--patience` | `20` | Epochs without a better validation macro F1 before stopping. |
| `--max-epochs` | `200` | Epoch budget. Cosine annealing runs over this budget. |
| `--limit` | `0` (every labelled row) | First N labelled rows of each split. Writes `<--out>/_smoke_<N>/<task>/`. |
| `--restart` | off | Retrain seeds that already have `metrics.json`. |
| `--dry-run` | off | Print the label setup and exit. |

Point `--embeddings` at an extraction folder and give the probe its own `--out`.
This writes `results/run_a/subcategory/`:

```powershell
python train_linear_probe.py --embeddings embeddings/run_a --out results/run_a
```

The 5-seed EnvSound scores are in [Results](docs/results.md#results). What to check
in those files is in [evaluation checks](docs/results.md#evaluation).

---

[Docs](docs/README.md) · [Method](docs/method.md) · [Data](docs/data.md) · [Results](docs/results.md) · [Conventions](docs/conventions.md)
