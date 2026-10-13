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

**On this page:** [Install](#install) · [Checkpoint](#checkpoint) · [Extract](#extract) · [Train](#train)

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
each clip through frozen CNN14, and saves
`embeddings/panns_cnn14/envsound_{train,val,test}.npz`. Run it as a script
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

First 50 rows of each split. Output goes to
`embeddings/panns_cnn14/_smoke_50/`, separate from the real cache:

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

Under `embeddings/panns_cnn14/`:

| File | Contents |
| --- | --- |
| `envsound_{train,val,test}.npz` | `X` plus `filename`, `Category`, `SubCategory`, `source`, `path` |
| `envsound_{split}_failed.csv` | clips that could not be read |
| `extract_info.json` | sample rate, embedding definition, and per-split counts |

While a split is still running, progress sits in `envsound_{split}.partial.npz`.
Stop the script and run the same command again to continue. `--restart` throws
that partial file away and starts the split over. `--save-every` (default 2000)
is how often the partial file is rewritten.

Other flags: `--out` (default `embeddings/panns_cnn14`), `--splits`
(default `train,val,test`), `--device` (`auto`, `cuda`, or `cpu`).

`embeddings/` is gitignored.

<a id="train"></a>

### Train the linear probe

`train_linear_probe.py` loads the `.npz` files only (never audio or CNN14).
It trains the paper's head with the [paper's settings](docs/method.md#paper-method).
Runs take minutes, even on CPU. Output goes to
`results/panns_cnn14_linear/<task>/`. That folder is gitignored, same as
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

First 200 labelled rows of each split, one seed, 5 epochs. Output goes to
`results/panns_cnn14_linear/_smoke_200/subcategory/`, separate from the real run:

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

Under `results/panns_cnn14_linear/<task>/`:

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

Other flags: `--embeddings` (default `embeddings/panns_cnn14`), `--out`
(default `results/panns_cnn14_linear`), `--device` (`auto`, `cuda`, or `cpu`),
`--batch-size`, `--lr`, `--weight-decay`, `--dropout`, `--gamma`, `--patience`,
`--max-epochs`, `--seeds`.

The 5-seed EnvSound scores are in [Results](docs/results.md#results). What to check
in those files is in [evaluation checks](docs/results.md#evaluation).

---

[Docs](docs/README.md) · [Method](docs/method.md) · [Data](docs/data.md) · [Results](docs/results.md) · [Conventions](docs/conventions.md)
