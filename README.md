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



## Getting started

How to install, extract embeddings, train the linear probe, and open the demo.
Run the scripts from this repo folder in PowerShell. Extract once, then train
on the cached file. The cache holds no labels: the probe joins the manifests
onto it, so a new label set does not need a new extraction.

**On this page:** [Install](#install) · [Checkpoint](#checkpoint) · [Extract](#extract) · [Extract flags](#extract-flags) · [Combine old caches](#combine-old-caches) · [Train](#train) · [Train flags](#train-flags) · [New label set](#new-label-set) · [Demo](#demo)



### Install

```powershell
pip install -r requirements.txt
```

`requirements.txt` pins the CUDA 13.0 build of PyTorch (`2.14.1+cu130`). A
plain `pip install torch` can install the `+cpu` wheel, and then
`torch.cuda.is_available()` stays False. On CPU the full set (~58k clips)
takes several hours.



### Checkpoint

`extract_panns_embeddings.py` looks for `models/Cnn14_mAP=0.431.pth` next to
itself (about 327 MB). That file is gitignored. If it is missing or smaller
than ~300 MB, the script downloads it from Zenodo into that path on the first
run. Pass `--checkpoint` only if the weights live somewhere else. Which
checkpoint to use is in the [code conventions](docs/conventions.md#conventions).



### Extract embeddings

`extract_panns_embeddings.py` reads the [manifests](docs/data.md#manifests) to
learn which clips to embed, runs each clip through frozen CNN14, and saves one
`embeddings.npz` under `--out` (default `embeddings/panns_cnn14/`). Run it as a
script (long, unattended, resumable). It reads `filename`, `source`, the path
column, and `exists`. It does not read or store `Category` or `SubCategory`.
What the cache contains is under
[embedding extraction](docs/method.md#extract-embeddings).

The [manifests](docs/data.md#manifests) have a `path` column. These CSVs sit
outside the audio tree, so pass `--audio-root` for the folder that holds
`FSD50K/`, `AudioSet/`, and `ESC-50/`. Relative paths resolve against that
folder. Absolute paths are used as written. Without `--audio-root`, relative
paths resolve against the parent of the folder passed to `--manifests`. When a
manifest has no path column (`path`, `filepath`, `file_path`, `audio_path`, or
`abs_path`), `--audio-root` is searched by filename. Rows with `exists` set to
false are skipped ([known data issues](docs/data.md#data-issues)).

#### Smoke test

First 50 rows of each split. Output is `<--out>/_smoke_50/` (default
`embeddings/panns_cnn14/_smoke_50/`), so a smoke test stays out of a full run
in the same folder:

```powershell
python extract_panns_embeddings.py --manifests "E:\Sounds for TA\Extracted\manifests" --limit 50
```

You should see `Device: cuda` (or `cpu`), then `audio root: E:\Sounds for TA\Extracted`,
then one `[split] embedded 50  failed 0` line per split and one
`saved ... embeddings.npz  X=(150, 2048)  failed=0` line. The first 50 rows
of each split exist on disk, so a non-zero `failed` count here is a path
problem. Open `envsound_*_failed.csv`: if `error` is `file not found` and
`path` contains `\manifests\`, the resolver used the manifests folder as the
base. Paths are relative to the data root, described in
[Manifests](docs/data.md#manifests).

#### Full run

```powershell
python extract_panns_embeddings.py --manifests "E:\Sounds for TA\Extracted\manifests"
```



#### Output

Under `--out` (default `embeddings/panns_cnn14/`):

| File | Contents |
| --- | --- |
| `embeddings.npz` | `X` (n_clips × 2048) plus `filename` and `source`, one row per `filename` + `source` |
| `envsound_{split}_failed.csv` | clips from that split that could not be read |
| `extract_info.json` | sample rate, embedding definition, per-split counts, and what is not stored |

A clip listed in more than one split is embedded once. If the same `filename`
and `source` point at two different files, the script stops and prints both
paths.

While the run is in progress, progress sits in `embeddings.partial.npz`.
Stop the script and run the same command again to continue. `--restart` throws
that partial file away and starts over. `--save-every` (default 2000) is how
often the partial file is rewritten.

`embeddings/` is gitignored.



#### Flags

| Flag | Default | What it does |
| --- | --- | --- |
| `--manifests` | required, unless `--combine-from` | Folder with `envsound_{train,val,test}.csv`. Each needs `filename` and `source`. |
| `--combine-from` | off | Folder of old `envsound_{split}.npz` files to stack into `<--out>/embeddings.npz`. No CNN14. See [Combine old caches](#combine-old-caches). |
| `--audio-root` | parent of that folder | Base for relative paths, or the tree to search by filename when the manifest has no path column. |
| `--out` | `embeddings/panns_cnn14` | Folder for this run. A new name starts a separate cache. The same name resumes it. |
| `--checkpoint` | `models/Cnn14_mAP=0.431.pth` | CNN14 weights, next to the script. Downloaded there if missing. See [Checkpoint](#checkpoint). |
| `--splits` | `train,val,test` | Comma-separated manifests to embed. |
| `--device` | `auto` | `auto`, `cuda`, or `cpu`. |
| `--limit` | `0` (every row) | First N rows of each split. Writes `<--out>/_smoke_<N>/`. |
| `--save-every` | `2000` | Rewrite `embeddings.partial.npz` every N new clips. |
| `--restart` | off | Ignore the partial file and start over. |

To keep two extractions apart, pass a different `--out`:

```powershell
python extract_panns_embeddings.py --manifests "E:\Sounds for TA\Extracted\manifests" --out embeddings/run_a
```

#### Combine old caches

Caches extracted before this layout hold one `envsound_{train,val,test}.npz`
per split, with labels inside. `--combine-from` stacks them into one
`embeddings.npz`, copying `X`, `filename`, and `source`. It does not load CNN14
or read manifests. It uses the `path` stored in the old files only to check
that one `filename` + `source` never points at two files, then drops it. It
ignores `*.partial.npz` and leaves the old files in place.

```powershell
python extract_panns_embeddings.py --combine-from embeddings/panns_cnn14 --out embeddings/panns_cnn14
```



### Train the linear probe

`train_linear_probe.py` loads `embeddings.npz` from `--embeddings` (never audio
or CNN14) and the three CSVs from `--manifests`. Labels and splits come from
the CSVs. Each row with `exists` not false is matched to the cache on
`filename` + `source`, and row order follows the CSV. It trains the paper's
head with the [paper's settings](docs/method.md#paper-method).
Runs take minutes, even on CPU. Output goes to `--out/<task>/` (default
`results/panns_cnn14_linear/<task>/`). That folder is gitignored, same as
`embeddings/`. Metric definitions are under [linear probe](docs/method.md#linear-probe).

The join stops, and prints the rows, if a CSV row has no vector in the cache
(for example a clip listed in `envsound_{split}_failed.csv`), or if the same
`filename` + `source` appears twice in one CSV or in more than one split.
Mark such clips `exists` false, or re-extract. Cache rows that no CSV uses are
counted and ignored.

Empty subcategories and shared subcategory names are covered in
[known data issues](docs/data.md#data-issues). Training choices that the paper leaves
unspecified are in the [code conventions](docs/conventions.md#conventions).



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
python train_linear_probe.py --manifests "E:\Sounds for TA\Extracted\manifests" --dry-run
```

#### Smoke test

First 200 labelled rows of each split, one seed, 5 epochs. Output is
`<--out>/_smoke_200/<task>/` (default
`results/panns_cnn14_linear/_smoke_200/subcategory/`):

```powershell
python train_linear_probe.py --manifests "E:\Sounds for TA\Extracted\manifests" --limit 200 --seeds 42 --max-epochs 5
```

#### Full subcategory run

The paper's default:

```powershell
python train_linear_probe.py --manifests "E:\Sounds for TA\Extracted\manifests"
```

#### The other heads

```powershell
python train_linear_probe.py --manifests "E:\Sounds for TA\Extracted\manifests" --task category
python train_linear_probe.py --manifests "E:\Sounds for TA\Extracted\manifests" --task pair
```



#### Output

Under `--out/<task>/` (default `results/panns_cnn14_linear/<task>/`):

| File | Contents |
| --- | --- |
| `summary.json` | mean ± sample std over seeds, class list, roll-up, settings (including `embeddings` and `manifests`) |
| `seed_<n>/metrics.json` | that seed's best epoch and test scores |
| `seed_<n>/history.csv` | train loss, val loss, val macro F1 |
| `seed_<n>/per_class_f1.csv` | precision, recall, F1, support |
| `seed_<n>/per_category_f1.csv` | same, after the category roll-up |
| `seed_<n>/per_source_f1.csv` | scores split by `source` |
| `seed_<n>/confusion_labels.csv` | confusion over the trained classes |
| `seed_<n>/confusion_category.csv` | confusion after roll-up |
| `seed_<n>/predictions.npz` | test logits, predictions, filename, source, path (from the CSV) |
| `seed_<n>/probe.pt` | best checkpoint, class list, parent map |



#### Flags

| Flag | Default | What it does |
| --- | --- | --- |
| `--embeddings` | `embeddings/panns_cnn14` | Folder holding `embeddings.npz` from the extractor. This is the `--out` of that run. |
| `--manifests` | required | Folder with `envsound_{train,val,test}.csv`. Needs `filename`, `source`, and `Category`. `SubCategory` and `path` are optional. |
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
python train_linear_probe.py --manifests "E:\Sounds for TA\Extracted\manifests" --embeddings embeddings/run_a --out results/run_a
```

#### New label set

The cache depends only on the audio and CNN14, so relabeling or resplitting
needs no extraction. Write the new `envsound_{train,val,test}.csv` to a new
folder, point `--manifests` at it, keep the same `--embeddings`, and give the
probe a new `--out`:

```powershell
python train_linear_probe.py --manifests "E:\Sounds for TA\Extracted\manifests_v2" --embeddings embeddings/panns_cnn14 --out results/labels_v2
```

Every clip in the new CSVs must already be in the cache, or the join stops. If
the new set adds clips, extract the new manifests into a new extractor `--out`
and point `--embeddings` there.

The 5-seed EnvSound scores are in [Results](docs/results.md#results). What to check
in those files is in [evaluation checks](docs/results.md#evaluation).



### Demo

`demo/app.py` serves one page on this machine. Drop a clip and the page plays
it and predicts a UCS tag. The clip goes through frozen CNN14, then the linear
probe selected on the page. The class names come from that `probe.pt`, so the
demo needs no embeddings cache or manifest. An upload has no stored label, so
the page shows predictions only. The script prints the URL (default
`http://127.0.0.1:8765`). It uses the packages already listed in
`requirements.txt`.

```powershell
python demo/app.py
```

The **Checkpoint** menu lists every `seed_<n>/probe.pt` under `results/`,
grouped by results run. Each option is the task, the seed, and that seed's test
macro F1 when `metrics.json` is present. Choosing one loads that probe, its
class list, and its parent map. The previous prediction is cleared. Reload the
page to pick up a probe saved while the server is running.

The page opens on `results/panns_cnn14_linear/subcategory/seed_42` when that
probe exists. Otherwise it uses the first subcategory seed 42, then the first
saved probe.

The first 30 seconds of a clip are scored. Files over 30 MB are refused.

#### Flags

| Flag | Default | What it does |
| --- | --- | --- |
| `--results` | `results` | Folder to scan for saved probes. |
| `--probe` | subcategory seed 42, else the first saved probe | Checkpoint to load first. |
| `--checkpoint` | `models/Cnn14_mAP=0.431.pth` | CNN14 weights. See [Checkpoint](#checkpoint). |
| `--device` | `auto` | `auto`, `cuda`, or `cpu`. |
| `--host` | `127.0.0.1` | |
| `--port` | `8765` | |
| `--top` | `5` | How many tags to show. |

Start on another run:

```powershell
python demo/app.py --probe results/run_a/subcategory/seed_42/probe.pt
```

---

[Docs](docs/README.md) · [Method](docs/method.md) · [Data](docs/data.md) · [Results](docs/results.md) · [Conventions](docs/conventions.md)
