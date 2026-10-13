# Project notes

Setup, data, pipeline, results, and conventions for the UCS sound tagger.
This document is context for coding assistants. The "Conventions" section
lists decisions that code in this repo should respect.

---

## Reference paper (the base case)

**Beck & Lerch, DAFx26 — arXiv [2606.05571](https://arxiv.org/pdf/2606.05571)**
Relabels AudioSet, FSD50K and ESC-50 into UCS v8.2.1 and benchmarks frozen
audio encoders on it. Their label CSVs and splits are used as-is, so this
project does **no label conversion of its own**.

- Repos: `JunWooBeck/envsound-ucs` (EnvSound splits), `JunWooBeck/audioset-ucs`,
`JunWooBeck/ucs-sfx-tools`
- Key finding reused here: predicting the **subcategory and rolling it up to the
category** beats predicting the category directly.



### The paper's method (what "replicate the paper" means)

The paper does **not fine-tune** PANNs. It is a **linear probe on frozen CNN14
embeddings**:


| Setting   | Value                                                                   |
| --------- | ----------------------------------------------------------------------- |
| Encoder   | PANNs CNN14, 32 kHz, checkpoint `Cnn14_mAP=0.431.pth` (~327 MB), frozen |
| Embedding | 2048-d per clip, extracted once and cached                              |
| Head      | Dropout 0.3 → one linear layer (2048 → n_classes)                       |
| Optimizer | AdamW, lr 1e-3, weight decay 1e-3, cosine schedule                      |
| Loss      | Focal loss, γ = 2, inverse-frequency class weights                      |
| Stopping  | Early stopping on val, patience 20, max 200 epochs                      |
| Seeds     | 42, 123, 456, 789, 1024 → report mean ± std                             |
| Metric    | Macro F1 on test                                                        |
| Balancing | None on the data; imbalance is handled by the loss + weights            |




### Target numbers (macro F1)


| Dataset / setup                                | Paper F1                                |
| ---------------------------------------------- | --------------------------------------- |
| EnvSound, Category classifier (59 classes)     | 0.46                                    |
| EnvSound, SubCategory classifier (144 classes) | 0.49                                    |
| EnvSound, SubCategory → Category               | **0.55**                                |
| FSD-env, flat subcategory / → category         | 0.66 / **0.72**                         |
| FSD-env, predict category directly             | 0.53                                    |
| FSD-env, hierarchical with true category given | 0.88 (upper bound, not reachable as is) |
| ESC-50, flat subcategory                       | 0.95 (too easy to separate models)      |


Being within a couple of points of these means the base case is reproduced.
The EnvSound linear-probe numbers from this repo are in Step 2.

---



## Data

**EnvSound-UCS** = AudioSet (balanced train + eval) + FSD50K (dev + eval pooled)

- ESC-50, all labelled with UCS. About 58k clips, 59 categories.

- **AudioSet** audio comes from the Hugging Face mirror
(`agkphysics/AudioSet`); ~15% of the listed clips are missing from it.
Manifests mark these with `exists = False`.
- **FSD50K** clips the manifests point at live in `FSD50K/FSD50K.dev_audio/`
and `FSD50K/FSD50K.eval_audio/`. The Zenodo unpack also leaves a second copy
one level deeper (`FSD50K.dev_audio/FSD50K.dev_audio/`); the `path` column
does not use that copy.
- **ESC-50**: only `ESC-50/audio/` is used; the copy inside `ESC-50-master/` is
ignored so nothing is counted twice.

On disk, under `E:\Sounds for TA\Extracted`:

```
AudioSet/audios/balanced_train_segments/Y*.wav
AudioSet/audios/eval_segments/Y*.wav
FSD50K/FSD50K.dev_audio/*.wav
FSD50K/FSD50K.eval_audio/*.wav
ESC-50/audio/*.wav
manifests/envsound_{train,val,test}.csv
```

### Manifests

`manifests/envsound_{train,val,test}.csv` — one row per clip. Columns:
`filename`, `Category`, `SubCategory`, `Keywords`, `source`, `path`, `exists`.
There is no `split` column; the split is which file the row is in.

`path` is relative to the data root, the parent of `manifests/`, and uses
forward slashes. Examples:

- `FSD50K/FSD50K.dev_audio/380669.wav`
- `AudioSet/audios/balanced_train_segments/YWz1z-vzrI-4.wav`
- `ESC-50/audio/1-26177-A-43.wav`

It is not relative to the manifests folder. `prepare_envsound_ucs.py` writes
these paths from its `--data` root, and the CSVs are saved in `<data>/manifests/`.
Joining a `path` onto `manifests/` builds a file that is not there
(`...\manifests\FSD50K\...`). Extraction then reports `file not found` for
every row, including rows with `exists` true. Rows with `exists` false are
real missing clips (mostly AudioSet) and are skipped before any file is opened.

### Known data issues

- **Empty SubCategory:** ~19.6k EnvSound clips (about a third; mostly FOLEY,
ANIMALS, MACHINES) have no SubCategory. The paper drops them for the
subcategory classifier. See open decisions below.
- **Duplicate subcategory names across categories:** e.g. `VEHICLES||HORN` vs
`ALARMS||HORN`. Subcategory name alone = 144 classes; `Category||SubCategory`
pairs = 165 classes.
- **Encoder leakage:** ~20% of EnvSound test clips come from AudioSet's balanced
train set, which CNN14 saw during pretraining. Always also report results per
`source` (the `source` column is kept in the embedding files for this).
- **Non-UCS labels:** `MUSIC` is not a UCS category (UCS uses `MUSICAL`); labels
like `GUNS-GUNSHOT` or `FOLEY-COIN` use subcategories not in UCS v8.2.
- **Licences:** ESC-50 and parts of FSD50K are CC-BY-NC; labels are CC-BY-SA.
Fine for a thesis, limiting for a commercial release.

---



## Pipeline

```
audio ──► frozen CNN14 ──► 2048-d embedding (.npz, cached) ──► dropout + linear head ──► SubCategory ──► Category
```



### Step 1 — Embedding extraction (`extract_panns_embeddings.py`)

Reads the manifests, runs each clip through frozen CNN14, saves
`embeddings/panns_cnn14/envsound_{train,val,test}.npz`.

- Each `.npz` holds `X` (n_clips × 2048, float32) plus `filename`, `Category`,
`SubCategory`, `source`, `path`, in the same row order. ~475 MB total.
- Saves progress every 2,000 clips and resumes if rerun.
- Failed clips go to `envsound_*_failed.csv`. Rows with `exists` false never
reach that file. A `file not found` error whose path starts with `manifests\`
means the relative `path` was joined onto the manifests folder instead of the
data root.
- Does **not** use the `panns_inference` package (on Windows it calls `wget` on
import and crashes). The CNN14 model code is copied in; embeddings were checked
to match the official model exactly.

Run as a script (long, unattended, resumable), not in a notebook.

#### How to run it

From this repo folder, in PowerShell.

**1. Install the packages** (once):

```powershell
pip install -r requirements.txt
```

`requirements.txt` pins the CUDA 13.0 build of PyTorch (`2.14.1+cu130`).
A plain `pip install torch` can install the `+cpu` wheel, and then
`torch.cuda.is_available()` stays False. On CPU the full set (~58k clips)
takes several hours.

**2. Put the checkpoint in** `models/`**.** The script looks for
`models/Cnn14_mAP=0.431.pth` next to itself (about 327 MB). That file is
gitignored. If it is missing or smaller than ~300 MB, the script downloads it
from Zenodo into that path on the first run. Pass `--checkpoint` only if the
weights live somewhere else.

**3. Smoke test** (first 50 rows of each split). Output goes to
`embeddings/panns_cnn14/_smoke_50/`, not the real cache:

```powershell
python extract_panns_embeddings.py --manifests "E:\Sounds for TA\Extracted\manifests" --limit 50
```

You should see `Device: cuda` (or `cpu`), then `audio root: E:\Sounds for TA\Extracted`,
then one `saved ... X=(50, 2048)  failed=0` line per split. The first 50 rows
of each split exist on disk, so a non-zero `failed` count here is a path
problem. Open `envsound_*_failed.csv`: if `error` is `file not found` and
`path` contains `\manifests\`, the resolver used the manifests folder as the
base.

**4. Full run:**

```powershell
python extract_panns_embeddings.py --manifests "E:\Sounds for TA\Extracted\manifests"
```

The manifests already have a `path` column, so `--audio-root` is not needed.
Relative paths resolve against the parent of the folder passed to `--manifests`
(see Manifests above). Absolute paths are used as written. Pass `--audio-root`
only to override that base, or when a manifest has no path column (`path`,
`filepath`, `file_path`, `audio_path`, or `abs_path`); in that case the script
searches the tree by filename. Rows with `exists` set to false are skipped.

**5. What you get** under `embeddings/panns_cnn14/`:


| File                            | Contents                                                         |
| ------------------------------- | ---------------------------------------------------------------- |
| `envsound_{train,val,test}.npz` | `X` plus `filename`, `Category`, `SubCategory`, `source`, `path` |
| `envsound_{split}_failed.csv`   | clips that could not be read                                     |
| `extract_info.json`             | sample rate, embedding definition, and per-split counts          |


While a split is still running, progress sits in `envsound_{split}.partial.npz`.
Stop the script and run the same command again to continue. `--restart` throws
that partial file away and starts the split over. `--save-every` (default 2000)
is how often the partial file is rewritten.

Other flags: `--out` (default `embeddings/panns_cnn14`), `--splits`
(default `train,val,test`), `--device` (`auto`, `cuda`, or `cpu`).

### Step 2 — Linear probe (`train_linear_probe.py`)

Loads the `.npz` files only (never audio or CNN14). Trains the paper's head with
the paper's settings. Runs take minutes, even on CPU. Output goes to
`results/panns_cnn14_linear/<task>/`. That folder is gitignored, same as
`embeddings/`.

`--task` (default `subcategory`):


| Task          | Labels                                                                                                                                              | EnvSound classes |
| ------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------- |
| `subcategory` | Drop empty SubCategory. Class = SubCategory name. Roll-up uses the most common parent category of that name in the training split; ties by name. | 144              |
| `pair`        | Drop empty SubCategory. Class = `Category\|\|SubCategory`. Roll-up splits the label on `\|\|`.                                                      | 165              |
| `category`    | Keep every clip. Class = Category.                                                                                                                  | 59               |


The paper does not state the batch size, the early-stopping metric, the
inverse-frequency formula, or the parent of a subcategory name that appears
under more than one category. This script uses batch size 256, early stopping
on validation macro F1, weights `w_c = N / (n_classes * count_c)`, and the
roll-up in the table. Those choices are written into `summary.json`.

Val is for early stopping only. Each seed scores the test split once, at that
seed's best validation checkpoint. A seed that already has `metrics.json` is
skipped. `--restart` retrains it.

#### How to run it

From this repo folder, after the embeddings exist.

**1. Check the label setup** (no training):

```powershell
python train_linear_probe.py --dry-run
```

**2. Smoke test** (first 200 labelled rows of each split, one seed, 5 epochs).
Output goes to `results/panns_cnn14_linear/_smoke_200/subcategory/`, not the
real run:

```powershell
python train_linear_probe.py --limit 200 --seeds 42 --max-epochs 5
```

**3. Full subcategory run** (the paper's default):

```powershell
python train_linear_probe.py
```

**4. The other heads:**

```powershell
python train_linear_probe.py --task category
python train_linear_probe.py --task pair
```

**5. What you get** under `results/panns_cnn14_linear/<task>/`:


| File                              | Contents                                                    |
| --------------------------------- | ----------------------------------------------------------- |
| `summary.json`                    | mean ± sample std over seeds, class list, roll-up, settings |
| `seed_<n>/metrics.json`           | that seed's best epoch and test scores                     |
| `seed_<n>/history.csv`            | train loss, val loss, val macro F1                          |
| `seed_<n>/per_class_f1.csv`       | precision, recall, F1, support                              |
| `seed_<n>/per_category_f1.csv`    | same, after the category roll-up                            |
| `seed_<n>/per_source_f1.csv`      | scores split by `source`                                    |
| `seed_<n>/confusion_labels.csv`   | confusion over the trained classes                          |
| `seed_<n>/confusion_category.csv` | confusion after roll-up                                     |
| `seed_<n>/predictions.npz`        | test logits, predictions, filename, source, path            |
| `seed_<n>/probe.pt`               | best checkpoint, class list, parent map                     |


Macro F1 is the mean of per-class F1 over classes with at least one true sample
in the scored split. `macro_f1_all_train_classes` scores a missing class as 0.
Std is the sample standard deviation over seeds (`ddof=1`).

Other flags: `--embeddings` (default `embeddings/panns_cnn14`), `--out`
(default `results/panns_cnn14_linear`), `--device` (`auto`, `cuda`, or `cpu`),
`--batch-size`, `--lr`, `--weight-decay`, `--dropout`, `--gamma`, `--patience`,
`--max-epochs`, `--seeds`.

#### EnvSound results (5 seeds)

| Setup                         | This run        | Paper |
| ----------------------------- | --------------- | ----- |
| SubCategory (144)             | 0.496 ± 0.007   | 0.49  |
| SubCategory → Category        | 0.487 ± 0.003   | 0.55  |
| Category classifier (59)      | 0.478 ± 0.005   | 0.46  |

Subcategory macro F1 matches the paper. Direct category is a bit above 0.46.
The roll-up lands at 0.487 against the paper's 0.55. On the subcategory head,
top-3 / top-5 accuracy is 0.777 ± 0.006 / 0.846 ± 0.006. The `pair` head is
implemented and has not been run.

---



## Evaluation and the "is the model the bottleneck?" checks

The probe writes these. Check them in order:

1. **Per-category F1** — is the error spread out or concentrated in small categories?
2. **Within-category confusion** — if most errors stay in the right category, the category tag is already usable.
3. **Top-3 accuracy** — if the right tag is usually in the top 3, the tool can *suggest* tags for a person to confirm.
4. **Fine-tuning gain** — if fine-tuned PANNs clearly beats the linear probe, more model work pays off; if not, the limit is data or labels.

Levers if accuracy stays low: stronger / fine-tuned encoders, narrower scope
(FSD-env's ~36 categories instead of 59), confidence thresholds, suggesting
instead of assigning tags.

## Open decisions

- **Main training set:** Option A, FSD-env (cleaner, higher F1) vs Option B,
EnvSound (59 categories, ~.17 lower F1, more practical).
- What F1 / top-3 accuracy counts as "good enough" for the tool.
- Assign tags vs suggest tags for confirmation.
- Delivery format of the tool (script over a folder, plugin, web app).

---



## Conventions for code in this repo

- **Base case = paper setup exactly.** Don't change splits, rebalance data, or
normalise loudness for the reproduction. Rebalancing / augmentation are later
experiments, applied to the **train split only**.
- **Call the frozen setup a "linear probe", not fine-tuning.**
- Val is for early stopping only; test is touched once, at the end.
- Use the 32 kHz `Cnn14_mAP=0.431.pth` checkpoint in `models/`; reject other PANNs checkpoints.
- Skip manifest rows with `exists = False`.
- Manifest `path` values are relative to the data root (parent of `manifests/`),
not to the manifests folder. Resolve them there unless the path is already absolute.
- Keep `source` alongside every prediction so results can be split per dataset.
- Unspecified paper details, fixed in `train_linear_probe.py`: batch size 256,
early stopping on validation macro F1, class weights
`w_c = N / (n_classes * count_c)`, shared subcategory names roll up to the
most common training parent (ties broken by category name). Default label
scheme is the paper's 144-class subcategory head; `--task category` and
`--task pair` are the other two.
- Long jobs (extraction, the linear probe, fine-tuning) are resumable scripts.
The probe skips a seed that already has `metrics.json`.
- Report macro F1 as mean ± std over the 5 seeds, plus top-k accuracy.
- `embeddings/`, `results/`, and `*.pth` stay out of git.
- Primary environment: Windows (PowerShell), RTX 4060; data under
`E:\Sounds for TA\Extracted`. Avoid tools that need `wget` or Unix-only paths.
