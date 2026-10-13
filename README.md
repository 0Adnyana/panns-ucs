# UCS Sound Tagger (Tugas Akhir)

Bachelor's thesis project (Tugas Akhir): a machine-learning tool that tags sound
effect files with **Universal Category System (UCS)** labels, the category /
subcategory scheme sound libraries are organised by (e.g. `DOORS-WOOD`,
`WEAPONS-GUNSHOT`).

Given an audio file, the tagger predicts its UCS **subcategory**, and the
**category** is looked up from that. The current work is building and
evaluating the models; how the tool reaches users (folder script, plugin, web
app) is not decided yet.

> This README is context for coding assistants (Cursor). The "Conventions"
> section at the bottom lists decisions that code in this repo should respect.

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

| Setting | Value |
|---|---|
| Encoder | PANNs CNN14, 32 kHz, checkpoint `Cnn14_mAP=0.431.pth` (~327 MB), frozen |
| Embedding | 2048-d per clip, extracted once and cached |
| Head | Dropout 0.3 → one linear layer (2048 → n_classes) |
| Optimizer | AdamW, lr 1e-3, weight decay 1e-3, cosine schedule |
| Loss | Focal loss, γ = 2, inverse-frequency class weights |
| Stopping | Early stopping on val, patience 20, max 200 epochs |
| Seeds | 42, 123, 456, 789, 1024 → report mean ± std |
| Metric | Macro F1 on test |
| Balancing | None on the data; imbalance is handled by the loss + weights |

### Target numbers (macro F1)

| Dataset / setup | Paper F1 |
|---|---|
| EnvSound, Category classifier (59 classes) | 0.46 |
| EnvSound, SubCategory classifier (144 classes) | 0.49 |
| EnvSound, SubCategory → Category | **0.55** |
| FSD-env, flat subcategory / → category | 0.66 / **0.72** |
| FSD-env, predict category directly | 0.53 |
| FSD-env, hierarchical with true category given | 0.88 (upper bound, not reachable as is) |
| ESC-50, flat subcategory | 0.95 (too easy to separate models) |

Being within a couple of points of these means the base case is reproduced.

---

## Data

**EnvSound-UCS** = AudioSet (balanced train + eval) + FSD50K (dev + eval pooled)
+ ESC-50, all labelled with UCS. About 58k clips, 59 categories.

- **AudioSet** audio comes from the Hugging Face mirror
  (`agkphysics/AudioSet`); ~15% of the listed clips are missing from it.
  Manifests mark these with `exists = False`.
- **FSD50K** audio is nested one level deep (`FSD50K.dev_audio/FSD50K.dev_audio/`).
- **ESC-50**: only `ESC-50/audio/` is used; the copy inside `ESC-50-master/` is
  ignored so nothing is counted twice.

### Manifests

`manifests/envsound_{train,val,test}.csv` — one row per clip. Relevant columns:
`filename`, `source`, `split`, `Category`, `SubCategory`, `Keywords`, `path`,
`exists`.

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
- Failed clips go to `envsound_*_failed.csv` (should only be the known-missing
  AudioSet clips; anything else is a path problem).
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

**2. Put the checkpoint in `models/`.** The script looks for
`models/Cnn14_mAP=0.431.pth` next to itself (about 327 MB). That file is
gitignored. If it is missing or smaller than ~300 MB, the script downloads it
from Zenodo into that path on the first run. Pass `--checkpoint` only if the
weights live somewhere else.

**3. Smoke test** (first 50 rows of each split). Output goes to
`embeddings/panns_cnn14/_smoke_50/`, not the real cache:

```powershell
python extract_panns_embeddings.py --manifests "E:\Sounds for TA\Extracted\manifests" --limit 50
```

You should see `Device: cuda` (or `cpu`), then one `saved ... X=(n, 2048)` line
per split. A non-zero `failed` count on the smoke test usually means a path
problem, not the known-missing AudioSet clips.

**4. Full run:**

```powershell
python extract_panns_embeddings.py --manifests "E:\Sounds for TA\Extracted\manifests"
```

The manifests already have a `path` column, so `--audio-root` is not needed.
Relative paths are resolved against the manifest folder. If a manifest has no
path column (`path`, `filepath`, `file_path`, `audio_path`, or `abs_path`),
pass `--audio-root` and the script searches that tree by filename. Rows with
`exists` set to false are skipped.

**5. What you get** under `embeddings/panns_cnn14/`:

| File | Contents |
|---|---|
| `envsound_{train,val,test}.npz` | `X` plus `filename`, `Category`, `SubCategory`, `source`, `path` |
| `envsound_{split}_failed.csv` | clips that could not be read |
| `extract_info.json` | sample rate, embedding definition, and per-split counts |

While a split is still running, progress sits in `envsound_{split}.partial.npz`.
Stop the script and run the same command again to continue. `--restart` throws
that partial file away and starts the split over. `--save-every` (default 2000)
is how often the partial file is rewritten.

Other flags: `--out` (default `embeddings/panns_cnn14`), `--splits`
(default `train,val,test`), `--device` (`auto`, `cuda`, or `cpu`).

### Step 2 — Classifier training (next, not written yet)

Loads the `.npz` files only (never audio or CNN14). Trains the paper's head with
the paper's settings. Runs take minutes, even on CPU. This is where experiments
happen (category vs subcategory, label scheme, seeds), so a notebook is fine
here. Outputs to produce: macro F1 per seed (mean ± std), SubCategory → Category
roll-up, per-category F1, per-source F1, confusion matrices, top-3 / top-5
accuracy.

### Later model experiments (planned)

1. Replicate the paper: frozen PANNs CNN14 + linear subcategory head ← **current**
2. Fine-tune PANNs CNN14 end to end (backbone lr ~1e-5, head ~1e-3; mixup,
   SpecAugment; early stopping on val F1). Needs on-the-fly audio, so hours per
   run on an RTX 4060.
3. Frozen SSLAM + subcategory head
4. Fine-tuned SSLAM
5. Compare all four against the paper, overall and per category

Other candidate encoders noted: EfficientAT (MobileNetV3 / DyMN, 32 kHz),
PaSST, HTS-AT, CED-Base (16 kHz), BEATs/EAT fine-tuned on AudioSet.

---

## Freesound UCS gap filler (`ucs_freesound/`)

Separate tool for adding training clips to UCS subcategories the dataset is
short on. Seven scripts, each writing a CSV to `output/` for review:

1. `step1_find_gaps.py` — counts clips per category / CatID (EnvSound manifests + `final/`)
2. `step2_build_queries.py` — Freesound queries from UCS names + synonyms, CLAP prompts (editable)
3. `step3_search.py` — searches, scores metadata, filters (supports `--dry-run`)
4. `step4_download.py` — downloads previews (~192 kbps ogg) to `downloads/<CatID>/`
5. `step5_verify_clap.py` — ranks each clip against all ~660 UCS prompts with LAION-CLAP
6. `step6_export_review.py` — `review.csv` for manual keep/reject (editable)
7. `step7_finalize.py` — copies kept clips to `final/`, writes `ATTRIBUTION.txt`, blacklists rejects

Settings live in `config.py`; the API key is read from `.env` only
(`FREESOUND_API_KEY`, never commit it). Freesound IDs already in EnvSound
(FSD50K/ESC-50 filenames) are excluded. `suggested_split` is assigned per pack /
uploader to avoid train–test leakage. CLAP is a filter, not an independent
evaluation. Freesound API limits: 60 requests/min, 2000/day.

---

## Evaluation and the "is the model the bottleneck?" checks

After the first runs, check in order:

1. **Per-category F1** — is the error spread out or concentrated in small categories?
2. **Within-category confusion** — if most errors stay in the right category, the category tag is already usable.
3. **Top-3 accuracy** — if the right tag is usually in the top 3, the tool can *suggest* tags for a person to confirm.
4. **Fine-tuning gain** — if fine-tuned PANNs clearly beats the linear probe, more model work pays off; if not, the limit is data or labels.

Levers if accuracy stays low: stronger / fine-tuned encoders, narrower scope
(FSD-env's ~36 categories instead of 59), confidence thresholds, suggesting
instead of assigning tags.

## Open decisions

- **Label scheme for step 2:** match the paper (drop empty SubCategory,
  subcategory name only, 144 classes) as default, with a flag for
  `Category||SubCategory` pairs (165 classes).
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
- Keep `source` alongside every prediction so results can be split per dataset.
- Long jobs (extraction, fine-tuning) are resumable scripts; analysis goes in notebooks.
- Report macro F1 as mean ± std over the 5 seeds, plus top-k accuracy.
- Primary environment: Windows (PowerShell), RTX 4060; data under
  `E:\Sounds for TA\Extracted`. Avoid tools that need `wget` or Unix-only paths.