# Data

[README](../README.md) · [Docs](README.md) · [Method](method.md) · **Data** · [Results](results.md) · [Conventions](conventions.md)

EnvSound-UCS: where the audio lives, how the manifests point at it, and the
label issues that change training.

**On this page:** [Layout](#layout) · [Manifests](#manifests) · [Known issues](#data-issues)

<a id="layout"></a>

## Layout

**EnvSound-UCS** is AudioSet (balanced train + eval) plus FSD50K (dev + eval
pooled) plus ESC-50, all labelled with UCS. About 58k clips, 59 categories.

- **AudioSet** audio comes from the Hugging Face mirror (`agkphysics/AudioSet`). About 15% of the listed clips are missing from it. Manifests mark these with `exists = False`.
- **FSD50K** clips the manifests point at live in `FSD50K/FSD50K.dev_audio/` and `FSD50K/FSD50K.eval_audio/`. The Zenodo unpack also leaves a second copy one level deeper (`FSD50K.dev_audio/FSD50K.dev_audio/`). The `path` column does not use that copy.
- **ESC-50** uses only `ESC-50/audio/`. The copy inside `ESC-50-master/` is ignored so nothing is counted twice.

On disk, under `E:\Sounds for TA\Extracted`:

```
AudioSet/audios/balanced_train_segments/Y*.wav
AudioSet/audios/eval_segments/Y*.wav
FSD50K/FSD50K.dev_audio/*.wav
FSD50K/FSD50K.eval_audio/*.wav
ESC-50/audio/*.wav
manifests/envsound_{train,val,test}.csv
```

<a id="manifests"></a>

## Manifests

`manifests/envsound_{train,val,test}.csv` — one row per clip. Columns:
`filename`, `Category`, `SubCategory`, `Keywords`, `source`, `path`, `exists`.
There is no `split` column. The split is which file the row is in.

`path` is relative to the data root (the parent of `manifests/`) and uses
forward slashes. Examples:

- `FSD50K/FSD50K.dev_audio/380669.wav`
- `AudioSet/audios/balanced_train_segments/YWz1z-vzrI-4.wav`
- `ESC-50/audio/1-26177-A-43.wav`

It is not relative to the manifests folder. `prepare_envsound_ucs.py` writes
these paths from its `--data` root, and the CSVs are saved in `<data>/manifests/`.
Joining a `path` onto `manifests/` builds a file that is not there
(`...\manifests\FSD50K\...`). Extraction then reports `file not found` for
every row, including rows with `exists` true.

Rows with `exists` false are real missing clips (mostly AudioSet) and are
skipped before any file is opened. How the extractor resolves paths is in
[Getting started](../README.md#extract).

<a id="data-issues"></a>

## Known issues

- **Empty SubCategory.** About 19.6k EnvSound clips (about a third; mostly FOLEY, ANIMALS, MACHINES) have no SubCategory. The paper drops them for the subcategory classifier. See [open decisions](conventions.md#open-decisions).
- **Duplicate subcategory names across categories.** For example `VEHICLES||HORN` vs `ALARMS||HORN`. Subcategory name alone is 144 classes. `Category||SubCategory` pairs are 165 classes. The three `--task` settings are in [Getting started](../README.md#tasks).
- **Encoder leakage.** About 20% of EnvSound test clips come from AudioSet's balanced train set, which CNN14 saw during pretraining. Always also report results per `source`. The `source` column is kept in the embedding files for this. Per-source scores land in `per_source_f1.csv` ([train output](../README.md#train-output)).
- **Non-UCS labels.** `MUSIC` is not a UCS category (UCS uses `MUSICAL`). Labels like `GUNS-GUNSHOT` or `FOLEY-COIN` use subcategories not in UCS v8.2.
- **Licences.** ESC-50 and parts of FSD50K are CC-BY-NC. Labels are CC-BY-SA. Fine for a thesis, limiting for a commercial release.

---

[README](../README.md) · [Docs](README.md) · [Method](method.md) · **Data** · [Results](results.md) · [Conventions](conventions.md)
