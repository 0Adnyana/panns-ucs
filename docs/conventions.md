# Conventions

[README](../README.md) · [Docs](README.md) · [Method](method.md) · [Data](data.md) · [Results](results.md) · **Conventions**

Decisions that code in this repo should respect, and choices the thesis has
not settled yet.

**On this page:** [Code conventions](#conventions) · [Open decisions](#open-decisions)

<a id="conventions"></a>

## Code conventions

- **Base case = paper setup exactly.** Don't change splits, rebalance data, or normalise loudness for the reproduction. Rebalancing and augmentation are later experiments, applied to the **train split only**. The paper's settings are in [Method](method.md#paper-method).
- **Call the frozen setup a "linear probe", not fine-tuning.**
- Val is for early stopping only. Test is touched once, at the end.
- Use the 32 kHz `Cnn14_mAP=0.431.pth` checkpoint in `models/`. Reject other PANNs checkpoints. The script downloads it if it is missing ([checkpoint](../README.md#checkpoint)).
- Skip manifest rows with `exists = False`.
- Manifest `path` values are relative to the data root (parent of `manifests/`), not to the manifests folder. Resolve them there unless the path is already absolute. See [Manifests](data.md#manifests).
- Keep `source` alongside every prediction so results can be split per dataset.
- Unspecified paper details, fixed in `train_linear_probe.py`: batch size 256, early stopping on validation macro F1, class weights `w_c = N / (n_classes * count_c)`, shared subcategory names roll up to the most common training parent (ties broken by category name). Default label scheme is the paper's 144-class subcategory head. `--task category` and `--task pair` are the other two ([tasks](../README.md#tasks)).
- Long jobs (extraction, the linear probe, fine-tuning) are resumable scripts. The probe skips a seed that already has `metrics.json`.
- Report macro F1 as mean ± std over the 5 seeds, plus top-k accuracy. The definition is under [linear probe](method.md#linear-probe).
- `embeddings/`, `results/`, and `*.pth` stay out of git.
- Primary environment: Windows (PowerShell), RTX 4060, data under `E:\Sounds for TA\Extracted`. Avoid tools that need `wget` or Unix-only paths.

<a id="open-decisions"></a>

## Open decisions

- **Main training set.** Option A, FSD-env (cleaner, higher F1) vs Option B, EnvSound (59 categories, about 0.17 lower F1, more practical). Paper numbers for both are in [target scores](method.md#target-numbers).
- What F1 / top-3 accuracy counts as "good enough" for the tool. Current EnvSound numbers are in [Results](results.md#results).
- Assign tags vs suggest tags for confirmation.
- Delivery format of the tool (script over a folder, plugin, web app).

---

[README](../README.md) · [Docs](README.md) · [Method](method.md) · [Data](data.md) · [Results](results.md) · **Conventions**
