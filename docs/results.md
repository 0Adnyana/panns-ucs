# Results

[README](../README.md) · [Docs](README.md) · [Method](method.md) · [Data](data.md) · **Results** · [Conventions](conventions.md)

EnvSound scores from the 5-seed linear probe, and what to look at if the
number is not the whole story. Paper targets are in
[Method](method.md#target-numbers). Output files are listed in
[Getting started](../README.md#train-output).

**On this page:** [Scores](#results) · [What to check](#evaluation)

<a id="results"></a>

## EnvSound scores

Five seeds. Macro F1, mean ± sample standard deviation.

| Setup | This run | Paper |
| --- | --- | --- |
| SubCategory (144) | 0.496 ± 0.007 | 0.49 |
| SubCategory → Category | 0.487 ± 0.003 | 0.55 |
| Category classifier (59) | 0.478 ± 0.005 | 0.46 |

Subcategory macro F1 matches the paper. Direct category is a bit above 0.46.
The roll-up lands at 0.487 against the paper's 0.55. On the subcategory head,
top-3 / top-5 accuracy is 0.777 ± 0.006 / 0.846 ± 0.006.

The `pair` head (165 classes, `Category||SubCategory`) is implemented and has
not been run. The command is in [Getting started](../README.md#train).

<a id="evaluation"></a>

## What to check

These files answer whether the model is the bottleneck. Read them in this order.

1. **Per-category F1** (`per_category_f1.csv`). Is the error spread out, or concentrated in small categories?
2. **Within-category confusion** (`confusion_category.csv`). If most errors stay in the right category, the category tag is already usable.
3. **Top-3 accuracy** (`metrics.json`). If the right tag is usually in the top 3, the tool can suggest tags for a person to confirm.
4. **Fine-tuning gain.** If fine-tuned PANNs clearly beats the linear probe, more model work pays off. If not, the limit is data or labels.

Levers if accuracy stays low: a stronger or fine-tuned encoder, a narrower
scope (FSD-env's ~36 categories instead of 59), confidence thresholds, or
suggesting tags instead of assigning them. Which of those to pursue is still
open — see [Open decisions](conventions.md#open-decisions).

Also split scores by `source` (`per_source_f1.csv`). About 20% of EnvSound
test clips overlap CNN14's pretraining data
([encoder leakage](data.md#data-issues)).

---

[README](../README.md) · [Docs](README.md) · [Method](method.md) · [Data](data.md) · **Results** · [Conventions](conventions.md)
