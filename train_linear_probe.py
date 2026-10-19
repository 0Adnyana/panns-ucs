"""
Step 2 of the Beck & Lerch (2026) benchmark: a linear probe on frozen CNN14 embeddings.

The paper does not fine-tune PANNs. This script never loads audio or the CNN14
checkpoint. It reads the cached 2048-d embeddings (embeddings.npz) and trains:

    Dropout(0.3) -> Linear(2048, n_classes)

with AdamW (lr 1e-3, weight decay 1e-3), cosine annealing over the epoch budget,
focal loss (gamma 2) and inverse-frequency class weights. Early stopping watches
validation macro F1 (patience 20, max 200 epochs). Seeds are 42, 123, 456, 789
and 1024. The test split is scored once per seed, at that seed's best
validation checkpoint. Macro F1 is reported as mean +/- sample std.

The paper does not state the batch size, the early-stopping metric, the exact
weight formula, or how a subcategory name maps to a category when the name is
shared. Those choices are recorded in summary.json.

Label schemes (--task):

  subcategory   drop empty SubCategory; class = SubCategory name
                (144 classes on EnvSound). This is the paper's flat subcategory
                classifier. A predicted name is rolled up to a category with the
                most common parent in the training split.
  pair          drop empty SubCategory; class = Category||SubCategory
                (165 classes). Roll-up splits on "||", so it is unambiguous.
  category      keep every clip; class = Category (59 classes). This is the
                paper's direct category classifier.

Labels and splits come from three CSVs, not from the cache. Pass each file
with --train, --eval, and --test. The filenames can be anything; the flag
chooses the split. --val is the same flag as --eval. --manifests FOLDER still
works and looks up envsound_{train,val,test}.csv (or {split}.csv) in that
folder. Do not pass the folder and the three files together.

Each CSV row (exists not false) is matched to embeddings.npz on
filename + source. A missing vector or a repeated pair stops the run.

Usage (from this folder):

  python train_linear_probe.py --train train.csv --eval eval.csv --test test.csv --dry-run
  python train_linear_probe.py --train train.csv --eval eval.csv --test test.csv
  python train_linear_probe.py --train a.csv --eval b.csv --test c.csv --task category
  python train_linear_probe.py --manifests M
  python train_linear_probe.py --train train.csv --eval eval.csv --test test.csv --limit 200 --seeds 42 --max-epochs 5
  python train_linear_probe.py --train train.csv --eval eval.csv --test test.csv --task pair --verbose

--verbose writes <out>/pair/verbose_test.csv (under _smoke_<N>/ when --limit
is set). Every row of the test CSV is kept, in the same order. The top 5
category and subcategory come from the pair head: average the softmax over
seeds, then take the five highest Category||SubCategory classes.
model_confidence_1..5 are those mean class probabilities. human_category
and human_subcategory are left blank. recording and pile
are copied when the test CSV has those columns. A row the pair head did not
score (exists false, empty label, label absent from train, or past --limit)
keeps blank prediction and confidence cells. If --task is not pair, the pair
head is trained as well.
"""
import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader, TensorDataset

EMB_DIM = 2048
DEFAULT_SEEDS = (42, 123, 456, 789, 1024)
EMPTY = {"", "nan", "none", "null", "<na>"}
PAPER_SUB_F1 = 0.49
PAPER_CAT_FROM_SUB_F1 = 0.55
PAPER_CAT_F1 = 0.46


class LinearProbe(nn.Module):
    def __init__(self, n_classes, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(EMB_DIM, n_classes)

    def forward(self, x):
        return self.linear(self.dropout(x))


def focal_loss(logits, target, weight, gamma):
    """Class-weighted focal loss. Reduction matches CrossEntropyLoss(weight=...)."""
    log_pt = F.log_softmax(logits, dim=-1).gather(1, target[:, None]).squeeze(1)
    pt = log_pt.exp()
    w = weight[target]
    per = -(1.0 - pt).pow(gamma) * log_pt * w
    return per.sum() / w.sum().clamp_min(1e-12)


def is_blank(values):
    return np.array([str(v).strip().lower() in EMPTY for v in values], dtype=bool)


def json_ready(value):
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    return value


def truthy(value):
    return str(value).strip().lower() in {"1", "true", "yes", "y", "t"}


def find_manifest(folder, split):
    for name in (f"envsound_{split}.csv", f"envsound/envsound_{split}.csv", f"{split}.csv"):
        path = Path(folder) / name
        if path.is_file():
            return path
    sys.exit(f"No manifest for '{split}' in {folder} (looked for envsound_{split}.csv)")


def read_manifest(path):
    path = Path(path)
    if not path.is_file():
        sys.exit(f"Missing manifest {path}")
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    for col in ("filename", "source", "Category"):
        if col not in df.columns:
            sys.exit(f"{path} has no '{col}' column. Columns: {list(df.columns)}")
    df["csv_line"] = np.arange(len(df)) + 2
    if "exists" in df.columns:
        df = df[df["exists"].map(truthy)]
    df = df.reset_index(drop=True)
    for col in ("filename", "source", "Category", "SubCategory", "path"):
        df[col] = df[col].str.strip() if col in df.columns else ""
    return path, df


def resolve_split_files(args):
    """Map train/val/test to CSV paths. Filenames are not inspected."""
    explicit = (("--train", args.train), ("--eval", args.eval), ("--test", args.test))
    given = [flag for flag, path in explicit if path]
    if args.manifests and given:
        sys.exit("Pass either --manifests or --train, --eval, and --test, not both.")
    if args.manifests:
        return {split: find_manifest(args.manifests, split) for split in ("train", "val", "test")}
    missing = [flag for flag, path in explicit if not path]
    if missing:
        sys.exit(
            "Pass --manifests, or all of --train, --eval, and --test. "
            f"Missing: {', '.join(missing)}"
        )
    return {"train": Path(args.train), "val": Path(args.eval), "test": Path(args.test)}


def load_embeddings(emb_dir, split_files):
    """Join the split CSVs onto embeddings.npz on filename + source."""
    path = Path(emb_dir) / "embeddings.npz"
    if not path.is_file():
        sys.exit(f"Missing {path}. Run extract_panns_embeddings.py first "
                 "(or --combine-from on a folder of old per-split caches).")
    with np.load(path, allow_pickle=True) as z:
        X = np.asarray(z["X"], dtype=np.float32)
        cache_fn = [str(v).strip() for v in z["filename"]]
        cache_src = [str(v).strip() for v in z["source"]]
    if X.ndim != 2 or X.shape[1] != EMB_DIM or len(X) != len(cache_fn):
        sys.exit(f"{path} has X shape {tuple(X.shape)} for {len(cache_fn)} filenames; "
                 f"expected (N, {EMB_DIM})")
    if not np.isfinite(X).all():
        sys.exit(f"{path} contains NaN or inf embeddings")
    row_of = {}
    for i, key in enumerate(zip(cache_fn, cache_src)):
        if key in row_of:
            sys.exit(f"{path} holds filename {key[0]!r} source {key[1]!r} twice")
        row_of[key] = i
    print(f"  cache {path}  X={tuple(X.shape)}", flush=True)

    frames = {split: read_manifest(path) for split, path in split_files.items()}

    seen = defaultdict(list)
    for split, (csv_path, df) in frames.items():
        for fn, src, line in zip(df["filename"], df["source"], df["csv_line"]):
            seen[(fn, src)].append(f"{csv_path.name} line {line}")
    repeated = {key: where for key, where in seen.items() if len(where) > 1}
    if repeated:
        lines = [f"  {fn!r} source {src!r}: " + ", ".join(where)
                 for (fn, src), where in list(repeated.items())[:10]]
        sys.exit(f"{len(repeated)} filename + source pair(s) appear more than once in the manifests:\n"
                 + "\n".join(lines))

    missing = [(csv_path.name, line, fn, src)
               for csv_path, df in frames.values()
               for fn, src, line in zip(df["filename"], df["source"], df["csv_line"])
               if (fn, src) not in row_of]
    if missing:
        lines = [f"  {name} line {line}: {fn!r} source {src!r}" for name, line, fn, src in missing[:10]]
        sys.exit(f"{len(missing)} manifest row(s) have no embedding in {path}:\n" + "\n".join(lines)
                 + "\nRe-run the extractor on these manifests, or drop the rows (exists=false).")

    cache_fn = np.array(cache_fn, dtype=str)
    cache_src = np.array(cache_src, dtype=str)
    splits = {}
    for split, (csv_path, df) in frames.items():
        idx = np.array([row_of[key] for key in zip(df["filename"], df["source"])], dtype=np.int64)
        splits[split] = {
            "X": X[idx],
            "filename": cache_fn[idx],
            "category": df["Category"].to_numpy(dtype=str),
            "subcategory": df["SubCategory"].to_numpy(dtype=str),
            "source": cache_src[idx],
            "path": df["path"].to_numpy(dtype=str),
            "csv_line": df["csv_line"].to_numpy(dtype=np.int64),
        }
        print(f"  joined {split:5}  {csv_path}  X={tuple(splits[split]['X'].shape)}", flush=True)
    unused = len(row_of) - len(seen)
    if unused:
        print(f"  {unused} cache row(s) are not in any manifest (not used)", flush=True)
    return splits


def prepare(raw, task, limit):
    """Filter rows, fit the class list on train, and build the category roll-up."""
    packs = {}
    dropped_empty = {}
    for split, data in raw.items():
        cat = data["category"]
        sub = data["subcategory"]
        blank_sub = is_blank(sub)
        blank_cat = is_blank(cat)
        if task == "category":
            labels = cat
            keep = ~blank_cat
        elif task == "subcategory":
            labels = sub
            keep = ~blank_sub
        elif task == "pair":
            labels = np.array([f"{c}||{s}" for c, s in zip(cat, sub)])
            keep = ~blank_sub & ~blank_cat
        else:
            sys.exit(f"unknown task {task}")
        dropped_empty[split] = int((~keep).sum())
        idx = np.flatnonzero(keep)
        if limit:
            idx = idx[:limit]
        pack = {key: data[key][idx] for key in (
            "X", "filename", "category", "subcategory", "source", "path", "csv_line")}
        pack["label"] = labels[idx]
        packs[split] = pack

    classes = sorted(set(packs["train"]["label"].tolist()))
    if not classes:
        sys.exit("Training split has no labelled rows.")
    index = {name: i for i, name in enumerate(classes)}

    dropped_unseen = {}
    for split, pack in packs.items():
        if split == "train":
            dropped_unseen[split] = 0
            known = np.ones(len(pack["label"]), dtype=bool)
        else:
            known = np.array([name in index for name in pack["label"]], dtype=bool)
            dropped_unseen[split] = int((~known).sum())
        if not known.all():
            for key in list(pack):
                pack[key] = pack[key][known]
        pack["y"] = np.array([index[name] for name in pack["label"]], dtype=np.int64)

    for split, pack in packs.items():
        if len(pack["y"]) == 0:
            sys.exit(f"{split} split has no labelled rows after filtering.")

    parent_of, ambiguous, rollup_rule = build_rollup(packs["train"], classes, task)
    return {
        "task": task,
        "classes": classes,
        "packs": packs,
        "parent_of": parent_of,
        "ambiguous": ambiguous,
        "rollup_rule": rollup_rule,
        "dropped_empty": dropped_empty,
        "dropped_unseen": dropped_unseen,
        "category_names": sorted(set(packs["train"]["category"].tolist())),
    }


def build_rollup(train, classes, task):
    """Map each trained class to the category used for SubCategory -> Category."""
    if task == "category":
        return list(classes), [], "the predicted class is already a category"

    if task == "pair":
        parents = [name.split("||", 1)[0] for name in classes]
        return parents, [], 'split the predicted "Category||SubCategory" label on "||"'

    counts = defaultdict(lambda: defaultdict(int))
    for cat, sub in zip(train["category"], train["subcategory"]):
        counts[sub][cat] += 1
    parents = []
    ambiguous = []
    for name in classes:
        per_cat = dict(counts[name])
        if not per_cat:
            sys.exit(f"subcategory {name!r} has no parent category in the training split")
        best = max(per_cat.values())
        chosen = sorted(cat for cat, n in per_cat.items() if n == best)[0]
        parents.append(chosen)
        if len(per_cat) > 1:
            ambiguous.append({
                "name": name,
                "parent_used": chosen,
                "train_counts": per_cat,
            })
    ambiguous.sort(key=lambda row: -sum(row["train_counts"].values()))
    rule = (
        "most common parent category of that subcategory name in the training split; "
        "ties broken by category name"
    )
    return parents, ambiguous, rule


def class_weights(y, n_classes):
    """w_c = N / (n_classes * count_c). Mean weight across samples is 1."""
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    if (counts == 0).any():
        missing = np.flatnonzero(counts == 0).tolist()
        sys.exit(f"training classes with no samples: {missing}")
    weights = counts.sum() / (n_classes * counts)
    return torch.tensor(weights, dtype=torch.float32), counts


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def topk_accuracy(logits, y, k):
    k = min(k, logits.shape[1])
    chosen = np.argpartition(logits, -k, axis=1)[:, -k:]
    return float(np.any(chosen == y[:, None], axis=1).mean())


def forward_split(model, X, y, weight, gamma, device, batch_size):
    model.eval()
    chunks = []
    total_loss = 0.0
    total_w = 0.0
    with torch.inference_mode():
        for start in range(0, len(X), batch_size):
            xb = torch.from_numpy(X[start:start + batch_size]).to(device, non_blocking=True)
            yb = torch.from_numpy(y[start:start + batch_size]).to(device, non_blocking=True)
            logits = model(xb)
            wsum = float(weight[yb].sum())
            total_loss += float(focal_loss(logits, yb, weight, gamma)) * wsum
            total_w += wsum
            chunks.append(logits.float().cpu().numpy())
    return np.concatenate(chunks, axis=0), total_loss / max(total_w, 1e-12)


def present_macro_f1(y_true, y_pred, labels):
    """Macro F1 over `labels`, plus the same mean restricted to classes with support."""
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0)
    present = support > 0
    macro_present = float(f1[present].mean()) if present.any() else 0.0
    macro_all = float(f1.mean()) if len(f1) else 0.0
    return macro_present, macro_all, precision, recall, f1, support


def score_split(y_true, y_pred, logits, classes):
    labels = list(range(len(classes)))
    macro, macro_all, precision, recall, f1, support = present_macro_f1(y_true, y_pred, labels)
    report = pd.DataFrame({
        "label": classes,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support.astype(int),
    })
    metrics = {
        "n": int(len(y_true)),
        "n_classes_with_support": int((support > 0).sum()),
        "macro_f1": macro,
        "macro_f1_all_train_classes": macro_all,
        "accuracy": float((y_true == y_pred).mean()),
        "top3_accuracy": topk_accuracy(logits, y_true, 3),
        "top5_accuracy": topk_accuracy(logits, y_true, 5),
    }
    return metrics, report


def score_categories(y_true_cat, y_pred_cat, category_names):
    # Include a category that shows up only at test time, so its errors are counted.
    names = list(category_names)
    known = set(names)
    extra = sorted((set(y_true_cat.tolist()) | set(y_pred_cat.tolist())) - known)
    labels = names + extra
    macro, _, precision, recall, f1, support = present_macro_f1(
        y_true_cat, y_pred_cat, labels)
    macro_all = float(f1[:len(names)].mean()) if names else 0.0
    report = pd.DataFrame({
        "category": labels,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support.astype(int),
    })
    metrics = {
        "category_macro_f1": macro,
        "category_macro_f1_all_train_categories": macro_all,
        "category_accuracy": float(np.mean(y_true_cat == y_pred_cat)),
        "n_categories_with_support": int((support > 0).sum()),
    }
    return metrics, report


def per_source_table(pack, y_pred, logits, parent_of):
    rows = []
    y_true = pack["y"]
    pred_cat = np.array([parent_of[i] for i in y_pred])
    for source in sorted(set(pack["source"].tolist())):
        mask = pack["source"] == source
        if not mask.any():
            continue
        yt, yp = y_true[mask], y_pred[mask]
        present = np.unique(yt).tolist()
        macro, _, _, _, _, _ = present_macro_f1(yt, yp, present)
        row = {
            "source": source,
            "n": int(mask.sum()),
            "macro_f1": macro,
            "accuracy": float((yt == yp).mean()),
            "top3_accuracy": topk_accuracy(logits[mask], yt, 3),
            "top5_accuracy": topk_accuracy(logits[mask], yt, 5),
        }
        ct, cp = pack["category"][mask], pred_cat[mask]
        present_cat = sorted(set(ct.tolist()))
        cat_macro, _, _, _, _, _ = present_macro_f1(ct, cp, present_cat)
        row["category_macro_f1"] = cat_macro
        row["category_accuracy"] = float(np.mean(ct == cp))
        rows.append(row)
    return pd.DataFrame(rows)


def train_one_seed(data, seed, device, args, seed_dir):
    metrics_path = seed_dir / "metrics.json"
    if metrics_path.is_file() and not args.restart:
        print(f"seed {seed}: already trained, skipping ({metrics_path})", flush=True)
        with open(metrics_path, encoding="utf-8") as f:
            return json.load(f)

    classes = data["classes"]
    n_classes = len(classes)
    train, val, test = data["packs"]["train"], data["packs"]["val"], data["packs"]["test"]
    set_seed(seed)
    model = LinearProbe(n_classes, args.dropout).to(device)
    weight, counts = class_weights(train["y"], n_classes)
    weight = weight.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.max_epochs)

    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(train["X"]), torch.from_numpy(train["y"])),
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        drop_last=False,
        pin_memory=(device == "cuda"),
    )

    best_f1 = -1.0
    best_epoch = 0
    best_state = None
    patience_left = args.patience
    stop_reason = "max_epochs"
    history = []
    t0 = time.time()

    for epoch in range(1, args.max_epochs + 1):
        model.train()
        running = 0.0
        running_w = 0.0
        lr = opt.param_groups[0]["lr"]
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = focal_loss(model(xb), yb, weight, args.gamma)
            loss.backward()
            opt.step()
            # Reconstruct the weighted sum from the mean so the epoch loss matches val.
            wsum = float(weight[yb].sum())
            running += float(loss.detach()) * wsum
            running_w += wsum
        schedule.step()

        val_logits, val_loss = forward_split(
            model, val["X"], val["y"], weight, args.gamma, device, args.batch_size)
        val_pred = val_logits.argmax(axis=1)
        val_f1, _, _, _, _, _ = present_macro_f1(val["y"], val_pred, list(range(n_classes)))
        improved = val_f1 > best_f1
        if improved:
            best_f1 = val_f1
            best_epoch = epoch
            patience_left = args.patience
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_left -= 1
        history.append({
            "epoch": epoch,
            "lr": lr,
            "train_loss": running / max(running_w, 1e-12),
            "val_loss": val_loss,
            "val_macro_f1": val_f1,
            "improved": int(improved),
        })
        print(
            f"  seed {seed}  epoch {epoch:3d}  lr {lr:.6f}  "
            f"train {history[-1]['train_loss']:.4f}  val {val_loss:.4f}  "
            f"val F1 {val_f1:.4f}" + ("  best" if improved else f"  patience {patience_left}"),
            flush=True,
        )
        if patience_left <= 0:
            stop_reason = "patience"
            break

    if best_state is None:
        sys.exit(f"seed {seed} produced no checkpoint")
    model.load_state_dict(best_state)
    model.to(device)
    test_logits, test_loss = forward_split(
        model, test["X"], test["y"], weight, args.gamma, device, args.batch_size)
    test_pred = test_logits.argmax(axis=1)
    test_metrics, class_report = score_split(test["y"], test_pred, test_logits, classes)

    parent_of = data["parent_of"]
    pred_cat = np.array([parent_of[i] for i in test_pred])
    cat_metrics, cat_report = score_categories(test["category"], pred_cat, data["category_names"])
    test_metrics.update(cat_metrics)
    test_metrics["loss"] = test_loss

    source_report = per_source_table(test, test_pred, test_logits, parent_of)
    cm = confusion_matrix(test["y"], test_pred, labels=list(range(n_classes)))
    cat_labels = cat_report["category"].tolist()
    cm_cat = confusion_matrix(test["category"], pred_cat, labels=cat_labels)

    seed_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(seed_dir / "history.csv", index=False)
    class_report.to_csv(seed_dir / "per_class_f1.csv", index=False)
    cat_report.to_csv(seed_dir / "per_category_f1.csv", index=False)
    source_report.to_csv(seed_dir / "per_source_f1.csv", index=False)
    pd.DataFrame(cm, index=classes, columns=classes).rename_axis("true").to_csv(
        seed_dir / "confusion_labels.csv")
    pd.DataFrame(cm_cat, index=cat_labels, columns=cat_labels).rename_axis(
        "true").to_csv(seed_dir / "confusion_category.csv")
    np.savez_compressed(
        seed_dir / "predictions.npz",
        logits=test_logits.astype(np.float32),
        y_true=test["y"],
        y_pred=test_pred.astype(np.int64),
        filename=test["filename"],
        source=test["source"],
        path=test["path"],
        category_true=test["category"],
        category_pred=pred_cat,
        subcategory_true=test["subcategory"],
        classes=np.array(classes),
        csv_line=test["csv_line"].astype(np.int64),
    )
    torch.save({
        "state_dict": best_state,
        "classes": classes,
        "parent_of": parent_of,
        "task": data["task"],
        "seed": seed,
        "counts": counts.tolist(),
    }, seed_dir / "probe.pt")

    record = {
        "seed": seed,
        "best_epoch": best_epoch,
        "epochs_ran": history[-1]["epoch"],
        "stop_reason": stop_reason,
        "best_val_macro_f1": best_f1,
        "seconds": round(time.time() - t0, 1),
        "test": test_metrics,
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(json_ready(record), f, indent=2)
    print(
        f"seed {seed} done  best epoch {best_epoch}  "
        f"test F1 {test_metrics['macro_f1']:.4f}  "
        f"category F1 {test_metrics['category_macro_f1']:.4f}  "
        f"top3 {test_metrics['top3_accuracy']:.4f}  top5 {test_metrics['top5_accuracy']:.4f}",
        flush=True,
    )
    return record


def mean_std(values):
    arr = np.asarray(values, dtype=np.float64)
    std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    return {"mean": float(arr.mean()), "std": std, "n": int(len(arr))}


def write_summary(data, records, out_dir, args, device):
    test_keys = [
        "macro_f1",
        "macro_f1_all_train_classes",
        "accuracy",
        "top3_accuracy",
        "top5_accuracy",
        "category_macro_f1",
        "category_macro_f1_all_train_categories",
        "category_accuracy",
    ]
    summary = {
        "task": data["task"],
        "n_classes": len(data["classes"]),
        "classes": data["classes"],
        "category_names": data["category_names"],
        "rollup_rule": data["rollup_rule"],
        "ambiguous_subcategories": data["ambiguous"],
        "dropped_empty": data["dropped_empty"],
        "dropped_unseen_label": data["dropped_unseen"],
        "rows": {split: int(len(data["packs"][split]["y"])) for split in ("train", "val", "test")},
        "device": device,
        "seeds": [r["seed"] for r in records],
        "per_seed": records,
        "mean_std": {},
        "std_definition": "sample standard deviation over seeds, ddof=1",
        "macro_f1_definition": (
            "mean of per-class F1 over classes with at least one true sample "
            "in the scored split. macro_f1_all_train_classes gives missing classes F1 0"
        ),
        "settings": {
            "embeddings": args.embeddings,
            "manifests": args.manifests,
            "train": str(args.split_files["train"]),
            "eval": str(args.split_files["val"]),
            "test": str(args.split_files["test"]),
            "verbose": bool(args.verbose),
            "dropout": args.dropout,
            "optimizer": "AdamW",
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "schedule": "CosineAnnealingLR, stepped once per epoch, T_max=max_epochs, eta_min=0",
            "loss": "focal",
            "gamma": args.gamma,
            "class_weight": "w_c = N / (n_classes * count_c) on the training split",
            "loss_reduction": "weight-normalized mean, same as torch.nn.CrossEntropyLoss(weight=...)",
            "early_stopping": "validation macro F1",
            "patience": args.patience,
            "max_epochs": args.max_epochs,
            "batch_size": args.batch_size,
            "limit": args.limit,
            "paper_does_not_specify": [
                "batch_size",
                "early_stopping metric (this script uses validation macro F1)",
                "inverse-frequency formula (this script uses balanced weights)",
                "parent category when a subcategory name is shared",
            ],
        },
    }
    if data["task"] == "subcategory":
        summary["paper_targets"] = {
            "subcategory_macro_f1": PAPER_SUB_F1,
            "category_from_subcategory_macro_f1": PAPER_CAT_FROM_SUB_F1,
        }
    elif data["task"] == "category":
        summary["paper_targets"] = {"category_macro_f1": PAPER_CAT_F1}

    for key in test_keys:
        summary["mean_std"][key] = mean_std([r["test"][key] for r in records])
    summary["mean_std"]["best_val_macro_f1"] = mean_std([r["best_val_macro_f1"] for r in records])

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(json_ready(summary), f, indent=2)
    return summary


def format_pm(stats):
    return f"{stats['mean']:.3f} +/- {stats['std']:.3f}"


def print_summary(summary):
    stats = summary["mean_std"]
    print("\nTest, mean +/- std over seeds", flush=True)
    print(f"  macro F1:                 {format_pm(stats['macro_f1'])}", flush=True)
    print(f"  accuracy:                 {format_pm(stats['accuracy'])}", flush=True)
    print(f"  top-3 accuracy:           {format_pm(stats['top3_accuracy'])}", flush=True)
    print(f"  top-5 accuracy:           {format_pm(stats['top5_accuracy'])}", flush=True)
    print(f"  category macro F1:        {format_pm(stats['category_macro_f1'])}", flush=True)
    print(f"  category accuracy:        {format_pm(stats['category_accuracy'])}", flush=True)
    targets = summary.get("paper_targets")
    if summary["task"] == "subcategory" and targets:
        print(
            f"  paper EnvSound targets:   subcategory {targets['subcategory_macro_f1']:.2f}, "
            f"subcategory -> category {targets['category_from_subcategory_macro_f1']:.2f}",
            flush=True,
        )
    elif summary["task"] == "category" and targets:
        print(f"  paper EnvSound target:    category {targets['category_macro_f1']:.2f}", flush=True)


def describe(data):
    rows = {split: len(data["packs"][split]["y"]) for split in ("train", "val", "test")}
    print(
        f"task {data['task']}  classes {len(data['classes'])}  "
        f"train {rows['train']}  val {rows['val']}  test {rows['test']}",
        flush=True,
    )
    print(
        "empty rows dropped: "
        + ", ".join(f"{split} {n}" for split, n in data["dropped_empty"].items()),
        flush=True,
    )
    unseen = data["dropped_unseen"]
    if any(unseen.values()):
        print(
            "val/test rows whose label is absent from train, dropped: "
            + ", ".join(f"{split} {n}" for split, n in unseen.items()),
            flush=True,
        )
    ambiguous = data["ambiguous"]
    if ambiguous:
        train_sub = data["packs"]["train"]["subcategory"]
        names = {row["name"] for row in ambiguous}
        n_hit = int(sum(name in names for name in train_sub))
        print(
            f"{len(ambiguous)} subcategory name{'s' if len(ambiguous) != 1 else ''} "
            f"ha{'ve' if len(ambiguous) != 1 else 's'} more than one parent "
            f"({n_hit} of {len(train_sub)} training clips). "
            "Roll-up uses the most common parent in train.",
            flush=True,
        )
        print("  shared names: " + ", ".join(row["name"] for row in ambiguous), flush=True)


VERBOSE_PASSTHROUGH = (
    "filename", "Category", "SubCategory", "Keywords", "source", "path", "exists",
    "recording", "pile",
)
VERBOSE_COLUMNS = (
    list(VERBOSE_PASSTHROUGH)
    + [f"model_prediction_{i}_{part}" for i in range(1, 6) for part in ("category", "subcategory")]
    + [f"model_confidence_{i}" for i in range(1, 6)]
    + ["human_category", "human_subcategory"]
)


def softmax_rows(logits):
    z = np.asarray(logits, dtype=np.float64)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    e /= e.sum(axis=1, keepdims=True)
    return e


def load_pair_seed(seed_dir):
    """Softmax of one pair-head seed, aligned by the test CSV line."""
    path = Path(seed_dir) / "predictions.npz"
    if not path.is_file():
        sys.exit(f"Missing {path}. Retrain this seed with --restart.")
    with np.load(path, allow_pickle=True) as z:
        if "csv_line" not in z.files:
            sys.exit(
                f"{path} has no csv_line. Retrain this seed with --restart "
                "so --verbose can match predictions to test rows."
            )
        classes = [str(c) for c in np.asarray(z["classes"]).tolist()]
        lines = np.asarray(z["csv_line"], dtype=np.int64)
        logits = np.asarray(z["logits"], dtype=np.float64)
    if logits.ndim != 2 or len(lines) != len(logits):
        sys.exit(f"{path} has logits {tuple(logits.shape)} for {len(lines)} test rows")
    if any("||" not in name for name in classes):
        sys.exit(f"{path} is not a pair-head probe. Class names have no '||'.")
    if len(classes) != logits.shape[1]:
        sys.exit(f"{path} has {len(classes)} classes and logits width {logits.shape[1]}")
    return classes, lines, softmax_rows(logits)


def mean_pair_topk(seed_dirs, k=5):
    """Average pair-head softmax over seeds and return the top k class indices."""
    loaded = [load_pair_seed(path) for path in seed_dirs]
    classes, lines, _ = loaded[0]
    for classes_i, lines_i, _ in loaded[1:]:
        if classes_i != classes:
            sys.exit("Pair seeds have different class lists, so their softmax cannot be averaged.")
        if not np.array_equal(lines_i, lines):
            sys.exit("Pair seeds scored different test rows, so their softmax cannot be averaged.")
    if len(set(int(line) for line in lines)) != len(lines):
        sys.exit("Pair predictions repeat a test CSV line.")
    mean = np.stack([probs for _, _, probs in loaded], axis=0).mean(axis=0)
    k_eff = min(k, mean.shape[1])
    order = np.argsort(-mean, axis=1, kind="mergesort")[:, :k_eff]
    by_line = {}
    for row, line in enumerate(lines):
        preds = []
        confs = []
        for rank in range(k):
            if rank >= k_eff:
                preds.append(("", ""))
                confs.append("")
                continue
            cls = int(order[row, rank])
            cat, sub = classes[cls].split("||", 1)
            preds.append((cat, sub))
            confs.append(f"{mean[row, cls]:.6f}")
        by_line[int(line)] = (preds, confs)
    return by_line


def write_verbose_test_csv(test_csv, seed_dirs, dest):
    """One row per test-CSV row. Unscored rows keep blank prediction cells."""
    by_line = mean_pair_topk(seed_dirs)
    raw = pd.read_csv(test_csv, dtype=str, keep_default_na=False)
    raw["_csv_line"] = np.arange(len(raw)) + 2
    file_lines = {int(line) for line in raw["_csv_line"]}
    missing = [line for line in by_line if line not in file_lines]
    if missing:
        sys.exit(
            f"{len(missing)} scored test row(s) are not in {test_csv}. "
            "The pair checkpoints were trained on a different file."
        )
    rows = []
    filled = 0
    for record in raw.to_dict("records"):
        out = {col: "" for col in VERBOSE_COLUMNS}
        for col in VERBOSE_PASSTHROUGH:
            if col in record:
                out[col] = record[col]
        hit = by_line.get(int(record["_csv_line"]))
        if hit is not None:
            filled += 1
            preds, confs = hit
            for i, ((cat, sub), conf) in enumerate(zip(preds, confs), start=1):
                out[f"model_prediction_{i}_category"] = cat
                out[f"model_prediction_{i}_subcategory"] = sub
                out[f"model_confidence_{i}"] = conf
        rows.append(out)
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=VERBOSE_COLUMNS).to_csv(dest, index=False, encoding="utf-8")
    return len(rows), filled


def note_verbose_csv(summary_path, csv_path, n_rows, n_filled):
    summary_path = Path(summary_path)
    if not summary_path.is_file():
        return
    with open(summary_path, encoding="utf-8") as f:
        summary = json.load(f)
    summary["verbose_test_csv"] = str(csv_path)
    summary["verbose_rule"] = (
        "Top 5 are the pair-head classes with the highest mean softmax over seeds. "
        "model_confidence_k is that class probability, written with 6 decimal places. "
        "Rows the pair head did not score have blank prediction and confidence cells."
    )
    summary["verbose_rows"] = {"test": n_rows, "with_predictions": n_filled}
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def task_output_dir(args, task):
    out_dir = Path(args.out)
    if args.limit:
        out_dir = out_dir / f"_smoke_{args.limit}"
    return out_dir / task


def train_task(data, args, device, seeds, out_dir):
    print(f"Writing {out_dir}", flush=True)
    records = []
    for seed in seeds:
        records.append(train_one_seed(data, seed, device, args, out_dir / f"seed_{seed}"))
    summary = write_summary(data, records, out_dir, args, device)
    print_summary(summary)
    print(f"Wrote {out_dir / 'summary.json'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--embeddings", default="embeddings/panns_cnn14",
                        help="folder holding embeddings.npz from the extractor")
    parser.add_argument("--manifests", default=None,
                        help="folder with envsound_{train,val,test}.csv. Alternative to --train/--eval/--test")
    parser.add_argument("--train", default=None, help="training CSV. Any filename")
    parser.add_argument("--eval", "--val", dest="eval", default=None,
                        help="validation CSV, used for early stopping. Any filename. --val is the same flag")
    parser.add_argument("--test", default=None, help="test CSV, scored once per seed. Any filename")
    parser.add_argument("--out", default="results/panns_cnn14_linear")
    parser.add_argument("--task", choices=("subcategory", "category", "pair"), default="subcategory")
    parser.add_argument("--seeds", default=",".join(str(s) for s in DEFAULT_SEEDS))
    parser.add_argument("--device", default="auto", help="auto, cuda, or cpu")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--gamma", type=float, default=2.0)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--limit", type=int, default=0, help="first N labelled rows per split (smoke test)")
    parser.add_argument("--restart", action="store_true", help="retrain seeds that already have metrics.json")
    parser.add_argument("--dry-run", action="store_true", help="print the label setup and exit")
    parser.add_argument(
        "--verbose", action="store_true",
        help="write <out>/pair/verbose_test.csv with the top 5 pair-head predictions "
             "for every test row. Probabilities are the mean softmax over seeds. "
             "Trains the pair head as well when --task is not pair",
    )
    args = parser.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        sys.exit("CUDA was requested, but torch.cuda.is_available() is False.")
    if device not in ("cuda", "cpu"):
        sys.exit(f"unknown device {device!r}")

    seeds = [int(part) for part in args.seeds.split(",") if part.strip()]
    if not seeds:
        sys.exit("pass at least one seed")

    print(f"Device: {device}", flush=True)
    args.split_files = resolve_split_files(args)
    raw = load_embeddings(args.embeddings, args.split_files)
    data = prepare(raw, args.task, args.limit)
    pair_data = prepare(raw, "pair", args.limit) if args.verbose and args.task != "pair" else None
    del raw
    describe(data)
    if args.dry_run:
        if pair_data is not None:
            print("Verbose predictions use the pair head:", flush=True)
            describe(pair_data)
        print("Dry run: no training.", flush=True)
        if args.verbose:
            print("No verbose_test.csv on a dry run.", flush=True)
        return

    train_task(data, args, device, seeds, task_output_dir(args, args.task))
    if not args.verbose:
        return
    if pair_data is not None:
        print("Verbose predictions use the pair head. Training that probe as well.", flush=True)
        describe(pair_data)
        pair_dir = task_output_dir(args, "pair")
        train_task(pair_data, args, device, seeds, pair_dir)
    else:
        pair_dir = task_output_dir(args, "pair")
    dest = pair_dir / "verbose_test.csv"
    n_rows, n_filled = write_verbose_test_csv(
        args.split_files["test"],
        [pair_dir / f"seed_{seed}" for seed in seeds],
        dest,
    )
    note_verbose_csv(pair_dir / "summary.json", dest, n_rows, n_filled)
    print(
        f"Wrote {dest}  ({n_filled} of {n_rows} test rows have a top-5 prediction)",
        flush=True,
    )


if __name__ == "__main__":
    main()
