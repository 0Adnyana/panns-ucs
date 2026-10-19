"""
Step 1 of the Beck & Lerch (2026) benchmark: frozen PANNs CNN14 embeddings.

The paper does not fine-tune PANNs. It runs every clip through the pretrained
CNN14 (AudioSet, Cnn14_mAP=0.431), keeps the 2048-d embedding, and trains a
dropout(0.3) + linear classifier on those vectors. This script does the first
half: one embedding per clip, in a single cache for every split.

The cache holds no labels. Labels come from the manifests at training time
(train_linear_probe.py --manifests), so a new label set reuses this cache.

Input : a folder of envsound_{train,val,test}.csv, or one CSV
        (only filename, source, path, exists are read; they say which clips to embed)
Output: <out>/embeddings.npz with
          X           float32 (N, 2048)
          filename, source   (str arrays; the pair is the lookup key)
        <out>/envsound_{split}_failed.csv   clips that could not be read
        (one CSV writes <stem>_failed.csv instead)
        <out>/extract_info.json             settings used (for your methods section)

The checkpoint defaults to models/Cnn14_mAP=0.431.pth next to this script.
If that file is missing or too small, it is downloaded there (~330 MB).

Usage (Windows, from the folder that holds this script):
  python extract_panns_embeddings.py --manifests "E:\\Sounds for TA\\Extracted\\manifests" --limit 50
  python extract_panns_embeddings.py --manifests "E:\\Sounds for TA\\Extracted\\manifests"
  python extract_panns_embeddings.py --manifests "E:\\Sounds for TA\\Extracted\\manifests\\envsound_train.csv"

Paths: if the manifests have a path column (path / filepath / audio_path),
it is used. Absolute paths are used as written. Relative paths (as written by
prepare_envsound_ucs.py, e.g. FSD50K/FSD50K.dev_audio/123.wav) are resolved
against --audio-root, or against the parent of the manifests folder (the data
root). Without a path column, --audio-root is searched for each filename.
Rows with exists = False/0 are skipped.

Resumable: progress is saved every --save-every clips; just run it again.

Old per-split caches (envsound_{train,val,test}.npz) become one embeddings.npz
without CNN14:
  python extract_panns_embeddings.py --combine-from embeddings/panns_cnn14

Requires: pip install -r requirements.txt
(The panns_inference package is not used: on Windows it tries to download
files with wget at import time and crashes.)
"""
import argparse
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

SR = 32000                 # CNN14 was trained at 32 kHz
MIN_SAMPLES = SR           # clips shorter than 1 s are zero-padded to 1 s
CKPT_NAME = "Cnn14_mAP=0.431.pth"
DEFAULT_CKPT = Path(__file__).resolve().parent / "models" / CKPT_NAME
CKPT_URL = "https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth?download=1"
PATH_COLUMNS = ["path", "filepath", "file_path", "audio_path", "abs_path"]
AUDIO_EXT = {".wav", ".flac", ".ogg", ".mp3", ".m4a", ".aiff", ".aif"}


# --------------------------------------------------------------------------
# CNN14 (copied from qiuqiangkong/audioset_tagging_cnn, inference parts only)
# --------------------------------------------------------------------------
def build_cnn14():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torchlibrosa.stft import Spectrogram, LogmelFilterBank

    class ConvBlock(nn.Module):
        def __init__(self, cin, cout):
            super().__init__()
            self.conv1 = nn.Conv2d(cin, cout, 3, 1, 1, bias=False)
            self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
            self.bn1 = nn.BatchNorm2d(cout)
            self.bn2 = nn.BatchNorm2d(cout)

        def forward(self, x, pool_size):
            x = F.relu_(self.bn1(self.conv1(x)))
            x = F.relu_(self.bn2(self.conv2(x)))
            return F.avg_pool2d(x, kernel_size=pool_size)

    class Cnn14(nn.Module):
        def __init__(self):
            super().__init__()
            self.spectrogram_extractor = Spectrogram(
                n_fft=1024, hop_length=320, win_length=1024, window="hann",
                center=True, pad_mode="reflect", freeze_parameters=True)
            self.logmel_extractor = LogmelFilterBank(
                sr=SR, n_fft=1024, n_mels=64, fmin=50, fmax=14000,
                ref=1.0, amin=1e-10, top_db=None, freeze_parameters=True)
            self.bn0 = nn.BatchNorm2d(64)
            chans = [1, 64, 128, 256, 512, 1024, 2048]
            for i in range(6):
                setattr(self, f"conv_block{i+1}", ConvBlock(chans[i], chans[i + 1]))
            self.fc1 = nn.Linear(2048, 2048)
            self.fc_audioset = nn.Linear(2048, 527)

        def forward(self, wav):                       # wav: (B, samples)
            x = self.logmel_extractor(self.spectrogram_extractor(wav))
            x = self.bn0(x.transpose(1, 3)).transpose(1, 3)
            for i in range(1, 6):
                x = getattr(self, f"conv_block{i}")(x, (2, 2))
            x = self.conv_block6(x, (1, 1))
            x = torch.mean(x, dim=3)
            x = torch.max(x, dim=2)[0] + torch.mean(x, dim=2)
            return F.relu_(self.fc1(x))               # the 2048-d "embedding"

    return Cnn14()


def load_model(ckpt_path, device):
    import torch
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists() or ckpt_path.stat().st_size < 3e8:
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading CNN14 checkpoint (~330 MB) to {ckpt_path} ...")
        tmp = ckpt_path.with_suffix(".part")
        urllib.request.urlretrieve(CKPT_URL, tmp)
        tmp.replace(ckpt_path)
    model = build_cnn14()
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = state.get("model", state)
    missing, unexpected = model.load_state_dict(state, strict=False)
    # spec_augmenter has no weights; anything else missing means a wrong file
    missing = [k for k in missing if not k.startswith("spec_augmenter")]
    if missing:
        sys.exit(f"Checkpoint does not match CNN14, missing weights: {missing[:5]}")
    return model.to(device).eval()


# --------------------------------------------------------------------------
# Manifests and paths
# --------------------------------------------------------------------------
def truthy(v):
    return str(v).strip().lower() in {"1", "true", "yes", "y", "t"}


def find_manifest(folder, split):
    for name in (f"envsound_{split}.csv", f"envsound/envsound_{split}.csv", f"{split}.csv"):
        p = Path(folder) / name
        if p.exists():
            return p
    sys.exit(f"No manifest for '{split}' in {folder} (looked for envsound_{split}.csv)")


def resolve_manifests(manifests, splits):
    """One CSV, or a folder of envsound_{split}.csv. Keys label the failed-clip files."""
    target = Path(manifests)
    if target.is_file():
        return {target.stem: target}
    if target.is_dir():
        return {split: find_manifest(target, split) for split in splits}
    sys.exit(f"--manifests is not a CSV or a folder: {target}")


_file_index = None


def index_audio_root(root):
    global _file_index
    if _file_index is None:
        print(f"Indexing audio files under {root} ...")
        _file_index = {}
        for dirpath, _, files in os.walk(root):
            for f in files:
                if Path(f).suffix.lower() in AUDIO_EXT:
                    _file_index.setdefault(f, os.path.join(dirpath, f))
        print(f"  {len(_file_index):,} audio files found")
    return _file_index


def path_bases(manifest_path, audio_root):
    """Folders to try for a relative path column, most likely first."""
    if audio_root:
        return [Path(audio_root)]
    # prepare_envsound_ucs.py writes paths relative to <data>/, and the CSVs
    # live in <data>/manifests/. Also accept CSVs that sit next to the audio.
    return [manifest_path.parent.parent, manifest_path.parent]


def resolve_paths(df, manifest_path, audio_root):
    col = next((c for c in PATH_COLUMNS if c in df.columns), None)
    if col:
        bases = path_bases(manifest_path, audio_root)
        out, used = [], None
        for raw in df[col].astype(str):
            p = raw.strip()
            if os.path.isabs(p):
                out.append(os.path.normpath(p))
                continue
            found = ""
            for base in bases:
                cand = os.path.normpath(str(base / p))
                if os.path.exists(cand):
                    found, used = cand, used or base
                    break
            out.append(found or os.path.normpath(str(bases[0] / p)))
        if used is not None:
            print(f"  audio root: {used}")
        return out
    if not audio_root:
        sys.exit(f"{manifest_path.name} has no path column ({', '.join(PATH_COLUMNS)}); "
                 "pass --audio-root so the files can be found by name.")
    idx = index_audio_root(audio_root)
    return [idx.get(str(f), "") for f in df["filename"]]


# --------------------------------------------------------------------------
# Main loop
# --------------------------------------------------------------------------
def load_audio(path):
    import librosa
    wav, _ = librosa.load(path, sr=SR, mono=True)
    if len(wav) < MIN_SAMPLES:
        wav = np.pad(wav, (0, MIN_SAMPLES - len(wav)))
    return wav.astype(np.float32)


def save_npz(path, emb, filenames, sources):
    tmp = str(path) + ".tmp.npz"
    np.savez(tmp, X=np.asarray(emb, dtype=np.float32).reshape(-1, 2048),
             filename=np.asarray(filenames, dtype=str), source=np.asarray(sources, dtype=str))
    os.replace(tmp, path)


def same_path(a, b):
    return os.path.normcase(os.path.normpath(a)) == os.path.normcase(os.path.normpath(b))


def check_paths(rows):
    """rows: (filename, source, path, where). Exit if one key points at two files."""
    first = {}
    clashes = []
    for fn, src, p, where in rows:
        key = (fn, src)
        if key not in first:
            first[key] = (p, where)
        elif not same_path(first[key][0], p):
            clashes.append((key, first[key], (p, where)))
    if clashes:
        lines = [f"  {fn!r} source {src!r}:\n    {p1}  ({w1})\n    {p2}  ({w2})"
                 for (fn, src), (p1, w1), (p2, w2) in clashes[:10]]
        sys.exit(f"{len(clashes)} filename + source pair(s) point at two different files:\n"
                 + "\n".join(lines))


def read_split(split, manifest, args):
    df = pd.read_csv(manifest, dtype=str, keep_default_na=False)
    for c in ("filename", "source"):
        if c not in df.columns:
            sys.exit(f"{manifest} has no '{c}' column. Columns: {list(df.columns)}")
    n_total = len(df)
    if "exists" in df.columns:
        df = df[df["exists"].map(truthy)]
    df = df.reset_index(drop=True)
    if args.limit:
        df = df.head(args.limit)
    print(f"[{split}] {manifest}: {n_total:,} rows, {len(df):,} to embed")
    df["filename"] = df["filename"].str.strip()
    df["source"] = df["source"].str.strip()
    df["resolved_path"] = resolve_paths(df, manifest, args.audio_root)
    return df, n_total


def extract(manifests, model, device, args):
    import torch
    single = Path(args.manifests).is_file()
    frames = {}
    for split, manifest in manifests.items():
        frames[split] = read_split(split, manifest, args)
    check_paths((r.filename, r.source, r.resolved_path, f"{split} manifest")
                for split, (df, _) in frames.items() for r in df.itertuples())

    out_dir = Path(args.out)
    final = out_dir / "embeddings.npz"
    partial = out_dir / "embeddings.partial.npz"
    emb, filenames, sources = [], [], []
    if partial.exists() and not args.restart:
        with np.load(partial) as z:
            emb = list(z["X"])
            filenames = [str(v) for v in z["filename"]]
            sources = [str(v) for v in z["source"]]
        print(f"resuming: {len(emb):,} already done")
    done = set(zip(filenames, sources))
    errors = {}                                       # (filename, source) -> (path, error)

    todo = {}
    for df, _ in frames.values():
        for r in df.itertuples():
            key = (r.filename, r.source)
            if key not in done:
                todo.setdefault(key, r.resolved_path)
    print(f"\n{len(done) + len(todo):,} unique clips, {len(todo):,} left to embed")

    t0, n_new = time.time(), 0
    for key, p in todo.items():
        if not p or not os.path.exists(p):
            errors[key] = (p, "file not found")
            continue
        try:
            wav = load_audio(p)
            with torch.inference_mode():
                e = model(torch.from_numpy(wav)[None].to(device))[0].cpu().numpy()
        except Exception as ex:                       # corrupt file, decoder error
            errors[key] = (p, f"{type(ex).__name__}: {ex}"[:200])
            continue
        emb.append(e)
        filenames.append(key[0])
        sources.append(key[1])
        n_new += 1
        if n_new % 200 == 0:
            rate = n_new / (time.time() - t0)
            left = (len(todo) - n_new - len(errors)) / max(rate, 1e-9)
            print(f"  {n_new:,}/{len(todo):,}  {rate:.1f} clips/s  ~{left/60:.0f} min left", flush=True)
        if n_new % args.save_every == 0:
            save_npz(partial, emb, filenames, sources)

    save_npz(final, emb, filenames, sources)
    if partial.exists():
        partial.unlink()
    have = set(zip(filenames, sources))
    stats = {}
    for split, (df, n_total) in frames.items():
        failed = [(r.filename, r.source, *errors[(r.filename, r.source)])
                  for r in df.itertuples() if (r.filename, r.source) in errors]
        failed_name = f"{split}_failed.csv" if single else f"envsound_{split}_failed.csv"
        pd.DataFrame(failed, columns=["filename", "source", "path", "error"]).to_csv(
            out_dir / failed_name, index=False)
        embedded = sum((r.filename, r.source) in have for r in df.itertuples())
        stats[split] = {"rows": n_total, "embedded": embedded, "failed": len(failed)}
        print(f"  [{split}] embedded {embedded:,}  failed {len(failed):,}")
    X = np.asarray(emb, dtype=np.float32)
    print(f"saved {final}  X={tuple(np.load(final)['X'].shape)}  failed={len(errors)}")
    if errors and not len(X):
        (fn, _), (p, err) = next(iter(errors.items()))
        print(f"  first error: {err}  ({fn} -> {p})")
    if len(X) and (np.abs(X).sum(1) == 0).mean() > 0.01:
        print("  WARNING: >1% all-zero embeddings, check the audio (silent or unreadable files?)")
    return stats


def combine(src_dir, splits, out_dir):
    """Stack old envsound_{split}.npz files into one embeddings.npz. No CNN14."""
    src_dir = Path(src_dir)
    emb, filenames, sources, rows = [], [], [], []
    seen = set()
    stats = {}
    for split in splits:
        path = src_dir / f"envsound_{split}.npz"
        if not path.is_file():
            sys.exit(f"Missing {path}")
        with np.load(path, allow_pickle=True) as z:
            X = np.asarray(z["X"], dtype=np.float32)
            fns = [str(v).strip() for v in z["filename"]]
            srcs = [str(v).strip() for v in z["source"]] if "source" in z.files else [""] * len(fns)
            paths = [str(v).strip() for v in z["path"]] if "path" in z.files else [""] * len(fns)
        if X.ndim != 2 or X.shape[1] != 2048 or len(X) != len(fns):
            sys.exit(f"{path}: X shape {tuple(X.shape)} for {len(fns)} filenames")
        added = 0
        for i, (fn, src, p) in enumerate(zip(fns, srcs, paths)):
            rows.append((fn, src, p, f"{path.name} row {i}"))
            if (fn, src) in seen:
                continue
            seen.add((fn, src))
            emb.append(X[i])
            filenames.append(fn)
            sources.append(src)
            added += 1
        stats[split] = {"rows": len(fns), "added": added}
        print(f"  {path.name}: {len(fns):,} rows, {added:,} new")
    check_paths(rows)
    final = Path(out_dir) / "embeddings.npz"
    save_npz(final, emb, filenames, sources)
    print(f"saved {final}  X={(len(emb), 2048)}")
    return stats


def write_info(out_dir, extra):
    info = {
        "model": "PANNs CNN14, AudioSet-pretrained, frozen (" + CKPT_NAME + ")",
        "embedding": "2048-d output of fc1 (after ReLU), eval mode, no dropout",
        "sample_rate": SR, "mono": True, "resampler": "librosa.load default",
        "clip_handling": "whole clip in one pass; clips < 1 s zero-padded to 1 s",
        "cache": "embeddings.npz: X, filename, source; one row per filename + source",
        "not_stored": "Category, SubCategory, and path; labels are joined from the manifests at training time",
    }
    info.update(extra)
    with open(Path(out_dir) / "extract_info.json", "w") as f:
        json.dump(info, f, indent=2)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifests", default="",
                    help="folder with envsound_{train,val,test}.csv, or one CSV")
    ap.add_argument("--combine-from", default="",
                    help="folder of old envsound_{split}.npz files to stack into <out>/embeddings.npz "
                         "(no CNN14, no manifests)")
    ap.add_argument("--audio-root", default="",
                    help="base folder for relative paths (default: parent of the manifests folder, "
                         "or the parent of a single CSV's folder), or the tree to search by filename")
    ap.add_argument("--out", default="embeddings/panns_cnn14")
    ap.add_argument("--checkpoint", default=str(DEFAULT_CKPT),
                    help="CNN14 weights (default: models/Cnn14_mAP=0.431.pth next to this script)")
    ap.add_argument("--splits", default="train,val,test",
                    help="comma-separated manifests to embed when --manifests is a folder")
    ap.add_argument("--device", default="auto", help="auto, cuda or cpu")
    ap.add_argument("--limit", type=int, default=0, help="only the first N rows per split (smoke test)")
    ap.add_argument("--save-every", type=int, default=2000)
    ap.add_argument("--restart", action="store_true", help="ignore saved progress")
    args = ap.parse_args()
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    if args.combine_from:
        if args.limit:
            sys.exit("--limit does not apply to --combine-from")
        Path(args.out).mkdir(parents=True, exist_ok=True)
        print(f"Combining {args.combine_from} -> {args.out}")
        old_info = Path(args.combine_from) / "extract_info.json"
        old = json.loads(old_info.read_text()) if old_info.is_file() else None
        stats = combine(args.combine_from, splits, args.out)
        write_info(args.out, {"combined_from": str(Path(args.combine_from).resolve()),
                              "splits": stats, "original_extract_info": old})
        print("\nDone.", json.dumps(stats))
        return
    if not args.manifests:
        sys.exit("pass --manifests (or --combine-from to stack old per-split caches)")

    import torch
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if args.limit:
        args.out = str(Path(args.out) / f"_smoke_{args.limit}")
    Path(args.out).mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}" + ("" if device == "cuda" else "  (no GPU: expect several hours for ~58k clips)"))
    model = load_model(args.checkpoint, device)

    stats = extract(resolve_manifests(args.manifests, splits), model, device, args)
    write_info(args.out, {"splits": stats})
    print("\nDone.", json.dumps(stats))


if __name__ == "__main__":
    main()
