"""
Step 1 of the Beck & Lerch (2026) benchmark: frozen PANNs CNN14 embeddings.

The paper does not fine-tune PANNs. It runs every clip through the pretrained
CNN14 (AudioSet, Cnn14_mAP=0.431), keeps the 2048-d embedding, and trains a
dropout(0.3) + linear classifier on those vectors. This script does the first
half: one embedding per clip, saved per split.

Input : manifests/envsound_{train,val,test}.csv  (your manifests)
Output: <out>/envsound_{split}.npz with
          X           float32 (N, 2048)
          filename, Category, SubCategory, source, path   (str arrays)
        <out>/envsound_{split}_failed.csv   clips that could not be read
        <out>/extract_info.json             settings used (for your methods section)

The checkpoint defaults to models/Cnn14_mAP=0.431.pth next to this script.
If that file is missing or too small, it is downloaded there (~330 MB).

Usage (Windows, from the folder that holds this script):
  python extract_panns_embeddings.py --manifests "E:\\Sounds for TA\\Extracted\\manifests" --limit 50
  python extract_panns_embeddings.py --manifests "E:\\Sounds for TA\\Extracted\\manifests"

Paths: if the manifests have a path column (path / filepath / audio_path),
it is used; relative paths are resolved against --audio-root (or the manifest
folder). Without a path column, --audio-root is searched for each filename.
Rows with exists = False/0 are skipped.

Resumable: progress is saved every --save-every clips; just run it again.

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


def resolve_paths(df, manifest_path, audio_root):
    col = next((c for c in PATH_COLUMNS if c in df.columns), None)
    base = Path(audio_root) if audio_root else manifest_path.parent
    if col:
        out = []
        for p in df[col].astype(str):
            p = p.strip().replace("\\", os.sep) if os.sep == "/" else p.strip()
            out.append(p if os.path.isabs(p) else str(base / p))
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


def save_npz(path, emb, meta):
    tmp = str(path) + ".tmp.npz"
    np.savez(tmp, X=np.asarray(emb, dtype=np.float32).reshape(-1, 2048),
             **{k: np.asarray(v, dtype=str) for k, v in meta.items()})
    os.replace(tmp, path)


def run_split(split, model, device, args):
    import torch
    manifest = find_manifest(args.manifests, split)
    df = pd.read_csv(manifest, dtype=str, keep_default_na=False)
    for c in ("filename", "Category"):
        if c not in df.columns:
            sys.exit(f"{manifest} has no '{c}' column. Columns: {list(df.columns)}")
    n_total = len(df)
    if "exists" in df.columns:
        df = df[df["exists"].map(truthy)]
    df = df.reset_index(drop=True)
    if args.limit:
        df = df.head(args.limit)
    paths = resolve_paths(df, manifest, args.audio_root)
    print(f"\n[{split}] {manifest}: {n_total:,} rows, {len(df):,} to embed")

    out_dir = Path(args.out)
    final = out_dir / f"envsound_{split}.npz"
    partial = out_dir / f"envsound_{split}.partial.npz"
    keys = ["filename", "Category", "SubCategory", "source", "path"]
    meta = {k: [] for k in keys}
    emb = []
    if partial.exists() and not args.restart:
        z = np.load(partial)
        emb = list(z["X"])
        meta = {k: list(z[k]) for k in keys}
        print(f"  resuming: {len(emb):,} already done")
    done = set(meta["filename"])
    failed = []

    t0, n_new = time.time(), 0
    for i, row in df.iterrows():
        fn = row["filename"]
        if fn in done:
            continue
        p = paths[i]
        if not p or not os.path.exists(p):
            failed.append((fn, p, "file not found"))
            continue
        try:
            wav = load_audio(p)
            with torch.inference_mode():
                e = model(torch.from_numpy(wav)[None].to(device))[0].cpu().numpy()
        except Exception as ex:                       # corrupt file, decoder error
            failed.append((fn, p, f"{type(ex).__name__}: {ex}"[:200]))
            continue
        emb.append(e)
        meta["filename"].append(fn)
        meta["Category"].append(row["Category"])
        meta["SubCategory"].append(row.get("SubCategory", ""))
        meta["source"].append(row.get("source", ""))
        meta["path"].append(p)
        n_new += 1
        if n_new % 200 == 0:
            rate = n_new / (time.time() - t0)
            left = (len(df) - len(emb) - len(failed)) / max(rate, 1e-9)
            print(f"  {len(emb):,}/{len(df):,}  {rate:.1f} clips/s  ~{left/60:.0f} min left", flush=True)
        if n_new % args.save_every == 0:
            save_npz(partial, emb, meta)

    save_npz(final, emb, meta)
    if partial.exists():
        partial.unlink()
    pd.DataFrame(failed, columns=["filename", "path", "error"]).to_csv(
        out_dir / f"envsound_{split}_failed.csv", index=False)
    X = np.asarray(emb, dtype=np.float32)
    print(f"  saved {final}  X={X.shape}  failed={len(failed)}")
    if len(X) and (np.abs(X).sum(1) == 0).mean() > 0.01:
        print("  WARNING: >1% all-zero embeddings, check the audio (silent or unreadable files?)")
    return {"rows": n_total, "embedded": len(X), "failed": len(failed)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifests", required=True, help="folder with envsound_{train,val,test}.csv")
    ap.add_argument("--audio-root", default="", help="base folder for relative paths, or to search by filename")
    ap.add_argument("--out", default="embeddings/panns_cnn14")
    ap.add_argument("--checkpoint", default=str(DEFAULT_CKPT),
                    help="CNN14 weights (default: models/Cnn14_mAP=0.431.pth next to this script)")
    ap.add_argument("--splits", default="train,val,test")
    ap.add_argument("--device", default="auto", help="auto, cuda or cpu")
    ap.add_argument("--limit", type=int, default=0, help="only the first N rows per split (smoke test)")
    ap.add_argument("--save-every", type=int, default=2000)
    ap.add_argument("--restart", action="store_true", help="ignore saved progress")
    args = ap.parse_args()

    import torch
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if args.limit:
        args.out = str(Path(args.out) / f"_smoke_{args.limit}")
    Path(args.out).mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}" + ("" if device == "cuda" else "  (no GPU: expect several hours for ~58k clips)"))
    model = load_model(args.checkpoint, device)

    stats = {s: run_split(s, model, device, args) for s in args.splits.split(",")}
    info = {
        "model": "PANNs CNN14, AudioSet-pretrained, frozen (" + CKPT_NAME + ")",
        "embedding": "2048-d output of fc1 (after ReLU), eval mode, no dropout",
        "sample_rate": SR, "mono": True, "resampler": "librosa.load default",
        "clip_handling": "whole clip in one pass; clips < 1 s zero-padded to 1 s",
        "splits": stats,
    }
    with open(Path(args.out) / "extract_info.json", "w") as f:
        json.dump(info, f, indent=2)
    print("\nDone.", json.dumps(stats))


if __name__ == "__main__":
    main()
