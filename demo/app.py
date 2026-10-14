"""
Local demo for the UCS tagger.

One page, on this machine. Drop a clip, or play one from the EnvSound test
split. The clip goes through frozen CNN14, then a saved linear probe. The
default probe is the subcategory head, seed 42.

    python demo/app.py

Open the URL it prints. Uses the packages already listed in requirements.txt.
"""
import argparse
import json
import os
import random
import shutil
import sys
import tempfile
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import numpy as np
import torch

DEMO = Path(__file__).resolve().parent
ROOT = DEMO.parent
sys.path.insert(0, str(ROOT))

from extract_panns_embeddings import AUDIO_EXT, DEFAULT_CKPT, MIN_SAMPLES, SR, load_model
from train_linear_probe import EMPTY, LinearProbe

DEFAULT_PROBE = ROOT / "results" / "panns_cnn14_linear" / "subcategory" / "seed_42" / "probe.pt"
DEFAULT_LIBRARY = ROOT / "embeddings" / "panns_cnn14" / "envsound_test.npz"
MAX_SECONDS = 30
MAX_BYTES = 30 * 1024 * 1024
PAGE = DEMO / "index.html"


class AudioError(Exception):
    def __init__(self, message, status=400):
        super().__init__(message)
        self.status = status


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--probe", default=str(DEFAULT_PROBE))
    ap.add_argument("--checkpoint", default=str(DEFAULT_CKPT))
    ap.add_argument("--library", default=str(DEFAULT_LIBRARY),
                    help="test-split .npz used by Play a test clip (default: embeddings/panns_cnn14/envsound_test.npz)")
    ap.add_argument("--device", default="auto", help="auto, cuda, or cpu")
    ap.add_argument("--top", type=int, default=5)
    return ap.parse_args(argv)


def blank(value):
    return str(value).strip().lower() in EMPTY


def load_probe(path, device):
    path = Path(path)
    if not path.is_file():
        sys.exit(f"No probe at {path}. Train the linear probe first, or pass --probe.")
    blob = torch.load(path, map_location="cpu", weights_only=False)
    classes = [str(name) for name in blob["classes"]]
    parent_of = [str(name) for name in blob["parent_of"]]
    if not classes:
        sys.exit(f"{path} has no classes.")
    if len(parent_of) != len(classes):
        sys.exit(f"{path} parent map has {len(parent_of)} entries for {len(classes)} classes.")
    probe = LinearProbe(len(classes), 0.0)
    probe.load_state_dict(blob["state_dict"])
    probe.to(device).eval()
    seed = blob.get("seed")
    return {
        "probe": probe,
        "classes": classes,
        "parent_of": parent_of,
        "task": str(blob.get("task") or "subcategory"),
        "seed": None if seed is None else int(seed),
    }


def load_library(path):
    path = Path(path)
    if not path.is_file():
        print(f"No test-split cache at {path}. Uploads still work.", flush=True)
        return []
    with np.load(path, allow_pickle=True) as z:
        filename = [str(v).strip() for v in z["filename"]]
        paths = [str(v).strip() for v in z["path"]]
        category = [str(v).strip() for v in z["Category"]]
        subcategory = ["" if blank(v) else str(v).strip() for v in z["SubCategory"]]
        source = ["" if blank(v) else str(v).strip() for v in z["source"]]
    rows = []
    for i, name in enumerate(filename):
        rows.append({
            "id": i,
            "filename": name,
            "path": paths[i],
            "category": category[i],
            "subcategory": subcategory[i],
            "source": source[i],
        })
    print(f"Test clips: {len(rows):,}", flush=True)
    return rows


def boot(args):
    if args.top < 1:
        sys.exit("--top must be at least 1.")
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        sys.exit("CUDA was requested, but torch.cuda.is_available() is False.")
    print(f"Device: {device}", flush=True)
    encoder = load_model(args.checkpoint, device)
    probe = load_probe(args.probe, device)
    print(
        f"Probe: {probe['task']}, seed {probe['seed']}, {len(probe['classes'])} classes",
        flush=True,
    )
    try:
        library = load_library(args.library)
    except Exception as ex:
        print(f"Could not read {args.library}: {ex}", flush=True)
        library = []
    probe.update({
        "encoder": encoder,
        "device": device,
        "library": library,
        "top": args.top,
        "lock": threading.Lock(),
    })
    return probe


def read_clip(path):
    import librosa
    try:
        wav, _ = librosa.load(path, sr=SR, mono=True, duration=MAX_SECONDS)
    except Exception as ex:
        raise AudioError(f"Could not read this file. {type(ex).__name__}: {ex}"[:200]) from ex
    if len(wav) == 0:
        raise AudioError("This file has no audio samples.")
    seconds = len(wav) / SR
    truncated = len(wav) >= int(SR * MAX_SECONDS) - 1
    try:
        import soundfile as sf
        duration = float(sf.info(path).duration)
        seconds = duration
        truncated = duration > MAX_SECONDS + 0.05
    except Exception:
        pass
    if len(wav) < MIN_SAMPLES:
        wav = np.pad(wav, (0, MIN_SAMPLES - len(wav)))
    return np.ascontiguousarray(wav, dtype=np.float32), round(float(seconds), 2), bool(truncated)


def as_tag(task, label, parent):
    if task == "pair" and "||" in label:
        category, subcategory = label.split("||", 1)
        return category, subcategory
    if task == "category":
        return label, ""
    return parent, label


def top_rows(state, probs):
    k = min(state["top"], len(probs))
    order = np.argpartition(-probs, k - 1)[:k]
    order = order[np.argsort(-probs[order])]
    rows = []
    for index in order:
        index = int(index)
        category, subcategory = as_tag(state["task"], state["classes"][index], state["parent_of"][index])
        rows.append({
            "category": category,
            "subcategory": subcategory,
            "prob": float(probs[index]),
        })
    return rows


def predict_wav(state, wav):
    with state["lock"]:
        with torch.inference_mode():
            emb = state["encoder"](torch.from_numpy(wav)[None].to(state["device"]))
            logits = state["probe"](emb)
            probs = torch.softmax(logits, dim=-1)[0].detach().cpu().numpy()
    if not np.isfinite(probs).all():
        raise AudioError("The model returned a non-finite score.", status=500)
    return top_rows(state, probs)


def predict_path(state, path):
    wav, seconds, truncated = read_clip(path)
    return predict_wav(state, wav), seconds, truncated


def result_payload(name, rows, seconds, truncated, task, labelled=None, source="", clip_id=None):
    payload = {
        "name": name,
        "seconds": seconds,
        "truncated": truncated,
        "task": task,
        "predictions": rows,
    }
    if labelled is not None:
        payload["labelled"] = labelled
    if source:
        payload["source"] = source
    if clip_id is not None:
        payload["id"] = clip_id
    return payload


def pick_clip(state):
    rows = state["library"]
    if not rows:
        raise AudioError("No test clips are indexed.", status=404)
    fallback = None
    for _ in range(30):
        row = rows[random.randrange(len(rows))]
        if not os.path.isfile(row["path"]):
            continue
        if row["subcategory"]:
            return row
        if fallback is None:
            fallback = row
    if fallback is not None:
        return fallback
    raise AudioError("No test clip was found on disk.", status=404)


def score_row(state, row):
    rows, seconds, truncated = predict_path(state, row["path"])
    return result_payload(
        row["filename"],
        rows,
        seconds,
        truncated,
        state["task"],
        labelled={"category": row["category"], "subcategory": row["subcategory"]},
        source=row["source"],
        clip_id=row["id"],
    )


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        path = urlparse(self.path).path
        try:
            if path == "/":
                self.send_bytes(200, "text/html; charset=utf-8", PAGE.read_bytes())
            elif path == "/favicon.ico":
                self.send_response(204)
                self.end_headers()
            elif path == "/api/info":
                state = self.server.state
                self.send_json(200, {
                    "task": state["task"],
                    "seed": state["seed"],
                    "n_classes": len(state["classes"]),
                    "device": state["device"],
                    "library": len(state["library"]),
                    "max_seconds": MAX_SECONDS,
                })
            elif path == "/api/random":
                self.send_json(200, score_row(self.server.state, pick_clip(self.server.state)))
            elif path.startswith("/api/clip/"):
                self.serve_clip(path[len("/api/clip/"):])
            else:
                self.send_json(404, {"error": "Not found."})
        except AudioError as ex:
            self.send_json(ex.status, {"error": str(ex)})
        except Exception as ex:
            traceback.print_exc()
            self.send_json(500, {"error": f"{type(ex).__name__}: {ex}"[:200]})

    def do_POST(self):
        if urlparse(self.path).path != "/api/predict":
            self.send_json(404, {"error": "Not found."})
            return
        try:
            self.predict_upload()
        except AudioError as ex:
            self.send_json(ex.status, {"error": str(ex)})
        except Exception as ex:
            traceback.print_exc()
            self.send_json(500, {"error": f"{type(ex).__name__}: {ex}"[:200]})

    def predict_upload(self):
        length = self.headers.get("Content-Length")
        if length is None:
            raise AudioError("Missing Content-Length.", status=411)
        try:
            size = int(length)
        except ValueError:
            raise AudioError("Missing Content-Length.", status=411) from None
        if size <= 0:
            raise AudioError("The upload was empty.")
        if size > MAX_BYTES:
            self.close_connection = True
            raise AudioError("That file is over 30 MB.", status=413)
        name = upload_name(self.path)
        suffix = Path(name).suffix.lower()
        if suffix not in AUDIO_EXT:
            raise AudioError("Use a wav, flac, ogg, mp3, m4a, aiff, or aif file.")
        body = self.rfile.read(size)
        fd, tmp = tempfile.mkstemp(suffix=suffix)
        os.close(fd)
        try:
            Path(tmp).write_bytes(body)
            rows, seconds, truncated = predict_path(self.server.state, tmp)
        finally:
            Path(tmp).unlink(missing_ok=True)
        self.send_json(200, result_payload(name, rows, seconds, truncated, self.server.state["task"]))

    def serve_clip(self, raw_id):
        if not raw_id.isdigit():
            self.send_json(404, {"error": "Not found."})
            return
        state = self.server.state
        clip_id = int(raw_id)
        if clip_id >= len(state["library"]):
            self.send_json(404, {"error": "Not found."})
            return
        path = Path(state["library"][clip_id]["path"])
        if not path.is_file():
            self.send_json(404, {"error": "This clip is not on disk."})
            return
        mime = {
            ".wav": "audio/wav",
            ".flac": "audio/flac",
            ".ogg": "audio/ogg",
            ".mp3": "audio/mpeg",
            ".m4a": "audio/mp4",
            ".aiff": "audio/aiff",
            ".aif": "audio/aiff",
        }.get(path.suffix.lower(), "application/octet-stream")
        try:
            self.send_response(200)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(path.stat().st_size))
            self.end_headers()
            with path.open("rb") as handle:
                shutil.copyfileobj(handle, self.wfile, length=256 * 1024)
        except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
            pass

    def send_json(self, status, payload):
        self.send_bytes(status, "application/json; charset=utf-8", json.dumps(payload).encode("utf-8"))

    def send_bytes(self, status, content_type, body):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        print(f"{self.address_string()}  {fmt % args}", flush=True)


def upload_name(url):
    query = parse_qs(urlparse(url).query)
    raw = unquote(query.get("name", ["clip.wav"])[0])[:180]
    name = Path(raw).name
    return name or "clip.wav"


class Server(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True


def serve(state, host, port):
    httpd = Server((host, port), Handler)
    httpd.state = state
    print(f"Open http://{host}:{port}", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.", flush=True)
    finally:
        httpd.server_close()


if __name__ == "__main__":
    args = parse_args()
    serve(boot(args), args.host, args.port)
