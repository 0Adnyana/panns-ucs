"""
Local demo for the UCS tagger.

One page, on this machine. Drop a clip, or play one from the EnvSound test
split. The clip goes through frozen CNN14, then a saved linear probe.

The page lists every probe saved under results/. Choosing one loads that
checkpoint and the test clips from the embeddings folder recorded in that
run's summary.json. The default is the subcategory head, seed 42.

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

DEFAULT_RESULTS = ROOT / "results"
DEFAULT_EMBEDDINGS = ROOT / "embeddings" / "panns_cnn14"
DEFAULT_PROBE = DEFAULT_RESULTS / "panns_cnn14_linear" / "subcategory" / "seed_42" / "probe.pt"
MAX_SECONDS = 30
MAX_BYTES = 30 * 1024 * 1024
PAGE = DEMO / "index.html"
TASK_RANK = {"subcategory": 0, "pair": 1, "category": 2}
_warned_embeddings = set()


class AudioError(Exception):
    def __init__(self, message, status=400):
        super().__init__(message)
        self.status = status


class ProbeError(Exception):
    pass


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--results", default=str(DEFAULT_RESULTS),
                    help="folder to scan for saved probes (default: results/)")
    ap.add_argument("--probe", default=None,
                    help="checkpoint to load first (default: subcategory seed 42, else the first saved probe)")
    ap.add_argument("--checkpoint", default=str(DEFAULT_CKPT))
    ap.add_argument("--library", default=None,
                    help="test-split .npz for the first checkpoint (default: envsound_test.npz in that run's embeddings folder)")
    ap.add_argument("--device", default="auto", help="auto, cuda, or cpu")
    ap.add_argument("--top", type=int, default=5)
    return ap.parse_args(argv)


def display_path(path):
    path = Path(path).resolve()
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def parse_seed(name):
    if not name.startswith("seed_"):
        return None
    try:
        return int(name[5:])
    except ValueError:
        return None


def embeddings_from_task_dir(task_dir):
    """Embeddings folder recorded for this training run.

    train_linear_probe.py writes it to summary.json under settings.embeddings.
    A seed folder can exist before that file does; those fall back to the
    default cache.
    """
    task_dir = Path(task_dir)
    summary_path = task_dir / "summary.json"
    raw = ""
    if summary_path.is_file():
        try:
            with open(summary_path, encoding="utf-8") as handle:
                summary = json.load(handle)
            raw = str((summary.get("settings") or {}).get("embeddings") or "").strip()
        except (OSError, json.JSONDecodeError, AttributeError):
            raw = ""
    if not raw:
        key = str(task_dir.resolve())
        if key not in _warned_embeddings:
            _warned_embeddings.add(key)
            print(
                f"No embeddings path in {summary_path}. Using {display_path(DEFAULT_EMBEDDINGS)}.",
                flush=True,
            )
        return DEFAULT_EMBEDDINGS.resolve()
    path = Path(raw)
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def seed_f1(seed_dir):
    path = Path(seed_dir) / "metrics.json"
    if not path.is_file():
        return None
    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
        return float(data["test"]["macro_f1"])
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None


def make_entry(probe_path, results_root):
    probe_path = Path(probe_path).resolve()
    results_root = Path(results_root).resolve()
    seed_dir = probe_path.parent
    task_dir = seed_dir.parent
    embeddings_dir = embeddings_from_task_dir(task_dir)
    try:
        ident = probe_path.relative_to(results_root).as_posix()
        parts = task_dir.relative_to(results_root).parts
        if len(parts) >= 2:
            run = Path(*parts[:-1]).as_posix()
            task = parts[-1]
        elif parts:
            run = parts[0]
            task = parts[0]
        else:
            run = task_dir.name
            task = task_dir.name
    except ValueError:
        ident = probe_path.as_posix()
        run = display_path(task_dir.parent)
        task = task_dir.name
    return {
        "id": ident,
        "run": run,
        "task": task,
        "seed": parse_seed(seed_dir.name),
        "f1": seed_f1(seed_dir),
        "probe": probe_path,
        "embeddings_dir": embeddings_dir,
        "library": (embeddings_dir / "envsound_test.npz").resolve(),
        "embeddings": display_path(embeddings_dir),
    }


def sort_key(entry):
    seed = entry["seed"]
    if seed is None:
        seed = 10 ** 9
    return (
        entry["embeddings"],
        entry["run"],
        TASK_RANK.get(entry["task"], 9),
        entry["task"],
        seed,
        entry["id"],
    )


def discover_checkpoints(results_root):
    results_root = Path(results_root).resolve()
    entries = []
    if not results_root.is_dir():
        return entries
    for probe in results_root.rglob("probe.pt"):
        if parse_seed(probe.parent.name) is None:
            continue
        entries.append(make_entry(probe, results_root))
    entries.sort(key=sort_key)
    return entries


def choose_initial(entries, probe_arg, results_root):
    """Pick the checkpoint the page loads first. Returns (entry, is_extra)."""
    if probe_arg:
        path = Path(probe_arg).resolve()
        if not path.is_file():
            sys.exit(f"No probe at {path}. Train the linear probe first, or pass --probe.")
        for entry in entries:
            if entry["probe"] == path:
                return entry, False
        return make_entry(path, results_root), True
    default = DEFAULT_PROBE.resolve()
    for entry in entries:
        if entry["probe"] == default:
            return entry, False
    preferred = [entry for entry in entries if entry["task"] == "subcategory" and entry["seed"] == 42]
    if preferred:
        return preferred[0], False
    if entries:
        return entries[0], False
    sys.exit(
        f"No saved probe under {results_root}. Train the linear probe first, or pass --probe."
    )


def point_library(entry, library_path):
    library_path = Path(library_path).resolve()
    entry["library"] = library_path
    entry["embeddings_dir"] = library_path.parent
    entry["embeddings"] = display_path(library_path.parent)
    return entry


def entry_public(entry):
    return {
        "id": entry["id"],
        "run": entry["run"],
        "task": entry["task"],
        "seed": entry["seed"],
        "f1": entry["f1"],
        "embeddings": entry["embeddings"],
    }


def blank(value):
    return str(value).strip().lower() in EMPTY


def load_probe(path, device):
    path = Path(path)
    if not path.is_file():
        raise ProbeError(f"No probe at {path}. Train the linear probe first, or pass --probe.")
    try:
        blob = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as ex:
        raise ProbeError(f"Could not read {path}. {type(ex).__name__}: {ex}") from ex
    for key in ("state_dict", "classes", "parent_of"):
        if key not in blob:
            raise ProbeError(f"{path} is missing {key}.")
    classes = [str(name) for name in blob["classes"]]
    parent_of = [str(name) for name in blob["parent_of"]]
    if not classes:
        raise ProbeError(f"{path} has no classes.")
    if len(parent_of) != len(classes):
        raise ProbeError(f"{path} parent map has {len(parent_of)} entries for {len(classes)} classes.")
    probe = LinearProbe(len(classes), 0.0)
    try:
        probe.load_state_dict(blob["state_dict"])
    except Exception as ex:
        raise ProbeError(f"Could not load {path}. {type(ex).__name__}: {ex}") from ex
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


def configure(httpd, encoder, device, top, results_root, extras=None, library_override=None):
    httpd.encoder = encoder
    httpd.device = device
    httpd.top = top
    httpd.results_root = Path(results_root).resolve()
    httpd.extras = list(extras or [])
    httpd.library_override = library_override
    httpd.libraries = {}
    httpd.infer_lock = threading.Lock()
    httpd.cache_lock = threading.Lock()
    httpd.switch_lock = threading.Lock()
    httpd.state = None


def boot(args):
    if args.top < 1:
        sys.exit("--top must be at least 1.")
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        sys.exit("CUDA was requested, but torch.cuda.is_available() is False.")
    if device not in ("cuda", "cpu"):
        sys.exit(f"unknown device {device!r}")
    print(f"Device: {device}", flush=True)
    results_root = Path(args.results).resolve()
    entries = discover_checkpoints(results_root)
    entry, is_extra = choose_initial(entries, args.probe, results_root)
    extras = [entry] if is_extra else []
    library_override = None
    if args.library and str(args.library).strip():
        library_override = (entry["probe"], Path(args.library).resolve())
        point_library(entry, library_override[1])
    encoder = load_model(args.checkpoint, device)
    httpd = Server((args.host, args.port), Handler)
    configure(httpd, encoder, device, args.top, results_root, extras, library_override)
    print(f"Checkpoints: {len(httpd.catalog())}", flush=True)
    try:
        httpd.activate(entry)
    except ProbeError as ex:
        httpd.server_close()
        sys.exit(str(ex))
    return httpd


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
                self.send_json(200, self.server.info())
            elif path == "/api/random":
                state = self.server.state
                self.send_json(200, score_row(state, pick_clip(state)))
            elif path.startswith("/api/clip/"):
                self.serve_clip(path[len("/api/clip/"):])
            else:
                self.send_json(404, {"error": "Not found."})
        except AudioError as ex:
            self.send_json(ex.status, {"error": str(ex)})
        except ProbeError as ex:
            self.send_json(500, {"error": str(ex)[:200]})
        except Exception as ex:
            traceback.print_exc()
            self.send_json(500, {"error": f"{type(ex).__name__}: {ex}"[:200]})

    def do_POST(self):
        path = urlparse(self.path).path
        try:
            if path == "/api/predict":
                self.predict_upload()
            elif path == "/api/checkpoint":
                self.switch_checkpoint()
            else:
                self.send_json(404, {"error": "Not found."})
        except AudioError as ex:
            self.send_json(ex.status, {"error": str(ex)})
        except ProbeError as ex:
            self.send_json(500, {"error": str(ex)[:200]})
        except Exception as ex:
            traceback.print_exc()
            self.send_json(500, {"error": f"{type(ex).__name__}: {ex}"[:200]})

    def switch_checkpoint(self):
        payload = self.read_json()
        ident = str(payload.get("id") or "").strip()
        if not ident:
            raise AudioError("Missing checkpoint.")
        entries = self.server.catalog()
        entry = next((row for row in entries if row["id"] == ident), None)
        if entry is None:
            raise AudioError("That checkpoint is not in the results folder.", status=404)
        with self.server.switch_lock:
            if self.server.state is not None and self.server.state["id"] == ident:
                payload = self.server.info()
            else:
                self.server.activate(entry)
                payload = self.server.info()
        self.send_json(200, payload)

    def read_json(self):
        length = self.headers.get("Content-Length")
        if length is None:
            raise AudioError("Missing Content-Length.", status=411)
        try:
            size = int(length)
        except ValueError:
            raise AudioError("Missing Content-Length.", status=411) from None
        if size <= 0:
            raise AudioError("The request was empty.")
        if size > 8192:
            raise AudioError("The request was too large.", status=413)
        try:
            payload = json.loads(self.rfile.read(size).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise AudioError("The request was not valid JSON.") from None
        if not isinstance(payload, dict):
            raise AudioError("The request was not a JSON object.")
        return payload

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
        state = self.server.state
        try:
            Path(tmp).write_bytes(body)
            rows, seconds, truncated = predict_path(state, tmp)
        finally:
            Path(tmp).unlink(missing_ok=True)
        self.send_json(200, result_payload(name, rows, seconds, truncated, state["task"]))

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
        self.send_header("Cache-Control", "no-store")
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

    def catalog(self):
        entries = discover_checkpoints(self.results_root)
        known = {entry["probe"] for entry in entries}
        for extra in self.extras:
            if extra["probe"] not in known and extra["probe"].is_file():
                entries.append(dict(extra))
        if self.library_override:
            probe_path, library_path = self.library_override
            for entry in entries:
                if entry["probe"] == probe_path:
                    point_library(entry, library_path)
        entries.sort(key=sort_key)
        return entries

    def library_rows(self, path):
        key = str(Path(path).resolve())
        with self.cache_lock:
            cached = self.libraries.get(key)
        if cached is not None:
            return cached
        try:
            rows = load_library(path)
        except Exception as ex:
            print(f"Could not read {path}: {ex}", flush=True)
            rows = []
        with self.cache_lock:
            return self.libraries.setdefault(key, rows)

    def activate(self, entry):
        library = self.library_rows(entry["library"])
        with self.infer_lock:
            loaded = load_probe(entry["probe"], self.device)
            self.state = {
                "probe": loaded["probe"],
                "classes": loaded["classes"],
                "parent_of": loaded["parent_of"],
                "task": loaded["task"],
                "seed": loaded["seed"],
                "encoder": self.encoder,
                "device": self.device,
                "library": library,
                "top": self.top,
                "lock": self.infer_lock,
                "id": entry["id"],
                "run": entry["run"],
                "embeddings": entry["embeddings"],
            }
        state = self.state
        print(
            f"Checkpoint: {entry['run']} · {state['task']}, seed {state['seed']}, "
            f"{len(state['classes'])} classes · {entry['embeddings']}",
            flush=True,
        )
        return state

    def info(self):
        state = self.state
        return {
            "task": state["task"],
            "seed": state["seed"],
            "n_classes": len(state["classes"]),
            "device": state["device"],
            "library": len(state["library"]),
            "max_seconds": MAX_SECONDS,
            "current": state["id"],
            "run": state["run"],
            "embeddings": state["embeddings"],
            "checkpoints": [entry_public(entry) for entry in self.catalog()],
        }


def serve(httpd):
    host, port = httpd.server_address[:2]
    print(f"Open http://{host}:{port}", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.", flush=True)
    finally:
        httpd.server_close()


if __name__ == "__main__":
    args = parse_args()
    serve(boot(args))
