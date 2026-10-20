"""
Local page for embedding extraction and the linear probe.

One page, on this machine. It starts extract_panns_embeddings.py and
train_linear_probe.py and shows their output. Paths are folders on this
machine, the same ones the scripts take on the command line. One job runs
at a time.

    python web/app.py

Open the URL it prints. Uses the packages already listed in requirements.txt.
"""
import argparse
import json
import math
import os
import re
import subprocess
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

WEB = Path(__file__).resolve().parent
ROOT = WEB.parent
EXTRACT = ROOT / "extract_panns_embeddings.py"
TRAIN = ROOT / "train_linear_probe.py"
PAGE = WEB / "index.html"

DEFAULT_EXTRACT_OUT = "embeddings/panns_cnn14"
DEFAULT_TRAIN_OUT = "results/panns_cnn14_linear"
DEFAULT_SEEDS = "42,123,456,789,1024"
DEFAULT_SPLITS = "train,val,test"
MAX_LOG = 4000
MAX_BODY = 64 * 1024
SAFE_TOKEN = re.compile(r"[A-Za-z0-9_./\\:=,+@-]+")


class JobError(Exception):
    def __init__(self, message, status=400):
        super().__init__(message)
        self.status = status


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8766)
    return ap.parse_args(argv)


def resolve(value):
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def clean(value):
    text = "" if value is None else str(value).strip()
    if any(ch in text for ch in "\x00\r\n"):
        raise JobError("A field contains a line break.")
    return text


def as_bool(body, key):
    value = body.get(key, False)
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def text_field(body, key, label, required=False, default=""):
    value = clean(body.get(key))
    if not value:
        if required:
            raise JobError(f"Enter {label}.")
        value = default
    if value.startswith("-"):
        raise JobError(f"{label} cannot start with '-'.")
    return value


def number_field(body, key, label, default, lo, hi, kind):
    raw = clean(body.get(key))
    if not raw:
        return str(default)
    try:
        value = kind(raw)
    except ValueError:
        raise JobError(f"{label} must be a number.") from None
    if kind is float and not math.isfinite(value):
        raise JobError(f"{label} must be a finite number.")
    if value < lo or value > hi:
        raise JobError(f"{label} must be between {lo} and {hi}.")
    return raw


def choice_field(body, key, label, choices, default):
    value = clean(body.get(key)) or default
    if value not in choices:
        raise JobError(f"{label} must be one of: {', '.join(choices)}.")
    return value


def existing(value, label, kind):
    path = resolve(value)
    if kind == "file" and not path.is_file():
        raise JobError(f"{label} is not a file: {value}")
    if kind == "dir" and not path.is_dir():
        raise JobError(f"{label} is not a folder: {value}")
    if kind == "any" and not path.exists():
        raise JobError(f"{label} was not found: {value}")
    return value


def output_dir(value, label):
    path = resolve(value)
    if path.exists() and not path.is_dir():
        raise JobError(f"{label} is a file, not a folder: {value}")
    return value


def parse_splits(raw):
    parts = [part.strip() for part in clean(raw).split(",") if part.strip()]
    if not parts:
        raise JobError("Enter at least one split.")
    for part in parts:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", part):
            raise JobError("Splits are names separated by commas.")
    return parts


def parse_seeds(raw):
    parts = [part.strip() for part in clean(raw).split(",") if part.strip()]
    if not parts:
        parts = DEFAULT_SEEDS.split(",")
    seeds = []
    for part in parts:
        if not re.fullmatch(r"\d+", part):
            raise JobError("Seeds are whole numbers separated by commas.")
        seeds.append(str(int(part)))
    return seeds


def quote_arg(text):
    if SAFE_TOKEN.fullmatch(text):
        return text
    return '"' + text.replace('"', '`"') + '"'


def display_command(cmd):
    parts = ["python", Path(cmd[2]).name, *cmd[3:]]
    return " ".join(quote_arg(part) for part in parts)


def build_extract(body):
    out = output_dir(text_field(body, "out", "Output folder", default=DEFAULT_EXTRACT_OUT), "Output folder")
    splits = parse_splits(body.get("splits") or DEFAULT_SPLITS)
    cmd = [sys.executable, "-u", str(EXTRACT), "--out", out, "--splits", ",".join(splits)]
    if as_bool(body, "combine"):
        src = existing(text_field(body, "combine_from", "a folder of old caches", required=True), "Combine from", "dir")
        cmd += ["--combine-from", src]
        return cmd, "combine"

    manifests = existing(
        text_field(body, "manifests", "a manifests folder or CSV", required=True),
        "Manifests",
        "any",
    )
    audio = text_field(body, "audio_root", "Audio root")
    if audio:
        existing(audio, "Audio root", "dir")
    checkpoint = text_field(body, "checkpoint", "Checkpoint")
    if checkpoint:
        if resolve(checkpoint).is_dir():
            raise JobError("Checkpoint is a folder, not a file.")
    device = choice_field(body, "device", "Device", ("auto", "cuda", "cpu"), "auto")
    limit = number_field(body, "limit", "Row limit", 0, 0, 10_000_000, int)
    save_every = number_field(body, "save_every", "Save every", 2000, 1, 1_000_000, int)
    cmd += [
        "--manifests", manifests,
        "--device", device,
        "--limit", limit,
        "--save-every", save_every,
    ]
    if audio:
        cmd += ["--audio-root", audio]
    if checkpoint:
        cmd += ["--checkpoint", checkpoint]
    if as_bool(body, "restart"):
        cmd += ["--restart"]
    return cmd, "extract"


def build_train(body):
    embeddings = text_field(body, "embeddings", "Embeddings folder", default=DEFAULT_EXTRACT_OUT)
    emb_dir = resolve(embeddings)
    if emb_dir.is_file():
        raise JobError("Point Embeddings at the folder that holds embeddings.npz.")
    if not emb_dir.is_dir() or not (emb_dir / "embeddings.npz").is_file():
        raise JobError(f"No embeddings.npz in {embeddings}. Extract embeddings first.")
    out = output_dir(text_field(body, "out", "Output folder", default=DEFAULT_TRAIN_OUT), "Output folder")
    task = choice_field(body, "task", "Task", ("subcategory", "category", "pair"), "subcategory")
    device = choice_field(body, "device", "Device", ("auto", "cuda", "cpu"), "auto")
    seeds = parse_seeds(body.get("seeds") or DEFAULT_SEEDS)
    mode = choice_field(body, "manifest_mode", "Manifests", ("folder", "files"), "folder")
    cmd = [
        sys.executable, "-u", str(TRAIN),
        "--embeddings", embeddings,
        "--out", out,
        "--task", task,
        "--device", device,
        "--seeds", ",".join(seeds),
        "--batch-size", number_field(body, "batch_size", "Batch size", 256, 1, 100_000, int),
        "--lr", number_field(body, "lr", "Learning rate", "1e-3", 1e-8, 10, float),
        "--weight-decay", number_field(body, "weight_decay", "Weight decay", "1e-3", 0, 10, float),
        "--dropout", number_field(body, "dropout", "Dropout", "0.3", 0, 0.99, float),
        "--gamma", number_field(body, "gamma", "Focal gamma", "2", 0, 100, float),
        "--patience", number_field(body, "patience", "Patience", 20, 1, 10_000, int),
        "--max-epochs", number_field(body, "max_epochs", "Max epochs", 200, 1, 10_000, int),
        "--limit", number_field(body, "limit", "Row limit", 0, 0, 10_000_000, int),
    ]
    if mode == "folder":
        manifests = existing(text_field(body, "manifests", "a manifests folder", required=True), "Manifests", "dir")
        cmd += ["--manifests", manifests]
    else:
        cmd += [
            "--train", existing(text_field(body, "train", "a training CSV", required=True), "Training CSV", "file"),
            "--eval", existing(text_field(body, "eval", "a validation CSV", required=True), "Validation CSV", "file"),
            "--test", existing(text_field(body, "test", "a test CSV", required=True), "Test CSV", "file"),
        ]
    dry = as_bool(body, "dry_run")
    if as_bool(body, "restart"):
        cmd.append("--restart")
    if dry:
        cmd.append("--dry-run")
    if as_bool(body, "verbose"):
        cmd.append("--verbose")
    return cmd, ("dry-run" if dry else "train")


def rel(path):
    path = Path(path).resolve()
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def embedding_dirs():
    root = ROOT / "embeddings"
    if not root.is_dir():
        return []
    found = [rel(path.parent) for path in root.rglob("embeddings.npz")]
    return sorted(set(found))


def result_dirs():
    root = ROOT / "results"
    if not root.is_dir():
        return []
    found = [rel(path.parent.parent) for path in root.rglob("summary.json")]
    return sorted(set(found))


def cuda_status():
    try:
        import torch
    except Exception as ex:
        return {"cuda": False, "gpu": "", "torch_error": f"{type(ex).__name__}: {ex}"}
    if not torch.cuda.is_available():
        return {"cuda": False, "gpu": "", "torch_error": ""}
    try:
        gpu = torch.cuda.get_device_name(0)
    except Exception:
        gpu = ""
    return {"cuda": True, "gpu": gpu, "torch_error": ""}


class Jobs:
    def __init__(self):
        self.lock = threading.Lock()
        self.job = None
        self.proc = None
        self.next_id = 1
        self._cuda = None

    def info(self):
        if self._cuda is None:
            self._cuda = cuda_status()
        cuda = self._cuda
        return {
            "cuda": cuda["cuda"],
            "gpu": cuda["gpu"],
            "torch_error": cuda["torch_error"],
            "embedding_dirs": embedding_dirs(),
            "result_dirs": result_dirs(),
            "defaults": {
                "extract_out": DEFAULT_EXTRACT_OUT,
                "train_embeddings": DEFAULT_EXTRACT_OUT,
                "train_out": DEFAULT_TRAIN_OUT,
                "seeds": DEFAULT_SEEDS,
                "splits": DEFAULT_SPLITS,
            },
        }

    def snapshot(self, after):
        with self.lock:
            job = self.job
            if job is None:
                return {
                    "id": 0,
                    "status": "idle",
                    "kind": "",
                    "returncode": None,
                    "command": "",
                    "lines": [],
                    "next": 0,
                    "skipped": False,
                }
            start = job["dropped"]
            lines = job["lines"]
            end = start + len(lines)
            try:
                after = int(after)
            except (TypeError, ValueError):
                after = 0
            if after < 0:
                after = 0
            # Behind the retained window, or ahead of this job: send what we still have.
            skipped = after < start or after > end
            chunk = list(lines) if skipped else lines[after - start:]
            return {
                "id": job["id"],
                "status": job["status"],
                "kind": job["kind"],
                "returncode": job["returncode"],
                "command": job["command"],
                "lines": chunk,
                "next": end,
                "skipped": skipped,
            }

    def start(self, kind, cmd):
        display = display_command(cmd)
        with self.lock:
            if self.job is not None and self.job["status"] == "running":
                raise JobError("A job is already running.", 409)
            job = {
                "id": self.next_id,
                "kind": kind,
                "status": "running",
                "returncode": None,
                "command": display,
                "lines": [display, ""],
                "dropped": 0,
                "started": time.time(),
            }
            self.next_id += 1
            self.job = job
        print(f"Job {job['id']} {kind}: {display}", flush=True)
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        kwargs = {
            "cwd": str(ROOT),
            "stdout": subprocess.PIPE,
            "stderr": subprocess.STDOUT,
            "stdin": subprocess.DEVNULL,
            "env": env,
            "text": True,
            "encoding": "utf-8",
            "errors": "replace",
            "bufsize": 1,
        }
        if os.name == "nt":
            kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
        try:
            proc = subprocess.Popen(cmd, **kwargs)
        except OSError as ex:
            with self.lock:
                job["status"] = "failed"
                job["returncode"] = -1
                job["lines"].append(f"Could not start the script. {ex}")
            return self.snapshot(0)
        with self.lock:
            self.proc = proc
        threading.Thread(target=self._read, args=(proc, job), daemon=True).start()
        return self.snapshot(0)

    def _append(self, job, text):
        job["lines"].append(text)
        extra = len(job["lines"]) - MAX_LOG
        if extra > 0:
            del job["lines"][:extra]
            job["dropped"] += extra

    def _read(self, proc, job):
        try:
            for line in proc.stdout:
                with self.lock:
                    self._append(job, line.rstrip("\r\n"))
            code = proc.wait()
        except Exception as ex:
            code = -1
            with self.lock:
                self._append(job, f"Could not read the script output. {type(ex).__name__}: {ex}")
        with self.lock:
            job["returncode"] = code
            if job["status"] == "running":
                job["status"] = "done" if code == 0 else "failed"
            status = job["status"]
        print(f"Job {job['id']} {status} (exit {code})", flush=True)

    def stop(self):
        with self.lock:
            proc = self.proc
            job = self.job
            if proc is None or job is None or job["status"] != "running":
                return False
            job["status"] = "stopped"
            self._append(job, "Stopping…")
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        return True


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
                self.send_json(200, self.server.jobs.info())
            elif path == "/api/job":
                query = parse_qs(urlparse(self.path).query)
                after = query.get("after", ["0"])[0]
                self.send_json(200, self.server.jobs.snapshot(after))
            else:
                self.send_json(404, {"error": "Not found."})
        except JobError as ex:
            self.send_json(ex.status, {"error": str(ex)})
        except Exception as ex:
            traceback.print_exc()
            self.send_json(500, {"error": f"{type(ex).__name__}: {ex}"[:200]})

    def do_POST(self):
        path = urlparse(self.path).path
        try:
            if path == "/api/extract":
                cmd, kind = build_extract(self.read_json())
                self.send_json(200, self.server.jobs.start(kind, cmd))
            elif path == "/api/train":
                cmd, kind = build_train(self.read_json())
                self.send_json(200, self.server.jobs.start(kind, cmd))
            elif path == "/api/stop":
                stopped = self.server.jobs.stop()
                snap = self.server.jobs.snapshot(0)
                if not stopped and snap["status"] == "idle":
                    raise JobError("Nothing is running.")
                self.send_json(200, snap)
            else:
                self.send_json(404, {"error": "Not found."})
        except JobError as ex:
            self.send_json(ex.status, {"error": str(ex)})
        except Exception as ex:
            traceback.print_exc()
            self.send_json(500, {"error": f"{type(ex).__name__}: {ex}"[:200]})

    def read_json(self):
        length = self.headers.get("Content-Length")
        if length is None:
            raise JobError("Missing Content-Length.", 411)
        try:
            size = int(length)
        except ValueError:
            raise JobError("Missing Content-Length.", 411) from None
        if size <= 0:
            raise JobError("The request was empty.")
        if size > MAX_BODY:
            raise JobError("The request was too large.", 413)
        try:
            payload = json.loads(self.rfile.read(size).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise JobError("The request was not valid JSON.") from None
        if not isinstance(payload, dict):
            raise JobError("The request was not a JSON object.")
        return payload

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


class Server(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, address, handler):
        super().__init__(address, handler)
        self.jobs = Jobs()


def serve(httpd):
    host, port = httpd.server_address[:2]
    print(f"Open http://{host}:{port}", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.", flush=True)
    finally:
        httpd.jobs.stop()
        httpd.server_close()


def main(argv=None):
    if not EXTRACT.is_file() or not TRAIN.is_file():
        sys.exit("extract_panns_embeddings.py and train_linear_probe.py must sit next to the web folder.")
    args = parse_args(argv)
    if not 1 <= args.port <= 65535:
        sys.exit("--port must be between 1 and 65535.")
    try:
        httpd = Server((args.host, args.port), Handler)
    except OSError as ex:
        sys.exit(f"Could not listen on {args.host}:{args.port}. {ex}")
    serve(httpd)


if __name__ == "__main__":
    main()
