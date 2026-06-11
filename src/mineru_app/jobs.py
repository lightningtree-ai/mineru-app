"""Job model, single worker thread, and SSE event bus.

One job per uploaded file: a batch of 30 PDFs shows per-file progress and one
failure doesn't abort the rest. A single worker consumes jobs sequentially —
MinerU's models load once on the first job and stay warm for the process
lifetime, and sequential execution avoids competing for the GPU.
"""
from __future__ import annotations

import contextlib
import json
import os
import queue
import re
import sys
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from .store import Store

ALLOWED_OPTIONS = {
    "lang": str,
    "backend": ("pipeline", "vlm-transformers"),
    "method": ("auto", "txt", "ocr"),
    "formula": bool,
    "table": bool,
    "start_page": int,
    "end_page": int,
    "device": ("auto", "cuda", "mps", "cpu"),
}


def sanitize_options(raw: dict) -> dict:
    """Whitelist + type-check user-supplied processing options."""
    opts: dict = {}
    for key, spec in ALLOWED_OPTIONS.items():
        val = raw.get(key)
        if val is None or val == "" or (key == "device" and val == "auto"):
            continue
        if isinstance(spec, tuple):
            if val not in spec:
                raise ValueError(f"Invalid value for {key!r}: {val!r}")
            opts[key] = val
        elif spec is bool:
            opts[key] = bool(val)
        elif spec is int:
            opts[key] = int(val)
        else:
            opts[key] = str(val)
    return opts


@dataclass
class Job:
    id: str
    doc_id: str
    name: str
    upload_path: str
    options: dict
    status: str = "queued"  # queued | running | done | failed | cancelled
    error: str | None = None
    device: str | None = None
    progress: str | None = None  # last log/progress line from MinerU
    queued_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None

    def public(self) -> dict:
        return {
            "id": self.id,
            "doc_id": self.doc_id,
            "name": self.name,
            "options": self.options,
            "status": self.status,
            "error": self.error,
            "device": self.device,
            "progress": self.progress,
            "queued_at": self.queued_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }


_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


@contextlib.contextmanager
def _capture_stderr_fd(on_line):
    """Tee OS-level stderr (fd 2) through a pipe, feeding complete lines (or \\r
    progress-bar refreshes, e.g. tqdm during VLM inference) to a callback, so
    long MinerU jobs show signs of life instead of sitting on "running" for hours.

    Capturing the file descriptor — not swapping the sys.stderr object — matters:
    loguru and tqdm grab a reference to the stderr *object* when first imported,
    so an object swap only ever catches output for the job that triggered the
    import. Everything ultimately writes to fd 2, which this sees process-wide.
    """
    read_fd, write_fd = os.pipe()
    saved_fd = os.dup(2)
    sys.stderr.flush()
    os.dup2(write_fd, 2)
    os.close(write_fd)

    def pump():
        buf = b""
        try:
            while True:
                chunk = os.read(read_fd, 4096)
                if not chunk:
                    break
                os.write(saved_fd, chunk)  # still reaches the real terminal
                buf += chunk
                *lines, buf = re.split(rb"[\r\n]", buf)
                for raw in lines:
                    line = _ANSI.sub("", raw.decode("utf-8", "replace")).strip()
                    if line:
                        on_line(line)
        except OSError:
            pass
        finally:
            os.close(read_fd)

    pump_thread = threading.Thread(target=pump, name="stderr-pump", daemon=True)
    pump_thread.start()
    try:
        yield
    finally:
        sys.stderr.flush()
        os.dup2(saved_fd, 2)  # closes the pipe's last write end -> pump sees EOF
        pump_thread.join(timeout=2)
        os.close(saved_fd)


class EventBus:
    """Fan-out of job events to SSE subscribers (thread-safe)."""

    def __init__(self):
        self._subscribers: list[queue.Queue] = []
        self._lock = threading.Lock()

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue()
        with self._lock:
            self._subscribers.append(q)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self._lock:
            if q in self._subscribers:
                self._subscribers.remove(q)

    def publish(self, event: dict) -> None:
        with self._lock:
            subs = list(self._subscribers)
        for q in subs:
            q.put(event)


class JobManager:
    def __init__(self, store: Store):
        self.store = store
        self.bus = EventBus()
        self.device: str | None = None  # set after the first job runs
        self._jobs: dict[str, Job] = {}
        self._queue: queue.Queue[Job] = queue.Queue()
        self._lock = threading.Lock()
        self._worker: threading.Thread | None = None

    def start(self) -> None:
        if self._worker is None:
            self._worker = threading.Thread(target=self._run, name="mineru-worker", daemon=True)
            self._worker.start()

    def submit(self, doc_id: str, name: str, upload_path: Path, options: dict) -> Job:
        job = Job(id=uuid.uuid4().hex[:12], doc_id=doc_id, name=name,
                  upload_path=str(upload_path), options=options)
        with self._lock:
            self._jobs[job.id] = job
        self._queue.put(job)
        self.bus.publish({"type": "job", "job": job.public()})
        return job

    def list_jobs(self) -> list[dict]:
        with self._lock:
            jobs = list(self._jobs.values())
        return [j.public() for j in sorted(jobs, key=lambda j: j.queued_at, reverse=True)]

    def cancel(self, job_id: str) -> str:
        """Cancel a queued job. Running jobs can't be interrupted (the model is
        mid-inference in this process); returns the job's current status."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return "missing"
            if job.status != "queued":
                return job.status
            job.status = "cancelled"
            job.finished_at = time.time()
        self._publish(job)
        return "cancelled"

    def _publish(self, job: Job) -> None:
        self.bus.publish({"type": "job", "job": job.public()})

    def _run(self) -> None:
        from .processing import preprocess  # heavy deps load in this thread only

        while True:
            job = self._queue.get()
            if job.status != "queued":  # cancelled while waiting
                continue
            job.status = "running"
            job.started_at = time.time()
            self._publish(job)

            last_pub = 0.0

            def on_line(line: str, job=job) -> None:
                nonlocal last_pub
                job.progress = line[:300]
                now = time.time()
                if now - last_pub >= 1.0:  # throttle SSE to ~1/s during bursts
                    last_pub = now
                    self._publish(job)

            try:
                opts = dict(job.options)
                device = opts.pop("device", None)
                with _capture_stderr_fd(on_line):
                    results = preprocess(
                        [job.upload_path],
                        output_dir=self.store.output_dir(job.doc_id),
                        device_mode=device,
                        **opts,
                    )
                result = results[0]
                if not result.get("markdown_path"):
                    raise RuntimeError("MinerU produced no markdown output")
                job.device = self.device = result.get("device")
                entry = Store.make_entry(
                    job.doc_id, job.name, job.options, result,
                    seconds=time.time() - job.started_at, store=self.store,
                )
                self.store.add_doc(entry)
                job.status = "done"
                job.progress = None
                job.finished_at = time.time()
                self._publish(job)
                self.bus.publish({"type": "doc", "doc": entry})
            except Exception as e:
                job.status = "failed"
                job.error = f"{type(e).__name__}: {e}"
                job.finished_at = time.time()
                traceback.print_exc()
                self._publish(job)


def sse_format(event: dict) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
