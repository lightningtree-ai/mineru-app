"""Job model, worker subprocess, and SSE event bus.

One job per uploaded file: a batch of 30 PDFs shows per-file progress and one
failure doesn't abort the rest. Jobs run sequentially in a single long-lived
**worker subprocess** — MinerU's models load once on the first job and stay
warm for as long as the worker lives, and sequential execution avoids
competing for the GPU. Running in a child process (rather than a thread) is
what makes cancellation possible: terminating the worker kills the inference
mid-flight and the OS reclaims its GPU memory; a fresh worker is spawned for
the next job. It also means a hard crash in torch can't take down the server.
"""
from __future__ import annotations

import contextlib
import json
import multiprocessing as mp
import os
import queue
import re
import signal
import subprocess
import sys
import threading
import time
import traceback
import uuid
from collections import deque
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


def _worker_main(job_q, event_q) -> None:
    """Worker subprocess: pull job specs, run MinerU, report over event_q.

    Events: ("progress", job_id, line) | ("done", job_id, slim_result)
          | ("failed", job_id, error_message)
    """
    if hasattr(os, "setpgrp"):
        # Lead our own process group: MinerU spawns helper subprocesses, and on
        # cancel the server kills the whole group so none of them get orphaned.
        os.setpgrp()

    from .processing import preprocess  # heavy import happens here, not in the server

    while True:
        spec = job_q.get()
        if spec is None:
            return
        job_id = spec["id"]
        last_sent = 0.0

        def on_line(line: str) -> None:
            nonlocal last_sent
            now = time.monotonic()
            if now - last_sent < 0.25:  # cap progress chatter (tqdm refreshes fast)
                return
            last_sent = now
            try:
                event_q.put_nowait(("progress", job_id, line[:300]))
            except Exception:
                pass

        try:
            opts = dict(spec["options"])
            device = opts.pop("device", None)
            with _capture_stderr_fd(on_line):
                results = preprocess(
                    [spec["upload_path"]],
                    output_dir=spec["output_dir"],
                    device_mode=device,
                    **opts,
                )
            result = results[0]
            if not result.get("markdown_path"):
                raise RuntimeError("MinerU produced no markdown output")
            slim = {k: result.get(k) for k in
                    ("source", "device", "parse_dir", "markdown_path",
                     "content_list_path", "images_dir")}
            slim["markdown_chars"] = len(result.get("markdown") or "")
            slim["blocks"] = len(result.get("content_list") or [])
            event_q.put(("done", job_id, slim))
        except Exception as e:
            traceback.print_exc()
            event_q.put(("failed", job_id, f"{type(e).__name__}: {e}"))


def _terminate_tree(proc) -> None:
    """Kill the worker and any helper processes it spawned (its process group)."""
    if proc is None or not proc.is_alive():
        return
    if sys.platform == "win32":
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)], capture_output=True)
        proc.join(timeout=5)
        return
    try:
        pgid = os.getpgid(proc.pid)
    except ProcessLookupError:
        return
    own_group = pgid == os.getpgid(0)  # worker hasn't run setpgrp() yet
    try:
        if own_group:
            proc.terminate()
        else:
            os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return
    proc.join(timeout=5)
    if proc.is_alive():
        try:
            if own_group:
                proc.kill()
            else:
                os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            return
        proc.join(timeout=2)


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
        self._pending: deque[str] = deque()
        self._current: str | None = None  # job id dispatched to the worker
        self._lock = threading.Lock()
        self._ctx = mp.get_context("spawn")
        self._proc: mp.process.BaseProcess | None = None
        self._job_q = None
        self._event_q = None
        self._pump_thread: threading.Thread | None = None
        self._last_prog_pub = 0.0

    def start(self) -> None:
        if self._pump_thread is None:
            self._pump_thread = threading.Thread(target=self._pump, name="mineru-events", daemon=True)
            self._pump_thread.start()

    def submit(self, doc_id: str, name: str, upload_path: Path, options: dict) -> Job:
        job = Job(id=uuid.uuid4().hex[:12], doc_id=doc_id, name=name,
                  upload_path=str(upload_path), options=options)
        with self._lock:
            self._jobs[job.id] = job
            self._pending.append(job.id)
        self._publish(job)
        self._dispatch()
        return job

    def list_jobs(self) -> list[dict]:
        with self._lock:
            jobs = list(self._jobs.values())
        return [j.public() for j in sorted(jobs, key=lambda j: j.queued_at, reverse=True)]

    def cancel(self, job_id: str) -> str:
        """Cancel a queued job, or terminate the worker to abort a running one.
        Returns the job's resulting status ("cancelled" on success)."""
        proc = None
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return "missing"
            if job.status == "queued":
                job.status = "cancelled"
                job.finished_at = time.time()
            elif job.status == "running" and self._current == job_id:
                proc, self._proc = self._proc, None  # next job spawns a fresh worker
                self._current = None
                job.status = "cancelled"
                job.progress = None
                job.finished_at = time.time()
            else:
                return job.status
        _terminate_tree(proc)
        self._publish(job)
        self._dispatch()
        return "cancelled"

    def shutdown(self) -> None:
        """Terminate the worker so the server process can exit cleanly."""
        with self._lock:
            proc, self._proc = self._proc, None
            self._current = None
        _terminate_tree(proc)

    # -- internals -------------------------------------------------------------

    def _publish(self, job: Job) -> None:
        self.bus.publish({"type": "job", "job": job.public()})

    def _ensure_worker_locked(self) -> None:
        if self._proc is not None and self._proc.is_alive():
            return
        self._job_q = self._ctx.Queue()
        self._event_q = self._ctx.Queue()
        # Not daemonic: torch/MinerU may spawn helper processes, which daemonic
        # processes are forbidden to do. shutdown() reaps it when the server exits.
        self._proc = self._ctx.Process(target=_worker_main, args=(self._job_q, self._event_q),
                                       name="mineru-worker")
        self._proc.start()

    def _dispatch(self) -> None:
        """Send the next queued job to the worker if it's idle."""
        with self._lock:
            if self._current is not None:
                return
            job = None
            while self._pending:
                candidate = self._jobs[self._pending.popleft()]
                if candidate.status == "queued":
                    job = candidate
                    break
            if job is None:
                return
            self._ensure_worker_locked()
            self._current = job.id
            job.status = "running"
            job.started_at = time.time()
            job_q = self._job_q
        job_q.put({
            "id": job.id,
            "upload_path": job.upload_path,
            "output_dir": str(self.store.output_dir(job.doc_id)),
            "options": job.options,
        })
        self._publish(job)

    def _pump(self) -> None:
        """Consume worker events; detect a worker that died mid-job."""
        while True:
            event_q = self._event_q
            if event_q is None:
                time.sleep(0.5)
                continue
            try:
                kind, job_id, payload = event_q.get(timeout=1.0)
            except queue.Empty:
                self._reap_dead_worker()
                continue
            except (EOFError, OSError):
                time.sleep(0.5)
                continue
            job = self._jobs.get(job_id)
            if job is None or job.status != "running":
                continue  # stale event from a cancelled/terminated run

            if kind == "progress":
                job.progress = payload
                now = time.time()
                if now - self._last_prog_pub >= 1.0:  # throttle SSE during bursts
                    self._last_prog_pub = now
                    self._publish(job)
            elif kind == "done":
                job.device = self.device = payload.get("device")
                entry = Store.make_entry(
                    job.doc_id, job.name, job.options, payload,
                    seconds=time.time() - job.started_at, store=self.store,
                )
                self.store.add_doc(entry)
                job.status = "done"
                job.progress = None
                job.finished_at = time.time()
                with self._lock:
                    if self._current == job_id:
                        self._current = None
                self._publish(job)
                self.bus.publish({"type": "doc", "doc": entry})
                self._dispatch()
            elif kind == "failed":
                job.status = "failed"
                job.error = payload
                job.finished_at = time.time()
                with self._lock:
                    if self._current == job_id:
                        self._current = None
                self._publish(job)
                self._dispatch()

    def _reap_dead_worker(self) -> None:
        with self._lock:
            proc, job_id = self._proc, self._current
            if job_id is None or (proc is not None and proc.is_alive()):
                return
            self._proc = None
            self._current = None
            job = self._jobs.get(job_id)
            if job is not None and job.status == "running":
                job.status = "failed"
                job.error = ("Worker process died unexpectedly "
                             "(possibly out of memory — try device=cpu or a smaller page range)")
                job.finished_at = time.time()
            else:
                job = None
        if job is not None:
            self._publish(job)
        self._dispatch()


def sse_format(event: dict) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
