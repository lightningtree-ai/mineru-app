"""FastAPI app: upload + queue API, document library, artifact serving, export."""
from __future__ import annotations

import json
import mimetypes
import queue
import subprocess
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from . import __version__
from .export import build_doc_zip, build_export_zip
from .jobs import JobManager, sanitize_options, sse_format
from .processing import SUPPORTED_SUFFIXES
from .store import Store

STATIC_DIR = Path(__file__).parent / "static"

store = Store()
manager = JobManager(store)


@asynccontextmanager
async def lifespan(app: FastAPI):
    manager.start()
    yield
    manager.shutdown()


app = FastAPI(title="mineru-app", version=__version__, lifespan=lifespan)


# -- jobs ---------------------------------------------------------------------

@app.post("/api/jobs")
async def create_jobs(files: list[UploadFile], options: str = Form("{}")):
    try:
        opts = sanitize_options(json.loads(options))
    except (ValueError, json.JSONDecodeError) as e:
        raise HTTPException(400, f"Bad options: {e}")

    jobs, skipped = [], []
    for f in files:
        name = Path(f.filename or "upload").name
        if Path(name).suffix.lower() not in SUPPORTED_SUFFIXES:
            skipped.append(name)
            continue
        doc_id = store.new_doc_id()
        dest = store.save_upload(doc_id, name, await f.read())
        jobs.append(manager.submit(doc_id, name, dest, opts).public())
    if not jobs and skipped:
        raise HTTPException(400, f"No supported files; skipped: {', '.join(skipped)}")
    return {"jobs": jobs, "skipped": skipped}


@app.get("/api/jobs")
def list_jobs():
    return {"jobs": manager.list_jobs(), "device": manager.device}


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str):
    status = manager.cancel(job_id)
    if status == "missing":
        raise HTTPException(404, "Job not found")
    if status != "cancelled":
        raise HTTPException(409, f"Job already {status}")
    return {"ok": True}


@app.get("/api/events")
def events():
    def stream():
        q = manager.bus.subscribe()
        try:
            yield sse_format({"type": "hello", "device": manager.device})
            while True:
                try:
                    yield sse_format(q.get(timeout=15))
                except queue.Empty:
                    yield ": keepalive\n\n"
        finally:
            manager.bus.unsubscribe(q)

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# -- document library -----------------------------------------------------------

@app.get("/api/docs")
def list_docs():
    return {"docs": store.list_docs()}


def _get_doc(doc_id: str) -> dict:
    doc = store.get_doc(doc_id)
    if not doc:
        raise HTTPException(404, "Document not found")
    return doc


def _doc_file(doc: dict, key: str) -> Path:
    rel = doc.get(key)
    if not rel:
        raise HTTPException(404, f"No {key} for this document")
    path = store.resolve(rel)
    if not path.exists():
        raise HTTPException(404, f"{key} missing on disk")
    return path


@app.get("/api/docs/{doc_id}/markdown")
def doc_markdown(doc_id: str):
    path = _doc_file(_get_doc(doc_id), "markdown_path")
    return PlainTextResponse(path.read_text(encoding="utf-8"), media_type="text/markdown")


@app.get("/api/docs/{doc_id}/content-list")
def doc_content_list(doc_id: str):
    path = _doc_file(_get_doc(doc_id), "content_list_path")
    return JSONResponse(json.loads(path.read_text(encoding="utf-8")))


@app.get("/api/docs/{doc_id}/source")
def doc_source(doc_id: str):
    path = _doc_file(_get_doc(doc_id), "source")
    media = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    return FileResponse(path, media_type=media, filename=path.name)


@app.get("/api/docs/{doc_id}/images/{name}")
def doc_image(doc_id: str, name: str):
    images_dir = _doc_file(_get_doc(doc_id), "images_dir")
    path = (images_dir / Path(name).name).resolve()
    if not path.is_relative_to(images_dir) or not path.exists():
        raise HTTPException(404, "Image not found")
    return FileResponse(path)


@app.get("/api/docs/{doc_id}/zip")
def doc_zip(doc_id: str):
    doc = _get_doc(doc_id)
    buf = build_doc_zip(store, doc_id)
    if buf is None:
        raise HTTPException(404, "No output for this document")
    return StreamingResponse(buf, media_type="application/zip", headers={
        "Content-Disposition": f'attachment; filename="{doc["stem"]}.zip"'})


@app.post("/api/docs/{doc_id}/reveal")
def reveal_doc(doc_id: str):
    """Open the OS file manager at the document's output (server is local-only)."""
    doc = _get_doc(doc_id)
    try:
        path = _doc_file(doc, "markdown_path")
        select = True
    except HTTPException:
        path = _doc_file(doc, "parse_dir")
        select = False
    if sys.platform == "darwin":
        cmd = ["open", "-R", str(path)] if select else ["open", str(path)]
    elif sys.platform == "win32":
        cmd = ["explorer", f"/select,{path}"] if select else ["explorer", str(path)]
    else:
        cmd = ["xdg-open", str(path.parent if select else path)]
    # explorer.exe exits nonzero even on success, so don't check the return code
    subprocess.Popen(cmd)
    return {"ok": True}


@app.delete("/api/docs/{doc_id}")
def delete_doc(doc_id: str):
    if not store.delete_doc(doc_id):
        raise HTTPException(404, "Document not found")
    return {"ok": True}


# -- export ---------------------------------------------------------------------

@app.post("/api/export")
async def export(payload: dict):
    ids = payload.get("ids") or []
    mode = payload.get("mode", "md-only")
    if mode not in ("md-only", "md+images"):
        raise HTTPException(400, f"Bad export mode: {mode}")
    if not ids:
        raise HTTPException(400, "No documents selected")
    buf, count = build_export_zip(store, ids, mode)
    if count == 0:
        raise HTTPException(404, "None of the selected documents have markdown output")
    return StreamingResponse(buf, media_type="application/zip", headers={
        "Content-Disposition": 'attachment; filename="mineru-export.zip"'})


# -- meta + frontend --------------------------------------------------------------

@app.get("/api/meta")
def meta():
    return {
        "version": __version__,
        "device": manager.device,  # known after the first job
        "platform": sys.platform,
        "supported_suffixes": sorted(SUPPORTED_SUFFIXES),
        "data_dir": str(store.root),
    }


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
