"""Document library: data-dir layout + manifest.json persistence.

Layout (root overridable via MINERU_APP_DATA, default ./data):
    data/
      uploads/<doc_id>/<safe stem>.<ext>     original upload
      output/<doc_id>/<safe stem>/<subdir>/  MinerU output tree
      manifest.json                          library index, survives restarts

Manifest paths are stored relative to the data root so the folder is relocatable.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import threading
import time
import uuid
from pathlib import Path

# Characters invalid on Windows filesystems (superset of POSIX restrictions).
_INVALID_FS_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
# Keep filesystem stems short: MinerU caps task stems at 200 bytes and Windows
# paths at 260 chars; Zotero names are routinely longer.
_MAX_STEM_BYTES = 96


def sanitize_stem(name: str) -> str:
    """A filesystem-safe, cross-platform stem derived from an original filename."""
    stem = _INVALID_FS_CHARS.sub(" ", Path(name).stem)
    stem = re.sub(r"\s+", " ", stem).strip().rstrip(". ")
    while len(stem.encode("utf-8")) > _MAX_STEM_BYTES:
        stem = stem[:-1].rstrip(". ")
    return stem or "document"


class Store:
    def __init__(self, root: str | os.PathLike | None = None):
        self.root = Path(root or os.environ.get("MINERU_APP_DATA", "data")).expanduser().resolve()
        self.uploads = self.root / "uploads"
        self.outputs = self.root / "output"
        self.manifest_path = self.root / "manifest.json"
        self._lock = threading.Lock()
        for d in (self.uploads, self.outputs):
            d.mkdir(parents=True, exist_ok=True)

    # -- manifest ------------------------------------------------------------

    def _read(self) -> dict:
        if self.manifest_path.exists():
            return json.loads(self.manifest_path.read_text(encoding="utf-8"))
        return {"docs": {}}

    def _write(self, manifest: dict) -> None:
        tmp = self.manifest_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.manifest_path)

    def list_docs(self) -> list[dict]:
        with self._lock:
            docs = list(self._read()["docs"].values())
        return sorted(docs, key=lambda d: d.get("created", 0), reverse=True)

    def get_doc(self, doc_id: str) -> dict | None:
        with self._lock:
            return self._read()["docs"].get(doc_id)

    def add_doc(self, entry: dict) -> None:
        with self._lock:
            manifest = self._read()
            manifest["docs"][entry["id"]] = entry
            self._write(manifest)

    def delete_doc(self, doc_id: str) -> bool:
        with self._lock:
            manifest = self._read()
            existed = manifest["docs"].pop(doc_id, None) is not None
            if existed:
                self._write(manifest)
        shutil.rmtree(self.uploads / doc_id, ignore_errors=True)
        shutil.rmtree(self.outputs / doc_id, ignore_errors=True)
        return existed

    # -- files ---------------------------------------------------------------

    def new_doc_id(self) -> str:
        return uuid.uuid4().hex[:12]

    def save_upload(self, doc_id: str, original_name: str, data: bytes) -> Path:
        suffix = Path(original_name).suffix.lower()
        dest_dir = self.uploads / doc_id
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / f"{sanitize_stem(original_name)}{suffix}"
        dest.write_bytes(data)
        return dest

    def output_dir(self, doc_id: str) -> Path:
        return self.outputs / doc_id

    def resolve(self, rel_path: str) -> Path:
        """Resolve a manifest-relative path, refusing escapes from the data root."""
        p = (self.root / rel_path).resolve()
        if not p.is_relative_to(self.root):
            raise ValueError(f"Path escapes data root: {rel_path}")
        return p

    def relativize(self, path: str | Path | None) -> str | None:
        if path is None:
            return None
        return os.path.relpath(Path(path).resolve(), self.root).replace(os.sep, "/")

    @staticmethod
    def make_entry(doc_id: str, original_name: str, options: dict, result: dict,
                   seconds: float, store: "Store") -> dict:
        """Build a manifest entry from a preprocess() result dict."""
        return {
            "id": doc_id,
            "original_name": original_name,
            "stem": sanitize_stem(original_name),
            "created": time.time(),
            "options": options,
            "device": result.get("device"),
            "seconds": round(seconds, 1),
            "source": store.relativize(result.get("source")),
            "parse_dir": store.relativize(result.get("parse_dir")),
            "markdown_path": store.relativize(result.get("markdown_path")),
            "content_list_path": store.relativize(result.get("content_list_path")),
            "images_dir": store.relativize(result.get("images_dir")),
            # the worker subprocess sends precomputed counts instead of the
            # (potentially huge) markdown/content_list payloads
            "markdown_chars": result.get("markdown_chars", len(result.get("markdown") or "")),
            "blocks": result.get("blocks", len(result.get("content_list") or [])),
        }
