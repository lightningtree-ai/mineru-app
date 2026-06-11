"""Flat export: selected docs -> one zip of <stem>.md files (optionally + images).

Built for the Zotero -> MinerU -> Claude Desktop flow: the zip has every
markdown file at the top level, named after the original upload, so the
unzipped folder can be dropped straight into a chat or an agent's context dir.
In "md+images" mode each doc's figures land in "<stem>_images/" and the image
links inside the markdown are rewritten to match.
"""
from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path

from .store import Store, sanitize_stem

# MinerU emits markdown/html image refs relative to the parse dir: images/<hash>.jpg
_IMG_MD = re.compile(r"(!\[[^\]]*\]\()images/")
_IMG_HTML = re.compile(r'(<img[^>]+src=["\'])images/')


def _rewrite_image_links(markdown: str, images_dirname: str) -> str:
    markdown = _IMG_MD.sub(rf"\g<1>{images_dirname}/", markdown)
    return _IMG_HTML.sub(rf"\g<1>{images_dirname}/", markdown)


def _unique(stem: str, taken: set[str]) -> str:
    name, n = stem, 1
    while name.lower() in taken:
        n += 1
        name = f"{stem} ({n})"
    taken.add(name.lower())
    return name


def build_export_zip(store: Store, doc_ids: list[str], mode: str = "md-only") -> tuple[io.BytesIO, int]:
    """Returns (zip buffer, number of docs included)."""
    buf = io.BytesIO()
    taken: set[str] = set()
    count = 0
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for doc_id in doc_ids:
            doc = store.get_doc(doc_id)
            if not doc or not doc.get("markdown_path"):
                continue
            md_path = store.resolve(doc["markdown_path"])
            if not md_path.exists():
                continue
            stem = _unique(sanitize_stem(doc.get("original_name") or doc["stem"]), taken)
            markdown = md_path.read_text(encoding="utf-8")

            if mode == "md+images" and doc.get("images_dir"):
                images_dir = store.resolve(doc["images_dir"])
                images = sorted(images_dir.glob("*")) if images_dir.exists() else []
                if images:
                    images_dirname = f"{stem}_images"
                    markdown = _rewrite_image_links(markdown, images_dirname)
                    for img in images:
                        if img.is_file():
                            zf.writestr(f"{images_dirname}/{img.name}", img.read_bytes())

            zf.writestr(f"{stem}.md", markdown)
            count += 1
    buf.seek(0)
    return buf, count


def build_doc_zip(store: Store, doc_id: str) -> io.BytesIO | None:
    """Full MinerU output tree for one document."""
    doc = store.get_doc(doc_id)
    if not doc:
        return None
    out_dir = store.output_dir(doc_id)
    if not out_dir.exists():
        return None
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(out_dir.rglob("*")):
            if f.is_file():
                zf.write(f, f.relative_to(out_dir))
    buf.seek(0)
    return buf
