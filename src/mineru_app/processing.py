#!/usr/bin/env python3
"""Preprocess documents into agent-ready Markdown + structured JSON using MinerU.

Use it three ways:

  Manually (CLI):
      python mineru_preprocess.py paper.pdf -o output
      python mineru_preprocess.py ./papers/ -o output --json

  Agentically (import):
      from mineru_app.processing import preprocess_pdf
      result = preprocess_pdf("paper.pdf")
      markdown = result["markdown"]            # feed to an LLM
      blocks   = result["content_list"]        # structured text/figure/table/equation blocks

  Web UI:
      mineru-app   # drag-and-drop server, see mineru_app.server

Device selection is automatic (cuda -> mps -> cpu), so the same code runs on
Windows/Linux CUDA boxes, Apple Silicon, and CPU-only machines. Override per call
with device_mode=..., or globally with the MINERU_DEVICE_MODE env var.
"""
from __future__ import annotations

import os
import sys

# --- Compute-device config. MUST be set before MinerU/torch are imported. ---
if sys.platform == "darwin":
    # Let ops not yet implemented on Apple-Silicon MPS fall back to CPU instead
    # of crashing. Harmless elsewhere but only relevant on macOS.
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
# Model weights are fetched on first run; default source is HuggingFace.
os.environ.setdefault("MINERU_MODEL_SOURCE", "huggingface")

import argparse
import json
import re
from pathlib import Path

PDF_SUFFIXES = {".pdf"}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".jp2", ".webp", ".gif", ".bmp", ".tiff"}
OFFICE_SUFFIXES = {".docx", ".pptx", ".xlsx"}
SUPPORTED_SUFFIXES = PDF_SUFFIXES | IMAGE_SUFFIXES | OFFICE_SUFFIXES

# Parsing quality tiers, cheapest first. `flash` reads the PDF text layer with no
# inference models at all; `basic` adds the small OCR/formula/table models; the last
# two add a vision-language model on top.
TIERS = ("flash", "basic", "standard", "advanced")
OCR_MODES = ("auto", "txt", "ocr")

# Filenames in the parse directory. save() writes the first two; we add the third.
MARKDOWN_NAME = "markdown.md"
STRUCTURED_NAME = "structured_content.json"
CONTENT_LIST_NAME = "content_list.json"

_MD_IMAGE = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")


def _log(*args, **kwargs):
    """Print progress to stderr so stdout stays clean for --json output."""
    print(*args, file=sys.stderr, **kwargs)


def _collect_inputs(paths: list[str]) -> list[Path]:
    """Expand files / directories into a flat list of supported document paths."""
    out: list[Path] = []
    for p in paths:
        path = Path(p).expanduser()
        if path.is_dir():
            out.extend(sorted(f for f in path.rglob("*") if f.suffix.lower() in SUPPORTED_SUFFIXES))
        elif path.is_file():
            out.append(path)
        else:
            raise FileNotFoundError(f"Input not found: {path}")
    if not out:
        raise ValueError("No supported document inputs found in the given paths.")
    return out


def _page_range(start_page: int, end_page: int | None) -> str:
    """Our 0-indexed inclusive bounds -> MinerU's 1-based range string.

    MinerU spells the last page `r1`, which is how an open-ended range is written.
    An empty string means the whole document.
    """
    start = max(0, int(start_page or 0))
    if end_page is None:
        return "" if start == 0 else f"{start + 1}-r1"
    return f"{start + 1}-{int(end_page) + 1}"


# MinerU 4 splits 3.x's single `text` type by role. Fold the two title types back into
# text + text_level so the Blocks tab keeps rendering them the way it always has.
_TITLE_LEVELS = {"doc_title": 1, "paragraph_title": 2}


def _flatten_blocks(structured: dict) -> list[dict]:
    """MinerU 4's page tree -> the flat block list this app publishes.

    `structured` is {"pages": [{"page_idx": n, "blocks": [...]}, ...]}. The web UI and
    any agentic caller want one ordered list carrying the field names the app has always
    used, so the adapter lives here and nothing downstream changes.
    """
    out: list[dict] = []
    for page in structured.get("pages", []):
        page_idx = page.get("page_idx")
        for block in page.get("blocks", []):
            btype = block.get("type")
            content = block.get("content", "")
            item = {"type": btype, "page_idx": page_idx}
            if "bbox" in block:
                item["bbox"] = block["bbox"]

            if btype in _TITLE_LEVELS:
                item["type"] = "text"
                item["text_level"] = block.get("level", _TITLE_LEVELS[btype])
            elif "level" in block:
                item["text_level"] = block["level"]
            item["text"] = content

            # Visual blocks carry a relative path once save() has externalised assets.
            source = block.get("image_source")
            if source and not source.startswith("data:"):
                item["img_path"] = source
            elif content:
                hit = _MD_IMAGE.search(content)
                if hit and not hit.group(1).startswith("data:"):
                    item["img_path"] = hit.group(1)

            captions = block.get("captions") or []
            footnotes = block.get("footnotes") or []
            if btype == "table":
                item["table_caption"], item["table_footnote"] = captions, footnotes
                item["table_body"] = content
            elif btype in ("image", "chart"):
                item[f"{btype}_caption"], item[f"{btype}_footnote"] = captions, footnotes
            elif captions or footnotes:
                item["captions"], item["footnotes"] = captions, footnotes

            out.append(item)
    return out


def _locate_outputs(parse_dir: Path) -> dict:
    """Read back what we wrote for one document."""
    md_path = parse_dir / MARKDOWN_NAME
    content_list_path = parse_dir / CONTENT_LIST_NAME
    images_dir = parse_dir / "images"

    markdown = md_path.read_text(encoding="utf-8") if md_path.exists() else ""
    content_list = []
    if content_list_path.exists():
        content_list = json.loads(content_list_path.read_text(encoding="utf-8"))

    return {
        "parse_dir": str(parse_dir),
        "markdown_path": str(md_path) if md_path.exists() else None,
        "markdown": markdown,
        "content_list_path": str(content_list_path) if content_list_path.exists() else None,
        "content_list": content_list,
        "images_dir": str(images_dir) if images_dir.exists() else None,
    }


def preprocess(
    inputs: list[str | Path],
    output_dir: str | Path = "output",
    *,
    tier: str = "basic",
    ocr_mode: str = "auto",
    image_analysis: bool = False,
    start_page: int = 0,
    end_page: int | None = None,
    device_mode: str | None = None,
) -> list[dict]:
    """Parse one or more documents with MinerU. Models load once for the whole batch.

    Returns a list of result dicts (one per input), each containing the output paths,
    the extracted Markdown text, and the structured `content_list` blocks.
    """
    if tier not in TIERS:
        raise ValueError(f"Unknown tier {tier!r}. Choose one of: {', '.join(TIERS)}")
    if ocr_mode not in OCR_MODES:
        raise ValueError(f"Unknown ocr_mode {ocr_mode!r}. Choose one of: {', '.join(OCR_MODES)}")
    if device_mode:
        os.environ["MINERU_DEVICE_MODE"] = device_mode

    # Imported lazily: pulls in torch/transformers and is slow; keep --help fast.
    from mineru.parser import parse
    from mineru.parser.writer import FileBasedDataWriter
    from mineru.model.runtime.device import get_device

    files = _collect_inputs([str(i) for i in inputs])
    device = get_device()
    page_range = _page_range(start_page, end_page)
    _log(f"[mineru-app] device={device} tier={tier} ocr_mode={ocr_mode} files={len(files)}")

    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for f in files:
        parse_dir = output_dir / f.stem / tier
        parse_dir.mkdir(parents=True, exist_ok=True)

        result = parse(
            str(f),
            tier=tier,
            ocr_mode=ocr_mode,
            image_analysis=image_analysis,
            page_range=page_range,
        )
        # save() externalises the images, so the Markdown references images/<name>
        # rather than inlining megabytes of base64. It writes markdown.md,
        # middle_json.json, structured_content.json and the image bytes.
        result.save(FileBasedDataWriter(str(parse_dir)))
        # Flatten the file save() just wrote, not result.structured_content(). save()
        # materialises a copy, so the live object still carries base64 data: URIs where
        # the written file has images/<name> paths.
        structured = json.loads((parse_dir / STRUCTURED_NAME).read_text(encoding="utf-8"))
        (parse_dir / CONTENT_LIST_NAME).write_text(
            json.dumps(_flatten_blocks(structured), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        # save() also left MinerU's intermediate protocol and the raw model dump behind.
        # Neither is read again by this app or the web UI, and together they run to
        # megabytes per document. Dropped once the outputs above exist, so a crash
        # mid-parse never costs a file we still needed. Keep them with
        # MINERU_APP_KEEP_INTERMEDIATE=1 to reproduce an upstream bug.
        if not os.environ.get("MINERU_APP_KEEP_INTERMEDIATE"):
            for name in ("middle_json.json", "model_output.json"):
                (parse_dir / name).unlink(missing_ok=True)

        results.append({"source": str(f.resolve()), "device": device, **_locate_outputs(parse_dir)})
    return results


def preprocess_pdf(pdf_path: str | Path, output_dir: str | Path = "output", **kwargs) -> dict:
    """Convenience wrapper for a single document. Returns one result dict."""
    return preprocess([pdf_path], output_dir, **kwargs)[0]


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Preprocess documents into agent-ready Markdown + JSON with MinerU.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("inputs", nargs="+", help="Document file(s) or directory(ies) to process.")
    p.add_argument("-o", "--output", default="output", help="Output directory.")
    p.add_argument("-t", "--tier", default="basic", choices=list(TIERS),
                   help="Parsing quality. 'flash' skips all models; 'standard' and "
                        "'advanced' add a vision-language model and are much slower.")
    p.add_argument("-m", "--ocr-mode", default="auto", choices=list(OCR_MODES),
                   help="How text is read: auto-detect, force the text layer, or force OCR.")
    p.add_argument("--image-analysis", action="store_true",
                   help="Enable figure captioning/analysis (extra models, slower).")
    p.add_argument("-s", "--start-page", type=int, default=0, help="First page (0-indexed).")
    p.add_argument("-e", "--end-page", type=int, default=None, help="Last page (0-indexed, inclusive).")
    p.add_argument("--device", default=None,
                   help="Override compute device: cuda | mps | cpu (default: auto-detect).")
    p.add_argument("--json", action="store_true",
                   help="Emit a JSON summary (paths + stats) to stdout for agentic use.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    results = preprocess(
        args.inputs,
        output_dir=args.output,
        tier=args.tier,
        ocr_mode=args.ocr_mode,
        image_analysis=args.image_analysis,
        start_page=args.start_page,
        end_page=args.end_page,
        device_mode=args.device,
    )

    if args.json:
        # Compact, machine-readable: omit the (potentially huge) inline markdown/content_list.
        summary = [
            {
                "source": r["source"],
                "device": r["device"],
                "parse_dir": r["parse_dir"],
                "markdown_path": r["markdown_path"],
                "content_list_path": r["content_list_path"],
                "images_dir": r["images_dir"],
                "markdown_chars": len(r["markdown"]),
                "blocks": len(r["content_list"]),
            }
            for r in results
        ]
        print(json.dumps(summary, indent=2))
    else:
        for r in results:
            _log("")
            _log(f"  source : {r['source']}")
            _log(f"  device : {r['device']}")
            _log(f"  markdown : {r['markdown_path']}  ({len(r['markdown'])} chars)")
            _log(f"  blocks   : {r['content_list_path']}  ({len(r['content_list'])} blocks)")
            _log(f"  images   : {r['images_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
