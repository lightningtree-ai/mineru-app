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
from pathlib import Path

PDF_SUFFIXES = {".pdf"}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".jp2", ".webp", ".gif", ".bmp", ".tiff"}
OFFICE_SUFFIXES = {".docx", ".pptx", ".xlsx"}
SUPPORTED_SUFFIXES = PDF_SUFFIXES | IMAGE_SUFFIXES | OFFICE_SUFFIXES


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


def _parse_subdir(source: Path, backend: str, method: str) -> str:
    """Subdirectory MinerU writes into for one document: <out>/<stem>/<subdir>/."""
    if source.suffix.lower() in OFFICE_SUFFIXES:
        return "office"
    return method if backend == "pipeline" else "vlm"


def _locate_outputs(output_dir: Path, stem: str, parse_dir: Path) -> dict:
    """Find the artifacts MinerU wrote for one document."""
    def _find(suffix: str) -> Path | None:
        # Prefer the canonical name, else any match under the parse dir.
        exact = parse_dir / f"{stem}{suffix}"
        if exact.exists():
            return exact
        hits = list(parse_dir.glob(f"*{suffix}"))
        return hits[0] if hits else None

    md_path = _find(".md")
    content_list_path = _find("_content_list.json")
    middle_path = _find("_middle.json")
    images_dir = parse_dir / "images"

    markdown = md_path.read_text(encoding="utf-8") if md_path and md_path.exists() else ""
    content_list = []
    if content_list_path and content_list_path.exists():
        content_list = json.loads(content_list_path.read_text(encoding="utf-8"))

    return {
        "parse_dir": str(parse_dir),
        "markdown_path": str(md_path) if md_path else None,
        "markdown": markdown,
        "content_list_path": str(content_list_path) if content_list_path else None,
        "content_list": content_list,
        "middle_json_path": str(middle_path) if middle_path else None,
        "images_dir": str(images_dir) if images_dir.exists() else None,
    }


def preprocess(
    inputs: list[str | Path],
    output_dir: str | Path = "output",
    *,
    lang: str = "en",
    backend: str = "pipeline",
    method: str = "auto",
    formula: bool = True,
    table: bool = True,
    image_analysis: bool = False,
    start_page: int = 0,
    end_page: int | None = None,
    device_mode: str | None = None,
) -> list[dict]:
    """Parse one or more documents with MinerU. Models load once for the whole batch.

    Returns a list of result dicts (one per input), each containing the output paths,
    the extracted Markdown text, and the structured `content_list` blocks.
    """
    if device_mode:
        os.environ["MINERU_DEVICE_MODE"] = device_mode

    # Imported lazily: pulls in torch/transformers and is slow; keep --help fast.
    from mineru.cli.common import do_parse, read_fn
    from mineru.utils.config_reader import get_device

    files = _collect_inputs([str(i) for i in inputs])
    device = get_device()
    _log(f"[mineru-app] device={device} backend={backend} method={method} files={len(files)}")

    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    stems = [f.stem for f in files]
    pdf_bytes_list = [read_fn(f) for f in files]
    lang_list = [lang] * len(files)

    do_parse(
        output_dir=str(output_dir),
        pdf_file_names=stems,
        pdf_bytes_list=pdf_bytes_list,
        p_lang_list=lang_list,
        backend=backend,
        parse_method=method,
        formula_enable=formula,
        table_enable=table,
        image_analysis=image_analysis,
        start_page_id=start_page,
        end_page_id=end_page,
    )

    results = []
    for f, stem in zip(files, stems):
        parse_dir = output_dir / stem / _parse_subdir(f, backend, method)
        result = {"source": str(f.resolve()), "device": device, **_locate_outputs(output_dir, stem, parse_dir)}
        results.append(result)
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
    p.add_argument("-l", "--lang", default="en", help="Document language (e.g. en, ch, japan).")
    p.add_argument("-b", "--backend", default="pipeline",
                   choices=["pipeline", "vlm-transformers"],
                   help="Parsing backend. 'pipeline' is the GPU-accelerated default.")
    p.add_argument("-m", "--method", default="auto", choices=["auto", "txt", "ocr"],
                   help="Pipeline parse method.")
    p.add_argument("--no-formula", action="store_true", help="Disable formula recognition.")
    p.add_argument("--no-table", action="store_true", help="Disable table recognition.")
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
        lang=args.lang,
        backend=args.backend,
        method=args.method,
        formula=not args.no_formula,
        table=not args.no_table,
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
