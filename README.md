# mineru-app

A local web app for turning **PDFs (and docx / pptx / xlsx / images) into agent-ready
Markdown + structured JSON** with [MinerU](https://github.com/opendatalab/MinerU).
Drag in files or whole folders (e.g. a Zotero export), watch the processing queue live,
spot-check the extraction side-by-side with the original, and export a flat zip of `.md`
files ready to drop into Claude Desktop or any other LLM tool.

Everything runs on your machine — no documents leave it. GPU acceleration is automatic:
CUDA on Windows/Linux, Metal (MPS) on Apple Silicon, CPU otherwise.

## Quick start

Install [uv](https://docs.astral.sh/uv/) once, then:

**macOS / Linux**
```bash
git clone https://github.com/LTAdvancedMaterials/mineru-app && cd mineru-app
uv sync
uv run mineru-app
```

**Windows (PowerShell)**
```powershell
git clone https://github.com/LTAdvancedMaterials/mineru-app; cd mineru-app
uv sync
uv run mineru-app
```

That's it — `uv sync` provisions Python 3.13 and all dependencies (on Windows it pulls
CUDA-enabled PyTorch automatically; you only need a reasonably current NVIDIA driver, not
the CUDA toolkit). `uv run mineru-app` starts the server on <http://127.0.0.1:8008> and
opens your browser.

> **First document is slow**: MinerU downloads its layout/formula/table/OCR models from
> HuggingFace into `~/.cache` on first use (a few GB). After that, everything is offline
> and fast — the models stay loaded while the app runs.

## Using it

1. **Drop** files or an entire folder onto the drop zone (a folder of PDFs exported from
   Zotero works great — subfolders are walked recursively).
2. Tweak **Options** if needed (language, OCR method, page range, formula/table toggles,
   device override). Defaults are right for English scientific PDFs.
3. **Process** — each file becomes a queue entry with live status. One bad PDF doesn't
   stop the rest.
4. Click any finished document to open the viewer:
   - **Markdown** — rendered output with figures and LaTeX math
   - **Side-by-side** — original PDF next to the extraction, for QA
   - **Blocks** — the structured `content_list` (type / page / content), filterable
   - **Raw** — the markdown source with one-click copy
5. Select documents in the **Library** (checkboxes) and export:
   - **md** — flat zip of `Name.md` files, named after the original uploads
   - **md + images** — same, plus each doc's figures in `Name_images/` with the
     markdown links rewritten to match

The unzipped export folder can be dropped straight into Claude Desktop, a RAG ingestion
folder, an Obsidian vault, etc.

Processed documents persist in `./data/` (uploads, outputs, `manifest.json`) and the
library survives restarts. Override the location with `--data DIR` or `MINERU_APP_DATA`.

```
mineru-app --help
  --host 127.0.0.1   bind address (localhost only by default)
  --port 8008
  --data DIR         data directory (default ./data)
  --no-browser
```

## CLI / agentic use

The web app is a wrapper around `mineru_app.processing`, which you can use directly:

```bash
# one PDF
uv run python mineru_preprocess.py paper.pdf -o output

# a whole folder, machine-readable summary on stdout
uv run python mineru_preprocess.py ./papers/ -o output --json
```

```python
from mineru_app.processing import preprocess_pdf, preprocess

result = preprocess_pdf("paper.pdf", output_dir="output")
markdown = result["markdown"]         # full Markdown — feed straight to an LLM
blocks   = result["content_list"]     # structured blocks: text / image / table / equation

# batch: models load once for the whole list
results = preprocess(["a.pdf", "b.pdf"], output_dir="output")
```

CLI options: `-l/--lang`, `-b/--backend` (`pipeline` | `vlm-transformers`), `-m/--method`
(`auto`/`txt`/`ocr`), `--no-formula`, `--no-table`, `-s/-e` page range, `--device`
(`cuda`/`mps`/`cpu`, default auto-detect), `--json`.

## Output layout

For `paper.pdf`, MinerU writes `paper.md`, `paper_content_list.json` (ordered structured
blocks with page numbers), `paper_middle.json` (full layout model), and `images/` into
`<output>/<stem>/<method>/` (`office/` for docx/pptx/xlsx, `vlm/` for VLM backends).

## How it works

- **Backend**: FastAPI ([src/mineru_app/server.py](src/mineru_app/server.py)) + a single
  worker thread ([jobs.py](src/mineru_app/jobs.py)) that runs MinerU jobs sequentially —
  models load once and stay warm. Live updates stream to the UI over SSE.
- **Frontend**: no-build vanilla JS ([src/mineru_app/static/](src/mineru_app/static/))
  with vendored `marked`, KaTeX, and pdf.js — works fully offline, no Node required.
- **Processing**: [processing.py](src/mineru_app/processing.py) wraps MinerU's `do_parse`;
  device auto-detects `cuda → mps → cpu` (`PYTORCH_ENABLE_MPS_FALLBACK` is set on macOS so
  unsupported MPS ops fall back to CPU instead of crashing).

## Troubleshooting

- **`uv sync` is huge on Windows** — that's the CUDA PyTorch build (~2.5 GB). It also
  runs fine on machines without an NVIDIA GPU (select device `cpu`, or let auto-detect
  handle it).
- **Out-of-memory on GPU** — set device to `cpu` in Options, or process fewer/smaller
  files; the queue is sequential so memory pressure doesn't stack.
- **A document fails** — the queue entry shows the error; the rest of the batch
  continues. The `pipeline` backend supports PDFs/images on all platforms; office files
  are converted natively (no LibreOffice needed).
