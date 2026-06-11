#!/usr/bin/env python3
"""Generate a tiny, valid single-page PDF for smoke-testing the pipeline (no deps)."""
from __future__ import annotations

import sys
from pathlib import Path

LINES = [
    "MinerU Smoke Test",
    "",
    "Abstract. This is a minimal scientific-style PDF used to verify that the",
    "local MinerU preprocessing pipeline runs end-to-end on Apple Silicon (MPS).",
    "",
    "1. Introduction",
    "We evaluate text extraction. The Pythagorean relation a^2 + b^2 = c^2 holds.",
    "",
    "2. Results",
    "Table 1 would appear here in a real paper. Extraction should yield Markdown.",
]


def build_pdf() -> bytes:
    # Build a text stream positioning each line.
    content_lines = ["BT", "/F1 14 Tf", "72 720 Td", "16 TL"]
    for line in LINES:
        safe = line.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")
        content_lines.append(f"({safe}) Tj")
        content_lines.append("T*")
    content_lines.append("ET")
    stream = "\n".join(content_lines).encode("latin-1")

    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]

    pdf = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for i, obj in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf += f"{i} 0 obj\n".encode() + obj + b"\nendobj\n"

    xref_pos = len(pdf)
    n = len(objects) + 1
    pdf += f"xref\n0 {n}\n".encode()
    pdf += b"0000000000 65535 f \n"
    for off in offsets[1:]:
        pdf += f"{off:010d} 00000 n \n".encode()
    pdf += b"trailer\n"
    pdf += f"<< /Size {n} /Root 1 0 R >>\n".encode()
    pdf += b"startxref\n" + str(xref_pos).encode() + b"\n%%EOF\n"
    return bytes(pdf)


def main() -> int:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("examples/sample.pdf")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(build_pdf())
    print(f"wrote {out} ({out.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
