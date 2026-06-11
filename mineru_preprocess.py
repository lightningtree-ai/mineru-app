#!/usr/bin/env python3
"""Back-compat shim: the implementation moved to mineru_app.processing.

Existing usage keeps working:
    python mineru_preprocess.py paper.pdf -o output
    from mineru_preprocess import preprocess_pdf
"""
from mineru_app.processing import (  # noqa: F401
    PDF_SUFFIXES,
    SUPPORTED_SUFFIXES,
    _collect_inputs,
    _locate_outputs,
    _log,
    main,
    preprocess,
    preprocess_pdf,
)

if __name__ == "__main__":
    raise SystemExit(main())
