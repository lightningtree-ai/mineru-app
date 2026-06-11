"""Entry point: `mineru-app` (or `python -m mineru_app`).

Starts the local server and opens the browser. Heavy MinerU/torch imports
happen lazily in the worker thread, so startup is fast.
"""
from __future__ import annotations

import argparse
import os
import threading
import webbrowser


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="mineru-app",
                                description="Local drag-and-drop MinerU preprocessing web app.")
    p.add_argument("--host", default="127.0.0.1", help="Bind address (default: localhost only).")
    p.add_argument("--port", type=int, default=8008, help="Port (default: 8008).")
    p.add_argument("--data", default=None, help="Data directory (default: ./data or $MINERU_APP_DATA).")
    p.add_argument("--no-browser", action="store_true", help="Don't open a browser tab.")
    args = p.parse_args(argv)

    if args.data:
        os.environ["MINERU_APP_DATA"] = args.data

    import uvicorn

    url = f"http://{'127.0.0.1' if args.host == '0.0.0.0' else args.host}:{args.port}"
    if not args.no_browser:
        threading.Timer(1.0, webbrowser.open, args=(url,)).start()
    print(f"mineru-app: {url}  (data dir: {os.environ.get('MINERU_APP_DATA', './data')})")

    uvicorn.run("mineru_app.server:app", host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
