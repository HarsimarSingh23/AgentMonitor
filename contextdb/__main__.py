"""``python -m contextdb`` — run the live dashboard server in the foreground.

Useful as the always-on collector that external producers (Claude Code hooks,
log shippers) POST events into via ``/ingest``.

    python -m contextdb            # serve on http://127.0.0.1:8765
    python -m contextdb --port 9000 --no-browser
"""

from __future__ import annotations

import argparse

from .dashboard import serve


def main() -> None:
    ap = argparse.ArgumentParser(prog="contextdb", description="contextdb live dashboard server")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--no-browser", action="store_true", help="don't open a browser")
    args = ap.parse_args()
    serve(port=args.port, host=args.host, open_browser=not args.no_browser, block=True)


if __name__ == "__main__":
    main()
