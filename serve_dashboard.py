from __future__ import annotations

import http.server
import json
import socketserver
import sys
from pathlib import Path

from resolve_issue import resolve


DEFAULT_PORT = 8765
OUTPUT_DIR = Path(__file__).resolve().parent / "ai_output"


class Handler(http.server.SimpleHTTPRequestHandler):
    def do_POST(self):
        if self.path != "/api/resolve":
            self.send_error(404, "Not found")
            return

        content_length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(content_length)
        try:
            payload = json.loads(body.decode("utf-8"))
            query = str(payload.get("query", "")).strip()
            if not query:
                raise ValueError("Missing query")
            result = resolve(query)
            data = json.dumps(result, ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except Exception as exc:
            data = json.dumps({"error": str(exc)}, ensure_ascii=False).encode("utf-8")
            self.send_response(400)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(OUTPUT_DIR), **kwargs)


def main() -> None:
    if not (OUTPUT_DIR / "dashboard.html").exists():
        raise SystemExit("Run pdf_ai_classifier.py first to generate ai_output/dashboard.html")

    port = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PORT
    with socketserver.TCPServer(("", port), Handler) as httpd:
        print(f"Dashboard available at http://localhost:{port}/dashboard.html")
        httpd.serve_forever()


if __name__ == "__main__":
    main()
