import json
import os
import sys
import traceback
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UI_DIR = os.path.join(BASE_DIR, "ui")
CACHE_DIR = os.path.join(sys.prefix, "cache")
TEMP_DIR = os.path.join(sys.prefix, "temp")

for directory in (
    CACHE_DIR,
    TEMP_DIR,
    os.path.join(CACHE_DIR, "huggingface"),
    os.path.join(CACHE_DIR, "torch"),
):
    os.makedirs(directory, exist_ok=True)

os.environ["HF_HOME"] = os.path.join(CACHE_DIR, "huggingface")
os.environ["TORCH_HOME"] = os.path.join(CACHE_DIR, "torch")
os.environ["TEMP"] = TEMP_DIR
os.environ["TMP"] = TEMP_DIR

from generator import generate_answer, prepare_evidence
from query import search


class UIRequestHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=UI_DIR, **kwargs)

    def _send_json(self, payload, status=200):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/health":
            self._send_json({"ok": True})
            return
        super().do_GET()

    def do_POST(self):
        if self.path != "/api/ask":
            self._send_json({"error": "Not found"}, status=404)
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._send_json({"error": "Invalid content length"}, status=400)
            return

        raw_body = self.rfile.read(content_length) if content_length else b"{}"
        try:
            payload = json.loads(raw_body.decode("utf-8") or "{}")
        except json.JSONDecodeError:
            self._send_json({"error": "Invalid JSON"}, status=400)
            return

        if not isinstance(payload, dict):
            self._send_json({"error": "The request body must be a JSON object"}, status=400)
            return

        question = str(payload.get("question", "")).strip()
        if not question:
            self._send_json({"error": "Please enter a question"}, status=400)
            return

        try:
            retrieved = search(question, mode="hybrid", use_reranker=False)
        except Exception as exc:
            traceback.print_exc()
            self._send_json({"error": str(exc)}, status=502)
            return

        try:
            result = generate_answer(question, retrieved)
        except Exception as exc:
            traceback.print_exc()
            sources, evidence = prepare_evidence(retrieved)
            result = {
                "answer": (
                    "Relevant evidence was retrieved, but answer generation failed: "
                    f"{exc}\n\nRetrieved evidence:\n\n{evidence}"
                ),
                "sources": sources,
                "warnings": ["The answer is a retrieval fallback, not a generated response."],
            }

        self._send_json(
            {
                "answer": result.get("answer") or "No answer generated.",
                "sources": result.get("sources") or [],
                "warnings": result.get("warnings") or [],
            }
        )


def main():
    port = int(os.getenv("PORT", "8000"))
    server = ThreadingHTTPServer(("127.0.0.1", port), UIRequestHandler)
    print(f"RAG UI is running on http://localhost:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Stopping RAG UI...")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
