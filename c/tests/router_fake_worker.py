"""A stand-in for `coli serve` that the router tests start as a worker: the same few endpoints, no model.

  python router_fake_worker.py --id text-a-colibri --port 9001 --modality text

Environment: FAKE_LOG (a file that gets a line when a request starts, ends, or is cut off),
FAKE_LOAD_DELAY (seconds before it listens), FAKE_FAIL (exit at once, saying why).
A request body may carry `sleep` (seconds the work takes) and `stream` (answer in three chunks).
"""
import argparse
import json
import os
import re
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

args = None
log_lock = threading.Lock()


def note(text):
    path = os.environ.get("FAKE_LOG")
    if not path:
        return
    with log_lock, open(path, "a", encoding="utf-8") as handle:
        handle.write(f"{time.time():.4f} {text}\n")


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *rest):
        pass

    def handle(self):
        try:
            super().handle()
        except ConnectionError:               # the router closed a connection it was done with
            pass

    def reply(self, status, body):
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")           # the router must not pass this on
        self.end_headers()
        self.wfile.write(data)

    def caps(self):
        return {"text": ["chat", "systemone"], "image": ["image_generation"], "audio": ["music_generation"]}[args.modality]

    def do_GET(self):
        if self.path.split("?")[0] == "/health":
            return self.reply(200, {"status": "ok", "arch": "fake", "kv_slots": 1, "capabilities": self.caps(),
                                    "scheduler": {"active": 0, "queued": 0, "capacity": 1}})
        if self.path.split("?")[0] == "/v1/models":
            item = {"id": args.id, "object": "model", "owned_by": "colibri", "capabilities": self.caps()}
            if args.modality == "image":
                item["image"] = {"default_width": 640}                  # what only the engine could say
            if args.modality == "audio":
                item["audio"] = {"tasks": ["text2music", "cover"]}
            return self.reply(200, {"object": "list", "data": [item],
                                    "models": [{"name": args.id, "description": "fake", "release_date": "2026-01-01"}]})
        if self.path.split("?")[0] == "/profile":
            return self.reply(200, {"seq": 7, "turns": [], "from": args.id})
        self.reply(404, {"error": {"message": "Not found."}})

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length)
        kind = self.headers.get("Content-Type", "")
        if "multipart" in kind:
            body = {"model": args.id}
            seen = raw.count(b"Content-Disposition")
            keys = sorted(m.decode() for m in re.findall(rb'name="([^"]+)"', raw))
        else:
            body = json.loads(raw)
            seen = 0
            keys = sorted(body)
            if body.get("model") != args.id:
                return self.reply(404, {"error": {"message": f"The model `{body.get('model')}` does not exist.",
                                                  "type": "invalid_request_error", "param": "model",
                                                  "code": "model_not_found"}})
        note(f"start {args.id}")
        origin = self.headers.get("Origin"), self.headers.get("Authorization"), self.headers.get("X-Test")
        try:
            if body.get("stream"):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                for index in range(int(body.get("chunks", 3))):
                    payload = f"data: {json.dumps({'model': args.id, 'n': index})}\n\n".encode()
                    self.wfile.write(b"%x\r\n%s\r\n" % (len(payload), payload))
                    self.wfile.flush()
                    time.sleep(float(body.get("sleep", 0)))
                tail = b"data: [DONE]\n\n"
                self.wfile.write(b"%x\r\n%s\r\n0\r\n\r\n" % (len(tail), tail))
                self.wfile.flush()
            else:
                time.sleep(float(body.get("sleep", 0)))
                self.reply(200, {"model": args.id, "pid": os.getpid(), "path": self.path, "parts": seen,
                                 "origin": origin[0], "authorization": origin[1], "x_test": origin[2],
                                 "size": length, "keys": keys})
        except OSError:
            note(f"cut {args.id}")
            return
        note(f"end {args.id}")


def main():
    global args
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--modality", default="text")
    args = parser.parse_args()
    if os.environ.get("FAKE_FAIL"):
        print("fake worker: " + os.environ["FAKE_FAIL"], file=sys.stderr, flush=True)
        sys.exit(3)
    time.sleep(float(os.environ.get("FAKE_LOAD_DELAY", "0")))
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    server.daemon_threads = True
    print(f"fake worker {args.id} listening", file=sys.stderr, flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
