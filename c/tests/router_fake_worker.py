"""A stand-in for `coli serve` that the router tests start as a worker: the same few endpoints, no model.

  python router_fake_worker.py --id text-a-colibri --port 9001 --modality text

Environment: FAKE_DEMOTE (ok, busy or no: answers POST /v1/demote like a gateway whose engine can give the card back; unset: 404 like an older one),
FAKE_LOG (a file that gets a line when a request starts, ends, or is cut off),
FAKE_LOAD_DELAY (seconds before it listens), FAKE_FAIL (exit at once, saying why).
A request body may carry `sleep` (seconds the work takes) and `stream` (answer in three chunks).
A picture or song request with `job_demo` in its body (or any, with FAKE_ENGINE=1; FAKE_STEP_SLEEP seconds a step) behaves like the real engines: `steps` progress events (`sleep` apart),
then a finished picture or song kept in COLI_IMAGE_DIR / COLI_MUSIC_DIR (FAKE_KEEP=0: not kept), or an error event if the body
says `fail`. A client that goes away mid-way is noted as `cut`, as the real worker's cancel is.
"""
import argparse
import json
import os
import re
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

args = None
LF = chr(10)
CRLF = bytes([13, 10])
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
        return {"text": ["chat", "systemone"], "image": ["image_generation"], "audio": ["music_generation"],
                "embedding": ["embeddings"]}[args.modality]

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
            if args.modality == "embedding":
                item["embedding"] = {"dimensions": 8}
            return self.reply(200, {"object": "list", "data": [item],
                                    "models": [{"name": args.id, "description": "fake", "release_date": "2026-01-01"}]})
        if self.path.split("?")[0] == "/profile":
            return self.reply(200, {"seq": 7, "turns": [], "from": args.id})
        self.reply(404, {"error": {"message": "Not found."}})

    def chunk(self, payload):
        self.wfile.write(b"%x" % len(payload) + CRLF + payload + CRLF)
        self.wfile.flush()

    def event(self, name, data):
        self.chunk(("event: " + name + LF + "data: " + json.dumps(data) + LF + LF).encode())

    def finish_stream(self):
        self.chunk(b"data: [DONE]" + LF.encode() + LF.encode())
        self.wfile.write(b"0" + CRLF + CRLF)
        self.wfile.flush()

    def job_demo(self, body):
        import base64
        import image_engine
        import music_engine
        music = self.path == "/v1/audio/music"
        steps, seed = int(body.get("steps", 3)), int(body.get("seed", 1))
        pause = float(body.get("sleep", os.environ.get("FAKE_STEP_SLEEP", 0)))
        prompt = body.get("prompt") or body.get("caption") or "x"
        stream = bool(body.get("stream"))
        events = "music_generation" if music else "image_generation"
        try:
            if stream:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
            for step in range(1, steps + 1):
                if stream:
                    self.event(events + ".progress", {"stage": "denoise", "step": step, "steps": steps, "elapsed": step * pause})
                time.sleep(pause)
            if body.get("fail"):
                error = {"error": {"message": str(body["fail"]), "type": "server_error", "param": None, "code": "engine_error"}}
                if stream:
                    self.event("error", error)
                    self.finish_stream()
                    return
                return self.reply(500, error)
            if music:
                data = music_engine.encode_wav(bytes(400), 8000, 2)
                info = {"seed": seed, "steps": steps, "duration": 0.05, "sample_rate": 8000}
            else:
                width, height = (int(n) for n in str(body.get("size", "8x6")).split("x"))
                data = image_engine.encode_png(8, 6, bytes(8 * 6 * 4), 4)
                info = {"width": width, "height": height, "seed": seed, "steps": steps}
            if os.environ.get("FAKE_KEEP", "1") != "0":
                if music:
                    info["file"] = os.path.basename(music_engine.save_wav(data, prompt, seed, music_engine.music_dir()))
                else:
                    info["file"] = os.path.basename(image_engine.save_png(data, prompt, seed, image_engine.image_dir()))
            done = {"created": int(time.time()), "data": [{"b64_json": base64.b64encode(data).decode()}], "colibri": info}
            if stream:
                self.event(events + ".completed", done)
                self.finish_stream()
            else:
                self.reply(200, done)
        except OSError:
            note("cut " + args.id)
            return
        note("end " + args.id)


    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length)
        kind = self.headers.get("Content-Type", "")
        if self.path == "/v1/demote" and os.environ.get("FAKE_DEMOTE"):
            mode = os.environ["FAKE_DEMOTE"]                      # ok, busy, or no: what the engine answers to "give the card back"
            note("demote " + args.id)
            reasons = {"busy": "a request is running or waiting", "no": "this engine cannot give the card back"}
            return self.reply(200, {"demoted": mode == "ok", "reason": None if mode == "ok" else reasons.get(mode, mode)})
        if "multipart" in kind:
            body = {"model": args.id}
            seen = raw.count(b"Content-Disposition")
            keys = sorted(m.decode() for m in re.findall(rb'name="([^"]+)"', raw))
            received = {k.decode(): v.decode("utf-8", "replace")      # the plain text fields (a file part has a filename)
                        for k, v in re.findall(rb'name="([^"]+)"\r\n\r\n(.*?)\r\n--', raw, re.S)}
        else:
            body = json.loads(raw)
            received = body
            seen = 0
            keys = sorted(body)
            if body.get("model") != args.id:
                return self.reply(404, {"error": {"message": f"The model `{body.get('model')}` does not exist.",
                                                  "type": "invalid_request_error", "param": "model",
                                                  "code": "model_not_found"}})
        note(f"start {args.id}")
        if (body.get("job_demo") or os.environ.get("FAKE_ENGINE")) and self.path in ("/v1/images/generations", "/v1/audio/music"):
            return self.job_demo(body)
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
                                 "size": length, "keys": keys, "received": received,
                                 "trace": self.headers.get("x-colibri-trace")})
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
    if os.environ.get("FAKE_FAIL_ON_CARD") and os.environ.get("COLI_VULKAN") == "1":
        print("fake worker: " + os.environ["FAKE_FAIL_ON_CARD"] + " (on the card)", file=sys.stderr, flush=True)
        sys.exit(3)
    time.sleep(float(os.environ.get("FAKE_LOAD_DELAY", "0")))
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    server.daemon_threads = True
    print(f"fake worker {args.id} listening", file=sys.stderr, flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
