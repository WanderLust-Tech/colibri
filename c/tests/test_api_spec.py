"""docs/api-reference/openapi.json must list every route the two servers answer, with the right method, and nothing they do not.

The dashboard and the editor extension are written against these routes, and they are deployed separately from the server, so a
route that is missing from the spec (or one the spec still lists after it was removed) goes unnoticed until a client hits it.
The routes are read out of the source: the `path == "..."` tests in each server's do_GET and do_POST, and the tuple of
compute paths the router forwards. `coli serve` (openai_server.py) and `coli router` (router.py) are checked against one
spec, since a client sees either.
"""
import json
import re
import sys
import unittest
from pathlib import Path

C_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(C_DIR))

SPEC = C_DIR.parent / "docs" / "api-reference" / "openapi.json"

# Routes a server answers that are not part of the API: the dashboard's own files and the browser's preflight.
NOT_API = {"/"}
# Spec paths that stand for a family of routes the source matches by prefix (`/v1/models/<id>`), not by an exact string.
PATTERNS = {"/v1/models/{model}": ("GET", "/v1/models/"), "/v1/library/{kind}/{name}": ("GET", "/v1/library/"),
            "/v1/jobs/{job}": ("GET", "/v1/jobs/"),
            "/v1/models/installs/{id}": ("GET", "/v1/models/installs/")}


def constants(source):
    """NAME = "/path" assignments, so `path == MUSIC_ENDPOINT` can be read."""
    return dict(re.findall(r'^([A-Z_]+)\s*=\s*"(/[^"]*)"', source, re.M))


def method_body(source, method):
    """The text of `def do_<method>` up to the next method of the handler class."""
    match = re.search(r"\n    def do_%s\(self\):.*?(?=\n    def do_|\nclass |\Z)" % method, source, re.S)
    return match.group(0) if match else ""


def routes_in(source, method):
    """The paths a handler's do_<method> tests for, as a set."""
    names = constants(source)
    body = method_body(source, method)
    found = set()
    for left, right in re.findall(r'\bpath\s*==\s*(?:"(/[^"]*)"|([A-Z_]+))', body):
        found.add(left or names.get(right))
    for group in re.findall(r'\bpath\s+in\s+\(([^)]*)\)', body):
        for literal, name in re.findall(r'"(/[^"]*)"|([A-Z_]+)', group):
            found.add(literal or names.get(name))
    for prefix in re.findall(r'path\.startswith\("(/v1/[^"]*/)"\)', body):      # a family of routes: /v1/library/<kind>/<name>
        found.add(prefix)
    found.discard(None)
    return found


def source_routes():
    routes = set()
    for name in ("openai_server.py", "router.py"):
        source = (C_DIR / name).read_text(encoding="utf-8")
        for method in ("GET", "POST"):
            for path in routes_in(source, method):
                routes.add((method, path))
    import router as router_module                    # the tuple of compute paths the router forwards to a model
    for path in router_module.COMPUTE_PATHS:
        routes.add(("POST", path))
    # `path.startswith("/v1/models/")` answers GET /v1/models/<id> in both servers.
    routes.add(("GET", "/v1/models/"))
    return {(m, p) for m, p in routes if p not in NOT_API}


def docs_text():
    return "\n".join(p.read_text(encoding="utf-8") for p in sorted((C_DIR.parent / "docs").glob("*.md")))


def spec_routes():
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    out = set()
    for path, item in spec["paths"].items():
        for method in item:
            if method in ("get", "post", "put", "delete", "patch"):
                out.add((method.upper(), PATTERNS.get(path, (None, path))[1] if path in PATTERNS else path))
    return out


class ApiSpecTest(unittest.TestCase):
    def test_finds_the_routes(self):
        found = source_routes()
        # a few that must be seen, or the extractor has gone blind
        for route in (("GET", "/health"), ("POST", "/v1/chat/completions"), ("POST", "/v1/audio/music"),
                      ("GET", "/v1/logs"), ("POST", "/v1/models/switch"), ("POST", "/v1/audio/transcriptions")):
            self.assertIn(route, found)

    def test_every_route_is_in_the_spec(self):
        missing = sorted(source_routes() - spec_routes())
        self.assertEqual(missing, [], "routes the servers answer that docs/api-reference/openapi.json does not list")

    def test_the_spec_lists_no_route_that_is_gone(self):
        stale = sorted(spec_routes() - source_routes())
        self.assertEqual(stale, [], "routes in docs/api-reference/openapi.json that no server answers")

    def test_every_route_is_written_up_in_docs(self):
        text = docs_text()
        undocumented = sorted(path for _method, path in source_routes() if not path.endswith("/") and path not in text)
        self.assertEqual(undocumented, [], "routes no page under docs/ mentions")


if __name__ == "__main__":
    unittest.main()
