"""`coli model available | install | uninstall` (model_cli.py): the flow a person sees, against a stub of the router's API and
against this machine's folders with a fake hub."""
import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import model_cli  # noqa: E402
import setup_catalog  # noqa: E402
import setup_download  # noqa: E402
from fake_hub import FakeHub  # noqa: E402

NO_COLOR = SimpleNamespace(dim="", r="", grn="", yel="")
REPO = "someone/tiny-image"


def row(**kw):
    base = {"id": "tiny-image", "name": "Tiny image", "kind": "image", "download_gb": 32.9, "status": "good", "fits": True,
            "reason": "the whole model fits in RAM (14 GB)", "license_note": "Apache-2.0", "installed": None,
            "unfinished": False, "job": None}
    base.update(kw)
    return base


class RouterStub:
    """The router's model routes, as far as the CLI uses them."""

    def __init__(self, rows=None, installed=None, jobs=None, status=200):
        self.rows, self.installed_items, self.jobs = rows or [row()], installed or [], list(jobs or [])
        self.calls, self.status = [], status
        self.removed = None
        self.folders = []

    def __call__(self, method, path, body=None):
        self.calls.append((method, path, body))
        if self.status != 200 and path == "/v1/models/catalog":
            return self.status, None
        if (method, path) == ("GET", "/v1/models/catalog"):
            return 200, {"data": self.rows, **({"folders": self.folders} if self.folders else {})}
        if (method, path) == ("GET", "/v1/models/installed"):
            return 200, {"data": self.installed_items}
        if (method, path) == ("POST", "/v1/models/install"):
            return 202, {"id": body["id"], "state": "starting"}
        if method == "GET" and path.startswith("/v1/models/installs/"):
            return 200, self.jobs.pop(0) if len(self.jobs) > 1 else self.jobs[0]
        if (method, path) == ("POST", "/v1/models/uninstall"):
            self.removed = body
            return 200, {"status": "removed", "freed_bytes": 3_000_000_000, "presets": ["terse"]}
        return 404, {"error": {"message": "not here"}}

    def posted(self, path):
        return [c for c in self.calls if c[0] == "POST" and c[1] == path]


def run(action, name=None, stub=None, yes=False, interactive=False, dest=None, **kw):
    stub = stub or RouterStub()
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        code = model_cli.run(SimpleNamespace(action=action, name=name, yes=yes, **({"dest": dest} if dest else {})), stub, NO_COLOR,
                             interactive=interactive, tty=False, poll=0, **kw)
    return code, out.getvalue(), stub


def job(state, done=0, total=1000, **kw):
    return {"id": "tiny-image", "state": state, "done": done, "total": total, "rate": 500.0, "eta_seconds": 5, "error": None,
            "dir": "/models/tiny-image", **kw}


class Available(unittest.TestCase):
    def test_lists_the_catalog_with_licence_and_state(self):
        stub = RouterStub(rows=[row(installed={"key": "k", "id": "tiny-image", "state": "idle"}),
                                row(id="big", name="Big", kind="text", download_gb=419.3, status="needs-ram", fits=False,
                                    reason="needs 120 GB of RAM", license_note="", job=job("downloading"))])
        code, text, _ = run("available", stub=stub)
        self.assertEqual(code, 0)
        self.assertRegex(text, r"tiny-image\s+image\s+32\.9 GB\s+yes\s+Apache-2\.0\s+installed")
        self.assertRegex(text, r"big\s+text\s+419\.3 GB\s+needs-ram\s+-\s+installing")
        self.assertIn("needs 120 GB of RAM", text)


class Install(unittest.TestCase):
    def test_nothing_happens_without_an_answer(self):
        code, text, stub = run("install", "tiny-image")
        self.assertEqual(code, 1)
        self.assertIn("add --yes", text)
        self.assertEqual(stub.posted("/v1/models/install"), [])

    def test_the_size_and_the_licence_are_shown_before_anything_is_fetched(self):
        code, text, stub = run("install", "tiny-image")
        self.assertIn("32.9 GB", text)
        self.assertIn("Apache-2.0", text)
        self.assertIn("fits in RAM", text)
        self.assertEqual(stub.posted("/v1/models/install"), [])

    def test_an_empty_answer_is_not_a_yes(self):
        # a null stdin that Windows calls a terminal: input() hits EOF, and that must not fetch 30 GB
        with mock.patch("builtins.input", side_effect=EOFError):
            code, text, stub = run("install", "tiny-image", interactive=True)
        self.assertEqual(code, 1)
        self.assertIn("nobody to ask", text)
        self.assertEqual(stub.posted("/v1/models/install"), [])

    def test_a_no_stops_it(self):
        with mock.patch("builtins.input", return_value="n"):
            code, text, stub = run("install", "tiny-image", interactive=True)
        self.assertEqual(code, 1)
        self.assertIn("not installed", text)
        self.assertEqual(stub.posted("/v1/models/install"), [])

    def test_it_asks_with_confirm_and_follows_the_job(self):
        stub = RouterStub(jobs=[job("downloading", 250), job("downloading", 800), job("done", 1000)])
        code, text, stub = run("install", "tiny-image", stub=stub, yes=True)
        self.assertEqual(code, 0, text)
        self.assertEqual(stub.posted("/v1/models/install")[0][2], {"id": "tiny-image", "confirm": True})
        self.assertIn("25.0%", text)
        self.assertIn("installed in /models/tiny-image", text)

    def test_dest_names_the_folder_in_the_request(self):
        stub = RouterStub(jobs=[job("done", 1000)])
        code, text, stub = run("install", "tiny-image", stub=stub, yes=True, dest="/big/models")
        self.assertEqual(code, 0, text)
        self.assertEqual(stub.posted("/v1/models/install")[0][2], {"id": "tiny-image", "confirm": True, "root": "/big/models"})

    def test_without_dest_the_request_has_no_folder(self):
        stub = RouterStub(jobs=[job("done", 1000)])
        run("install", "tiny-image", stub=stub, yes=True)
        self.assertNotIn("root", stub.posted("/v1/models/install")[0][2])

    def test_a_server_that_refuses_the_folder_is_told_to_the_user(self):
        class Refusing(RouterStub):
            def __call__(self, method, path, body=None):
                if (method, path) == ("POST", "/v1/models/install"):
                    return 409, {"error": {"message": "/nope is not one of the models folders (/a, /b)"}}
                return super().__call__(method, path, body)
        code, text, _ = run("install", "tiny-image", stub=Refusing(), yes=True, dest="/nope")
        self.assertEqual(code, 1)
        self.assertIn("is not one of the models folders", text)

    def test_no_room_in_the_default_folder_does_not_stop_a_chosen_one(self):
        full = row(status="needs-disk", fits=False, reason="needs 20 GB free; 3 GB is free")
        stub = RouterStub(rows=[full], jobs=[job("done", 1000)])
        code, text, stub = run("install", "tiny-image", stub=stub, yes=True)
        self.assertEqual(code, 1)
        self.assertEqual(stub.posted("/v1/models/install"), [])
        stub = RouterStub(rows=[full], jobs=[job("done", 1000)])
        code, text, stub = run("install", "tiny-image", stub=stub, yes=True, dest="/big")
        self.assertEqual(code, 0, text)
        self.assertEqual(stub.posted("/v1/models/install")[0][2]["root"], "/big")

    def test_available_lists_the_folders_when_there_are_several(self):
        stub = RouterStub()
        stub.folders = [{"path": "/small", "free_gb": 12.0, "preferred": False}, {"path": "/big", "free_gb": 800.0, "preferred": True}]
        code, text, _ = run("available", stub=stub)
        self.assertEqual(code, 0)
        self.assertIn("/big  (800 GB free)  <- default", text)
        self.assertIn("/small  (12 GB free)", text)
        self.assertIn("--dest FOLDER", text)
        one = RouterStub()
        one.folders = [{"path": "/only", "free_gb": 5.0, "preferred": True}]
        self.assertNotIn("/only", run("available", stub=one)[1])

    def test_a_failed_job_says_why(self):
        stub = RouterStub(jobs=[job("downloading", 100), job("failed", error="disk full")])
        code, text, _ = run("install", "tiny-image", stub=stub, yes=True)
        self.assertEqual(code, 1)
        self.assertIn("disk full", text)

    def test_what_is_already_here_is_not_fetched_again(self):
        stub = RouterStub(rows=[row(installed={"key": "k", "id": "tiny-image", "state": "idle"})])
        code, text, stub = run("install", "tiny-image", stub=stub, yes=True)
        self.assertEqual(code, 0)
        self.assertIn("already installed", text)
        self.assertEqual(stub.posted("/v1/models/install"), [])

    def test_an_unknown_id_lists_the_known_ones(self):
        code, text, _ = run("install", "nope", yes=True)
        self.assertEqual(code, 1)
        self.assertIn("tiny-image", text)

    def test_no_room_on_the_disk_is_refused_even_with_yes(self):
        stub = RouterStub(rows=[row(status="needs-disk", fits=False, reason="needs 34 GB free; 10 GB free there")])
        code, text, stub = run("install", "tiny-image", stub=stub, yes=True)
        self.assertEqual(code, 1)
        self.assertIn("needs 34 GB free", text)
        self.assertEqual(stub.posted("/v1/models/install"), [])

    def test_too_little_ram_asks_separately(self):
        stub = RouterStub(rows=[row(status="needs-ram", fits=False, reason="needs 120 GB of RAM")],
                          jobs=[job("done", 1000)])
        with mock.patch("builtins.input", side_effect=["n"]) as asked:
            code, text, stub = run("install", "tiny-image", stub=stub, interactive=True)
        self.assertEqual(code, 1)
        self.assertEqual(asked.call_count, 1)
        self.assertEqual(stub.posted("/v1/models/install"), [])


class Uninstall(unittest.TestCase):
    ITEM = {"id": "tiny-image-colibri", "key": "abc123", "name": "Tiny image", "state": "idle", "size_bytes": 3_000_000_000}

    def test_it_names_what_goes_and_asks(self):
        stub = RouterStub(installed=[self.ITEM])
        code, text, stub = run("uninstall", "tiny-image", stub=stub)
        self.assertEqual(code, 1)
        self.assertIn("frees 3.0 GB", text)
        self.assertIsNone(stub.removed)

    def test_yes_removes_it_and_reports_the_presets(self):
        stub = RouterStub(installed=[self.ITEM])
        code, text, stub = run("uninstall", "tiny-image", stub=stub, yes=True)
        self.assertEqual(code, 0, text)
        self.assertEqual(stub.removed, {"key": "abc123", "confirm": True})
        self.assertIn("3.0 GB freed", text)
        self.assertIn("terse", text)

    def test_a_loaded_model_is_refused_with_the_way_forward(self):
        stub = RouterStub(installed=[dict(self.ITEM, state="ready")])
        code, text, stub = run("uninstall", "tiny-image", stub=stub, yes=True)
        self.assertEqual(code, 1)
        self.assertIn("coli stop", text)
        self.assertIsNone(stub.removed)

    def test_the_catalog_id_names_it_too(self):
        # the router calls the folder after its family ("z-image-turbo-colibri"); the catalog calls it by the id it was fetched as
        item = dict(self.ITEM, id="z-image-turbo-colibri")
        stub = RouterStub(rows=[row(installed={"key": "abc123", "id": "z-image-turbo-colibri", "state": "idle"})],
                          installed=[item])
        code, text, stub = run("uninstall", "tiny-image", stub=stub, yes=True)
        self.assertEqual(code, 0, text)
        self.assertEqual(stub.removed, {"key": "abc123", "confirm": True})

    def test_an_unknown_name(self):
        code, text, _ = run("uninstall", "nope", stub=RouterStub(installed=[self.ITEM]), yes=True)
        self.assertEqual(code, 1)
        self.assertIn("no installed model is called nope", text)


class Reaching(unittest.TestCase):
    def test_a_server_that_wants_a_key(self):
        code, text, _ = run("available", stub=RouterStub(status=401))
        self.assertEqual(code, 1)
        self.assertIn("API key", text)

    def test_another_machine_is_never_worked_on_directly(self):
        code, text, _ = run("available", stub=RouterStub(status=404), local_ok=False)
        self.assertEqual(code, 1)
        self.assertIn("install routes", text)
        code, text, _ = run("available", stub=RouterStub(status=0), local_ok=False)
        self.assertEqual(code, 1)


class ThisMachine(unittest.TestCase):
    """No router: the same installer, here, against a models folder and a fake hub."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        src = os.path.join(self.tmp.name, "repo")
        os.makedirs(src)
        Path(src, "model_index.json").write_text(json.dumps({"_class_name": "ZImagePipeline"}))
        Path(src, "tokenizer.json").write_text("{}")
        Path(src, "weights.safetensors").write_bytes(os.urandom(200_000))
        hub = FakeHub({REPO: src})
        hub.__enter__()
        self.addCleanup(hub.__exit__)
        catalog = os.path.join(self.tmp.name, "catalog.json")
        Path(catalog).write_text(json.dumps([{
            "id": "tiny-image", "family": "zimage", "name": "Tiny image", "repo": REPO, "disk_gb": 0.001, "ram_min_gb": 0.1,
            "ram_good_gb": 0.2, "dense_gb": 0.01, "rank": 1, "size_class": "small", "summary": "fixture",
            "modality": "image", "license_note": "Apache-2.0"}]))
        for patch in (mock.patch.dict(os.environ, {"HF_ENDPOINT": hub.base, "COLI_SETUP_CATALOG": catalog}),
                      mock.patch.object(setup_download.time, "sleep")):
            patch.start()
            self.addCleanup(patch.stop)
        self.root = os.path.join(self.tmp.name, "models")
        os.makedirs(self.root)
        self.backend = model_cli.LocalBackend(roots=[self.root])

    def test_install_then_uninstall_without_a_router(self):
        code, text, _ = run("install", "tiny-image", stub=lambda *a: (None, None), yes=True, backend=self.backend)
        self.assertEqual(code, 0, text)
        self.assertTrue(setup_download.is_complete(os.path.join(self.root, "tiny-image")))
        code, text, _ = run("available", stub=lambda *a: (None, None), backend=self.backend)
        self.assertRegex(text, r"tiny-image\s+image\s+\S+ GB\s+yes\s+Apache-2\.0\s+installed")
        code, text, _ = run("uninstall", "tiny-image", stub=lambda *a: (None, None), yes=True, backend=self.backend)
        self.assertEqual(code, 0, text)
        self.assertIn("freed", text)
        self.assertEqual(os.listdir(self.root), [])


if __name__ == "__main__":
    unittest.main()
