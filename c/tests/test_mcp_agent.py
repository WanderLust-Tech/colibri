"""The tools of a chat session (c/mcp_agent.py): the toolbox, the stream accumulator and the loop, with a scripted model.

    python -m unittest tests.test_mcp_agent
"""
import atexit
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import mcp_agent as ag  # noqa: E402
import mcp_client  # noqa: E402
import mcp_registry as reg  # noqa: E402


_TEMPS = []
atexit.register(lambda: [shutil.rmtree(folder, ignore_errors=True) for folder in _TEMPS])      # the folders the tests made


def scratch(prefix):
    folder = tempfile.mkdtemp(prefix=prefix)
    _TEMPS.append(folder)
    return folder

FAKE = str(HERE / "mcp_fake_server.py")


def server(*flags, **extra):
    return dict({"command": sys.executable, "args": [FAKE, *flags]}, **extra)


def trust_file():
    return os.path.join(scratch("coli-trust-"), "trust.json")


def box_of(servers, approve=lambda n, e: True, **kw):
    box = ag.ToolBox(servers, approve_server=approve, trust_file=kw.pop("trust_file", None) or trust_file(), **kw)
    box.open()
    return box


class Scripted:
    """A model that answers from a script: each item is (text, calls) or a function of the messages."""

    def __init__(self, *steps):
        self.steps, self.seen = list(steps), []

    def __call__(self, messages, tools):
        self.seen.append((json.loads(json.dumps(messages)), [t["function"]["name"] for t in tools]))
        step = self.steps.pop(0) if self.steps else ("(out of script)", [])
        return step(messages) if callable(step) else step


def call(name, arguments, ident="c1"):
    return {"id": ident, "name": name, "arguments": arguments if isinstance(arguments, str) else json.dumps(arguments)}


class AccumulatorTest(unittest.TestCase):
    def test_fragments_become_calls(self):
        acc = ag.Accumulator()
        acc.feed([{"index": 0, "id": "a1", "function": {"name": "fake__ec", "arguments": ""}}])
        acc.feed([{"index": 0, "function": {"name": "ho", "arguments": '{"te'}}])
        acc.feed([{"index": 1, "id": "b2", "function": {"name": "fake__add", "arguments": "{}"}}, {"index": 0, "function": {"arguments": 'xt": "hi"}'}}])
        self.assertEqual(acc.result(), [{"id": "a1", "name": "fake__echo", "arguments": '{"text": "hi"}'},
                                        {"id": "b2", "name": "fake__add", "arguments": "{}"}])

    def test_a_missing_id_is_made_up_a_nameless_call_is_dropped_and_arguments_may_be_an_object(self):
        acc = ag.Accumulator()
        acc.feed([{"function": {"name": "x", "arguments": {"a": 1}}}, {"function": {"arguments": "{}"}}, "junk"])
        self.assertEqual(acc.result(), [{"id": "call_0", "name": "x", "arguments": '{"a": 1}'}])
        self.assertEqual(ag.Accumulator().result(), [])


class ClipTest(unittest.TestCase):
    def test_the_middle_goes_and_the_cut_is_said(self):
        text = "a" * 100 + "b" * 100
        out = ag.clip_middle(text, 60)
        self.assertTrue(out.startswith("a" * 30) and out.endswith("b" * 30))
        self.assertIn("140 characters left out", out)
        self.assertEqual(ag.clip_middle("short", 60), "short")

    def test_tool_filters(self):
        self.assertTrue(ag.allowed("read_file", None, None))
        self.assertTrue(ag.allowed("read_file", ["read_*"], None))
        self.assertFalse(ag.allowed("write_file", ["read_*"], None))
        self.assertFalse(ag.allowed("read_secret", ["read_*"], ["*secret"]))
        self.assertFalse(ag.allowed("x", [], ["x"]))


class ToolBoxTest(unittest.TestCase):
    def test_tools_are_named_by_server_and_described_for_the_model(self):
        box = box_of({"fake": server()})
        try:
            self.assertEqual(sorted(box.tools), ["fake__add", "fake__crash", "fake__echo", "fake__fail", "fake__image", "fake__slow"])
            spec = next(s for s in box.specs() if s["function"]["name"] == "fake__echo")
            self.assertEqual(spec["type"], "function")
            self.assertTrue(spec["function"]["description"].startswith("[fake] Say the text back"))
            self.assertEqual(spec["function"]["parameters"]["required"], ["text"])
            self.assertTrue(box.tools["fake__echo"].read_only)
            self.assertFalse(box.tools["fake__add"].read_only)
            self.assertEqual(box.instructions(), {"fake": "be careful"})
        finally:
            box.close()

    def test_include_and_exclude_apply_to_the_servers_own_names(self):
        box = box_of({"fake": server(includeTools=["e*", "a*"], excludeTools=["add"])})
        try:
            self.assertEqual(sorted(box.tools), ["fake__echo"])
        finally:
            box.close()

    def test_two_tools_that_come_to_one_name_are_both_kept(self):
        box = ag.ToolBox({}, trust_file=trust_file())
        box._add("s", {"name": "a.b"})
        box._add("s", {"name": "a_b"})
        self.assertEqual(sorted(box.tools), ["s__a_b", "s__a_b_2"])
        self.assertEqual(box.tools["s__a_b_2"].tool, "a_b")

    def test_a_name_is_never_longer_than_64(self):
        box = ag.ToolBox({}, trust_file=trust_file())
        box._add("server", {"name": "x" * 100})
        box._add("server", {"name": "x" * 100})
        self.assertTrue(all(len(n) <= 64 for n in box.tools), box.tools.keys())
        self.assertEqual(len(box.tools), 2)

    def test_a_tool_with_no_usable_schema_gets_an_empty_one(self):
        box = ag.ToolBox({}, trust_file=trust_file())
        box._add("s", {"name": "t", "inputSchema": "nonsense"})
        self.assertEqual(box.tools["s__t"].schema, {"type": "object", "properties": {}})
        self.assertIn("t (from s)", box.tools["s__t"].description)


class ApprovalTest(unittest.TestCase):
    def test_an_unapproved_server_is_asked_about_once_and_then_remembered(self):
        asked, tf = [], trust_file()
        entry = server()
        box = ag.ToolBox({"fake": entry}, approve_server=lambda n, e: asked.append((n, reg.describe(e))) or True, trust_file=tf)
        try:
            self.assertEqual(box.open(), {"fake": None})
        finally:
            box.close()
        self.assertEqual(len(asked), 1)
        self.assertEqual(asked[0][0], "fake")
        self.assertIn("mcp_fake_server.py", asked[0][1])
        again = ag.ToolBox({"fake": entry}, approve_server=lambda n, e: self.fail("asked again"), trust_file=tf)
        try:
            self.assertEqual(again.open(), {"fake": None})
        finally:
            again.close()

    def test_a_refusal_means_the_program_is_not_started(self):
        box = ag.ToolBox({"fake": {"command": "definitely-not-run-this", "args": []}}, approve_server=lambda n, e: False, trust_file=trust_file())
        self.assertEqual(box.open(), {"fake": "not approved on this machine"})
        self.assertEqual(box.tools, {})

    def test_with_nobody_to_ask_nothing_unapproved_runs(self):
        box = ag.ToolBox({"fake": server()}, approve_server=None, trust_file=trust_file())
        self.assertEqual(box.open(), {"fake": "not approved on this machine"})

    def test_a_changed_definition_is_asked_about_again(self):
        tf, asked = trust_file(), []
        for args in ([FAKE], [FAKE, "noise"]):
            box = ag.ToolBox({"fake": {"command": sys.executable, "args": args}}, approve_server=lambda n, e: asked.append(n) or True, trust_file=tf)
            box.open()
            box.close()
        self.assertEqual(asked, ["fake", "fake"])

    def test_a_tool_filter_change_is_not_a_new_program(self):
        tf, asked = trust_file(), []
        for extra in ({}, {"excludeTools": ["add"]}, {"timeoutSeconds": 5}):
            box = ag.ToolBox({"fake": server(**extra)}, approve_server=lambda n, e: asked.append(n) or True, trust_file=tf)
            box.open()
            box.close()
        self.assertEqual(asked, ["fake"])

    def test_disabled_servers_are_skipped_without_a_question(self):
        box = ag.ToolBox({"off": server(disabled=True)}, approve_server=lambda n, e: self.fail("asked"), trust_file=trust_file())
        self.assertEqual(box.open(), {"off": "disabled"})

    def test_names_picks_some_and_says_about_one_that_is_not_there(self):
        box = ag.ToolBox({"a": server(), "b": server()}, approve_server=lambda n, e: True, trust_file=trust_file())
        try:
            self.assertEqual(box.open(["a", "ghost"]), {"a": None, "ghost": "no such server in the list"})
            self.assertTrue(all(t.server == "a" for t in box.tools.values()))
        finally:
            box.close()

    def test_a_missing_variable_is_named(self):
        entry = {"command": sys.executable, "args": [FAKE], "env": {"MY_TOKEN": "${COLI_SURELY_NOT_SET_XYZ}"}}
        box = ag.ToolBox({"s": entry}, approve_server=lambda n, e: True, environ={}, trust_file=trust_file())
        self.assertIn("COLI_SURELY_NOT_SET_XYZ is not set", box.open()["s"])

    def test_a_server_that_does_not_start_is_a_problem_not_a_crash_and_the_others_still_open(self):
        box = ag.ToolBox({"bad": server("exit-on-start"), "good": server()}, approve_server=lambda n, e: True, trust_file=trust_file())
        try:
            result = box.open()
            self.assertIn("database file is missing", result["bad"])
            self.assertIsNone(result["good"])
            self.assertIn("good__echo", box.tools)
        finally:
            box.close()


class OnDemandTest(unittest.TestCase):
    """A server with toolMode "search": its tools are not listed to the model; tools_search finds them and tools_call runs one."""

    def setUp(self):
        self.box = box_of({"fake": server(toolMode="search")})
        self.addCleanup(self.box.close)

    def run_loop(self, model, approve=None, box=None):
        messages = [{"role": "user", "content": "go"}]
        seen_calls, seen_results = [], []
        outcome = ag.run_turn(model, messages, box or self.box, approve, lambda i, a, how: seen_calls.append((i.name, a, how)),
                              lambda i, text: seen_results.append((i.name, text)))
        return outcome, messages, seen_calls, seen_results

    def tool_results(self, messages):
        return [m["content"] for m in messages if m["role"] == "tool"]

    def test_the_model_is_offered_two_small_tools_and_the_servers_tools_are_kept_back(self):
        self.assertEqual(sorted(self.box.deferred), ["fake__add", "fake__crash", "fake__echo", "fake__fail", "fake__image", "fake__slow"])
        self.assertEqual(self.box.tools, {})
        names = [s["function"]["name"] for s in self.box.specs()]
        self.assertEqual(names, [ag.SEARCH_TOOL, ag.CALL_TOOL])
        self.assertIn("fake (6 tools)", self.box.specs()[0]["function"]["description"])
        self.assertEqual(json.dumps(self.box.specs()), json.dumps(self.box.specs()), "the list is the same every time (the model's cache of the conversation stays good)")

    def test_other_servers_are_listed_as_before_beside_it(self):
        box = box_of({"fake": server(toolMode="search"), "plain": server()})
        self.addCleanup(box.close)
        names = [s["function"]["name"] for s in box.specs()]
        self.assertIn("plain__echo", names)
        self.assertNotIn("fake__echo", names)
        self.assertEqual(names[-2:], [ag.SEARCH_TOOL, ag.CALL_TOOL])

    def test_a_server_that_is_not_on_demand_adds_no_search_tools(self):
        box = box_of({"plain": server(toolMode="all")})
        self.addCleanup(box.close)
        self.assertEqual(box.meta_specs(), [])
        self.assertNotIn(ag.SEARCH_TOOL, [s["function"]["name"] for s in box.specs()])

    def test_a_search_finds_a_tool_by_what_it_does_with_its_arguments(self):
        model = Scripted(("", [call(ag.SEARCH_TOOL, {"query": "add two numbers"})]), ("found it", []))
        outcome, messages, calls, results = self.run_loop(model)
        text = self.tool_results(messages)[0]
        self.assertTrue(text.startswith("fake__add"), text)
        self.assertIn("asks the person first", text)
        self.assertIn("a: number, b: number", text, "the arguments are there in short, so the model can call it")
        self.assertIn('"a":{"type":"number"}', self.box.search("add two numbers", full=True), "and in full when asked")
        self.assertEqual(calls, [(ag.SEARCH_TOOL, {"query": "add two numbers"}, "read-only")], "a search is not asked about")

    def test_a_search_says_which_tools_are_read_only_and_how_many_more_there_are(self):
        text = self.box.search("a", limit=1)
        self.assertIn("more match", text)
        self.assertIn("read-only", self.box.search("say the text back"))
        self.assertEqual(self.box.search("echo").count("arguments:"), 1)

    def test_an_empty_search_lists_every_name(self):
        text = self.box.search("")
        self.assertIn("fake (6 tools): ", text)
        for name in self.box.deferred:
            self.assertIn(name, text)
        self.assertEqual(self.box.search(None), text)

    def test_a_search_with_no_match_says_so_and_offers_near_names_or_the_list(self):
        self.assertIn("No tool matches", self.box.search("zzzzqq"))
        self.assertIn("empty query lists", self.box.search("zzzzqq"))
        self.assertIn("ech", self.box.search("ekho"))

    def test_a_long_schema_is_short_by_default_and_cut_and_says_so_when_asked_in_full(self):
        info = next(iter(self.box.deferred.values()))
        info.schema = {"type": "object", "properties": {f"p{i}": {"type": "string", "description": "x" * 50} for i in range(200)}, "required": ["p0"]}
        short = self.box.search(info.tool)
        self.assertIn("p0*: string", short)
        self.assertIn("+186 more", short)
        self.assertLess(len(short), 1500)
        text = self.box.search(info.tool, full=True)
        self.assertIn("the rest of the schema left out", text)
        self.assertLess(len(text), ag.SEARCH_SCHEMA_CHARS + 1500)

    def test_arguments_in_short(self):
        self.assertEqual(ag.signature({"type": "object", "properties": {"path": {"type": "string"}, "detail": {"enum": ["summary", "complete"]},
                                                                       "x": {"type": ["string", "null"]}, "y": {"anyOf": []}}, "required": ["path"]}),
                         "path*: string, detail: summary|complete, x: string/null, y: any")
        self.assertEqual(ag.signature({"type": "object", "properties": {}}), "(none)")
        self.assertEqual(ag.signature(None), "(none)")
        self.assertIn("a|b|c|d|e|f|...", ag.signature({"properties": {"k": {"enum": list("abcdefgh")}}}))

    def test_a_word_in_a_name_and_a_rare_word_count_for_more(self):
        box = ag.ToolBox({})
        for name, text in (("address_to_file_offset", "Map one address to a file offset in the address space"), ("list_procedures", "List every analyzed procedure address"),
                           ("procedure_pseudo_code", "Decompile one procedure at an address"), ("search_bytes", "Find a byte pattern at an address")):
            box.deferred["s__" + name] = ag.ToolInfo("s__" + name, "s", name, f"[s] {text}", {"type": "object", "properties": {}}, True)
        first = lambda q: box.search(q, limit=1).split("  ")[0]
        self.assertEqual(first("decompile a procedure"), "s__procedure_pseudo_code")
        self.assertEqual(first("list procedures"), "s__list_procedures")
        self.assertEqual(first("pattern of bytes"), "s__search_bytes")
        self.assertEqual(first("address"), "s__address_to_file_offset", "the name that says it first")

    def test_a_match_shows_the_first_sentence_only_and_only_the_first_match_can_be_in_full(self):
        box = ag.ToolBox({})
        schema = {"type": "object", "properties": {"x": {"type": "string"}}}
        box.deferred["s__one"] = ag.ToolInfo("s__one", "s", "one", "[s] Reads the thing. " + "Then a very long second sentence. " * 30, schema, True)
        box.deferred["s__two"] = ag.ToolInfo("s__two", "s", "two", "[s] Reads the other thing. " + "More text. " * 30, schema, True)
        text = box.search("reads thing", full=True)
        self.assertNotIn("second sentence", text)
        self.assertNotIn("More text", text)
        self.assertEqual(text.count('{"type":"object"'), 1, "the second match stays short")
        self.assertEqual(text.count("x: string"), 1)
        box.deferred["s__long"] = ag.ToolInfo("s__long", "s", "long", "[s] " + "word " * 200, schema, True)
        self.assertLess(len(box.search("word").split("arguments:")[0]), ag.SEARCH_DESCRIPTION_CHARS + 60)

    def test_two_tools_whose_names_become_the_same_both_stay(self):
        box = ag.ToolBox({})
        box._add("s", {"name": "x y", "description": "d"}, deferred=True)
        box._add("s", {"name": "x_y", "description": "d"}, deferred=True)
        self.assertEqual(sorted(box.deferred), ["s__x_y", "s__x_y_2"])
        box._add("s", {"name": "x_y", "description": "d"})
        self.assertIn("s__x_y_3", box.tools)

    def test_a_word_few_tools_use_beats_a_word_in_one_tools_name_that_every_description_uses(self):
        box = ag.ToolBox({})
        schema = {"type": "object", "properties": {}}
        for i in range(200):
            box.deferred[f"s__t{i}"] = ag.ToolInfo(f"s__t{i}", "s", f"t{i}", "[s] works on an address", schema, True)
        box.deferred["s__address_tool"] = ag.ToolInfo("s__address_tool", "s", "address_tool", "[s] works on an address", schema, True)
        box.deferred["s__decompiler"] = ag.ToolInfo("s__decompiler", "s", "decompiler", "[s] works on an address and can decompile", schema, True)
        self.assertEqual(box.search("address decompile", limit=1).split("  ")[0], "s__decompiler",
                         "`address` is in every description, `decompile` in one: the rare word decides")

    def test_the_same_search_twice_is_told_it_is_the_same(self):
        first = self.box.search("add two numbers")
        again = self.box.search("add two numbers")
        self.assertNotIn("searched for exactly this", first)
        self.assertIn("searched for exactly this", again)
        self.assertTrue(again.endswith(first))
        self.assertNotIn("searched for exactly this", self.box.search("add two numbers", full=True), "a different request is not a repeat")

    def test_calling_through_runs_a_read_only_tool_without_asking(self):
        asked = []
        model = Scripted(("", [call(ag.CALL_TOOL, {"name": "fake__echo", "arguments": {"text": "hi"}})]), ("it said hi", []))
        outcome, messages, calls, results = self.run_loop(model, approve=lambda i, a: asked.append(i.name) or "no")
        self.assertEqual(self.tool_results(messages), ["hi"])
        self.assertEqual((asked, calls, results), ([], [("fake__echo", {"text": "hi"}, "read-only")], [("fake__echo", "hi")]),
                         "the screen is told about the real tool, not tools_call")

    def test_calling_through_asks_about_the_real_tool_and_a_refusal_is_a_result(self):
        asked = []
        model = Scripted(("", [call(ag.CALL_TOOL, {"name": "fake__add", "arguments": {"a": 1, "b": 2}})]), ("ok, not then", []))
        outcome, messages, calls, results = self.run_loop(model, approve=lambda i, a: asked.append((i.name, a)) or "no")
        self.assertEqual(asked, [("fake__add", {"a": 1, "b": 2})])
        self.assertIn("declined", self.tool_results(messages)[0])
        self.assertEqual(results, [], "it did not run")

    def test_yes_runs_it_and_always_covers_later_calls_through(self):
        asked = []
        answers = iter(["always"])
        model = Scripted(("", [call(ag.CALL_TOOL, {"name": "fake__add", "arguments": {"a": 1, "b": 2}}, "x1")]),
                         ("", [call(ag.CALL_TOOL, {"name": "fake__add", "arguments": {"a": 5, "b": 5}}, "x2")]), ("done", []))
        outcome, messages, calls, results = self.run_loop(model, approve=lambda i, a: asked.append(1) or next(answers))
        self.assertEqual(len(asked), 1)
        self.assertEqual(self.tool_results(messages), ['{"sum": 3}', '{"sum": 10}'])
        self.assertEqual([c[2] for c in calls], ["approved", "approved for this session"])

    def test_with_nobody_to_ask_a_tool_that_changes_things_is_refused_through_the_door_too(self):
        model = Scripted(("", [call(ag.CALL_TOOL, {"name": "fake__add", "arguments": {}})]), ("ok", []))
        _, messages, _, results = self.run_loop(model, approve=None)
        self.assertIn("declined", self.tool_results(messages)[0])
        self.assertEqual(results, [])

    def test_the_tools_own_name_is_enough_when_only_one_server_has_it_and_arguments_may_be_text(self):
        model = Scripted(("", [call(ag.CALL_TOOL, {"name": "echo", "arguments": '{"text": "bare"}'})]), ("ok", []))
        _, messages, _, _ = self.run_loop(model)
        self.assertEqual(self.tool_results(messages), ["bare"])
        two = box_of({"a": server(toolMode="search"), "b": server(toolMode="search")})
        self.addCleanup(two.close)
        self.assertIsNone(two.find("echo"), "two servers have it: the full name is needed")
        self.assertEqual(two.find("a__echo").server, "a")

    def test_a_name_that_is_not_there_and_bad_arguments_are_errors_to_correct(self):
        model = Scripted(("", [call(ag.CALL_TOOL, {"name": "fake__ech", "arguments": {}}, "1"),
                               call(ag.CALL_TOOL, {"name": "nothing"}, "2"),
                               call(ag.CALL_TOOL, {"arguments": {}}, "3"),
                               call(ag.CALL_TOOL, {"name": "fake__echo", "arguments": [1]}, "4"),
                               call(ag.CALL_TOOL, {"name": "fake__echo", "arguments": "{not json"}, "5"),
                               call(ag.CALL_TOOL, "{broken", "6"),
                               call(ag.SEARCH_TOOL, "[1]", "7")]), ("fixed", []))
        _, messages, _, results = self.run_loop(model)
        got = self.tool_results(messages)
        self.assertIn("Did you mean: fake__echo", got[0])
        self.assertIn("tools_search", got[1])
        self.assertIn("needs the `name`", got[2])
        self.assertIn("must be an object", got[3])
        self.assertIn("not valid JSON", got[4])
        self.assertIn("not valid JSON", got[5])
        self.assertIn("JSON object", got[6])
        self.assertEqual(results, [])

    def test_a_kept_back_tool_cannot_be_called_by_its_own_name(self):
        model = Scripted(("", [call("fake__echo", {"text": "sneak"})]), ("ok", []))
        _, messages, calls, results = self.run_loop(model)
        self.assertIn("there is no tool called `fake__echo`", self.tool_results(messages)[0])
        self.assertEqual((calls, results), ([], []))

    def test_without_an_on_demand_server_the_search_names_are_ordinary_unknown_tools(self):
        box = box_of({"plain": server()})
        self.addCleanup(box.close)
        messages = [{"role": "user", "content": "go"}]
        ag.run_turn(Scripted(("", [call(ag.SEARCH_TOOL, {"query": "x"})]), ("ok", [])), messages, box)
        self.assertIn("there is no tool called `tools_search`", self.tool_results(messages)[0])

    def test_a_result_through_the_door_is_cut_like_any_other(self):
        messages = [{"role": "user", "content": "go"}]
        model = Scripted(("", [call(ag.CALL_TOOL, {"name": "fake__echo", "arguments": {"text": "x" * 500}})]), ("ok", []))
        ag.run_turn(model, messages, self.box, max_chars=100)
        self.assertIn("characters left out of the middle", self.tool_results(messages)[0])

    def test_the_toolmode_setting_is_checked_and_does_not_change_what_is_approved(self):
        base = {"command": "x", "args": ["y"]}
        self.assertEqual(reg.check("s", dict(base, toolMode="search"))["toolMode"], "search")
        self.assertEqual(reg.check("s", dict(base, toolMode="all"))["toolMode"], "all")
        for bad in ("lazy", True, None, 1):
            with self.assertRaises(reg.RegistryError):
                reg.check("s", dict(base, toolMode=bad))
        self.assertEqual(reg.digest("s", base), reg.digest("s", dict(base, toolMode="search")), "how tools are offered is not what runs")


class LoopTest(unittest.TestCase):
    def setUp(self):
        self.box = box_of({"fake": server()})
        self.addCleanup(self.box.close)

    def run_loop(self, model, messages=None, approve=None, **kw):
        messages = messages if messages is not None else [{"role": "user", "content": "go"}]
        seen_calls, seen_results = [], []
        outcome = ag.run_turn(model, messages, self.box, approve, lambda i, a, how: seen_calls.append((i.name, a, how)),
                              lambda i, text: seen_results.append((i.name, text)), **kw)
        return outcome, messages, seen_calls, seen_results

    def test_a_read_only_tool_runs_and_its_result_goes_back(self):
        model = Scripted(("let me check", [call("fake__echo", {"text": "hello"})]), ("it said hello", []))
        outcome, messages, calls, results = self.run_loop(model)
        self.assertEqual(outcome, "done")
        self.assertEqual([m["role"] for m in messages], ["user", "assistant", "tool", "assistant"])
        self.assertEqual(messages[1]["tool_calls"][0], {"id": "c1", "type": "function", "function": {"name": "fake__echo", "arguments": '{"text": "hello"}'}})
        self.assertEqual(messages[2], {"role": "tool", "tool_call_id": "c1", "content": "hello"})
        self.assertEqual(calls, [("fake__echo", {"text": "hello"}, "read-only")])
        self.assertEqual(results, [("fake__echo", "hello")])
        self.assertIn("fake__echo", model.seen[0][1])
        self.assertEqual(model.seen[1][0][2]["content"], "hello", "the second request carried the result")

    def test_a_tool_that_changes_things_is_asked_about_and_a_refusal_is_a_result(self):
        asked = []
        model = Scripted(("", [call("fake__add", {"a": 1, "b": 2})]), ("fine, I will not", []))
        outcome, messages, calls, results = self.run_loop(model, approve=lambda i, a: asked.append((i.name, a)) or "no")
        self.assertEqual(asked, [("fake__add", {"a": 1, "b": 2})])
        self.assertIn("declined", messages[2]["content"])
        self.assertEqual((calls, results), ([("fake__add", {"a": 1, "b": 2}, "declined")], []), "it did not run")

    def test_yes_runs_it_and_always_stops_the_questions(self):
        asked = []
        answers = iter(["always"])
        model = Scripted(("", [call("fake__add", {"a": 1, "b": 2}, "x1")]), ("", [call("fake__add", {"a": 5, "b": 5}, "x2")]), ("done", []))
        outcome, messages, calls, results = self.run_loop(model, approve=lambda i, a: asked.append(1) or next(answers))
        self.assertEqual(len(asked), 1)
        self.assertEqual([m["content"] for m in messages if m["role"] == "tool"], ['{"sum": 3}', '{"sum": 10}'])
        self.assertEqual([c[2] for c in calls], ["approved", "approved for this session"])

    def test_with_nobody_to_ask_a_tool_that_changes_things_is_refused(self):
        model = Scripted(("", [call("fake__add", {})]), ("ok", []))
        _, messages, _, results = self.run_loop(model, approve=None)
        self.assertIn("declined", messages[2]["content"])
        self.assertEqual(results, [])

    def test_several_calls_in_one_answer_each_get_their_own_result_in_order(self):
        model = Scripted(("", [call("fake__echo", {"text": "one"}, "a"), call("fake__echo", {"text": "two"}, "b")]), ("both", []))
        _, messages, _, _ = self.run_loop(model)
        self.assertEqual([(m["tool_call_id"], m["content"]) for m in messages if m["role"] == "tool"], [("a", "one"), ("b", "two")])

    def test_a_tool_that_is_not_there_and_arguments_that_are_not_json_are_errors_to_correct(self):
        model = Scripted(("", [call("fake__ecko", {}, "a"), call("fake__echo", "{not json", "b"), call("fake__echo", "[1]", "c"), call("nope", {}, "d")]),
                         ("ok", []))
        _, messages, _, results = self.run_loop(model)
        tool = {m["tool_call_id"]: m["content"] for m in messages if m["role"] == "tool"}
        self.assertIn("no tool called `fake__ecko`", tool["a"])
        self.assertIn("Did you mean: fake__echo", tool["a"])
        self.assertIn("not valid JSON", tool["b"])
        self.assertIn("must be a JSON object", tool["c"])
        self.assertIn("The tools are:", tool["d"])
        self.assertEqual(results, [], "none of them ran")

    def test_empty_arguments_mean_none(self):
        model = Scripted(("", [call("fake__fail", "", "a")]), ("ok", []))
        _, messages, _, _ = self.run_loop(model, approve=lambda i, a: "yes")
        self.assertEqual(messages[2]["content"], "Error: it did not work")

    def test_a_tool_that_reports_an_error_says_so(self):
        model = Scripted(("", [call("fake__fail", {}, "a")]), ("ok", []))
        _, messages, _, _ = self.run_loop(model, approve=lambda i, a: "yes")
        self.assertTrue(messages[2]["content"].startswith("Error: "))

    def test_a_long_result_is_cut_in_the_middle_and_the_cut_is_said(self):
        model = Scripted(("", [call("fake__echo", {"text": "x" * 5000}, "a")]), ("ok", []))
        _, messages, _, results = self.run_loop(model, max_chars=1000)
        self.assertLess(len(messages[2]["content"]), 1200)
        self.assertIn("characters left out of the middle", messages[2]["content"])
        self.assertEqual(len(results[0][1]), 5000, "the screen is given the whole text")

    def test_a_server_that_dies_is_a_result_and_the_session_goes_on(self):
        model = Scripted(("", [call("fake__crash", {}, "a")]), ("", [call("fake__echo", {"text": "again"}, "b")]), ("sorry", []))
        _, messages, _, _ = self.run_loop(model, approve=lambda i, a: "yes")
        tool = {m["tool_call_id"]: m["content"] for m in messages if m["role"] == "tool"}
        self.assertIn("segmentation fault", tool["a"])
        self.assertTrue(tool["b"].startswith("Error:"), "the dead server stays dead and says so")

    def test_a_timeout_is_a_result(self):
        self.box.clients["fake"].call_timeout = 0.5
        model = Scripted(("", [call("fake__slow", {"seconds": 4}, "a")]), ("ok", []))
        _, messages, _, _ = self.run_loop(model, approve=lambda i, a: "yes")
        self.assertIn("did not answer", messages[2]["content"])

    def test_the_rounds_are_limited_and_the_model_is_asked_once_more_with_no_tools(self):
        model = Scripted(*[("", [call("fake__echo", {"text": str(i)}, f"c{i}")]) for i in range(3)], ("here is what I have", [call("fake__echo", {}, "late")]))
        outcome, messages, _, results = self.run_loop(model, max_rounds=3)
        self.assertEqual(outcome, "rounds")
        self.assertEqual(len(results), 3)
        self.assertEqual(model.seen[-1][1], [], "no tools are offered in the last request")
        self.assertIn("limit of 3 rounds", model.seen[-1][0][-1]["content"])
        self.assertNotIn("tool_calls", messages[-1], "a call it makes anyway is not recorded: it has no result")
        self.assertEqual(messages[-1]["content"], "here is what I have")

    def test_ctrl_c_during_a_tool_leaves_every_call_answered(self):
        model = Scripted(("", [call("fake__echo", {"text": "a"}, "a"), call("fake__echo", {"text": "b"}, "b"), call("fake__echo", {"text": "c"}, "c")]))
        seen = []

        def interrupting(info, text):
            seen.append(text)
            if len(seen) == 1:
                raise KeyboardInterrupt

        messages = [{"role": "user", "content": "go"}]
        outcome = ag.run_turn(model, messages, self.box, None, None, interrupting)
        self.assertEqual(outcome, "interrupted")
        tools = [(m["tool_call_id"], m["content"]) for m in messages if m["role"] == "tool"]
        self.assertEqual([i for i, _ in tools], ["a", "b", "c"])
        self.assertTrue(all("interrupted" in c for _, c in tools))

    def test_every_tool_call_in_the_transcript_has_a_result(self):
        model = Scripted(("", [call("fake__echo", {"text": "a"}, "a"), call("fake__ecko", {}, "b")]), ("", [call("fake__add", {}, "c")]), ("end", []))
        _, messages, _, _ = self.run_loop(model, approve=lambda i, a: "no")
        wanted = [c["id"] for m in messages if m.get("tool_calls") for c in m["tool_calls"]]
        got = [m["tool_call_id"] for m in messages if m["role"] == "tool"]
        self.assertEqual(wanted, got)


if __name__ == "__main__":
    unittest.main()
