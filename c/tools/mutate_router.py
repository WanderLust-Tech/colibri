#!/usr/bin/env python3
"""Proof that the router's tests can fail: break router.py one line at a time and see that the tests say so.

    mutate_router.py [--only NAME,NAME] [--timeout SECONDS] [--skip-baseline]

A mutant is a small, plausible mistake in the router: the gate that lets one model compute at a time, the memory check
and the order of eviction, the rule that a busy model is never stopped, keep-alive, the rewrite of the model name, the
guards, streaming, cancelling for a client that left, and what a failed load leaves behind. Each is applied to
router.py in place, run against the tests that are meant to catch it, and the file is put back (and checked) afterwards.
CAUGHT when any of them fails or hangs. The unmodified file is run first and has to pass. Exits non-zero if a mutant
survives."""
import argparse
import hashlib
import os
import subprocess
import sys
import time

C_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TARGET = os.path.join(C_DIR, "router.py")
T = "tests.test_router."

# name: (text in router.py, its replacement, [the tests that must notice])
MUTANTS = {
    # --- one model at a time -----------------------------------------------------------------------------------------
    "gate-lets-models-overlap": (
        "        if any(k != key and n > 0 for k, n in self.active.items()):\n            return False\n", "",
        [T + "OneModelAtATimeTest.test_two_models_never_compute_together"]),
    "gate-queue-unbounded": (
        "if len(self.waiting) >= self.max_queue:", "if False:",
        [T + "OneModelAtATimeTest.test_waiting_requests_are_served_in_order_and_the_queue_is_bounded"]),
    "gate-never-times-out": (
        "deadline = time.monotonic() + self.timeout", "deadline = time.monotonic() + 1e9",
        [T + "OneModelAtATimeTest.test_a_request_that_waits_too_long_times_out"]),
    # (a waiting request that ignores same-model waiters ahead of it is not a mutant: they start the moment it is its
    # turn anyway, so the requests still overlap and nothing observable changes.)
    # --- making room ---------------------------------------------------------------------------------------------------
    "evicts-a-model-in-use": (
        'victims = [w for w in others if w.state == "ready" and w.inflight == 0]',
        'victims = [w for w in others if w.state == "ready"]',
        [T + "MakingRoomTest.test_a_model_in_use_is_never_stopped_to_make_room"]),
    "no-memory-check": (
        "return free is None or self.estimate(entry) <= free * FIT_HEADROOM", "return True",
        [T + "MakingRoomTest.test_a_model_that_does_not_fit_stops_the_idle_one"]),
    "no-headroom": (
        "FIT_HEADROOM = 0.8 ", "FIT_HEADROOM = 1.0 ",
        [T + "MakingRoomTest.test_the_headroom_counts"]),
    "evicts-the-newest": (
        "victim = min(victims, key=lambda w: w.last_used)", "victim = max(victims, key=lambda w: w.last_used)",
        [T + "MakingRoomTest.test_the_least_recently_used_goes_first"]),
    "ignores-the-model-limit": (
        "room = self.max_loaded <= 0 or len(counted) < self.max_loaded", "room = True",
        [T + "MakingRoomTest.test_a_limit_on_loaded_models_is_kept"]),
    "gives-up-when-nothing-fits": (
        "        if not others:                                   # nothing left to free: try anyway\n",
        "        if False:\n",
        [T + "MakingRoomTest.test_a_model_bigger_than_memory_is_loaded_anyway_when_nothing_else_is"]),
    "forgets-what-a-model-took": (
        "        known = self.state.resident(entry[\"key\"])\n", "        known = None\n",
        [T + "MakingRoomTest.test_what_a_model_really_took_is_remembered"]),
    # --- keep-alive ------------------------------------------------------------------------------------------------------
    "stops-a-model-in-use": (
        'elif (worker.state == "ready" and worker.inflight == 0 and worker.keep_alive is not None',
        'elif (worker.state == "ready" and worker.keep_alive is not None',
        [T + "KeepAliveTest.test_a_model_in_use_is_not_stopped"]),
    "forever-is-not-forever": (
        "worker.keep_alive is not None\n                      and now - worker.last_used >= worker.keep_alive",
        "True\n                      and now - worker.last_used >= (worker.keep_alive or 0)",
        [T + "KeepAliveTest.test_forever_keeps_it"]),
    "release-forgets-the-clock": (
        "            worker.inflight = max(0, worker.inflight - 1)\n            worker.last_used = time.monotonic()\n",
        "            worker.inflight = max(0, worker.inflight - 1)\n",
        [T + "KeepAliveTest.test_the_clock_runs_from_the_end_of_a_request_not_its_start"]),
    "idle-models-stay": (
        "and now - worker.last_used >= worker.keep_alive", "and now - worker.last_used >= worker.keep_alive + 1e9",
        [T + "KeepAliveTest.test_an_idle_model_is_stopped_after_its_keep_alive"]),
    # --- the request and the answer ------------------------------------------------------------------------------------
    "model-name-passed-as-given": (
        '                raw = with_model(self.headers.get("Content-Type"), raw, entry["id"])\n',
        "                pass\n",
        [T + "RoutingTest.test_the_workers_get_the_model_by_its_own_id",
         T + "RoutingTest.test_a_single_model_is_used_when_the_request_names_none"]),
    "keys-are-not-aliases": (
        'names = {entry["id"], entry["key"],', 'names = {entry["id"],',
        [T + "RoutingTest.test_the_model_is_found_by_id_key_folder_name_or_the_id_without_the_suffix"]),
    "answer-is-buffered": (
        "                    chunk = response.read1(65536)", "                    chunk = response.read(1 << 30)",
        [T + "ProxyTest.test_a_streamed_answer_arrives_as_it_is_made"]),
    "client-leaving-is-ignored": (
        "                    self.wfile.write(chunk)\n                    self.wfile.flush()\n",
        "                    try:\n                        self.wfile.write(chunk)\n                        self.wfile.flush()\n"
        "                    except OSError:\n                        pass\n",
        [T + "ProxyTest.test_a_client_that_leaves_cancels_the_workers_request"]),
    "workers-see-the-key": (
        'DROP_REQUEST = HOP_BY_HOP | {"authorization", "x-api-key", "origin", "referer", "accept-encoding"}',
        'DROP_REQUEST = HOP_BY_HOP | {"origin", "referer", "accept-encoding"}',
        [T + "ProxyTest.test_the_workers_headers_are_not_passed_on_and_the_clients_secrets_are_not_forwarded"]),
    "workers-cors-passed-on": (
        'if lower in DROP_RESPONSE or lower.startswith("access-control-"):', "if lower in DROP_RESPONSE:",
        [T + "ProxyTest.test_the_client_gets_one_cors_header_and_it_is_the_routers"]),
    # --- the guards -----------------------------------------------------------------------------------------------------
    "no-key-check": (
        "        key = self.server.api_key\n        if not key:\n            return True\n",
        "        return True\n        key = self.server.api_key\n        if not key:\n            return True\n",
        [T + "GuardsTest.test_the_key_is_checked_once_at_the_door"]),
    "no-host-check": (
        "        if name not in allowed:", "        if False:",
        [T + "GuardsTest.test_an_unknown_host_header_is_refused"]),
    "health-shows-all-without-key": (
        "        if not self._authed():\n            return payload\n", "",
        [T + "GuardsTest.test_health_is_public_but_shows_nothing_without_the_key"]),
    "cross-origin-switch-allowed": (
        'raise ApiError(403, "Cross-origin model switches are not allowed.", "forbidden")', "pass",
        [T + "GuardsTest.test_a_cross_origin_switch_is_refused"]),
    "chunked-bodies-accepted": (
        '        if self.headers.get_all("Transfer-Encoding"):\n            raise ApiError(400, "Transfer-Encoding is not supported; send one Content-Length.")\n', "",
        [T + "GuardsTest.test_a_chunked_request_is_refused"]),
    # --- failures --------------------------------------------------------------------------------------------------------
    "failed-load-stays-registered": (
        "                if self.workers.get(worker.key) is worker:\n                    del self.workers[worker.key]\n                self.last_error[worker.key] = api.message",
        "                self.last_error[worker.key] = api.message",
        [T + "FailureTest.test_a_failed_load_is_tried_again_by_the_next_request",
         T + "FailureTest.test_a_worker_that_does_not_start_is_reported_with_its_reason"]),
    "dead-worker-is-used": (
        '            if worker.state == "ready" and worker.alive():', '            if worker.state == "ready":',
        [T + "FailureTest.test_a_worker_that_dies_is_started_again"]),
    "second-request-loads-again": (
        '            if worker.state in ("starting", "stopping"):\n                return "wait", worker\n', "",
        [T + "FailureTest.test_requests_that_wait_for_a_loading_model_share_the_load"]),
    "close-leaves-workers-running": (
        "        for thread in threads:\n            thread.start()\n", "        for thread in threads:\n            pass\n",
        [T + "FailureTest.test_the_router_stops_its_workers_when_it_closes"]),
    "load-error-not-reported": (
        'tail = " | ".join(list(self.log)[-6:]) or f"exit code {self.proc.returncode}"',
        'tail = f"exit code {self.proc.returncode}"',
        [T + "FailureTest.test_a_worker_that_does_not_start_is_reported_with_its_reason"]),
    # --- what the dashboard reads ------------------------------------------------------------------------------------
    "unloaded-models-not-remembered": (
        '                item, card = state.cached_entry(entry["key"])\n                item = dict(item) if item else self.stub(entry)',
        '                item, card = None, None\n                item = self.stub(entry)',
        [T + "RoutingTest.test_a_model_that_is_not_loaded_is_listed_from_what_it_said_last_time"]),
    "switch-does-not-wait": (
        "            worker = pool.acquire(entry)\n            pool.release(worker)\n",
        "            pass\n",
        [T + "SwitchTest.test_switch_loads_a_model_and_answers_when_it_is_ready"]),
    "multipart-model-not-found": (
        "        at = raw.find(b'name=\"model\"')\n        if at < 0:\n            return None",
        "        at = raw.find(b'name=\"modelx\"')\n        if at < 0:\n            return None",
        [T + "RoutingTest.test_multipart_bodies_are_routed_by_their_model_field"]),
}


def digest(path):
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def run(tests, timeout):
    command = [sys.executable, "-m", "unittest", *tests]
    env = dict(os.environ, PYTHONUTF8="1")
    try:
        result = subprocess.run(command, cwd=C_DIR, env=env, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return False, "timed out"
    return result.returncode == 0, result.stderr.strip().splitlines()[-1] if result.stderr.strip() else ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default="")
    parser.add_argument("--timeout", type=float, default=150)
    parser.add_argument("--skip-baseline", action="store_true", help="do not run the unmodified file first")
    args = parser.parse_args()
    names = [n for n in args.only.split(",") if n] or list(MUTANTS)
    unknown = [n for n in names if n not in MUTANTS]
    if unknown:
        sys.exit("unknown mutants: " + ", ".join(unknown))
    with open(TARGET, encoding="utf-8", newline="") as handle:
        original = handle.read()
    before = digest(TARGET)
    for name in names:                              # every text must be there exactly once, before anything runs
        count = original.count(MUTANTS[name][0])
        if count != 1:
            sys.exit(f"{name}: the text to change occurs {count} times, not once")
    every = sorted({test for name in names for test in MUTANTS[name][2]})
    if not args.skip_baseline:
        print(f"unmodified: {len(every)} test(s)", flush=True)
        ok, why = run(every, args.timeout * 4)
        if not ok:
            sys.exit(f"the unmodified router fails its tests: {why}")
        print("unmodified pass", flush=True)
    survivors = []
    try:
        for name in names:
            old, new, tests = MUTANTS[name]
            with open(TARGET, "w", encoding="utf-8", newline="") as handle:
                handle.write(original.replace(old, new, 1))
            began = time.time()
            ok, why = run(tests, args.timeout)
            verdict = "SURVIVED" if ok else "caught"
            print(f"  {verdict:9} {name} ({time.time() - began:.0f} s){'' if ok else ': ' + why[:70]}", flush=True)
            if ok:
                survivors.append(name)
    finally:
        with open(TARGET, "w", encoding="utf-8", newline="") as handle:
            handle.write(original)
    if digest(TARGET) != before:
        sys.exit("router.py was not restored: check git diff")
    print(f"{len(names) - len(survivors)} of {len(names)} mutants caught")
    if survivors:
        print("survived: " + ", ".join(survivors))
        sys.exit(1)


if __name__ == "__main__":
    main()
