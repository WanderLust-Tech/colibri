"""What this server can do, as the `features` block of an authenticated GET /health.

The dashboard and the editor extension are deployed apart from the server, so a new client meets an old server (and the other
way round). A client that reads this block can hide a control the server lacks instead of finding out from a 404; a server
that predates the block has no `features` at all, and a client then falls back to asking (a 404 means the route is not there).

Only what a client cannot tell otherwise is listed. What the loaded model can do (pictures, music, speech, tools, the context
window) is in `capabilities` and on the model card, because it changes with the model and not with the server.

A value is a boolean, or a short word where the answer has more than two cases (`model_switch`). Names are added, never
reused for something else; `api` is the version of this list and goes up only when a name changes its meaning.
"""

API_LEVEL = 1


def describe(*, router=False, presets=False, request_log=False, model_settings=False, jobs=False):
    """The block for one server. The arguments say what a router was started with; `coli serve` has none of them."""
    return {
        "api": API_LEVEL,
        # A `coli router` serves several models on one port; `coli serve` serves one.
        "router": bool(router),
        # GET /v1/models/installed: the models on this machine and what each is doing.
        "installed_models": True,
        # POST /v1/models/switch: "load" starts the model beside the others; "restart" replaces the one served.
        "model_switch": "load" if router else "restart",
        # POST /v1/models/unload, and per-model on/off and GPU pins (GET and POST /v1/models/settings).
        "model_unload": bool(router),
        "model_settings": bool(router and model_settings),
        # GET, POST /v1/presets: named model + defaults + system text.
        "presets": bool(router and presets),
        # GET /v1/requests: only when the router was started with a request log.
        "request_log": bool(router and request_log),
        # GET /v1/logs: what the router and its models printed.
        "server_log": bool(router),
        # X-Colibri-Priority, and pictures and songs waiting behind a chat.
        "priority": bool(router),
        # GET /v1/library and /v1/library/{kind}/{name}: the kept pictures and songs, listed and served.
        "library": True,
        # POST with `Prefer: respond-async` (pictures and songs) answers 202 with a job; GET /v1/jobs, GET /v1/jobs/{id},
        # POST /v1/jobs/cancel. A render goes on when the page that asked for it is closed.
        "jobs": bool(router and jobs),
        # `response_format` json_object and json_schema.
        "json_mode": True,
        # usage.prompt_tokens_details.cached_tokens: how much of the prompt was already in the cache.
        "cached_tokens": True,
    }


FEATURES = tuple(describe())
