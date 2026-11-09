# Request log

One JSON line per request, in a file, for the times you want to know what was asked of the server, how long it took and
what it cost. It is off unless you ask for it.

```
coli router --request-log C:\Users\you\colibri-requests.log
coli serve  --request-log requests.log          # the same for a single server
set COLI_REQUEST_LOG=requests.log               # or the environment, which `coli router` hands to its workers
```

## What a line holds

What happened, not what was said. A line from the router:

```json
{"ts":"2026-11-09T14:02:11.482Z","source":"router","trace":"9f3c1a07be52d4aa","method":"POST","path":"/v1/chat/completions",
 "model":"terse","preset":"terse","resolved":"qwen3-coder-30b-a3b-colibri","stream":false,"bytes_in":212,
 "status":200,"seconds":4.213,"client":"192.168.1.20"}
```

and the gateway behind it writes its own for the same request, with the tokens only it knows:

```json
{"ts":"2026-11-09T14:02:15.690Z","source":"gateway","trace":"9f3c1a07be52d4aa","id":"req_6b1d...","method":"POST",
 "path":"/v1/chat/completions","model":"qwen3-coder-30b-a3b-colibri","stream":false,"status":200,"seconds":4.18,
 "prompt_tokens":38,"completion_tokens":61,"finish":"stop","client":"127.0.0.1"}
```

| Field | Meaning |
|---|---|
| `ts` | UTC, to the millisecond. It is the time the line was written, when the request had finished. |
| `source` | `router` or `gateway`. |
| `trace` | The router's id for the request, handed to the worker, so the two lines can be joined. A single `coli serve` has none, unless a client sends an `x-colibri-trace` header. |
| `id` | The gateway's own request id (`x-request-id`). |
| `path`, `model`, `stream` | What was asked. `model` is what the client wrote. |
| `preset`, `resolved` | Router only: the preset the request named, and the model it ran on. |
| `status` | The HTTP status that went out. A streamed answer that fails half way is a `200`. |
| `error` | The error's code, or its message when it has none, for a refused or failed request. |
| `seconds` | Wall time for the request, including a model load and a wait in the queue. |
| `prompt_tokens`, `completion_tokens`, `finish` | Gateway only, for requests that generate text: tokens in and out, `stop` or `length`. A transcription of a long file adds its pieces. |
| `bytes_in` | Router only: the size of the request body. |
| `client` | The address the request came from (the router's, for a worker). |

Only requests that compute are logged (chat, completions, messages, systemone, images, music, transcriptions), not
`GET`s, the dashboard or the management calls.

## The text of requests

`COLI_REQUEST_LOG_BODIES=1` adds a `request` object to each line: the JSON body, with every string cut to 500
characters (a picture or a song sent as base64 is not a log line). It is off by default because a log of prompts is a
privacy decision: it is a plain file, and anyone who can read it reads what was asked. Multipart bodies are not
copied.

## The file

It is appended to and, when it would pass `COLI_REQUEST_LOG_MAX` bytes (50 MB by default), moved to `FILE.1`
(replacing an older one) and started again, so a log never grows past about twice that. A log that cannot be written (a
full disk, a folder that is not there) says so once on the server's stderr and never fails a request.

Reading it: every line is a JSON object, so `jq`, or in PowerShell
`Get-Content requests.log | ConvertFrom-Json | Where-Object source -eq gateway | Measure-Object completion_tokens -Sum`.
