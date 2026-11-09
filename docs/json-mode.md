# JSON mode: an answer that cannot be anything else

`response_format` on Gemma 4, Qwen3.6 and Qwen3-Coder limits what the model may say. The engine takes the request's grammar,
and at every step only tokens that keep the answer inside it can be chosen, so what comes back is valid JSON, and has the
shape of the schema when a schema was given. That is different from GLM-5.2, where the same field feeds
[grammar-forced drafts](grammar-draft.md): a speed-up the model still verifies, which promises nothing about the output.

```
curl http://localhost:8000/v1/chat/completions -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" -d '{
  "model": "qwen3-coder-30b-a3b-colibri",
  "messages": [{"role": "user", "content": "Where is the Colosseum?"}],
  "response_format": {"type": "json_schema", "json_schema": {"name": "place", "schema": {
    "type": "object",
    "properties": {"city": {"type": "string"}, "country": {"type": "string"}},
    "required": ["city", "country"]}}}}'
```

The three forms are the gateway's: `{"type": "json_object"}` (a JSON object), `{"type": "json_schema", ...}` (a schema), and
`{"type": "gbnf", "grammar": "root ::= ..."}` (a grammar of your own, for example `root ::= "yes" | "no"`).

## How it works

[`c/constrain.h`](../c/constrain.h) walks the grammar one byte at a time (the byte-level pushdown automaton in `c/grammar.h`).
Before each token is chosen it sets to minus infinity the logit of every token whose first byte the grammar cannot take, one
pass over the vocabulary. The engine's own sampler (greedy, temperature, top-p) then chooses among what is left. A token of
several bytes can still run off the grammar in its second byte, so the choice is checked byte by byte on a copy of the walker;
a token that fails is removed and the sampler asked again, which is sampling from the model's own distribution restricted to
the tokens that are valid. The end-of-turn token is allowed only where the grammar can end, and once nothing but the end can
follow, it is all that is left. A request without a grammar is not touched.

The cost is that one pass over the vocabulary (about a quarter of a million numbers) per token, small beside the forward pass,
and a token table built from the tokenizer the first time a request brings a grammar.

## What is promised, and what is not

- Every byte the engine sends keeps the answer a valid beginning of the grammar. An answer that ends by itself (the model
  chose the end token, which was allowed only where the grammar can end) is valid JSON, and with a schema, has its shape: the
  properties in the order the schema lists them, with values of the types it names. The tests check this against Python's
  `json.loads`, with random-weight models whose unconstrained output is noise.
- **Not promised: that it finishes.** `max_tokens` can cut an answer short; it then ends with `finish_reason: "length"` and
  is a valid beginning, not a whole document. Give a structured answer room.
- **Whitespace is bounded, not thrifty.** The grammar allows whitespace between tokens (the compiler in `c/schema_gbnf.h`
  adds it at every separator), but a run of it is at most 20 bytes, after which only a token that goes on is legal. Without
  the bound, a model that could not say what it wanted (every token it preferred masked) filled its whole budget with spaces
  and newlines before the object opened: a real Gemma E2B did exactly that with a schema, found on the first real run
  (2026-11-09). An answer may still hold up to 20 bytes of whitespace at each separator.
- **Strings hold valid UTF-8.** Bytes above 0x7f in a string must form whole, well-formed characters (RFC 3629), so a token
  that stops inside one is followed by its rest. (An answer cut by `max_tokens` can still stop inside a character.)
- **`json_object` is an object.** The root is an object, not any value, as OpenAI's mode promises.

## What is refused (HTTP 400, parameter `response_format`)

The engine never goes on without the constraint, so what it cannot honour is a 400, not a quietly unconstrained answer.

- **Thinking on.** The grammar would limit the reasoning, not only the answer. Turn thinking off for the request.
- **`tools`.** The grammar would forbid the tokens of a tool call. (Tools the client forbade with `tool_choice: "none"`
  do not count.)
- **A schema the compiler does not have.** The subset: objects with `properties` (all of them in `required`, or no `required`),
  strings (with `enum` or `const`), numbers, integers, booleans, `null`, arrays with `items` (`minItems` 0 or 1), enums, and
  nesting to 32 levels. `anyOf`, `oneOf`, `$ref`, `pattern`, `format`, `minimum`, optional properties and the like are refused
  with the engine's account of which. Use `{"type": "gbnf"}` for something the schema subset cannot say.
- **A grammar that does not compile**, or is too ambiguous to follow.
- **An answer that gets stuck**: part way through, no token of the vocabulary can continue the grammar (it asks for a byte no
  token carries), or the model's end token does not exist in the vocabulary of a grammar that can only end.
- **`logprobs` together with a grammar** (engine level): the numbers would be of the masked distribution.

## Where

| Engine | `response_format` |
|---|---|
| Gemma 4 (26B-A4B, 31B, E4B, E2B) | constrained |
| Qwen3.6, Qwen3-Coder (family `qwen36`) | constrained |
| GLM-5.2 | a speed-up only |
| every other engine | HTTP 400 |

An engine says it takes a grammar in its `CAPS` line (`grammar=1`, [serve_protocol.md](serve_protocol.md)); the grammar travels
on `SUBMIT` as the seventh numeric field, its bytes right after the prompt's, the way GLM's does. The table in
[api.md](api.md) is held to the registry by `c/tests/test_api_matrix_docs.py`.

Tests: `c/tests/test_constrain.c` and `test_constrain.py` (the mechanism, with random logits), `test_gemma4.py`
(`ServeGrammarTest`, on the tiny Gemma 4 models), `test_qwen36_grammar.py` (the tiny Qwen3.6 model), and
`test_gemma4_gateway.py` (`GrammarTest`, the gateway's side).
