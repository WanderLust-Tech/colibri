"""Gemma 4's chat format: the prompt a conversation becomes, and the tool calls a reply holds.

The renderer is google/gemma-4-26B-A4B-it's chat_template.jinja ported by hand, byte for byte (the
gateway renders prompts without jinja, and tests/test_gemma4_chat_template.py renders the release's own
template with jinja2 and compares). What the template does, in order:

  <bos>
  [<|turn>system\\n  <|think|>\\n  system text  <|tool>declaration:...<tool|>...  <turn|>\\n]
  <|turn>user\\n text <turn|>\\n
  <|turn>model\\n [<|channel>thought\\n reasoning \\n<channel|>] text [<|tool_call>call:f{k:v}<tool_call|>
      <|tool_response>response:f{...}<tool_response|>] <turn|>\\n
  ... and the cue  <|turn>model\\n  followed, with thinking off, by an empty thought channel
  <|channel>thought\\n<channel|>  that tells the model to answer at once.

Tool arguments are written in Gemma's own syntax: keys bare, strings between <|"|> marks, numbers
and booleans bare, objects in braces, lists in brackets, keys of an object in sorted order.

Pure functions, no I/O. Errors are ChatError(message, param), which the gateway turns into a 400.
"""
import json
import re
import uuid

BOS = "<bos>"
Q = '<|"|>'                                     # the string delimiter token
CALL_OPEN, CALL_CLOSE = "<|tool_call>", "<tool_call|>"
CHANNEL_OPEN, CHANNEL_CLOSE = "<|channel>", "<channel|>"


class ChatError(ValueError):
    def __init__(self, message, param=None, code=None):
        super().__init__(message)
        self.param = param
        self.code = code


# ----------------------------------------------------------------------------- jinja's ways

def _s(value):
    """How jinja prints a value: None as None, True as True, an undefined name as nothing."""
    return "" if value is _MISSING else str(value)


_MISSING = object()


def _get(mapping, key):
    return mapping.get(key, _MISSING) if isinstance(mapping, dict) else _MISSING


def _truthy(value):
    return value is not _MISSING and bool(value)


def _upper(value):
    return "" if value is _MISSING else _s(value).upper()


def _dictsort(mapping):
    """jinja's dictsort: by key, case-insensitive, stable."""
    return sorted(mapping.items(), key=lambda item: str(item[0]).lower())


def _is_mapping(value):
    return isinstance(value, dict)


def _is_string(value):
    return isinstance(value, str)


def _is_sequence(value):
    return isinstance(value, (list, tuple, str))


def format_argument(argument, escape_keys=True):
    if argument is None:
        return "null"
    if isinstance(argument, str):
        return Q + argument + Q
    if isinstance(argument, bool):
        return "true" if argument else "false"
    if isinstance(argument, dict):
        parts = []
        for key, value in _dictsort(argument):
            shown = Q + str(key) + Q if escape_keys else str(key)
            parts.append(shown + ":" + format_argument(value, escape_keys))
        return "{" + ",".join(parts) + "}"
    if isinstance(argument, (list, tuple)):
        return "[" + ",".join(format_argument(item, escape_keys) for item in argument) + "]"
    return _s(argument)


_STANDARD_KEYS = ("description", "type", "properties", "required", "nullable")


def _required_list(value):
    required = _get(value, "required")
    return [] if required is _MISSING else required


def format_parameters(properties, required=None, filter_keys=False):
    out = []
    found_first = False
    for key, value in _dictsort(properties):
        if filter_keys and key in _STANDARD_KEYS:
            continue
        if found_first:
            out.append(",")
        found_first = True
        out.append(f"{key}:{{")
        add_comma = False
        if _truthy(_get(value, "description")):
            out.append(f"description:{Q}{_s(value['description'])}{Q}")
            add_comma = True

        def comma():
            nonlocal add_comma
            if add_comma:
                out.append(",")
            add_comma = True

        kind = _upper(_get(value, "type"))
        if kind == "STRING":
            if _truthy(_get(value, "enum")):
                comma()
                out.append("enum:" + format_argument(value["enum"]))
        elif kind == "ARRAY":
            items = _get(value, "items")
            if _is_mapping(items) and items:
                comma()
                out.append("items:{")
                first_item = False
                for item_key, item_value in _dictsort(items):
                    if item_value is None:
                        continue
                    if first_item:
                        out.append(",")
                    first_item = True
                    if item_key == "properties":
                        out.append("properties:{")
                        if _is_mapping(item_value):
                            out.append(format_parameters(item_value, _required_list(items)))
                        out.append("}")
                    elif item_key == "required":
                        out.append("required:[" + ",".join(Q + _s(r) + Q for r in item_value) + "]")
                    elif item_key == "type":
                        if _is_string(item_value):
                            out.append("type:" + format_argument(item_value.upper()))
                        else:
                            out.append("type:" + format_argument([_s(t).upper() for t in item_value]))
                    else:
                        out.append(f"{item_key}:{format_argument(item_value)}")
                out.append("}")
        if _truthy(_get(value, "nullable")):
            comma()
            out.append("nullable:true")
        if kind == "OBJECT":
            if _is_mapping(_get(value, "properties")):
                comma()
                out.append("properties:{" + format_parameters(value["properties"], _required_list(value)) + "}")
            elif _is_mapping(value):
                comma()
                out.append("properties:{" + format_parameters(value, _required_list(value), True) + "}")
            if _truthy(_get(value, "required")):
                comma()
                out.append("required:[" + ",".join(Q + _s(r) + Q for r in _required_list(value)) + "]")
        comma()
        out.append(f"type:{Q}{_upper(_get(value, 'type'))}{Q}}}")
    return "".join(out)


def format_function_declaration(tool):
    function = tool["function"]
    out = [f"declaration:{_s(_get(function, 'name'))}{{description:{Q}{_s(_get(function, 'description'))}{Q}"]
    params = _get(function, "parameters")
    if _truthy(params):
        out.append(",parameters:{")
        if _truthy(_get(params, "properties")):
            out.append("properties:{" + format_parameters(params["properties"], _get(params, "required")) + "},")
        if _truthy(_get(params, "required")):
            out.append("required:[" + ",".join(Q + _s(item) + Q for item in params["required"]) + "],")
        if _truthy(_get(params, "type")):
            out.append(f"type:{Q}{_upper(params['type'])}{Q}}}")
    if "response" in function:
        response = function["response"]
        out.append(",response:{")
        if _truthy(_get(response, "description")):
            out.append(f"description:{Q}{_s(response['description'])}{Q},")
        if _upper(_get(response, "type")) == "OBJECT":
            out.append(f"type:{Q}{_upper(response['type'])}{Q}}}")
    out.append("}")
    return "".join(out)


def strip_thinking(text):
    result = ""
    for part in text.split(CHANNEL_CLOSE):
        result += part.split(CHANNEL_OPEN)[0] if CHANNEL_OPEN in part else part
    return result.strip()


def format_tool_response_block(tool_name, response):
    out = ["<|tool_response>"]
    if isinstance(response, dict):
        pairs = [f"{key}:{format_argument(value, False)}" for key, value in _dictsort(response)]
        out.append(f"response:{tool_name}{{" + ",".join(pairs) + "}")
    else:
        out.append(f"response:{tool_name}{{value:{format_argument(response, False)}}}")
    out.append("<tool_response|>")
    return "".join(out)


# ------------------------------------------------------------------------------ the prompt

def _text_of(content, where, trim_each=False, model=False):
    """The text of a message's content: a string, or the text parts of a list (a picture is refused:
    this engine reads text)."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for index, part in enumerate(content):
            if isinstance(part, dict) and part.get("type") == "text":
                out.append(_s(part.get("text", "")))
            elif isinstance(part, dict) and part.get("type") in ("image", "image_url", "input_audio", "audio", "video"):
                raise ChatError("This Gemma 4 engine reads text only: pictures, audio and video are not supported yet.",
                                f"{where}.{index}", "unsupported_content_type")
            else:
                raise ChatError("Message content parts must be text.", f"{where}.{index}", "unsupported_content_type")
        return "".join(out)
    raise ChatError("Message content must be a string or an array of parts.", where)


def _captured(content, model, where):
    """The template's captured_content: a string trimmed (an assistant's, with its thinking channels
    removed first), a list of parts one by one."""
    if isinstance(content, str):
        return strip_thinking(content) if model else content.strip()
    if isinstance(content, list):
        out = []
        for index, part in enumerate(content):
            if isinstance(part, dict) and part.get("type") == "text":
                text = _s(part.get("text", ""))
                out.append(strip_thinking(text) if model else text.strip())
            elif isinstance(part, dict) and part.get("type") in ("image", "image_url", "input_audio", "audio", "video"):
                raise ChatError("This Gemma 4 engine reads text only: pictures, audio and video are not supported yet.",
                                f"{where}.{index}", "unsupported_content_type")
            else:
                raise ChatError("Message content parts must be text.", f"{where}.{index}", "unsupported_content_type")
        return "".join(out)
    if content is None:
        return ""
    raise ChatError("Message content must be a string or an array of parts.", where)


def _arguments(call, where):
    function = call.get("function") if isinstance(call, dict) else None
    if not isinstance(function, dict) or not isinstance(function.get("name"), str) or not function["name"]:
        raise ChatError("Each tool call needs a `function.name`.", where)
    args = function.get("arguments")
    if isinstance(args, str):
        try:
            args = json.loads(args) if args.strip() else {}
        except ValueError:
            raise ChatError("`function.arguments` must be a JSON object.", f"{where}.function.arguments")
    if args is not None and not isinstance(args, dict):
        raise ChatError("`function.arguments` must be a JSON object.", f"{where}.function.arguments")
    return function["name"], args


def render(messages, tools=None, enable_thinking=False, add_generation_prompt=True, preserve_thinking=False):
    """The prompt for a conversation, as chat_template.jinja renders it."""
    if not isinstance(messages, list) or not messages:
        raise ChatError("`messages` must be a non-empty array.", "messages")
    if tools is not None and not isinstance(tools, list):
        raise ChatError("`tools` must be an array.", "tools")
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ChatError("Each message must be an object.", f"messages.{index}")
        if message.get("role") not in ("system", "developer", "user", "assistant", "tool"):
            raise ChatError(f"Unsupported role {message.get('role')!r}.", f"messages.{index}.role")
    out = [BOS]
    first = messages[0]
    system_first = first["role"] in ("system", "developer")
    if enable_thinking or tools or system_first:
        out.append("<|turn>system\n")
        if enable_thinking:
            out.append("<|think|>\n")
        if system_first:
            content = first.get("content")
            if isinstance(content, str):
                out.append(content.strip())
            elif isinstance(content, list):
                for item in content:
                    out.append(_s(item.get("text", "") if isinstance(item, dict) else "").strip() + " ")
        if tools:
            for index, tool in enumerate(tools):
                if not isinstance(tool, dict) or not isinstance(tool.get("function"), dict):
                    raise ChatError("Each tool must be an object with a `function`.", f"tools.{index}")
                out.append("<|tool>" + format_function_declaration(tool).strip() + "<tool|>")
        out.append("<turn|>\n")
    loop = messages[1:] if system_first else messages
    last_user = -1
    for i, message in enumerate(loop):
        if message["role"] == "user":
            last_user = i
    prev_type = "think" if enable_thinking else None       # the template's namespace state, and what the cue reads
    if tools:
        prev_type = "tool"
    prev_non_tool_role = None
    for i, message in enumerate(loop):
        role_in = message["role"]
        if role_in == "tool":
            continue
        where = f"messages.{i + (1 if system_first else 0)}"
        prev_type = None
        role = "model" if role_in == "assistant" else role_in
        continue_same = role == "model" and prev_non_tool_role == "assistant"
        if not continue_same:
            out.append("<|turn>" + role + "\n")
        thinking = message.get("reasoning") or message.get("reasoning_content")
        gate = i > last_user or (preserve_thinking and message.get("tool_calls"))
        if thinking and gate:
            out.append("<|channel>thought\n" + _s(thinking) + "\n<channel|>")
        calls = message.get("tool_calls")
        if calls:
            for position, call in enumerate(calls):
                name, args = _arguments(call, f"{where}.tool_calls.{position}")
                out.append("<|tool_call>call:" + name + "{")
                if args:
                    out.append(",".join(f"{key}:{format_argument(value, False)}" for key, value in _dictsort(args)))
                out.append("}<tool_call|>")
            prev_type = "tool_call"
        responded = False
        if message.get("tool_responses"):
            for response in message["tool_responses"]:
                out.append(format_tool_response_block(response.get("name") or "unknown", response.get("response")))
                responded = True
                prev_type = "tool_response"
        elif calls:
            for k in range(i + 1, len(loop)):
                follow = loop[k]
                if follow["role"] != "tool":
                    break
                name = follow.get("name") or "unknown"
                for call in calls:
                    if isinstance(call, dict) and call.get("id") == follow.get("tool_call_id"):
                        name = call["function"]["name"]
                body = follow.get("content")
                if isinstance(body, str):
                    out.append(format_tool_response_block(name, body))
                elif isinstance(body, list):
                    out.append(format_tool_response_block(name, _text_of(body, f"messages.{k}.content")))
                else:
                    out.append(format_tool_response_block(name, body))
                responded = True
                prev_type = "tool_response"
        captured = _captured(message.get("content"), role == "model", f"{where}.content")
        out.append(captured)
        has_content = len(captured.strip()) > 0
        next_role, found = None, False
        for j in range(i + 1, len(loop)):
            if loop[j]["role"] != "tool":
                next_role, found = loop[j]["role"], True
                break
        continues_into_next = role == "model" and next_role == "assistant" and (not calls or responded)
        if prev_type == "tool_call" and not responded:
            out.append("<|tool_response>")
        elif continues_into_next:
            pass
        elif not (responded and not has_content and not found):
            out.append("<turn|>\n")
        prev_non_tool_role = role_in
    if add_generation_prompt:
        if prev_type not in ("tool_response", "tool_call"):
            out.append("<|turn>model\n")
            if not enable_thinking:
                out.append("<|channel>thought\n<channel|>")
        elif prev_type == "tool_response" and enable_thinking:
            out.append("<|channel>thought\n")
    return "".join(out)


# ----------------------------------------------------------------------------- the reply

class _Parser:
    """Gemma's tool-call argument syntax: {key:value,...} with keys bare or quoted, strings between
    <|"|> marks, numbers, true, false, null, nested objects and lists."""

    def __init__(self, text):
        self.s = text
        self.i = 0

    def error(self, why):
        raise ValueError(f"{why} at {self.i} in {self.s[:60]!r}")

    def ws(self):
        while self.i < len(self.s) and self.s[self.i] in " \t\r\n":
            self.i += 1

    def value(self):
        self.ws()
        s, i = self.s, self.i
        if s.startswith(Q, i):
            end = s.find(Q, i + len(Q))
            if end < 0:
                self.error("unterminated string")
            self.i = end + len(Q)
            return s[i + len(Q):end]
        if s.startswith("{", i):
            return self.obj()
        if s.startswith("[", i):
            return self.array()
        j = i
        while j < len(s) and s[j] not in ",}]":
            j += 1
        raw = s[i:j].strip()
        self.i = j
        if raw == "true":
            return True
        if raw == "false":
            return False
        if raw in ("null", "None", ""):
            return None
        try:
            return int(raw)
        except ValueError:
            try:
                return float(raw)
            except ValueError:
                return raw

    def key(self):
        self.ws()
        s, i = self.s, self.i
        if s.startswith(Q, i):
            end = s.find(Q, i + len(Q))
            if end < 0:
                self.error("unterminated key")
            self.i = end + len(Q)
            return s[i + len(Q):end]
        j = s.find(":", i)
        if j < 0:
            self.error("a key with no value")
        self.i = j
        return s[i:j].strip()

    def obj(self):
        self.i += 1                                   # {
        out = {}
        self.ws()
        if self.s.startswith("}", self.i):
            self.i += 1
            return out
        while True:
            key = self.key()
            self.ws()
            if not self.s.startswith(":", self.i):
                self.error("expected :")
            self.i += 1
            out[key] = self.value()
            self.ws()
            if self.s.startswith(",", self.i):
                self.i += 1
                continue
            if self.s.startswith("}", self.i):
                self.i += 1
                return out
            self.error("expected , or }")

    def array(self):
        self.i += 1                                   # [
        out = []
        self.ws()
        if self.s.startswith("]", self.i):
            self.i += 1
            return out
        while True:
            out.append(self.value())
            self.ws()
            if self.s.startswith(",", self.i):
                self.i += 1
                continue
            if self.s.startswith("]", self.i):
                self.i += 1
                return out
            self.error("expected , or ]")


def parse_arguments(text):
    """`{k:v,...}` (Gemma's syntax) to a dict."""
    parser = _Parser(text)
    parser.ws()
    value = parser.obj()
    return value


_CALL = re.compile(re.escape(CALL_OPEN) + r"call:(\w+)(\{.*?\})" + re.escape(CALL_CLOSE), re.S)


def parse_tool_calls(reply, tools=None):
    """(content, calls): the reply without its tool calls, and each call as an OpenAI `tool_calls`
    entry. A call whose arguments do not parse is left in the content, not guessed at."""
    calls, kept, pos = [], [], 0
    reply = reply or ""
    for match in _CALL.finditer(reply):
        kept.append(reply[pos:match.start()])
        pos = match.end()
        try:
            args = parse_arguments(match.group(2))
        except ValueError:
            kept.append(match.group(0))
            continue
        calls.append({"id": f"call_{uuid.uuid4().hex[:24]}", "type": "function",
                      "function": {"name": match.group(1), "arguments": json.dumps(args, ensure_ascii=False)}})
    kept.append(reply[pos:])
    return "".join(kept).strip(), calls


class StreamDialect:
    """The engine's words for the thinking channel, in the gateway's: the gateway already splits
    `<think>...</think>`, so the channel's opening `<|channel>thought\\n` (which the reply starts with when
    thinking is on, and which the splitter's own initial state stands for) is dropped and its closing
    `<channel|>` becomes `</think>`.

    Each of those control tokens arrives as a frame of its own. feed() takes a frame's bytes and returns
    the bytes to pass on (possibly none)."""
    EXPECT = b"thought\n"

    def __init__(self):
        self.pending = None                           # bytes after <|channel> still being matched

    def feed(self, data):
        if self.pending is not None:
            self.pending += data
            if self.EXPECT.startswith(self.pending):
                if self.pending == self.EXPECT:
                    self.pending = None
                return b""
            out, self.pending = self.pending, None
            if out.startswith(self.EXPECT):
                return out[len(self.EXPECT):]
            return out
        if data == CHANNEL_OPEN.encode():
            self.pending = b""
            return b""
        if data == CHANNEL_CLOSE.encode():
            return b"</think>"
        return data

    def flush(self):
        out, self.pending = self.pending, None
        return out or b""
