---
name: disassemble-exe
description: Disassemble a Windows program (.exe or .dll) into assembly instructions with the REA tools, one function at a time. Gives instructions only, no C or C++ style code. Use for "disassemble", "show the assembly of", "what instructions does this function run".
argument-hint: <full path to the exe or dll> [function name or address, default: entry]
---

Disassemble a native program with the REA tools (Ghidra underneath) and show **assembly only**.

Arguments: `$1` is the full path of the program on this PC. `$2`, if given, is the function (a name such as `main`, or a hex address such as `0x140001000`); with none, use `entry`.

If the path above reads as empty (``), ask the user for the full path of the program and do nothing else until they give it.

How to work:

1. The REA tools may not be in your tool list: they are found with `tools_search` and run with `tools_call` (their names look like `rea__open_binary`). Search by what you want to do ("open a binary", "assembly of a procedure") and use the names the search shows. Never run the program itself.
2. `open_binary` with `path` = `$1`. Leave out `document` in every later call: the program just opened is the one used. If it fails, say why (wrong path, not a PE file) and stop.
3. `binary_overview` once, only to learn the format, architecture and size. Do not paste its output; say the format and architecture in one line.
4. Find the function. For `entry`, call `procedure_info` with `procedure` = `entry`: it gives the entry address, signature and size. For another name, call `procedure_info` with that name; if it is not found, call `search_procedures` with `pattern` = the name and `detail` = `summary`, and pick from what it returns. Never guess an address.
5. `procedure_assembly` with `procedure` = the name or address. Show the instructions in one code block exactly as returned, with their addresses. If there are more than about 150 lines, show the first 120, say how many there are in all and where you stopped, and offer to continue or to look at a part.
6. After the listing, say in plain sentences what the instructions do (the calls, the jumps, what is loaded and stored). Do **not** write C, C++ or pseudo code, and do not call `procedure_pseudo_code`; if the user wants that, tell them to use the `decompile-exe` skill.
7. Offer next steps: `procedure_callees` or `procedure_callers` for what it calls or who calls it, `xrefs` for references to an address, `search_strings` for a text the program uses.

Rules: one function at a time; where a tool offers `detail`, use `summary` unless you need the whole result; say so if the tool reports that the analysis may be incomplete; names like `FUN_140001000` are Ghidra's placeholders, not the original names; if a tool is declined or fails, say that and what you would do instead.
