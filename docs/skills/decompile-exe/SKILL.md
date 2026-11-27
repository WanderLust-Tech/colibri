---
name: decompile-exe
description: Decompile a function of a Windows program (.exe or .dll) into C or C++ style code with the REA tools (Ghidra's decompiler), one function at a time, with the assembly only where the code is unclear. Use for "decompile", "show it as C", "what does this function do in code".
argument-hint: <full path to the exe or dll> [function name or address, default: entry]
---

Decompile a native program with the REA tools (Ghidra underneath) and show **C or C++ style code**.

Arguments: `$1` is the full path of the program on this PC. `$2`, if given, is the function (a name such as `main`, or a hex address such as `0x140001000`); with none, use `entry`.

If the path above reads as empty (``), ask the user for the full path of the program and do nothing else until they give it.

How to work:

1. The REA tools may not be in your tool list: they are found with `tools_search` and run with `tools_call` (their names look like `rea__open_binary`). Search by what you want to do ("open a binary", "decompile a procedure") and use the names the search shows. Never run the program itself.
2. `open_binary` with `path` = `$1`. Leave out `document` in every later call: the program just opened is the one used. If it fails, say why (wrong path, not a PE file) and stop.
3. `binary_overview` once, only to learn the format, architecture and size. Do not paste its output; say the format and architecture in one line.
4. Find the function. For `entry`, call `procedure_info` with `procedure` = `entry`: it gives the entry address, signature and size. For another name, call `procedure_info` with that name; if it is not found, call `search_procedures` with `pattern` = the name and `detail` = `summary`, and pick from what it returns. Never guess an address.
5. `procedure_pseudo_code` with `procedure` = the name or address. Show the code in one `c` code block exactly as returned. If it is longer than about 150 lines, show the first 120, say how many lines there are and where you stopped, and offer to continue.
6. Then explain what the function does in a few plain sentences, in terms of the code (the arguments, the calls it makes, the loops and branches). Say clearly that this is a **reconstruction by a decompiler, not the original source**: names like `FUN_140001000`, `param_1` and `local_28` are placeholders, and types may be wrong.
7. If the user asks for a cleaner version, rewrite it as readable C or C++ with meaningful names and types, and mark every name or type you inferred (a comment such as `/* inferred */`) so it is not mistaken for something the tool reported. Do not claim it compiles or matches the original.
8. Where the code is unclear (inline assembly, odd calling conventions, a branch that looks wrong), call `procedure_assembly` for that function and quote only the few lines in question.
9. If the user wants the code saved, write it to the file or folder they name with the file-writing tool, as shown (a header comment with the function's name and address is fine), and say where it went. For many functions at once, point them to the `decompile-to-files` skill: it has REA write every function to a folder directly.
10. Offer next steps: `procedure_callees` or `procedure_callers`, `xrefs` for references to an address, `search_strings` for a text the program uses, `analyze_function` for a summary of one function.

Rules: one function at a time; where a tool offers `detail`, use `summary` unless you need the whole result; say so if the tool reports that the analysis may be incomplete; if a tool is declined or fails, say that and what you would do instead.
