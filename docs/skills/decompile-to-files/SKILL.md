---
name: decompile-to-files
description: Decompile a Windows program (.exe or .dll) with the REA tools and write the C or C++ style code of its functions into a folder, one .c file per function, without passing the code through the chat. Use for "decompile the whole exe to files", "save the decompiled code".
argument-hint: <full path to the exe or dll> <output folder> [function name to limit it to]
---

Decompile a native program with the REA tools (Ghidra underneath) and have REA write the code of its functions into a folder. The code goes from REA to the files directly; you only handle the summary.

Arguments: `$1` is the full path of the program on this PC. `$2` is the folder to put the export in. `$3`, if given, limits the export to functions with that name.

If the program path or the folder above reads as empty (``), ask the user for it and do nothing else until they give it. Never choose a folder the user did not name.

**The one rule that matters: REA makes the export folder itself and refuses one that already exists, even an empty one. So do NOT create it, not with `mkdir`, not with a file write, and do NOT run `dir` to check it first.** Just call the tool and read its answer.

How to work:

1. The REA tools may not be in your tool list: they are found with `tools_search` and run with `tools_call` (their names look like `rea__open_binary`). Search by what you want to do ("open a binary", "export pseudocode of all functions to a directory") and use the names the search shows. Never run the program itself.
2. `open_binary` with `path` = `$1`. Its answer includes a `sha256`. If it fails, say why (wrong path, not a PE file) and stop.
3. Work out the export folder's name: `$2`, then a backslash, then the program's file name without its extension, then `-`, then the first 8 characters of that `sha256`. For example `D:\decomp\Game-7bf6bdd2`. (The name holds the program's hash, so a folder with that name can only be an earlier export of this same program.)
4. Call `decompile_all` with `output_directory` = that folder. Add `name` = `$3` only if it was given. Leave out the other options the first time. Read the answer:
   - It worked: REA made the folder. Each function is a file `functions/0x<address>.c` (named by address) that starts with a comment giving its name, address range and the provider's limitations; the folder also has `decompile-index.json` (the manifest) and `decompile-journal.jsonl`. Imports from other libraries get no file.
   - "The output directory's parent does not exist": then, and only then, create `$2` itself with `mkdir` (the parent, not the export folder) and repeat the same call.
   - "Output directory already exists": an earlier export of this program is there. Repeat the same call with `resume` = `true`; verified files are skipped and the rest is done. If that is refused too, add `-2` to the folder name (then `-3`...) and call again without `resume`. Never delete or overwrite the user's files to get around an error, and never create the export folder yourself.
5. The answer is a manifest, not the code. Do not paste it. Say how many functions were written, how many failed or were unavailable, and the full path of the folder.
6. A large program can take longer than one call is allowed. If the call times out or stops early, repeat it with the same `output_directory` and `resume` = `true`. To work in parts, `address_start` and `address_end` or `limit` select a range.
7. Offer next steps: read one file and explain it, decompile one function in chat (`/decompile-exe`), or look at what a function calls (`procedure_callees`). Do not read every file back into the chat.

Remind the user once that this is **a decompiler's reconstruction, not the original source**: names such as `FUN_140001000`, `param_1` and `local_28` are placeholders, types may be wrong, and the files are not meant to compile as they are.

Rules: if a tool is declined or fails, say that and what you would do instead; say so if the manifest reports that analysis may be incomplete.
