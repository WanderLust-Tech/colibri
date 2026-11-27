# Skills for taking programs and games apart

Five skills for the VS Code extension (a skill is a `SKILL.md` folder; the extension finds these in `~/.colibri/skills/` for every workspace, lists them for the
model and runs one with `/name ARGUMENTS`). The folders here are copies. They use tools installed on the PC, none of them part of Colibri.

| Skill | Run with | What it does | Tool behind it |
|---|---|---|---|
| `disassemble-exe` | `/disassemble-exe PATH [FUNCTION]` | assembly of one function of a native program, explained in words | REA `procedure_assembly` (Ghidra) |
| `decompile-exe` | `/decompile-exe PATH [FUNCTION]` | one function as C or C++ style code, saved to a file if asked | REA `procedure_pseudo_code` (Ghidra) |
| `decompile-to-files` | `/decompile-to-files PATH FOLDER [NAME]` | every function as a `.c` file, written by REA straight to disk | REA `decompile_all` |
| `decompile-unity-game` | `/decompile-unity-game GAMEFOLDER OUTFOLDER [EXTRA]` | the game's own .NET assemblies as C# projects | ilspycmd |
| `extract-unity-assets` | `/extract-unity-assets SOURCE OUTFOLDER [MODE]` | images, audio, text, models, scenes out of a Unity game, folder or bundle | AssetRipper (script in the skill) |

For a Unity game made with Mono the game's logic is .NET code in `<name>_Data\Managed\Assembly-CSharp.dll`; the `.exe` is only a launcher, so use
`decompile-unity-game`, not the native ones. `GameAssembly.dll` instead means IL2CPP, which none of these handle.

## What is installed (this PC, 2026-11-27)

- REA 6.10.1 in `C:\Users\kp-mi\tools\rea` (an MCP server, `coli tools` list entry `rea`, on demand), Ghidra 12.2 and JDK 25 beside it: [mcp-client.md](../mcp-client.md).
- ilspycmd 11.2 (`dotnet tool install -g ilspycmd`, in `%USERPROFILE%\.dotnet\tools`).
- AssetRipper 2.0.0 free edition (GPL-3.0) in `C:\Users\kp-mi\tools\AssetRipper-2.0.0`: `AssetRipper_win_x64.zip` from the project's releases, its SHA-256 checked against
  the digest GitHub shows (`9a7ef0e7...01baf7`); one self-contained 124 MB `exe`.

## AssetRipper without its window

`AssetRipper.GUI.Free.exe --headless --port N [--log-path FILE]` runs only its web server, on `127.0.0.1` (checked: it listens on loopback only). The page is an ordinary web
API (`/openapi.json`, `/swagger` on the running server): `POST /LoadFolder` or `/LoadFile` with a form field `Path`, then `POST /Export/PrimaryContent` or `/Export/UnityProject`
with a form field `Path` (the output folder). The export call blocks until it is finished; the log shows `Processing : Finished processing assets` when loading is done and
`ExportProgress : (n/total) ...` while exporting. None of this is in AssetRipper's documentation; it was read from the running program.
`extract-unity-assets/Export-UnityAssets.ps1` does all of it (free port, load, export, a progress line every 20 s, a summary, then stops the process) and refuses an output
folder that is not new or empty, a missing parent, or too little free space. Tried on a 14.6 MB bundle: 431 files (88 png, 12 jpg, 331 json) in a second; and on a
13 KB one. **Not tried on the whole game** (3 GB of bundles in `StandaloneWindows64`, 143 MB of `.assets`): time and output size are unknown.

## Lessons from running these through a model

- REA's `decompile_all` refuses an output folder that already exists, even an empty one; a model that made the folder first got refused twice. The skill now names a new
  folder from the program's hash and forbids creating it. The AssetRipper script has the same rule (new or empty folder, make only a missing parent).
- A model with no tool for a job wanders (it ran `dir` and `where` looking for REA when a same-named setting hid it). If a skill does nothing useful, look at the
  extension's Colibri output channel first.
- Everything here only reads the game's files; the only writes are to the output folders you name.
