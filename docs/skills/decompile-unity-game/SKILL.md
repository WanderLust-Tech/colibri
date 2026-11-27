---
name: decompile-unity-game
description: Decompile the game code of a Unity (Mono) game folder to real C# source files with ilspycmd, and say what the other parts of the folder are (engine DLLs, third-party DLLs, asset bundles). Use for "decompile this Unity game", "get the C# of the game", "decompile the exe and its dlls".
argument-hint: <game folder, the one with the .exe> <output folder> [extra assembly names, comma separated]
---

A Unity game made with the Mono backend keeps its code in .NET assemblies, which decompile to real C# (names, classes, methods), not to the pseudo code of a native decompiler. The `.exe` in the game folder is only a small launcher and `UnityPlayer.dll` is the engine, so this skill does not decompile those.

Arguments: `$1` is the game folder (the one holding the `.exe` and a `<name>_Data` folder). `$2` is the folder to put the C# in. `$3`, if given, names extra assemblies to decompile besides the game's own (comma separated, with or without `.dll`).

If the game folder or the output folder above reads as empty (``), ask the user for it and do nothing else until they give it. Never choose an output folder the user did not name.

The commands below are for PowerShell. Run them one at a time.

How to work:

1. Find the game's code. Run `dir "$1\*_Data\Managed\Assembly-CSharp*.dll"`. If nothing is found, look for `GameAssembly.dll` in `$1` with `dir "$1\GameAssembly.dll"`: if it exists the game is IL2CPP, whose code is not .NET, so say that this skill cannot decompile it (a tool such as Il2CppDumper is needed) and stop. If neither exists, say this does not look like a Unity game folder and stop.
2. The assemblies to decompile are the game's own: `Assembly-CSharp.dll` and, if it exists, `Assembly-CSharp-firstpass.dll`, plus anything named in `$3`. Do not decompile the Unity, System, Microsoft, Sentry, DOTween, Zenject or other third-party assemblies unless `$3` names them.
3. For each assembly, run exactly this (change only the names; the first path is the `Managed` folder you found, and it is also the reference path so types resolve):

   `& "$env:USERPROFILE\.dotnet\tools\ilspycmd.cmd" -p -r "<Managed folder>" -o "$2\<assembly name without .dll>" "<Managed folder>\<assembly name>.dll"`

   `-p` writes a whole project, one `.cs` file per type, in folders by namespace. Running it again on an existing output folder is fine. Do not create the output folders first; ilspycmd makes them. If the command says ilspycmd is not recognized, tell the user to run `dotnet tool install -g ilspycmd` and stop; do not install it yourself.
4. When each one finishes, count what was written: `(Get-ChildItem "$2\<assembly name>" -Recurse -Filter *.cs).Count`, and list the top-level folders (the namespaces): `Get-ChildItem "$2\<assembly name>" -Directory | Select-Object -ExpandProperty Name`. Do not read every file into the chat.
5. Report: for each assembly the full output path, how many `.cs` files, and the namespaces. Then explain what else is in the game folder, in a few lines, without touching it:
   - the `.exe` is the Unity launcher and `UnityPlayer.dll` is the engine: not the game's logic, so not decompiled here (for a native DLL the user does want, the `decompile-to-files` skill does it);
   - the other `.dll` files in `Managed` are Unity, .NET and third-party libraries (name a few you saw);
   - the `.bundle`, `.assets`, `.resS` and `.resource` files are Unity asset bundles (textures, audio, sprites, scenes). Neither this skill nor ilspycmd opens them; a tool such as AssetRipper or AssetStudio does, which the user has to run themselves. Do not try to open them.
6. Remind the user once that the decompiled C# is a reconstruction: names the compiler threw away (locals, some lambdas) are invented, and it will not necessarily rebuild as it is. It is for reading and study.
7. Offer next steps: read one class and explain it, search the output for a name (`Select-String -Path "$2\<assembly name>\*.cs" -Pattern "<text>" -Recurse`), or list the biggest files (the main game classes).

Rules: if a command fails or is declined, say so and what you would do instead.
