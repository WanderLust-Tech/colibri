---
name: extract-unity-assets
description: Extract the assets of a Unity game (images, audio, text, models, scenes) into a folder with AssetRipper, for a whole game folder, a sub-folder, or a single .bundle or .assets file. Use for "extract the assets", "rip the textures and audio", "unpack the asset bundles".
argument-hint: <game folder, sub-folder, or one .bundle/.assets file> <output folder> [PrimaryContent|UnityProject]
---

AssetRipper (installed on this PC) reads Unity games and writes their assets out as ordinary files. A prepared script runs it for you, headless, on a free local port, and stops it when done.

Arguments: `$1` is what to read: the game folder, a folder inside it (for example `<game>\<name>_Data\StreamingAssets`), or one `.bundle` or `.assets` file. `$2` is the folder to write into. `$3`, if given, is the mode: `PrimaryContent` (the default: the assets as png/jpg images, audio, text, models, and json for the rest) or `UnityProject` (a full Unity project with the scripts, settings and scenes; much bigger).

If the source or the output folder above reads as empty (``), ask the user for it and do nothing else until they give it. Never choose an output folder the user did not name.

The commands below are for PowerShell. How to work:

1. **Do not create the output folder.** The script requires that it does not exist, or is empty, and its parent exists. If it answers that the parent does not exist, then (and only then) make that parent folder with `mkdir` and run the same command again. If it says the output folder is not empty, ask the user for another folder; never delete anything to get around it.
2. A whole game can take a long time and gigabytes, so tell the user what you are about to do (the source, the mode, the output folder) in one line, then start the script **in the background** with `run_command` (`background` = true, `wait_seconds` = 30):

   `powershell.exe -NoProfile -ExecutionPolicy Bypass -File "$env:USERPROFILE\.colibri\skills\extract-unity-assets\Export-UnityAssets.ps1" -Source "$1" -OutFolder "$2" -Mode <the mode, PrimaryContent if none was given>`

   It first prints the source size, the free space and the mode, and refuses if the drive has too little space. Note the process id it returns.
3. Watch it with `process_output` (the process id). It prints a progress line about every 20 seconds, such as `(1204/5113) Exporting '...'`. Between checks wait with `run_command` `Start-Sleep -Seconds 120` (foreground, `timeout_seconds` = 200); do not check more often than that. Do not start a second copy while one is running, and do not stop it early unless the user asks; if the user asks, use `stop_process`.
4. When the output shows `finished in`, it also lists the number of files and their size, a table by file type, and the top-level folders. If it shows an error instead, say what it says and what you would do about it (for example a missing AssetRipper: it is meant to be at `%USERPROFILE%\tools\AssetRipper-2.0.0\`; do not download anything yourself).
5. Report to the user: the output folder, how many files and how big, the types (for example `.png`, `.jpg`, audio, `.json`), and what the top-level folders hold. A few "could not export" warnings are normal (Unity's built-in resources). Do not read the exported files into the chat.
6. Offer next steps: list or open a folder of interest, look at one file, or extract a different part of the game (a smaller `$1`) with another output folder.

Notes to mention once: these are the files of a game the user has, for their own study; the project's own disclaimer excludes copying protected assets, so do not redistribute them. Asset names come from the game's own data; textures that were packed into atlases come out as atlases.

Rules: if a command fails or is declined, say so and what you would do instead.
