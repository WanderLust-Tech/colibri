"""A look inside a model folder, and a check that its files are whole (`coli model info`, `coli model verify`).

Both read as little as they can. `inspect` reads config.json, the manifest the downloader wrote, and only the headers of the
safetensors files (the first few kilobytes of each: a length and a JSON table of tensors), never a weight, so it is quick on a slow
drive and works on a model that has never been loaded. `verify` re-checks every file the manifest lists, by size and by hash, and
for a model fetched some other way (no manifest) checks that each safetensors file is as long as its own header says it should be,
which is what a download cut short looks like.
"""
import json
import os
import struct

HEADER_LIMIT = 64 << 20          # a safetensors header larger than this is not one
MAX_FILES = 600                  # safetensors files looked at in one folder
CONTEXT_KEYS = ("max_position_embeddings", "n_positions", "seq_length", "context_length", "max_seq_len", "model_max_length")
DTYPE_BYTES = {"F64": 8, "F32": 4, "BF16": 2, "F16": 2, "F8_E4M3": 1, "F8_E5M2": 1, "I8": 1, "U8": 1, "I16": 2, "I32": 4, "I64": 8, "BOOL": 1}
# What a dominant dtype is called for people. U8 is not "8-bit": the containers pack quantised weights (int4 included) into bytes,
# so all the header can honestly say is that they are packed bytes.
DTYPE_NAMES = {"BF16": "bf16", "F16": "fp16", "F32": "fp32", "F8_E4M3": "fp8", "F8_E5M2": "fp8", "I8": "int8", "U8": "u8 packed", "F64": "fp64"}


class HeaderError(Exception):
    """A safetensors file whose header cannot be read, with what is wrong in plain words."""


def read_header(path):
    """(header dict, data offset, file size) of one safetensors file, reading only the header."""
    size = os.path.getsize(path)
    with open(path, "rb") as handle:
        raw = handle.read(8)
        if len(raw) != 8:
            raise HeaderError("shorter than a safetensors header")
        length = struct.unpack("<Q", raw)[0]
        if length < 2 or length > HEADER_LIMIT or 8 + length > size:
            raise HeaderError(f"header length {length} does not fit in a file of {size} bytes")
        body = handle.read(length)
    try:
        header = json.loads(body)
    except ValueError:
        raise HeaderError("header is not JSON") from None
    if not isinstance(header, dict):
        raise HeaderError("header is not a table")
    return header, 8 + length, size


def check_safetensors(path):
    """None when the file is as long as its own header says; else what is wrong. A file cut short is the usual way a download fails."""
    try:
        header, offset, size = read_header(path)
    except HeaderError as error:
        return str(error)
    except OSError as error:
        return f"cannot be read ({error.strerror or error})"
    end = 0
    for name, tensor in header.items():
        if name == "__metadata__":
            continue
        try:
            end = max(end, int(tensor["data_offsets"][1]))
        except (KeyError, TypeError, ValueError, IndexError):
            return f"tensor {name} has no readable offsets"
    if offset + end != size:
        return f"is {size} bytes but its header accounts for {offset + end}" + (" (cut short)" if size < offset + end else "")
    return None


def _json(path):
    try:
        with open(path, encoding="utf-8") as handle:
            found = json.load(handle)
        return found if isinstance(found, dict) else None
    except (OSError, ValueError):
        return None


def _safetensors_files(folder):
    """The safetensors files in a model folder: at its top, else one level down (a pipeline keeps transformer/, vae/, ...)."""
    out = []
    try:
        names = sorted(os.listdir(folder))
    except OSError:
        return out
    for name in names:
        path = os.path.join(folder, name)
        if name.endswith(".safetensors") and os.path.isfile(path):
            out.append(path)
    if out:
        return out[:MAX_FILES]
    for name in names:
        path = os.path.join(folder, name)
        if os.path.isdir(path):
            try:
                for inner in sorted(os.listdir(path)):
                    if inner.endswith(".safetensors"):
                        out.append(os.path.join(path, inner))
            except OSError:
                continue
        if len(out) >= MAX_FILES:
            break
    return out[:MAX_FILES]


def _context_length(config):
    for holder in (config, config.get("text_config"), config.get("llm_config"), config.get("language_config")):
        if isinstance(holder, dict):
            for key in CONTEXT_KEYS:
                value = holder.get(key)
                if isinstance(value, int) and not isinstance(value, bool) and 0 < value < 1 << 31:
                    return value
    return None


def _architecture(config):
    names = config.get("architectures")
    if isinstance(names, list) and names and isinstance(names[0], str):
        return names[0]
    for key in ("model_type", "_class_name"):
        if isinstance(config.get(key), str):
            return config[key]
    return None


def _quantisation(by_bytes, config):
    """A short label: the dominant dtype, or the quantisation method the config names."""
    method = None
    for holder in (config, config.get("text_config") if isinstance(config.get("text_config"), dict) else {}):
        quant = holder.get("quantization_config") or holder.get("quantization")
        if isinstance(quant, dict):
            method = quant.get("quant_method") or quant.get("method") or quant.get("type")
            bits = quant.get("bits")
            if method and bits:
                method = f"{method} {bits}-bit"
            break
    if not by_bytes:
        return method
    dominant = max(by_bytes, key=by_bytes.get)
    label = DTYPE_NAMES.get(dominant, dominant.lower())
    mixed = [DTYPE_NAMES.get(d, d.lower()) for d in sorted(by_bytes, key=by_bytes.get, reverse=True)[1:] if by_bytes[d] > 0.02 * sum(by_bytes.values())]
    label += (" with " + ", ".join(mixed)) if mixed else ""
    return f"{method} ({label})" if method else label


def inspect(folder):
    """What can be said about a model folder from its small files. `problems` names, in plain words, what is missing or unreadable;
    a folder with none is one the engines can be expected to read. Never raises on a bad folder."""
    folder = os.path.abspath(folder)
    out = {"path": folder, "problems": []}
    problems = out["problems"]
    if not os.path.isdir(folder):
        problems.append("the folder is not there")
        return out
    config = _json(os.path.join(folder, "config.json")) or _json(os.path.join(folder, "model_index.json"))
    if config is None:
        problems.append("no config.json (or model_index.json) that can be read: the architecture cannot be told")
        config = {}
    out["architecture"] = _architecture(config)
    out["context_length"] = _context_length(config)
    try:
        from family_registry import display_for, resolve_model
        resolved = resolve_model(folder)
        out["family"], out["modality"] = resolved.descriptor.id, resolved.descriptor.modality
        out["name"] = display_for(resolved)[0]
    except Exception as error:                      # a folder no family claims: say which piece is missing
        if config:
            problems.append(f"Colibri has no engine for it: {str(error)[:160] or type(error).__name__}")
    by_bytes, tensors, total, shards = {}, 0, 0, 0
    for path in _safetensors_files(folder):
        try:
            header, _offset, size = read_header(path)
        except (HeaderError, OSError) as error:
            problems.append(f"{os.path.relpath(path, folder)}: {error}")
            continue
        shards += 1
        total += size
        for name, tensor in header.items():
            if name == "__metadata__" or not isinstance(tensor, dict):
                continue
            tensors += 1
            try:
                count = 1
                for dim in tensor.get("shape", []):
                    count *= int(dim)
                by_bytes[tensor["dtype"]] = by_bytes.get(tensor["dtype"], 0) + count * DTYPE_BYTES.get(tensor["dtype"], 1)
            except (KeyError, TypeError, ValueError):
                continue
    if not shards and "family" not in out:
        problems.append("no safetensors files: the weights are not here")
    out.update(shards=shards, tensors=tensors, bytes=total, quantisation=_quantisation(by_bytes, config))
    try:
        import setup_download
        manifest = setup_download.read_manifest(folder)
    except Exception:
        manifest = None
    if manifest:
        out["source"] = {"repo": manifest.get("repo"), "revision": manifest.get("revision")}
        out["complete"] = bool(manifest.get("complete"))
        if not manifest.get("complete"):
            listed = len(manifest.get("files") or [])
            problems.append(f"the download did not finish (the manifest lists {listed} files)")
    return out


def verify_model(folder, quick=False, progress=None):
    """Check that a model's files are whole. Returns {"path", "status", "checked", "bad": [{"path", "problem"}], "how"}, with status
    `ok`, `damaged` or `unverifiable`. With a manifest, every listed file must be there at its size and (unless `quick`) hash; without
    one, each safetensors file must be as long as its header says. `progress(done, total, path)` is called before each file."""
    folder = os.path.abspath(folder)
    out = {"path": folder, "bad": [], "checked": 0, "hashed": 0}
    manifest = None
    try:
        import setup_download
        manifest = setup_download.read_manifest(folder)
    except Exception:
        setup_download = None
    if manifest and manifest.get("files"):
        files = manifest["files"]
        for index, spec in enumerate(files):
            relative = spec.get("path", "")
            if progress:
                progress(index, len(files), relative)
            try:
                path = setup_download.safe_join(folder, relative)
            except Exception:
                out["bad"].append({"path": relative, "problem": "a name that leaves the folder"})
                continue
            if not os.path.exists(path):
                out["bad"].append({"path": relative, "problem": "missing"})
                continue
            size = os.path.getsize(path)
            if size != int(spec.get("size") or 0):
                out["bad"].append({"path": relative, "problem": f"is {size} bytes, should be {spec.get('size')}"})
                continue
            if not quick:
                try:
                    good = setup_download.verify_file(path, spec)
                except OSError as error:
                    out["bad"].append({"path": relative, "problem": f"cannot be read ({error.strerror or error})"})
                    continue
                if not good:
                    out["bad"].append({"path": relative, "problem": "checksum mismatch"})
                    continue
                if spec.get("sha256") or spec.get("git_oid"):
                    out["hashed"] += 1
            if relative.endswith(".safetensors"):
                problem = check_safetensors(path)
                if problem:
                    out["bad"].append({"path": relative, "problem": problem})
                    continue
            out["checked"] += 1
        if not manifest.get("complete"):
            out["bad"].append({"path": "(manifest)", "problem": "the download never finished"})
        # say what was really checked: a manifest written by an older download lists sizes but no checksums
        out["how"] = ("sizes and safetensors headers (quick: no checksums)" if quick else
                      "sizes, checksums and safetensors headers" if out["hashed"] else
                      "sizes and safetensors headers (the manifest holds no checksums)")
        out["status"] = "damaged" if out["bad"] else "ok"
        return out
    files = _safetensors_files(folder)
    out["how"] = "each safetensors file is as long as its header says (this model has no manifest to check against)"
    if not files:
        out["status"] = "unverifiable"
        out["reason"] = "no manifest and no safetensors files to check"
        return out
    for index, path in enumerate(files):
        relative = os.path.relpath(path, folder)
        if progress:
            progress(index, len(files), relative)
        problem = check_safetensors(path)
        if problem:
            out["bad"].append({"path": relative, "problem": problem})
        else:
            out["checked"] += 1
    out["status"] = "damaged" if out["bad"] else "ok"
    return out


def gigabytes(count):
    return f"{count / 2**30:.1f} GB" if count >= 2**30 else f"{count / 2**20:.0f} MB"


def describe(info):
    """The lines `coli model info` prints."""
    lines = [info.get("name") or os.path.basename(info["path"].rstrip("/\\")), f"  folder        {info['path']}"]
    if info.get("family"):
        lines.append(f"  engine        {info['family']} ({info.get('modality')})")
    if info.get("architecture"):
        lines.append(f"  architecture  {info['architecture']}")
    if info.get("quantisation"):
        lines.append(f"  weights       {info['quantisation']}")
    if info.get("context_length"):
        lines.append(f"  context       up to {info['context_length']:,} tokens (the model's own limit; the server may serve less)")
    if info.get("shards"):
        lines.append(f"  files         {info['shards']} safetensors, {info['tensors']:,} tensors, {gigabytes(info['bytes'])}")
    source = info.get("source")
    if source and source.get("repo"):
        lines.append(f"  source        {source['repo']}" + (f" @ {str(source['revision'])[:12]}" if source.get("revision") else "")
                     + ("" if info.get("complete") else "  (download unfinished)"))
    elif "complete" not in info:
        lines.append("  source        not recorded (it was not fetched by `coli setup`)")
    for problem in info["problems"]:
        lines.append(f"  ! {problem}")
    return lines
