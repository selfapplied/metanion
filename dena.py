# dena.py — generate_dena()
# Python 3.13; stdlib + msgpack; expects reflex.py in path (from earlier)
from pathlib import Path
from typing import Iterable, Dict, Optional, Any
from collections import Counter
import io, os, zipfile, msgpack, marshal, time

from zipflex import build_reflex_from_members  # forward+reverse solid packer

def _read_dir_recursive(root: Path) -> Dict[str, bytes]:
    """Collect files under root, names are posix paths relative to root."""
    root = root.resolve()
    out: Dict[str, bytes] = {}
    for p in root.rglob("*"):
        if p.is_file():
            rel = p.relative_to(root).as_posix()
            out[rel] = p.read_bytes()
    return out

def _read_zip(path: Path) -> Dict[str, bytes]:
    out: Dict[str, bytes] = {}
    with zipfile.ZipFile(path, "r") as zf:
        for n in zf.namelist():
            out[n] = zf.read(n)
    return out

def _compile_dir_to_bytecode(src_dir: Path) -> Dict[str, bytes]:
    """Compile *.py to marshalled code objects under __bc__/module.path."""
    members: Dict[str, bytes] = {}
    for p in src_dir.rglob("*.py"):
        mod = p.relative_to(src_dir).with_suffix("").as_posix().replace("/", ".")
        src = p.read_text(encoding="utf-8")
        co = compile(src, filename=str(p), mode="exec", optimize=2, dont_inherit=True)
        members[f"__bc__/{mod}"] = marshal.dumps(co, version=5)  # 3.13 marshal
    return members

def _family_counts_for_dir(dir_path: Path, limit_bytes: int = 1<<16) -> Counter[str]:
    """Ultra-light token counts for a single directory (non-recursive)."""
    import re
    tok = re.compile(r"[A-Za-z0-9_]+")
    c = Counter()
    for p in dir_path.iterdir():
        if p.is_file():
            try:
                txt = p.read_bytes()[:limit_bytes].decode("utf-8", errors="ignore")
            except Exception:
                continue
            for t in tok.findall(txt):
                c[t] += 1
    return c

def generate_dena(
    sources: Iterable[str | os.PathLike],
    *,
    py_tools_dir: Optional[str | os.PathLike] = None,   # compile & embed as __bc__/...
    include_family: bool = True,                         # write per-dir counts
    chunk: int = 64 * 1024,                              # waypoint spacing (swap later w/ FFT)
    with_reverse: bool = True,                           # double-ended
    level: int = 6,                                      # deflate level
    out_path: Optional[str | os.PathLike] = None,        # if set, write file & return path
    manifest_extra: Optional[Dict[str, Any]] = None,     # user metadata
) -> bytes | str:
    """
    Build a DENA: a reflex double-ended solid + manifest + (optional) bytecode & families.

    - sources: mix of directories and .zip files; directory contents stored under their
      relative paths; zip members stored as-is.
    - py_tools_dir: if set, *.py compiled to marshalled code objects under __bc__/.
    - include_family: adds registry/family/<dirname>.mpk (simple unigram counts).
    - manifest at registry/dena.mpk: {version, created, entries, opts, families, modules}
    - returns bytes (if out_path=None) or writes file and returns its path.
    """
    # 1) Gather members
    members: Dict[str, bytes] = {}
    families: Dict[str, dict] = {}
    src_list = [Path(s) for s in sources]

    for src in src_list:
        if src.is_dir():
            # stash files with a directory prefix
            base = src.resolve()
            for rel, data in _read_dir_recursive(base).items():
                members[f"{src.name}/{rel}"] = data
            if include_family:
                counts = _family_counts_for_dir(base)
                families[src.name] = dict(counts)
        elif src.is_file() and src.suffix.lower() == ".zip":
            for n, b in _read_zip(src).items():
                members[n] = b
        else:
            # single file: drop at top-level by name
            members[src.name] = src.read_bytes()

    # 2) Embed Python tool bytecode (optional)
    modules: Dict[str, int] = {}
    if py_tools_dir:
        bc = _compile_dir_to_bytecode(Path(py_tools_dir))
        members.update(bc)
        # manifest summarization: sizes only (no code)
        modules = {k: len(v) for k, v in bc.items()}

    # 3) Manifest (tucked inside the archive)
    manifest: Dict[str, Any] = {
        "version": 1,
        "created": int(time.time()),
        "entries": len(members),
        "opts": {"chunk": chunk, "with_reverse": with_reverse, "level": level},
        "families": {k: {"version": 1, "unigrams": v} for k, v in families.items()},
        "modules": modules,
    }
    if manifest_extra:
        manifest.update({"extra": manifest_extra})
    members["registry/dena.mpk"] = msgpack.packb(manifest, use_bin_type=True, strict_types=True)

    # 4) Build the reflex payload (forward+reverse solid)
    blob = build_reflex_from_members(members, chunk=chunk, with_reverse=with_reverse, level=level)

    # 5) Output
    if out_path:
        outp = Path(out_path)
        outp.write_bytes(blob)
        return outp.as_posix()
    return blob

# 1) From a directory (recursively) + Python tools
dena_bytes = generate_dena(
    ["./data_dir", "./assets.zip"], 
    py_tools_dir="./toolsrc",
    include_family=True,
    chunk=65536, with_reverse=True, level=6
)

# 2) Write straight to file
dst = generate_dena(["./corpus", "./plugin.zip"], py_tools_dir="./bytecode", out_path="bundle.dena")
print("wrote:", dst)
