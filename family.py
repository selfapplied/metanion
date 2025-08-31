# build_bytecode_reflex.py (Python 3.13)
import marshal, types, pathlib
from eonyx import zipflex
from zipflex import build_reflex_from_members, Reflex # your file above

def compile_dir_to_bc(src_dir: str) -> dict[str, bytes]:
    members = {}
    for p in pathlib.Path(src_dir).rglob("*.py"):
        name = p.relative_to(src_dir).with_suffix("").as_posix().replace("/", ".")
        src = p.read_text(encoding="utf-8")
        co = compile(src, filename=str(p), mode="exec", optimize=2, dont_inherit=True)
        blob = marshal.dumps(co, version=5)  # Py3.13 default
        members[f"__bc__/{name}"] = blob
    return members

def write_self_hosting_reflex(src_dir: str, extras: dict[str, bytes], out_path: str):
    members = compile_dir_to_bc(src_dir) | extras  # include your bootstrap, config, etc.
    blob = build_reflex_from_members(members, chunk=64*1024, with_reverse=True, level=6)
    pathlib.Path(out_path).write_bytes(blob)

# usage:
# write_self_hosting_reflex("toolsrc", {"__boot__/entry": b""}, "tool.reflex")
#  2) Meta-path loader (runtime inside Python)
#  Finder+Loader that reads marshalled code objects from an opened reflex blob.
# reflex_loader.py (ship this bytecode too)
import importlib.abc, importlib.util, marshal, types, sys
import importlib.machinery

class ReflexModuleSpec(importlib.machinery.ModuleSpec):
    pass

class ReflexBytecodeLoader(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def __init__(self, blob: bytes):
        self.rx = Reflex.unpack(blob)
        self.tape = _inflate_raw(self.rx.payload_f)
        # index: mod -> memoryview of marshalled code
        self.index = {}
        for e in self.rx.entries:
            if e.name.startswith("__bc__/"):
                mv = memoryview(self.tape)[e.off:e.off+e.length]
                self.index[e.name[7:]] = bytes(mv)

    # Finder
    def find_spec(self, fullname, path, target=None):
        if fullname in self.index:
            return ReflexModuleSpec(fullname, self)
        return None

    # Loader
    def create_module(self, spec):  # default module is fine
        return None

    def exec_module(self, module):
        blob = self.index[module.__spec__.name]
        co = marshal.loads(blob)
        module.__loader__ = self
        module.__package__ = module.__spec__.name.rpartition(".")[0]
        exec(co, module.__dict__)

def install_reflex_loader(reflex_bytes: bytes, priority: bool = True):
    loader = ReflexBytecodeLoader(reflex_bytes)
    if priority:
        sys.meta_path.insert(0, loader)
    else:
        sys.meta_path.append(loader)
    return loader

# 3) Bootstrap entry (self-contained CLI inside the archive)
# This tiny script installs the loader for itself, then exposes converters.

# __main__.py (include in your packed sources)
import sys, pathlib
from zipflex import read_zip_members, build_reflex_from_members, extract_members_from_reflex

def main():
    # open the current .reflex file and install loader
    me = pathlib.Path(sys.argv[0])
    blob = me.read_bytes() if me.suffix == ".reflex" else pathlib.Path(sys.argv[1]).read_bytes()
    install_reflex_loader(blob)

    if sys.argv[-2:] and sys.argv[1] == "zip2reflex":
        src, dst = sys.argv[2], sys.argv[3]
        pathlib.Path(dst).write_bytes(zip_to_reflex(src))
        return
    if sys.argv[-2:] and sys.argv[1] == "reflex2zip":
        src, dst = sys.argv[2], sys.argv[3]
        reflex_to_zip(pathlib.Path(src).read_bytes(), dst)
        return
    print("usage:\n  reflex.RUN zip2reflex <zip> <out.reflex>\n  reflex.RUN reflex2zip <in.reflex> <out.zip>")

if __name__ == "__main__":
    main()

# 4) Family genes per directory (inside archive)
# Let the bytecode define how families are made; loader makes them importable.

# tools/family.py (compiled & packed)
from collections import Counter
from pathlib import Path
import msgpack

def counts_from_dir(d: str) -> Counter:
    c = Counter()
    for p in Path(d).iterdir():
        if p.is_file():
            try:
                txt = p.read_text("utf-8", errors="ignore")
                for tok in __import__("re").findall(r"[A-Za-z0-9_]+", txt):
                    c[tok] += 1
            except Exception:
                pass
    return c

def build_family(dir_path: str) -> bytes:
    # returns msgpack bytes; caller decides where to store
    c = counts_from_dir(dir_path)
    obj = {"version":1, "counts": dict(c)}
    return msgpack.packb(obj, use_bin_type=True, strict_types=True)

# 5) Converters that the archive “writes itself”
# These live as bytecode inside __bc__/tools/zipconv.py.

# tools/zipconv.py
import zipfile, io, pathlib, msgpack
from collections import Counter
from zipflex import build_reflex_from_members, extract_members_from_reflex

def zip_to_reflex(zip_path: str, *, with_reverse=True, level=6, chunk=65536) -> bytes:
    members = {}
    with zipfile.ZipFile(zip_path, "r") as zf:
        for n in zf.namelist():
            members[n] = zf.read(n)
    # Optionally add a per-dir family gene:
    fam = _family_for_members(members)
    if fam:
        members["registry/family.mpk"] = fam
    return build_reflex_from_members(members, chunk=chunk, with_reverse=with_reverse, level=level)

def reflex_to_zip(reflex_blob: bytes, dst_zip_path: str) -> None:
    parts = extract_members_from_reflex(reflex_blob)
    with zipfile.ZipFile(dst_zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for n, mv in parts.items():
            zf.writestr(n, bytes(mv))

def _family_for_members(members: dict[str, bytes]) -> bytes | None:
    # Build one family over top-level names only
    # (You can group by prefix here for per-subdir families)
    tmpdir = pathlib.Path(".").resolve()
    return build_family(str(tmpdir))  # or derive directly from members if you prefer


