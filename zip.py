from pathlib import Path
from typing import NamedTuple, List, Tuple, Optional, Iterable, Callable, Dict, Union, IO
import zipfile
import zlib
import time
from collections import namedtuple
import struct
import io
import tempfile
import shutil
from aspire import Opix
from emits import em, shield, attempt

# --- Core Data Structures as NamedTuples ---

# Represents a single file to be added to the archive.
# path: The destination path inside the zip file.
# data: The raw bytes of the file's content.
FileOp = namedtuple('FileOp', 'path data')

# Represents the complete manifest for creating a zip archive.
# ops: A list of FileOp tuples that define the archive's content.
# comment: An optional comment to embed in the zip file.
ZipFlexManifest = namedtuple('ZipFlexManifest', 'ops comment')

# A structure to hold the results of an energy-budgeted scan.
ScanResult = namedtuple('ScanResult', 'manifest frontier energy_spent')
CompressedAsset = namedtuple(
    'CompressedAsset', 'raw_deflate compressed_size uncompressed_size crc32 flags')

# --- Custom Reflex Archive Format ---

ReflexEntry = namedtuple('ReflexEntry', 'name off length crc')
Reflex = namedtuple(
    'Reflex', 'entries total_len f_index r_index payload_f payload_r')

MAGIC = b"DE1\x00"  # 4-byte canonical magic
ALT_MAGIC4 = b"DE1\\"  # legacy 4-byte prefix when a stray '0' byte followed


def _deflate_raw(b: bytes, level: int = 6) -> bytes:
    c = zlib.compressobj(level=level, wbits=-15)
    return c.compress(b) + c.flush()


def _inflate_raw(b: bytes) -> bytes:
    d = zlib.decompressobj(wbits=-15)
    return d.decompress(b) + d.flush()


def _two_file_zip_bytes() -> bytes:
    """Build an in-memory standard ZIP with two small files."""
    bio = io.BytesIO()
    with zipfile.ZipFile(bio, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr('u', '1')
        zf.writestr('v', '2')
    return bio.getvalue()

@em("magic: DE1\\x00", via=lambda f: (lambda b: {"magic": (b[:4].decode('latin1'))})(f(ZipFlexManifest([FileOp('a', b'a')], b't'))))
def blit_to_reflex_bytes(manifest: ZipFlexManifest, with_reverse: bool = True, level: int = 6, stats: Optional[Opix] = None) -> bytes:
    """Blits a manifest to an in-memory Reflex archive and returns its bytes."""
    stats = stats if stats is not None else Opix()
    tape = io.BytesIO()
    entries = []
    pos = 0
    for op in manifest.ops:
        entries.append(ReflexEntry(
            op.path, pos, len(op.data), zlib.crc32(op.data)))
        tape.write(op.data)
        pos += len(op.data)

    tape_bytes = tape.getvalue()

    payload_f = _deflate_raw(tape_bytes, level=level)
    payload_r = _deflate_raw(
        tape_bytes[::-1], level=level) if with_reverse else b""

    f_index, r_index = [], []

    stats['ops'] += len(manifest.ops)
    stats['uncompressed'] += len(tape_bytes)
    stats['compressed_f'] += len(payload_f)
    if with_reverse:
        stats['compressed_r'] += len(payload_r)

    reflex = Reflex(entries, len(tape_bytes),
                    f_index, r_index, payload_f, payload_r)
    return _pack_reflex(reflex)


def _pack_reflex(reflex: Reflex) -> bytes:
    # Simplified packing logic
    flags = 1 if reflex.payload_r else 0
    toc = io.BytesIO()
    for e in reflex.entries:
        nb = e.name.encode("utf-8")
        toc.write(struct.pack("<H", len(nb)))
        toc.write(nb)
        toc.write(struct.pack("<QQI", e.off, e.length, e.crc))

    header = [
        MAGIC,
        struct.pack("<HIQ", flags, len(reflex.entries), reflex.total_len),
        toc.getvalue(),
        struct.pack("<Q", len(reflex.payload_f)),
        reflex.payload_f,
        struct.pack("<Q", len(reflex.payload_r)),
        reflex.payload_r,
    ]
    return b"".join(header)


@em("assets: 1", via=lambda f: (lambda d: {"assets": len(d)})(f(blit_to_reflex_bytes(ZipFlexManifest([FileOp('x', b'1')], b't')))))
def unpack_reflex(blob: bytes, stats: Optional[Opix] = None) -> Dict[str, bytes]:
    """Unpacks a Reflex archive into a dictionary of files."""
    stats = stats if stats is not None else Opix()
    br = io.BytesIO(blob)

    sig4 = br.read(4)
    if sig4 == ALT_MAGIC4:
        # Legacy: skip one stray byte to realign
        _ = br.read(1)
    elif sig4 != MAGIC:
        stats['!'] += 1  # Invalid magic number
        return {}

    try:
        flags, n_entries, total_len = struct.unpack("<HIQ", br.read(14))
        entries = []
        for _ in range(n_entries):
            (nl,) = struct.unpack("<H", br.read(2))
            name = br.read(nl).decode("utf-8")
            off, length, crc = struct.unpack("<QQI", br.read(20))
            entries.append(ReflexEntry(name, off, length, crc))

        (f_pay_len,) = struct.unpack("<Q", br.read(8))
        payload_f = br.read(f_pay_len)

        tape = _inflate_raw(payload_f)

        files = {}
        for e in entries:
            data = tape[e.off:e.off + e.length]
            if zlib.crc32(data) != e.crc:
                stats[f"≠:{e.name}"] += 1  # CRC mismatch
                continue
            files[e.name] = data

        return files
    except (struct.error, zlib.error, UnicodeDecodeError):
        stats['⟂'] += 1  # Corrupted data
        return {}


# --- Spec-as-test: Reflex and ZIP recovery flows ---


@em("assets: 2")
def test_reflex_roundtrip_small() -> Dict[str, int]:
    """Roundtrip two small files via Reflex bytes and ensure both return."""
    a = FileOp('a.txt', b'hi')
    b = FileOp('b.txt', b'bye')
    manifest = ZipFlexManifest([a, b], comment=b'test')
    blob = blit_to_reflex_bytes(manifest)
    files = unpack_reflex(blob, Opix())
    return {"assets": len(files)}


@em("assets: 2")
def test_standard_zip_extract_in_memory() -> Dict[str, int]:
    """Create a standard ZIP in memory and extract assets without writing to disk."""
    bio = io.BytesIO()
    with zipfile.ZipFile(bio, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr('x.txt', 'alpha')
        zf.writestr('y.txt', 'beta')
    data = bio.getvalue()
    assets, _ = extract_zip_deflate_streams(data, Opix())
    return {"assets": len(assets)}


@em("bang: 1")
def test_reflex_recovery_invalid_magic() -> Dict[str, int]:
    """Feed invalid bytes and ensure the loader flags an error."""
    stats = Opix()
    files = unpack_reflex(b"NOPE", stats)
    return {"bang": int(stats.get('!', 0)), "files": len(files)}


@em("crc_mismatches: 2")
def test_reflex_crc_mismatch_detection() -> Dict[str, int]:
    """Corrupt the payload and ensure CRC mismatches are counted."""
    a = FileOp('a.txt', b'hi')
    b = FileOp('b.txt', b'bye')
    manifest = ZipFlexManifest([a, b], comment=b'test')
    blob = blit_to_reflex_bytes(manifest)
    # Corrupt a byte in the forward payload region (after header+toc+len)
    br = io.BytesIO(blob)
    try:
        br.seek(4)
        hdr = br.read(14)
        if len(hdr) != 14:
            return {"crc_mismatches": 0}
        _flags, n_entries, _total = struct.unpack('<HIQ', hdr)
        # walk toc to the end
        for _ in range(n_entries):
            nl_b = br.read(2)
            if len(nl_b) != 2:
                return {"crc_mismatches": 0}
            (nl,) = struct.unpack('<H', nl_b)
            br.seek(nl, io.SEEK_CUR)
            br.seek(20, io.SEEK_CUR)
        # f_len position
        f_len_b = br.read(8)
        if len(f_len_b) != 8:
            return {"crc_mismatches": 0}
        (f_len,) = struct.unpack('<Q', f_len_b)
        payload_off = br.tell()
    except Exception:
        return {"crc_mismatches": 0}

    mutated = bytearray(blob)
    if f_len >= 4:
        mid = payload_off + f_len // 2
        if mid < len(mutated):
            mutated[mid] ^= 0xFF
    stats = Opix()
    files = unpack_reflex(bytes(mutated), stats)
    crc_mismatches = sum(1 for k in stats.keys() if isinstance(k, str) and k.startswith('≠:'))
    return {"crc_mismatches": int(crc_mismatches), "files": len(files)}


# Additional compressed asset coverage
@em("assets: 2")
def test_zip_deflate_streams_compressed_assets() -> Dict[str, int]:
    data = _two_file_zip_bytes()
    assets, compressed = extract_zip_deflate_streams(data, Opix())
    return {"assets": len(assets)}


# --- Manifest Creation (Energy-Budgeted Scan) ---


class DirectoryScanner:
    def __init__(self, root: Path, energy_budget: int, filter_fn: Optional[Callable[[Path], bool]] = None, comment: str = '', stats: Optional[Opix] = None):
        self.root = root
        self.energy_budget = energy_budget
        self.filter_fn = filter_fn
        self.comment = comment
        self.stats = stats if stats is not None else Opix()
        self.ops: List[FileOp] = []
        self.frontier: List[Path] = []
        self.energy_spent = 0
        self.paths_to_scan: List[Tuple[Path, str]] = [(root.resolve(), '')]

    def scan(self) -> ScanResult:
        """Creates a manifest by recursively scanning a directory within an energy budget."""
        while self.paths_to_scan:
            current_root, current_prefix = self.paths_to_scan.pop(0)

            if self.energy_spent >= self.energy_budget and not current_root.is_dir():
                self.frontier.append(current_root)
                continue

            self._scan_directory(current_root, current_prefix)

        return ScanResult(ZipFlexManifest(self.ops, self.comment), self.frontier, self.energy_spent)

    def _scan_directory(self, current_root: Path, current_prefix: str):
        """Helper to scan a single directory."""
        try:
            paths = sorted(current_root.iterdir())
            for p in paths:
                if self.filter_fn and not self.filter_fn(p):
                    continue

                if p.is_dir():
                    self.paths_to_scan.insert(0, (p, current_prefix))
                elif p.is_file():
                    self._process_file(p, current_root, current_prefix)
        except OSError:
            self.stats['!'] += 1  # Directory not readable

    def _process_file(self, p: Path, current_root: Path, prefix: str):
        """Helper to process a single file."""
        relative_path = Path(prefix) / p.relative_to(current_root)

        try:
            file_size = p.stat().st_size
        except IOError:
            self.stats['∅'] += 1  # File not stat-able
            return

        if self.energy_spent + file_size > self.energy_budget:
            self.frontier.append(p)
            return

        if p.suffix.lower() == '.zip':
            self._process_zip_file(p, relative_path)
            return

        data = attempt(lambda: p.read_bytes(), alerts=self.stats, glyph='∅', tag='read', default=None, exceptions=(IOError, OSError))
        if data is None:
            return
        self.ops.append(FileOp(relative_path.as_posix(), data))
        self.energy_spent += file_size

    def _process_zip_file(self, p: Path, relative_path: Path):
        """Helper to process a zip file."""
        # In-memory processing only; never extract to disk
        zf = attempt(lambda: zipfile.ZipFile(p, 'r'), alerts=self.stats, glyph='⟂', tag='zip', default=None, exceptions=(zipfile.BadZipFile,))
        if zf is None:
            return
        try:
            with zf:
                for info in zf.infolist():
                    name = info.filename
                    # Skip directories and unsafe paths
                    if name.endswith('/') or name.startswith('/') or any(part == '..' for part in name.split('/')):
                        continue
                    try:
                        data = zf.read(name)
                    except Exception:
                        self.stats['∅'] += 1
                        continue
                    file_size = len(data)
                    if self.energy_spent + file_size > self.energy_budget:
                        self.frontier.append(p)
                        return
                    op_path = (relative_path / name).as_posix()
                    self.ops.append(FileOp(op_path, data))
                    self.energy_spent += file_size
        except Exception:
            self.stats['⟂'] += 1


# --- Zip Archive Reading ---


def _handle_zip_entry(zf: zipfile.ZipFile, info: zipfile.ZipInfo, f_obj: IO[bytes], stats: Opix) -> Tuple[Optional[FileOp], Optional[Tuple[str, CompressedAsset]]]:
    """Processes a single entry in a zip file."""
    name = info.filename
    try:
        data = zf.read(name)
    except (KeyError, zlib.error):
        stats['∅'] += 1
        return None, None

    file_op = FileOp(name, data)
    compressed_asset = None

    if info.compress_type == zipfile.ZIP_DEFLATED:
        raw_deflate = _extract_raw_deflate(f_obj, info)
        if raw_deflate:
            actual_crc = zlib.crc32(data) & 0xffffffff
            if actual_crc != info.CRC:
                stats['≠'] += 1

            compressed_asset = (name, CompressedAsset(
                raw_deflate=raw_deflate,
                compressed_size=info.compress_size,
                uncompressed_size=info.file_size,
                crc32=info.CRC,
                flags=info.flag_bits
            ))
        else:
            stats['⟂'] += 1

    return file_op, compressed_asset


@em("assets: 2", via=lambda f: {"assets": len(f(_two_file_zip_bytes(), Opix())[0])})
@em("compressed: 2", via=lambda f: {"compressed": len(f(_two_file_zip_bytes(), Opix())[1])})
def extract_zip_deflate_streams(zip_path_or_bytes: Union[str, bytes], stats: Opix) -> Tuple[Dict[str, bytes], Dict[str, CompressedAsset]]:
    """
    Extracts raw DEFLATE streams from a ZIP file or bytes without decompression.
    """
    extra_assets = {}
    compressed_assets = {}
    f_obj = None

    try:
        if isinstance(zip_path_or_bytes, str):
            f_obj = attempt(lambda: open(zip_path_or_bytes, 'rb'), alerts=stats, glyph='!', tag='open', default=None, exceptions=(FileNotFoundError, OSError))
            if f_obj is None:
                return extra_assets, compressed_assets
        else:
            f_obj = io.BytesIO(zip_path_or_bytes)
        
        zf = attempt(lambda: zipfile.ZipFile(f_obj, 'r'), alerts=stats, glyph='!', tag='zip', default=None, exceptions=(zipfile.BadZipFile,))
        if zf is None:
            return extra_assets, compressed_assets

        with zf:
            for info in zf.infolist():
                file_op, compressed_asset_tuple = _handle_zip_entry(
                    zf, info, f_obj, stats)
                if file_op:
                    extra_assets[file_op.path] = file_op.data
                if compressed_asset_tuple:
                    name, asset = compressed_asset_tuple
                    compressed_assets[name] = asset

    finally:
        if f_obj and isinstance(zip_path_or_bytes, str):
            try:
                f_obj.close()
            except Exception:
                pass

    return extra_assets, compressed_assets


def _extract_raw_deflate(f, zip_info) -> Optional[bytes]:
    """Extracts raw deflate data from a ZIP entry."""
    f.seek(zip_info.header_offset)
    hdr = f.read(30)
    if len(hdr) != 30:
        return None

    sig, ver, flag, comp, mt, md, crc32, comp_size, file_size, nlen, xlen = \
        struct.unpack('<IHHHHHIIIHH', hdr)

    if sig != 0x04034b50:
        return None

    f.seek(zip_info.header_offset + 30 + nlen + xlen)
    raw_deflate = f.read(zip_info.compress_size)

    if len(raw_deflate) == zip_info.compress_size:
        return raw_deflate

    return None
