from pathlib import Path
from typing import NamedTuple, List, Tuple, Optional, Iterable, Callable, Dict, Union, IO, Any
import zipfile
import zlib
import time
from collections import namedtuple
import struct
import io
import tempfile
import shutil
from aspire import Opix
from loader import shield, attempt, unpack_reflex
from emits import em
import types
from sprixel2 import gene

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

# Magic number for Reflex archives
MAGIC = b"DE1\x00"

# Reflex entry structure
ReflexEntry = namedtuple('ReflexEntry', 'name off length crc')

# Reflex archive structure


class Reflex:
    def __init__(self, entries: List[ReflexEntry], total_len: int,
                 f_index: List[int], r_index: List[int],
                 payload_f: bytes, payload_r: bytes):
        self.entries = entries
        self.total_len = total_len
        self.f_index = f_index
        self.r_index = r_index
        self.payload_f = payload_f
        self.payload_r = payload_r


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


@em("magic: DE1\\x00")
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


@em("assets: 2")
@em("compressed: 2")
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

# --- Import Utilities ---


def import_zip(zip_path: Path) -> Dict[str, Callable[..., Any]]:
    """Import all Python modules from a zip file.
    
    Style Vector: Integration Over Protection
    - Trust the zip file structure
    - Create natural flow for dynamic imports
    """
    imported = {}
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            for info in zf.infolist():
                if info.filename.endswith('.py') and not info.filename.startswith('__'):
                    try:
                        # Read the Python code
                        code_bytes = zf.read(info.filename)
                        code_str = code_bytes.decode('utf-8')

                        # Create a module name from the file path
                        module_name = info.filename.replace(
                            '/', '.').replace('.py', '')

                        # Compile and execute the code
                        compiled = compile(
                            code_str, f'<{info.filename}>', 'exec')

                        # Create a new module
                        module = types.ModuleType(module_name)
                        exec(compiled, module.__dict__)

                        # Find callable functions in the module
                        for name, obj in module.__dict__.items():
                            if callable(obj) and not name.startswith('_'):
                                imported[name] = obj

                    except Exception:
                        # Skip problematic files, continue with others
                        continue
    except Exception:
        # If zip can't be read, return empty dict
        pass

    return imported


def import_dir(dir_path: Path) -> Dict[str, Callable[..., Any]]:
    """Import all Python modules from a directory tree.
    
    Style Vector: Object Tuning Over Attribute Checking
    - Trust the directory structure
    - Create natural flow for dynamic imports
    """
    imported = {}

    try:
        for py_file in dir_path.rglob('*.py'):
            if py_file.name.startswith('__'):
                continue

            try:
                # Read the Python code
                code_str = py_file.read_text(encoding='utf-8')

                # Create a module name from the file path
                rel_path = py_file.relative_to(dir_path)
                module_name = str(rel_path).replace(
                    '/', '.').replace('.py', '')

                # Compile and execute the code
                compiled = compile(code_str, f'<{py_file}>', 'exec')

                # Create a new module
                module = types.ModuleType(module_name)
                exec(compiled, module.__dict__)

                # Find callable functions in the module
                for name, obj in module.__dict__.items():
                    if callable(obj) and not name.startswith('_'):
                        imported[name] = obj

            except Exception:
                # Skip problematic files, continue with others
                continue

    except Exception:
        # If directory can't be read, return empty dict
        pass

    return imported


# --- Manifest Functions ---

def manifest_from_directory(root: Path, energy_budget: int, filter_fn: Optional[Callable[[Path], bool]] = None, comment: str = '', stats: Optional[Opix] = None) -> ScanResult:
    """Creates a manifest by scanning a directory within an energy budget.
    
    Style Vector: Integration Over Protection
    - Trust the DirectoryScanner to handle the scanning process
    - Return structured results for natural flow
    """
    scanner = DirectoryScanner(root, energy_budget, filter_fn, comment, stats)
    return scanner.scan()


def blit_to_reflex(manifest: ZipFlexManifest, output_path: Path, with_reverse: bool = True, level: int = 6, stats: Optional[Opix] = None):
    """Blits a manifest to a Reflex archive file.
    
    Style Vector: Object Tuning Over Attribute Checking
    - Trust the manifest has the structure it needs
    - Create natural flow from bytes to file
    """
    bytes_data = blit_to_reflex_bytes(manifest, with_reverse, level, stats)
    output_path.write_bytes(bytes_data)


# --- Hilbert Space Expansion Genes ---

# Hilbert space data structures
HilbertCoords = namedtuple('HilbertCoords', [
                           'file_density', 'deflate_entropy', 'depth_variance', 'extension_diversity'])
ExpansionVector = namedtuple('ExpansionVector', ['x', 'y', 'z', 'w'])
GenePoolMetrics = namedtuple('GenePoolMetrics', [
                             'coverage', 'diversity', 'expansion_potential', 'num_points'])

# Type analysis data structures
TypeSignature = namedtuple(
    'TypeSignature', ['name', 'params', 'return_type', 'arity', 'complexity'])
TypeCompatibility = namedtuple('TypeCompatibility', [
                               'score', 'can_compose', 'shared_types', 'conversion_path'])


@gene
@em("zip_path: str := hilbert_coords: HilbertCoords")
def zip_to_hilbert_coords(zip_path: str) -> HilbertCoords:
    """Extract Hilbert space coordinates from zip file structure.
    
    Each zip file represents a compressed measurement of a code space.
    We extract dimensional coordinates that can expand the gene pool.
    """
    import numpy as np

    try:
        assets, compressed = extract_zip_deflate_streams(zip_path, Opix())

        # Dimension 1: File count density (structural complexity)
        file_density = len(assets) / \
            max(1, len([k for k in assets.keys() if '/' in k]))

        # Dimension 2: Deflate entropy (compression information)
        deflate_entropy = 0.0
        if compressed:
            total_compressed = sum(
                asset.compressed_size for asset in compressed.values())
            total_uncompressed = sum(
                asset.uncompressed_size for asset in compressed.values())
            deflate_entropy = total_compressed / max(1, total_uncompressed)

        # Dimension 3: Path depth variance (hierarchical complexity)
        if assets:
            depths = [len(path.split('/')) for path in assets.keys()]
            depth_variance = np.var(depths) if len(depths) > 1 else 0.0
        else:
            depth_variance = 0.0

        # Dimension 4: Extension diversity (semantic breadth)
        extensions = set()
        for path in assets.keys():
            if '.' in path:
                extensions.add(path.split('.')[-1].lower())
        extension_diversity = float(len(extensions))

        # Normalize to unit hypersphere
        coords = np.array([file_density, deflate_entropy,
                          depth_variance, extension_diversity])
        norm = np.linalg.norm(coords)
        if norm > 0:
            coords = coords / norm
        else:
            coords = np.array([0.25, 0.25, 0.25, 0.25])

        return HilbertCoords(
            file_density=float(coords[0]),
            deflate_entropy=float(coords[1]),
            depth_variance=float(coords[2]),
            extension_diversity=float(coords[3])
        )

    except Exception:
        # Fallback coordinates for invalid zips
        return HilbertCoords(
            file_density=0.0,
            deflate_entropy=0.0,
            depth_variance=0.0,
            extension_diversity=1.0
        )


@gene
@em("coords_a: List[float], coords_b: List[float] := distance: float")
def hilbert_distance(coords_a: List[float], coords_b: List[float]) -> float:
    """Calculate distance between two points in Hilbert space."""
    import numpy as np

    a = np.array(coords_a)
    b = np.array(coords_b)

    # Use geodesic distance on unit hypersphere
    dot_product = np.clip(np.dot(a, b), -1.0, 1.0)
    return np.arccos(dot_product)


@gene
@em("zip_paths: List[str], target_coords: List[float] := expansion_vector: List[float]")
def compute_expansion_vector(zip_paths: List[str], target_coords: List[float]) -> List[float]:
    """Compute expansion vector for gene pool growth in Hilbert space.
    
    Given a set of zip files and target coordinates, compute the direction
    vector that maximally expands the gene pool coverage in Hilbert space.
    """
    import numpy as np

    if not zip_paths:
        return target_coords

    # Extract coordinates for all zip files
    all_coords = []
    for zip_path in zip_paths:
        coords = zip_to_hilbert_coords(zip_path)
        all_coords.append(coords)

    if not all_coords:
        return target_coords

    # Find centroid of existing points
    centroid = np.mean(all_coords, axis=0)

    # Expansion vector points from centroid toward target
    target = np.array(target_coords)
    expansion = target - centroid

    # Normalize expansion vector
    norm = np.linalg.norm(expansion)
    return (expansion / norm).tolist() if norm > 0 else target_coords


@gene
@em("zip_dir: str, max_files: int := selected_zips: List[str]")
def select_expansion_zips(zip_dir: str, max_files: int = 10) -> List[str]:
    """Select zip files that maximally expand Hilbert space coverage.
    
    Use greedy algorithm to select zip files that provide maximum
    diversity in the gene pool's Hilbert space representation.
    """
    import os
    import numpy as np

    # Find all zip files
    zip_files = []
    if os.path.isdir(zip_dir):
        for root, dirs, files in os.walk(zip_dir):
            for file in files:
                if file.lower().endswith('.zip'):
                    zip_files.append(os.path.join(root, file))

    if not zip_files:
        return []

    # Extract coordinates for all zips
    zip_coords = {}
    for zip_path in zip_files:
        coords = zip_to_hilbert_coords(zip_path)
        zip_coords[zip_path] = coords

    # Greedy selection for maximum coverage
    selected = []
    selected_coords = []

    # Start with zip closest to origin
    if zip_coords:
        origin = [0.0] * len(next(iter(zip_coords.values())))
        first_zip = min(zip_coords.items(),
                        key=lambda x: hilbert_distance(x[1], origin))
        selected.append(first_zip[0])
        selected_coords.append(first_zip[1])

    # Greedily add zips that maximize minimum distance to selected set
    remaining = {k: v for k, v in zip_coords.items() if k not in selected}

    while len(selected) < max_files and remaining:
        best_zip = None
        best_min_dist = -1

        for zip_path, coords in remaining.items():
            # Find minimum distance to any selected zip
            min_dist = min(hilbert_distance(coords, sel_coords)
                           for sel_coords in selected_coords)

            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_zip = zip_path

        if best_zip:
            selected.append(best_zip)
            selected_coords.append(remaining[best_zip])
            del remaining[best_zip]
        else:
            break

    return selected


@gene
@em("zip_paths: List[str] := gene_pool_metrics: Dict[str, float]")
def analyze_gene_pool_expansion(zip_paths: List[str]) -> Dict[str, float]:
    """Analyze how zip files expand the gene pool in Hilbert space.
    
    Returns metrics about the gene pool's coverage, diversity, and
    expansion potential in the Hilbert space of code structures.
    """
    import numpy as np

    if not zip_paths:
        return {"coverage": 0.0, "diversity": 0.0, "expansion_potential": 0.0}

    # Extract all coordinates
    all_coords = []
    for zip_path in zip_paths:
        coords = zip_to_hilbert_coords(zip_path)
        all_coords.append(coords)

    coords_array = np.array(all_coords)

    # Coverage: volume of convex hull (approximated by determinant)
    if len(all_coords) >= len(all_coords[0]):
        try:
            coverage = abs(np.linalg.det(coords_array[:len(all_coords[0])]))
        except:
            coverage = 0.0
    else:
        coverage = 0.0

    # Diversity: average pairwise distance
    if len(all_coords) > 1:
        distances = []
        for i in range(len(all_coords)):
            for j in range(i + 1, len(all_coords)):
                dist = hilbert_distance(all_coords[i], all_coords[j])
                distances.append(dist)
        diversity = np.mean(distances) if distances else 0.0
    else:
        diversity = 0.0

    # Expansion potential: distance from centroid to unit sphere boundary
    centroid = np.mean(coords_array, axis=0)
    centroid_norm = np.linalg.norm(centroid)
    expansion_potential = max(0.0, 1.0 - centroid_norm)

    return GenePoolMetrics(
        coverage=float(coverage),
        diversity=float(diversity),
        expansion_potential=float(expansion_potential),
        num_points=len(all_coords)
    )


# --- Type Understanding Genes ---

@gene
@em("fn: Callable := type_signature: TypeSignature")
def analyze_function_types(fn: Callable[..., Any]) -> TypeSignature:
    """Analyze the type signature of a function for compatibility assessment.
    
    Extracts parameter types, return type, arity, and complexity metrics
    that can be used for gene recombination and composition.
    """
    import inspect
    from typing import get_type_hints

    try:
        sig = inspect.signature(fn)
        name = getattr(fn, '__name__', 'anonymous')

        # Get type hints if available
        try:
            type_hints = get_type_hints(fn)
        except:
            type_hints = {}

        # Analyze parameters
        params = []
        required_count = 0
        for param_name, param in sig.parameters.items():
            param_type = type_hints.get(param_name, 'Any')
            is_required = param.default is inspect.Parameter.empty
            if is_required:
                required_count += 1

            params.append({
                'name': param_name,
                'type': str(param_type),
                'required': is_required,
                'has_default': not is_required
            })

        # Get return type
        return_type = str(type_hints.get('return', 'Any'))

        # Calculate complexity score
        complexity = len(params) + (2 if return_type != 'Any' else 0)
        if any('Union' in p['type'] or 'Optional' in p['type'] for p in params):
            complexity += 1
        if 'Union' in return_type or 'Optional' in return_type:
            complexity += 1

        return TypeSignature(
            name=name,
            params=tuple(params),
            return_type=return_type,
            arity=required_count,
            complexity=complexity
        )

    except Exception:
        # Fallback for functions we can't analyze
        return TypeSignature(
            name=getattr(fn, '__name__', 'unknown'),
            params=(),
            return_type='Any',
            arity=0,
            complexity=1
        )


@gene
@em("sig_a: TypeSignature, sig_b: TypeSignature := compatibility: TypeCompatibility")
def compute_type_compatibility(sig_a: TypeSignature, sig_b: TypeSignature) -> TypeCompatibility:
    """Compute compatibility score between two function type signatures.
    
    Determines if functions can be composed, what types they share,
    and potential conversion paths for gene recombination.
    """

    # Extract type strings from both signatures
    types_a = set()
    types_b = set()

    # Add parameter types
    for param in sig_a.params:
        types_a.add(param['type'])
    for param in sig_b.params:
        types_b.add(param['type'])

    # Add return types
    types_a.add(sig_a.return_type)
    types_b.add(sig_b.return_type)

    # Calculate shared types
    shared_types = types_a.intersection(types_b)

    # Base compatibility score (Jaccard similarity)
    all_types = types_a.union(types_b)
    base_score = len(shared_types) / len(all_types) if all_types else 0.0

    # Check if functions can compose (output of A matches input of B)
    can_compose_ab = False
    can_compose_ba = False

    for param in sig_b.params:
        if param['type'] == sig_a.return_type or sig_a.return_type == 'Any':
            can_compose_ab = True
            break

    for param in sig_a.params:
        if param['type'] == sig_b.return_type or sig_b.return_type == 'Any':
            can_compose_ba = True
            break

    can_compose = can_compose_ab or can_compose_ba

    # Boost score for composable functions
    if can_compose:
        base_score += 0.3

    # Boost score for similar arity
    arity_diff = abs(sig_a.arity - sig_b.arity)
    if arity_diff <= 1:
        base_score += 0.2
    elif arity_diff <= 2:
        base_score += 0.1

    # Simple conversion path heuristic
    conversion_path = []
    if can_compose_ab:
        conversion_path.append(f"{sig_a.name} -> {sig_b.name}")
    if can_compose_ba:
        conversion_path.append(f"{sig_b.name} -> {sig_a.name}")

    return TypeCompatibility(
        score=min(1.0, base_score),
        can_compose=can_compose,
        shared_types=tuple(sorted(shared_types)),
        conversion_path=tuple(conversion_path)
    )


@gene
@em("functions: List[Callable] := type_graph: Dict[str, List[str]]")
def build_type_composition_graph(functions: List[Callable[..., Any]]) -> Dict[str, List[str]]:
    """Build a graph of type-compatible functions for composition chains.
    
    Creates a directed graph where edges represent type compatibility,
    enabling discovery of function composition chains in the gene pool.
    """

    # Analyze all function signatures
    signatures = {}
    for fn in functions:
        sig = analyze_function_types(fn)
        signatures[sig.name] = sig

    # Build adjacency list for composition graph
    graph = {name: [] for name in signatures.keys()}

    for name_a, sig_a in signatures.items():
        for name_b, sig_b in signatures.items():
            if name_a != name_b:
                compatibility = compute_type_compatibility(sig_a, sig_b)

                # Add edge if functions can compose with good compatibility
                if compatibility.can_compose and compatibility.score > 0.5:
                    graph[name_a].append(name_b)

    return graph


@gene
@em("type_str: str := canonical_type: str")
def canonicalize_type(type_str: str) -> str:
    """Convert type string to canonical form for better matching.
    
    Normalizes type representations to improve compatibility detection
    between functions with equivalent but differently expressed types.
    """

    # Remove whitespace
    canonical = type_str.replace(' ', '')

    # Normalize common patterns
    replacements = {
        'typing.': '',
        'builtins.': '',
        'collections.abc.': '',
        'numpy.ndarray': 'ndarray',
        'numpy.array': 'ndarray',
        'Dict[str,': 'Dict[str, ',
        'List[': 'List[',
        'Optional[': 'Union[None, ',
    }

    for old, new in replacements.items():
        canonical = canonical.replace(old, new)

    # Sort Union types for consistency
    if 'Union[' in canonical:
        try:
            # Simple heuristic: sort comma-separated types in Union
            import re
            union_match = re.search(r'Union\[([^\]]+)\]', canonical)
            if union_match:
                union_content = union_match.group(1)
                sorted_types = ', '.join(sorted(union_content.split(', ')))
                canonical = canonical.replace(
                    union_match.group(0), f'Union[{sorted_types}]')
        except:
            pass  # Keep original if sorting fails

    return canonical


@gene
@em("fn: Callable, target_type: str := conversion_fn: Optional[Callable]")
def suggest_type_conversion(fn: Callable[..., Any], target_type: str) -> Optional[Callable[..., Any]]:
    """Suggest a conversion function to bridge type gaps.
    
    Analyzes function output type and suggests built-in or common
    conversion functions to reach the target type.
    """

    sig = analyze_function_types(fn)
    source_type = canonicalize_type(sig.return_type)
    target_canonical = canonicalize_type(target_type)

    # Common conversion mappings
    conversions = {
        ('str', 'int'): int,
        ('str', 'float'): float,
        ('int', 'str'): str,
        ('float', 'str'): str,
        ('int', 'float'): float,
        ('float', 'int'): int,
        ('list', 'tuple'): tuple,
        ('tuple', 'list'): list,
        ('str', 'list'): list,
        ('Any', target_canonical): lambda x: x,  # Identity for Any
        (source_type, 'Any'): lambda x: x,  # Identity to Any
    }

    # Direct conversion available
    conversion_key = (source_type, target_canonical)
    if conversion_key in conversions:
        return conversions[conversion_key]

    # Check for numpy conversions
    if 'ndarray' in source_type and target_canonical in ['list', 'tuple']:
        return lambda x: x.tolist() if hasattr(x, 'tolist') else list(x)

    if source_type in ['list', 'tuple'] and 'ndarray' in target_canonical:
        try:
            import numpy as np
            return np.array
        except ImportError:
            pass

    # No conversion found
    return None
