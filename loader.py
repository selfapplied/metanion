from pathlib import Path
from typing import Dict, Callable, Any, Optional, Union, IO, Tuple, Iterable, cast
from collections import Counter, namedtuple
import types
from aspire import Opix
import zipfile
import zlib
import io
import struct
from contextlib import contextmanager
import warnings
import numpy as np
from collections import Counter
from bitstring import ConstBitStream, ReadError
from typing import List


# Structured event logging with namedtuple oneliners
EventLog = namedtuple('EventLog', 'glyph exception_name module_name')
ZipBuffer = namedtuple('ZipBuffer', 'data stats')
ModuleResult = namedtuple('ModuleResult', 'name callables success')


# Global event counter for tracking issues
event_counter: Counter[str] = Counter()


def note_event(glyph: str, exception_name: str, module_name: str) -> None:
    """Record an event in our structured log."""
    # TODO: use buffer surface instead]=
    event = EventLog(glyph, exception_name, module_name)
    event_counter[f"{glyph}:{exception_name}:{module_name}"] += 1
    event_counter[f"{glyph}:{module_name}"] += 1
    event_counter[glyph] += 1


def make_zip_buffer(zip_path: Path) -> ZipBuffer:
    """Create a manipulable zip buffer using our custom parser."""
    stats = Opix()

    zip_bytes = attempt(lambda: zip_path.read_bytes(), alerts=stats, glyph='📖', tag='read', default=None)
    if zip_bytes:
        assets = unpack_reflex(zip_bytes, stats)
        return ZipBuffer(assets, stats)
    
    assets, compressed = extract_zip_deflate_streams(str(zip_path), stats)
    return ZipBuffer(assets, stats)


def compile_safe(code_str: str, filename: str, module_name: str) -> Optional[types.CodeType]:
    """Safely compile Python code with event logging."""
    return attempt(
        lambda: compile(code_str, filename, 'exec'),
        alerts=event_counter,
        glyph='⟂',
        tag=module_name
    )


def exec_safe(compiled: types.CodeType, module: types.ModuleType, module_name: str) -> bool:
    """Safely execute compiled code with event logging."""
    success = attempt(
        lambda: exec(compiled, module.__dict__),
        alerts=event_counter,
        glyph='⚡',
        tag=module_name,
        default=False
    ) or True
    
    # Run test runner if available (after successful execution)
    if success:
        from loader import run_specs
        # Get callables from the module and run tests
        callables = {name: obj for name, obj in module.__dict__.items() 
                    if callable(obj) and not name.startswith('_')}
        if callables:
            run_specs(callables)
    
    return success


def pick_callables(module: types.ModuleType, module_name: str) -> Dict[str, Callable[..., Any]]:
    """Extract callable functions from module."""
    return {name: obj for name, obj in module.__dict__.items() 
            if callable(obj) and not name.startswith('_')}


def load_module_from_bytes(name: str, data: bytes, module_name: str) -> ModuleResult:
    """Process a single module with comprehensive error handling."""
    if not name.endswith('.py') or name.startswith('__'):
        return ModuleResult(name, {}, False)
    
    code_str = attempt(
        lambda: data.decode('utf-8'),
        alerts=event_counter,
        glyph='🔤',
        tag=module_name,
        default=None
    )
    if not code_str:
        return ModuleResult(name, {}, False)
    
    compiled = compile_safe(code_str, f'<{name}>', module_name)
    if not compiled:
        return ModuleResult(name, {}, False)
    
    module = types.ModuleType(module_name)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        warnings.simplefilter("ignore", UserWarning)
        if exec_safe(compiled, module, module_name):
            callables = pick_callables(module, module_name)
            return ModuleResult(name, callables, True)
    
    return ModuleResult(name, {}, False)


def load_zip_buffer(zip_buffer: ZipBuffer, zip_path: Path) -> Dict[str, Callable[..., Any]]:
    """Process zip buffer using sprixel composition to eliminate nesting."""
    imported = {}
    
    # Compose operations without nesting
    for name, data in zip_buffer.data.items():
        module_name = name.replace('/', '.').replace('.py', '')
        result = load_module_from_bytes(name, data, module_name)
        if result.success:
            imported.update(result.callables)
    
    return imported


def load_directory(dir_path: Path) -> Dict[str, Callable[..., Any]]:
    """Process a directory tree with comprehensive error handling."""
    imported = {}
    
    with shield(alerts=event_counter, glyph='📁', tag=str(dir_path)):
        for py_file in dir_path.rglob('*.py'):
            if py_file.name.startswith('__'):
                continue
            
            module_name = str(py_file.relative_to(dir_path)).replace('/', '.').replace('.py', '')
            
            # Read file content
            code_str = attempt(
                lambda: py_file.read_text(encoding='utf-8'),
                alerts=event_counter,
                glyph='📄',
                tag=module_name
            )
            if not code_str:
                continue
            
            # Compile and execute
            compiled = compile_safe(code_str, f'<{py_file}>', module_name)
            if not compiled:
                continue
            
            module = types.ModuleType(module_name)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                warnings.simplefilter("ignore", UserWarning)
                if exec_safe(compiled, module, module_name):
                    callables = pick_callables(module, module_name)
                    imported.update(callables)
    
    return imported


def import_zip(zip_path: Path) -> Dict[str, Callable[..., Any]]:
    """Import all Python modules from a zip file using custom parser.
    
    Style Vector: Integration Over Protection
    - Trust the zip file structure
    - Create natural flow for dynamic imports
    - Uses custom zip parser for powerful buffer manipulation
    """
    zip_buffer = make_zip_buffer(zip_path)
    return load_zip_buffer(zip_buffer, zip_path)


def import_dir(dir_path: Path) -> Dict[str, Callable[..., Any]]:
    """Import all Python modules from a directory tree.
    
    Style Vector: Object Tuning Over Attribute Checking
    - Trust the directory structure
    - Create natural flow for dynamic imports
    - Comprehensive error handling with event logging
    """
    return load_directory(dir_path)


def import_paths(paths: List[Path]) -> Dict[str, Callable[..., Any]]:
    """Import Python callables from a mixed list of directories, zip files, or .py files.

    - Directories are scanned recursively for .py files
    - Zip files are parsed via custom buffer importer
    - Single .py files are compiled and executed directly
    """
    imported: Dict[str, Callable[..., Any]] = {}
    for p in paths:
        try:
            if p.is_dir():
                imported.update(import_dir(p))
            elif p.suffix == '.zip' and p.exists():
                imported.update(import_zip(p))
            elif p.suffix == '.py' and p.exists():
                module_name = p.stem
                code_str = attempt(
                    lambda: p.read_text(encoding='utf-8'),
                    alerts=event_counter,
                    glyph='📄',
                    tag=str(p)
                )
                if not code_str:
                    continue
                compiled = compile_safe(code_str, f'<{p}>', module_name)
                if not compiled:
                    continue
                module = types.ModuleType(module_name)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", DeprecationWarning)
                    warnings.simplefilter("ignore", UserWarning)
                    if exec_safe(compiled, module, module_name):
                        callables = pick_callables(module, module_name)
                        imported.update(callables)
        except Exception:
            # Keep flowing on individual path failures
            continue
    return imported


def run_specs(name_to_fn: Dict[str, Callable[..., Any]]) -> Dict[str, int]:
    """Execute registered specs using emits registries and planning.

    - Delegates parsing/planning/matching to emits
    - Uses attempt/shield for boundary safety
    - Returns glyph counters for summary
    """
    from emits import genes as _genes_reg
    from emits import invokers as _invokers
    from emits import trust_scores as _trust
    from emits import discoveries as _disc
    from emits import plan as _plan
    from emits import match as _match
    from typing import Mapping

    glyphs: Dict[str, int] = {'✓': 0, '⊘': 0, '∅': 0, '↷': 0, '🗜': 0}
    prev = Counter(_trust)

    for name in list(_genes_reg.keys()):
        fn = name_to_fn.get(name)
        if fn is None:
            glyphs['∅'] += 1
            continue
        calls = _invokers.get(name, [])
        specs_for_fn = _genes_reg.get(name, [])
        if calls or specs_for_fn:
            for sp in specs_for_fn:
                args, kwargs, _ = _plan(sp)
                events = Counter()
                _sentinel = object()
                if args or kwargs:
                    safe_fn = cast(Callable[..., Any], fn)
                    value = attempt(lambda: safe_fn(*args, **kwargs, __em_spec=sp.raw), alerts=events, glyph='↷', tag=name, default=_sentinel)
                elif calls and len(calls) > 0 and callable(calls[0]):
                    inv = cast(Callable[[Callable[..., Any]], Any], calls[0])
                    safe_fn = cast(Callable[..., Any], fn)
                    value = attempt(lambda: inv(safe_fn), alerts=events, glyph='↷', tag=name, default=_sentinel)
                else:
                    safe_fn = cast(Callable[..., Any], fn)
                    value = attempt(lambda: safe_fn(__em_spec=sp.raw), alerts=events, glyph='↷', tag=name, default=_sentinel)
                if value is _sentinel:
                    if any(k.startswith('↷:AssertionError') for k in events.keys()):
                        glyphs['⊘'] += 1
                    else:
                        glyphs['↷'] += 1
                    continue
                m = Counter()
                did = attempt(lambda: _match(value, sp.outputs), alerts=m, glyph='↷', tag=f"match:{name}", default=False)
                if did:
                    _trust[name] += 1
                    attempt(lambda: _disc[name].append(value), glyph='✎', tag=name)
                    if isinstance(value, Mapping):
                        comp = value.get('compressed')
                        if isinstance(comp, (int, float, str)):
                            glyphs['🗜'] += int(comp)
                    glyphs['✓'] += 1
                else:
                    glyphs['⊘'] += 1
        else:
            ev = Counter()
            _sentinel = object()
            safe_fn2 = cast(Callable[..., Any], fn)
            val = attempt(lambda: safe_fn2(), alerts=ev, glyph='↷', tag=name, default=_sentinel)
            if val is _sentinel:
                if any(k.startswith('↷:AssertionError') for k in ev.keys()):
                    glyphs['⊘'] += 1
                else:
                    glyphs['↷'] += 1
            else:
                glyphs['✓'] += 1
                if isinstance(val, Mapping):
                    comp = val.get('compressed')
                    if isinstance(comp, (int, float, str)):
                        glyphs['🗜'] += int(comp)

    # α count
    alpha = 0
    for k, v in _trust.items():
        dv = v - prev.get(k, 0)
        if dv > 0:
            alpha += dv
    if alpha:
        glyphs['α'] = alpha
    return glyphs


def get_event_summary() -> Dict[str, int]:
    """Get a summary of all recorded events."""
    return dict(event_counter)


def clear_events() -> None:
    """Clear all recorded events."""
    event_counter.clear()


def note_alert(alerts: Optional[Any], glyph: str, tag: Optional[str], exc: Optional[BaseException]) -> None:
    try:
        if alerts is None:
            return
        alerts[str(glyph)] += 1
        if tag:
            alerts[f"{glyph}:{tag}"] += 1
        if exc is not None:
            cname = exc.__class__.__name__
            alerts[f"{glyph}:{cname}"] += 1
    except Exception:
        pass

def attempt(run: Callable[[], Any], *, alerts: Optional[Any] = None, glyph: str = '!', tag: Optional[str] = None,
            default: Any = None, exceptions: tuple[type[BaseException], ...] = (Exception,), rethrow: bool = False) -> Any:
    """Run a callable once; on failure, mint glyphs into alerts and return default.

    - Use ONLY at external boundaries (IO/CLI/Net). Internal code should raise.
    - glyph: short key like '!' or '⟂'; tag adds context like '!:{op}'.
    - If rethrow=True, the exception is re-raised after noting alerts.
    """
    try:
        return run()
    except exceptions as e:  # type: ignore[catching-non-exception]
        note_alert(alerts, glyph, tag, e)
        if rethrow:
            raise
        return default

@contextmanager
def shield(*, alerts: Optional[Any] = None, glyph: str = '!', tag: Optional[str] = None,
           exceptions: tuple[type[BaseException], ...] = (Exception,), rethrow: bool = False):
    """Context manager to unify try/catch. See attempt().

    Usage:
      with shield(alerts=asp, glyph='!', tag='open'):
          data = Path(p).read_bytes()
    """
    try:
        yield
    except exceptions as e:  # type: ignore[catching-non-exception]
        note_alert(alerts, glyph, tag, e)
        if rethrow:
            raise


CompressedAsset = namedtuple(
    'CompressedAsset', 'raw_deflate compressed_size uncompressed_size crc32 flags')

FileOp = namedtuple('FileOp', 'path data')

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

        tape = inflate_raw(payload_f)

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

def inflate_raw(b: bytes) -> bytes:
    d = zlib.decompressobj(wbits=-15)
    return d.decompress(b) + d.flush()

ReflexEntry = namedtuple('ReflexEntry', 'name off length crc')
MAGIC = b"DE1\x00"
ALT_MAGIC4 = b"DE1\\"

def handle_zip_entry(zf: zipfile.ZipFile, info: zipfile.ZipInfo, f_obj: IO[bytes], stats: Opix) -> Tuple[Optional[FileOp], Optional[Tuple[str, CompressedAsset]]]:
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
        raw_deflate = extract_raw_deflate(f_obj, info)
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
                file_op, compressed_asset_tuple = handle_zip_entry(
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

def extract_raw_deflate(f, zip_info) -> Optional[bytes]:
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


class BitLSB:
    """LSB-first bit reader over a ConstBitStream."""
    def __init__(self, stream: ConstBitStream):
        self.stream = stream
        self._bit_buf = 0
        self._bit_count = 0

    def _fill_byte(self):
        if self._bit_count == 0:
            byte_val = self.stream.read('uint:8')
            self._bit_buf = byte_val
            self._bit_count = 8

    def read_bit(self) -> int:
        if self._bit_count == 0:
            self._fill_byte()
        bit = self._bit_buf & 1
        self._bit_buf >>= 1
        self._bit_count -= 1
        return bit

    def read_bits(self, n: int) -> int:
        val = 0
        shift = 0
        while n > 0:
            if self._bit_count == 0:
                self._fill_byte()
            take = min(n, self._bit_count)
            mask = (1 << take) - 1
            val |= (self._bit_buf & mask) << shift
            self._bit_buf >>= take
            self._bit_count -= take
            n -= take
            shift += take
        return val

    def align_to_byte(self):
        self._bit_buf = 0
        self._bit_count = 0
        misalign = (-self.stream.pos) % 8
        if misalign:
            self.stream.pos += misalign

    def read_bytes(self, n: int) -> bytes:
        self.align_to_byte()
        return self.stream.read(f'bytes:{n}')

class ErrorTracker:
    def __init__(self):
        self.error_counts: Counter = Counter()  # Use standard Counter
        self._error_window: List[str] = []
        self._seen_states: Dict[tuple, int] = {}
        self._cycle_count: int = 0
        self._step_idx: int = 0

    def count_error(self, key: str):
        self.error_counts[key] += 1

    def record_error(self, key: str, attempted_parse: str, stream_pos: int) -> bool:
        self.count_error(key)
        self._error_window.append(key)
        if len(self._error_window) > 4:
            self._error_window.pop(0)
        
        phase = int(stream_pos % 8)
        byte_pos = int(stream_pos // 8)
        recent_hash = 0
        for k in self._error_window:
            recent_hash = hash(recent_hash + hash(k))  # Simplified hash
        
        state = (phase, attempted_parse, byte_pos, recent_hash)
        prev = self._seen_states.get(state)
        
        if prev is not None and (self._step_idx - prev) <= 128:
            self._cycle_count += 1
            if self._cycle_count >= 2:
                self.count_error('∞')
                return True
        self._seen_states[state] = self._step_idx
        self._step_idx += 1
        return False

def parse_stored_block(reader: BitLSB, on_warning: Callable[[str], None] = lambda _k: None) -> bytes:
    reader.align_to_byte()
    if (len(reader.stream) - reader.stream.pos) // 8 < 4:
        on_warning('⟂')
        return b""
    len_bytes = reader.read_bytes(2)
    nlen_bytes = reader.read_bytes(2)
    length = int.from_bytes(len_bytes, 'little')
    nlen = int.from_bytes(nlen_bytes, 'little')
    if (length ^ 0xFFFF) != nlen:
        on_warning('≠')
        return b""
    if (len(reader.stream) - reader.stream.pos) // 8 < length:
        on_warning('⋯')
        return b""
    return reader.read_bytes(length)

def compute_resonance(data: bytes) -> float:
    """Placeholder resonance computation (e.g., compression ratio)."""
    import zlib
    compressed = len(zlib.compress(data))
    return len(data) / compressed if compressed else 1.0


class BufferSurface:
    def __init__(self):
        self.buffers: Dict[str, bytes] = {}
        self.stats: Opix = Opix()

    def add_buffer(self, node: str, data: bytes) -> None:
        self.buffers[node] = data
        self.stats['📄'] += 1 # Count buffer additions

    def get_buffer(self, node: str) -> Optional[bytes]:
        return self.buffers.get(node)

    def to_floatbuffer(self, node: str) -> Optional['np.ndarray']:
        """Cast buffer to float array for continuous ops."""
        data = self.buffers.get(node)
        if data is None:
            return None
        return np.frombuffer(data, dtype=np.uint8).astype(np.float32)  # Byte-to-float view

    def yield_events(self, node: str) -> Iterable[Tuple[str, Union[str, Tuple[int, int], Dict]]]:
        """Yield DEFLATE-parsed events from buffer, enhanced with phase/resonance."""
        data = self.buffers.get(node)
        if data is None:
            return
        
        # Adapt deflate's stream_processor logic
        stream = ConstBitStream(bytes=data)
        reader = BitLSB(stream)  # Assuming BitLSB class is moved/imported
        error_tracker = ErrorTracker()  # Assuming ErrorTracker is moved
        
        block_count = 0
        state_phi = 0.0
        
        while reader.stream.pos < len(reader.stream):
            last_block = bool(reader.read_bits(1))
            block_type = reader.read_bits(2)
            
            # Simplified parsing (full from deflate.py, adapted)
            if block_type == 0:  # Stored
                stored_data = parse_stored_block(reader, lambda k: error_tracker.count_error(k))
                yield ("stored", {"data": stored_data})
            elif block_type == 1:  # Static
                yield ("static", {})
            elif block_type == 2:  # Dynamic
                # Parse trees/meta as in parse_dynamic_huffman_block
                meta = {}  # Placeholder; implement full parsing
                yield ("dynamic", meta)
            
            # Enhance with phase/resonance
            phase = int(reader.stream.pos % 8)
            resonance = compute_resonance(bytes(data))  # Your func
            yield ("enhanced", {"phase": phase, "resonance": resonance})
            
            if last_block:
                break
        
        yield ("summary", dict(error_tracker.error_counts))

