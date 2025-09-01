#!/usr/bin/env python3
"""
loom: Hilbert Grammar CLI with Metanion Field Theory
Treats code as particles in dual Boolean-continuous lattice with quaternion fibers.

    CE1{
  lens=QCONV↔METANION | mode=InflateWalk | Ξ=eonyx:metanion |
  data={m=0xffffffffffffffffffffffffffffffffffffffffffffffffff, α=0.000, E=33727.0} |
  ops=[Measure; Transport; Convolve; Route; Match; Reconcile; Inflate] |
  emit=CE1c{α=0.000, energy=33727.0, matches=200}
}
"""
from __future__ import annotations
from collections import namedtuple
from typing import List, Optional
from pathlib import Path
import zipfile
import numpy as np
import quaternion
from sprixel2 import gene
from loader import attempt
from metanion import em, Metanion, HilbertPoint, SpatialEntry

# Core structures (imported from metanion)
Transform = namedtuple('Transform', ['f', 'U', 'w', 'c', 'y'])
CliFlag = namedtuple('CliFlag', ['name', 'value', 'is_boolean'])
CliArgs = namedtuple('CliArgs', ['target', 'flags', 'tail_emits'])

@gene
@em("path_str: str, data: bytes := coords: HilbertPoint")
def hilbert_coords(path_str: str, data: bytes) -> HilbertPoint:
    """Map file to 4D Hilbert space coordinates."""
    path_parts = path_str.split('/')
    density = len(path_parts) / max(1, len(path_str))
    
    if len(data) > 0:
        # Convert to uint8 array safely
        data_array = np.frombuffer(data, dtype=np.uint8) if isinstance(data, bytes) else np.array(data, dtype=np.uint8)
        if len(data_array) > 0:
            byte_counts = np.bincount(data_array, minlength=256)
            probs = byte_counts / len(data_array)
            entropy = -np.sum(probs * np.log2(probs + 1e-10)) / 8.0
        else:
            entropy = 0.0
    else:
        entropy = 0.0
    
    depth = len(path_parts)
    diversity = len(set(path_str)) / 256.0
    
    return HilbertPoint(
        min(density, 1.0),
        min(entropy, 1.0), 
        min(depth / 10.0, 1.0),
        min(diversity, 1.0)
    )

@gene
@em("coords: HilbertPoint := quat: np.ndarray")
def quaternion_lift(coords: HilbertPoint) -> np.ndarray:
    """Convert Hilbert coordinates to unit quaternion."""
    w = np.sqrt(max(0, 1 - coords.density**2 - coords.entropy**2 - coords.depth**2))
    return np.array([w, coords.density, coords.entropy, coords.depth])

@gene
@em("target: str, energy_budget: int := entries: List[SpatialEntry]")
def hilbert_walk(target: str, energy_budget: int) -> List[SpatialEntry]:
    """Walk directory/zip in Hilbert space with energy tracking."""
    entries = []
    energy = energy_budget
    target_path = Path(target)
    
    if target_path.is_file() and target_path.suffix.lower() == '.zip':
        # Walk zip file
        with zipfile.ZipFile(target_path, 'r') as zf:
            for info in zf.infolist()[:energy//10]:  # Energy-limited
                if energy <= 0:
                    break
                
                data = attempt(lambda: zf.read(info.filename), 
                             default=b'', exceptions=(Exception,))
                coords = hilbert_coords(info.filename, data)
                quat = quaternion_lift(coords)
                
                entries.append(SpatialEntry(info.filename, data, coords, quat))
                energy -= 10
    
    elif target_path.is_dir():
        # Walk directory
        for path in target_path.rglob('*'):
            if energy <= 0:
                break
            if path.is_file():
                data = attempt(lambda: path.read_bytes(), 
                             default=b'', exceptions=(Exception,))
                coords = hilbert_coords(str(path), data)
                quat = quaternion_lift(coords)
                
                entries.append(SpatialEntry(str(path), data, coords, quat))
                energy -= 5
    
    return entries

@gene
@em("entries: List[SpatialEntry], regex_a: str, regex_b: str := matches: List[str]")
def basis_filter(entries: List[SpatialEntry], regex_a: str, regex_b: str) -> List[str]:
    """Filter entries by dual regex basis vectors."""
    import re
    pattern_a = re.compile(regex_a)
    pattern_b = re.compile(regex_b)
    
    matches = []
    for entry in entries:
        content = entry.data.decode('utf-8', errors='ignore')
        match_a = bool(pattern_a.search(content))
        match_b = bool(pattern_b.search(content))
        
        if match_a and match_b:
            basis_match = 'both'
        elif match_a:
            basis_match = 'a'
        elif match_b:
            basis_match = 'b'
        else:
            basis_match = 'neither'
        
        matches.append(f"{entry.path}:{basis_match}")
    
    return matches

@gene
@em("entries: List[SpatialEntry] := metanion: Metanion")
def metanion_genesis(entries: List[SpatialEntry]) -> Metanion:
    """Convert spatial entries to Metanion particle."""
    transforms = []
    
    for i, entry in enumerate(entries):
        weight = len(entry.data)
        cost = weight // 100 + 1
        yield_ = 1
        
        transform = Transform(
            f=f"compress_{i}",
            U=entry.quat,
            w=weight,
            c=cost,
            y=yield_
        )
        transforms.append(transform)
    
    # All compressed initially (m_i = 1)
    n = len(transforms)
    m = (1 << n) - 1 if n > 0 else 0
    
    # Calculate α(m) = Σ(1-m_i)w_i / Σw_i
    total_weight = sum(t.w for t in transforms) if transforms else 1
    alpha = 0.0  # All compressed
    
    # Calculate Q(m) = Π U_i^σ_i
    Q = np.array([1.0, 0.0, 0.0, 0.0])
    for transform in transforms:
        Q = quaternion.quat_mul(Q, transform.U)
    
    # Energy E(m) = Σ m_i(c_i - y_i)
    E = sum(t.c - t.y for t in transforms)
    
    return Metanion(entries, transforms, m, alpha, Q, E)

@gene
@em("metanion: Metanion := ce1_block: str")
def ce1_emission(metanion: Metanion) -> str:
    """Emit CE1 specification block."""
    return f'''CE1{{
  lens=QCONV↔METANION | mode=InflateWalk | Ξ=eonyx:metanion |
  data={{m=0x{metanion.m:x}, α={metanion.alpha:.3f}, E={metanion.E:.1f}}} |
  ops=[Measure; Transport; Convolve; Route; Match; Reconcile; Inflate] |
  emit=CE1c{{α={metanion.alpha:.3f}, energy={metanion.E:.1f}, matches={len(metanion.S)}}}
}}'''

@gene
@em("argv: List[str] := args: CliArgs")
def unix_parse(argv: List[str]) -> CliArgs:
    """Parse Unix-style command line arguments."""
    if not argv:
        return CliArgs('.', [], [])
    
    # Find -- separator
    try:
        sep_idx = argv.index('--')
        args_part = argv[:sep_idx]
        tail_emits = argv[sep_idx + 1:]
    except ValueError:
        args_part = argv
        tail_emits = []
    
    target = args_part[0] if args_part else '.'
    flags = []
    
    i = 1
    while i < len(args_part):
        arg = args_part[i]
        
        if arg.startswith('--'):
            if '=' in arg:
                name, value = arg[2:].split('=', 1)
                flags.append(CliFlag(name, value, False))
            else:
                name = arg[2:]
                if i + 1 < len(args_part) and not args_part[i + 1].startswith('-'):
                    flags.append(CliFlag(name, args_part[i + 1], False))
                    i += 1
                else:
                    flags.append(CliFlag(name, True, True))
        
        elif arg.startswith('-') and len(arg) > 1:
            name = arg[1:]
            if i + 1 < len(args_part) and not args_part[i + 1].startswith('-'):
                flags.append(CliFlag(name, args_part[i + 1], False))
                i += 1
            else:
                flags.append(CliFlag(name, True, True))
        
        i += 1
    
    return CliArgs(target, flags, tail_emits)

@gene
@em("args: CliArgs := result: (str, Optional[Metanion])")
def pipeline_flow(args: CliArgs) -> tuple[str, Optional[Metanion]]:
    """Execute complete Hilbert grammar pipeline."""
    # Parse flags
    flag_dict = {flag.name: flag.value for flag in args.flags}
    
    basis_a = flag_dict.get('a', flag_dict.get('basis-a', '.*'))
    basis_b = flag_dict.get('b', flag_dict.get('basis-b', '.*'))
    energy = int(flag_dict.get('e', flag_dict.get('energy', '1000')))
    verbose = flag_dict.get('v', flag_dict.get('verbose', False))
    show_metanion = flag_dict.get('metanion', False)
    show_ce1 = flag_dict.get('ce1', False)
    
    # Execute pipeline
    entries = hilbert_walk(args.target, energy)
    matches = basis_filter(entries, basis_a, basis_b)
    
    # Build result
    result = f"📊 Hilbert Grammar Results:\n"
    result += f"   Target: {args.target}\n"
    result += f"   Basis A: {basis_a}\n"
    result += f"   Basis B: {basis_b}\n"
    result += f"   Total entries: {len(entries)}\n"
    result += f"   Matches: {len(matches)}\n"
    
    if verbose:
        result += f"\n📝 Match Details:\n"
        for match in matches[:10]:  # Limit output
            result += f"   {match}\n"
    
    metanion = None
    if show_metanion or show_ce1:
        metanion = metanion_genesis(entries)

    if show_metanion and metanion:
        result += f"\n🧬 Metanion Particle:\n"
        result += f"   Bitmask: 0x{metanion.m:x}\n"
        result += f"   Inflation α: {metanion.alpha:.3f}\n"
        result += f"   Energy E: {metanion.E:.1f}\n"
        result += f"   Transforms: {len(metanion.T)}\n"
        
        if show_ce1:
            result += f"\n📋 CE1 Specification:\n"
            result += ce1_emission(metanion)
    
    if args.tail_emits:
        result += f"\n🔗 Tail Emits: {' '.join(args.tail_emits)}\n"
    
    return result, metanion



@gene
@em("func: callable := decorated_func: callable")
def grammar_parser(func):
    """Decorator that creates a grammar parser from a function's docstring."""
    def wrapper(*args, **kwargs):
        docstring = func.__doc__ or "No docstring available."
        doc_data = docstring.encode('utf-8')
        doc_coords = hilbert_coords(f"{func.__name__}_doc", doc_data)
        doc_quat = quaternion_lift(doc_coords)
        doc_entry = SpatialEntry(f"{func.__name__}_doc", doc_data, doc_coords, doc_quat)
        grammar_matches = basis_filter([doc_entry], r'[A-Z][A-Z]+:|"""|\w+\(', r'->|:=|\w+\s*=')
        doc_metanion = metanion_genesis([doc_entry])
        result = func(*args, **kwargs)
        
        if isinstance(result, str):
            analysis = f"\n🧬 Grammar Parser Analysis for {func.__name__}:\n"
            analysis += f"   Coords: {doc_coords}\n"
            analysis += f"   Quaternion: [{doc_quat[0]:.3f}, {doc_quat[1]:.3f}, {doc_quat[2]:.3f}, {doc_quat[3]:.3f}]\n"
            analysis += f"   Matches: {len(grammar_matches)}, Metanion: m=0x{doc_metanion.m:x}, E={doc_metanion.E:.1f}\n"
            analysis += f"   Complexity: {'HIGH' if doc_coords.entropy > 0.5 else 'LOW'} (entropy={doc_coords.entropy:.3f})\n"
            return result + analysis
        return result
    
    wrapper.__name__ = func.__name__
    wrapper.__doc__ = func.__doc__
    return wrapper

@gene
@em("text: str := grammar_gene: callable")
def grammar_genesis(text: str):
    """Convert any text into a grammar-analyzing gene automatically."""
    def grammar_gene() -> str:
        # Walk the text through Hilbert space
        data = text.encode('utf-8')
        coords = hilbert_coords("grammar_text", data)
        quat = quaternion_lift(coords)
        entry = SpatialEntry("grammar_text", data, coords, quat)
        matches = basis_filter([entry], r'[A-Z][A-Z]+:|->|:=', r'\w+\s*\(|\w+:')
        metanion = metanion_genesis([entry])
        
        analysis = f"🧬 Auto-Generated Grammar Gene:\n"
        analysis += f"   Text coords: {coords}\n"
        analysis += f"   Quaternion: [{quat[0]:.3f}, {quat[1]:.3f}, {quat[2]:.3f}, {quat[3]:.3f}]\n"
        analysis += f"   Grammar patterns: {len(matches)}\n"
        analysis += f"   Metanion: m=0x{metanion.m:x}, E={metanion.E:.1f}\n"
        analysis += f"   Original text: {text[:100]}{'...' if len(text) > 100 else ''}\n"
        
        return analysis
    
    # Register the grammar as a gene
    from sprixel2 import genes
    import hashlib
    gene_name = f"grammar_{hashlib.sha256(text.encode()).hexdigest()[:8]}"
    genes[gene_name] = grammar_gene
    
    return grammar_gene

@gene
@em("$ := auto_gene_count: int")
def grammar_awakening() -> int:
    """Automatically convert all discovered grammars into genes."""
    count = 0
    
    # Auto-geneticize all docstrings in the current module
    import inspect
    current_module = inspect.getmodule(grammar_awakening)
    
    for name, obj in inspect.getmembers(current_module):
        if inspect.isfunction(obj) and obj.__doc__:
            # Convert function docstring to gene
            grammar_genesis(obj.__doc__)
            count += 1
    
    # Auto-geneticize all help text patterns found in files
    import glob
    for md_file in glob.glob("*.md"):
        try:
            with open(md_file, 'r') as f:
                content = f.read()
                # Look for grammar patterns
                if any(pattern in content for pattern in ['USAGE:', 'SYNTAX:', '->>', '::=', 'PIPELINE:']):
                    grammar_genesis(content[:500])  # First 500 chars
                    count += 1
        except:
            pass
    
    return count

@gene
@em("ce1_block: str")
def embed_ce1(ce1_block: str):
    """Embeds the CE1 block into the loom.py docstring."""
    file_path = 'loom.py'
    
    # Read current content
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Find the main docstring
    import re
    docstring_match = re.search(r'"""(.+?)"""', content, re.DOTALL)
    if not docstring_match:
        print("Could not find docstring to embed CE1 block.")
        return

    docstring_content = docstring_match.group(1)
    
    # Check for existing CE1 block and replace it, or append a new one
    ce1_block_pattern = re.compile(r'(\n\n    CE1\{.+?\n    \}\n)', re.DOTALL)
    
    if ce1_block_pattern.search(docstring_content):
        new_docstring_content = ce1_block_pattern.sub(f'\n\n    {ce1_block}\n', docstring_content)
    else:
        new_docstring_content = docstring_content.rstrip() + f'\n\n    {ce1_block}\n'
        
    # Create the new full docstring
    new_docstring = f'"""{new_docstring_content}"""'
    old_docstring = docstring_match.group(0)

    # Replace the old docstring with the new one in the file content
    new_content = content.replace(old_docstring, new_docstring)
    
    # Write the updated content back to the file
    with open(file_path, 'w') as f:
        f.write(new_content)

@gene
@em("$ := spec_gene_count: int")
def em_awakening() -> int:
    """Convert all @em specifications into executable genes."""
    count = 0
    
    # Import emits registry to access all specs
    from emits import genes as em_genes
    
    for func_name, spec_list in em_genes.items():
        for spec in spec_list:
            # Create a gene from each em specification
            spec_text = spec.raw
            
            def spec_gene(spec_text=spec_text, func_name=func_name) -> str:
                """Auto-generated gene from @em specification."""
                # Walk the spec through Hilbert space
                data = spec_text.encode('utf-8')
                coords = hilbert_coords(f"em_spec_{func_name}", data)
                quat = quaternion_lift(coords)
                entry = SpatialEntry(f"em_spec_{func_name}", data, coords, quat)
                matches = basis_filter([entry], r':=|->|\w+:', r'\w+\s*,|\w+\s*\)')
                metanion = metanion_genesis([entry])
                
                analysis = f"🧬 Em Spec Gene for {func_name}:\n"
                analysis += f"   Spec: {spec_text}\n"
                analysis += f"   Coords: {coords}\n"
                analysis += f"   Quaternion: [{quat[0]:.3f}, {quat[1]:.3f}, {quat[2]:.3f}, {quat[3]:.3f}]\n"
                analysis += f"   Type patterns: {len(matches)}\n"
                analysis += f"   Metanion: m=0x{metanion.m:x}, E={metanion.E:.1f}\n"
                analysis += f"   Mint: {spec.mint}\n"
                
                return analysis
            
            # Register the spec as a gene
            from sprixel2 import genes
            import hashlib
            spec_hash = hashlib.sha256(spec_text.encode()).hexdigest()[:8]
            gene_name = f"em_{func_name}_{spec_hash}"
            genes[gene_name] = spec_gene
            count += 1
    
    return count

@grammar
def cli_main() -> str:
    """🧬 Loom: Hilbert Grammar CLI

    USAGE:
        python loom.py [regex1] [weight1] ...
        -e 

    EXAMPLES:
        loom.py "\t" 10
        loom.py . -v -- trace        # Verbose with tail emits
    
    OUTPUT:
        
    """
    import sys    
    argv = sys.argv[1:]
    args = unix_parse(argv)
    
    # Check for help - return docstring with grammar analysis
    if not argv or '--help' in argv or '-h' in argv or any(f.name in ('h', 'help') for f in args.flags):
        return cli_main.__doc__ or "No help available"
    
    # Execute normal pipeline
    result, metanion = pipeline_flow(args)
    
    # Check if we need to embed the CE1 block
    if any(f.name == 'embed-ce1' for f in args.flags) and metanion:
        ce1_block = ce1_emission(metanion)
        embed_ce1(ce1_block)
        result += "\n\n✅ CE1 block embedded in file docstring."

    return result

if __name__ == '__main__':
    print(cli_main())
