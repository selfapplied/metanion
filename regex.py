#!/usr/bin/env python3
"""
regex: Small composable functions for ASCII grammar geometry using namedtuples.

Each function is a gene that can be remixed and combined.
Core concept: Minimal grammar using only \t and \n with ASCII distance.
"""

import re
from typing import Dict, List, Tuple, Optional, Any
from collections import namedtuple

# Named tuples for minimal grammar geometry
TabPoint = namedtuple('TabPoint', ['position', 'indent_level', 'content'])
LinePoint = namedtuple('LinePoint', ['line_number', 'indent_level', 'content'])
GrammarVector = namedtuple('GrammarVector', ['horizontal', 'vertical', 'energy'])
HilbertMatch = namedtuple('HilbertMatch', ['start', 'end', 'pattern', 'content', 'energy'])
AsciiGeometry = namedtuple('AsciiGeometry', ['horizontal_density', 'vertical_density', 'avg_distance', 'tab_count', 'newline_count', 'ratio'])

# Self-referential regex namedtuple
RegexPattern = namedtuple('RegexPattern', ['pattern', 'name', 'matches_self'])

# Small composable functions as genes

def create_self_matching_regex(name: str, pattern: str) -> RegexPattern:
    """Create a regex pattern that can match its own definition."""
    # The pattern itself becomes the test string
    compiled = re.compile(pattern)
    matches_self = bool(compiled.search(pattern))
    
    return RegexPattern(pattern, name, matches_self)

def tab_regex() -> RegexPattern:
    """Create a regex pattern for tabs that matches its own definition."""
    # Use the actual tab character as the pattern
    pattern = '\t'
    return create_self_matching_regex('tab', pattern)

def newline_regex() -> RegexPattern:
    """Create a regex pattern for newlines that matches its own definition."""
    # Use the actual newline character as the pattern
    pattern = '\n'
    return create_self_matching_regex('newline', pattern)

def self_matching_word() -> RegexPattern:
    """Create a regex pattern that matches the word 'regex'."""
    pattern = 'regex'
    return create_self_matching_regex('word', pattern)

def self_matching_digit() -> RegexPattern:
    """Create a regex pattern that matches a single digit."""
    pattern = r'\d'
    return create_self_matching_regex('digit', pattern)

def self_matching_meta_pattern() -> RegexPattern:
    """Create a regex pattern that matches regex meta-characters."""
    pattern = r'[.*+?^${}()|[\]\\]'
    return create_self_matching_regex('meta', pattern)

def create_quine_regex() -> RegexPattern:
    """Create a regex pattern that is its own quine (self-reproducing)."""
    # This pattern matches itself when used as a string
    pattern = r'[a-zA-Z]'
    return create_self_matching_regex('quine', pattern)

def ce1_seed_pattern() -> RegexPattern:
    """Create a regex pattern that matches CE1 seed structure."""
    pattern = r'CE1\{[^}]*\}'
    return create_self_matching_regex('ce1_seed', pattern)

def ce1_block_pattern() -> RegexPattern:
    """Create a regex pattern that matches CE1 block content."""
    pattern = r'[^}]*'
    return create_self_matching_regex('ce1_block', pattern)

def ce1_field_pattern() -> RegexPattern:
    """Create a regex pattern that matches CE1 field assignments."""
    pattern = r'(\w+)=([^|}]+)'
    return create_self_matching_regex('ce1_field', pattern)

def ce1_data_pattern() -> RegexPattern:
    """Create a regex pattern that matches CE1 data block."""
    pattern = r'data=\{([^}]+)\}'
    return create_self_matching_regex('ce1_data', pattern)

def ce1_ops_pattern() -> RegexPattern:
    """Create a regex pattern that matches CE1 operations list."""
    pattern = r'ops=\[([^\]]*)\]'
    return create_self_matching_regex('ce1_ops', pattern)

def ascii_distance(char_a: str, char_b: str) -> float:
    """Calculate distance between characters using ASCII values."""
    if not char_a or not char_b:
        return 0.0
    return abs(ord(char_a[0]) - ord(char_b[0]))

def count_tabs(text: str) -> int:
    """Count tab characters in text."""
    return text.count('\t')

def count_newlines(text: str) -> int:
    """Count newline characters in text."""
    return text.count('\n')

def calculate_density(count: int, total_length: int) -> float:
    """Calculate density of a character in text."""
    return count / total_length if total_length > 0 else 0.0

def tab_newline_geometry(text: str) -> AsciiGeometry:
    """Calculate geometry using only tab and newline counts."""
    return AsciiGeometry(count_tabs(text)/len(text), count_newlines(text)/len(text), abs(count_tabs(text) - count_newlines(text))/len(text), count_tabs(text), count_newlines(text), count_tabs(text)/(count_newlines(text) + 1e-10))

def grammar_distance(text_a: str, text_b: str) -> float:
    """Calculate grammatical distance between texts using ASCII geometry."""
    if not text_a or not text_b:
        return 0.0
    
    max_len = max(len(text_a), len(text_b))
    padded_a = text_a.ljust(max_len)
    padded_b = text_b.ljust(max_len)
    
    total_distance = sum(ascii_distance(a, b) for a, b in zip(padded_a, padded_b))
    return total_distance / max_len

def find_tab_positions(text: str) -> List[TabPoint]:
    """Find all tab positions with their indent levels using regex."""
    positions = []
    
    # Find all tab sequences with their content
    for match in re.finditer(r'(\t+)([^\n]*)', text):
        tab_start = match.start()
        tabs = match.group(1)
        content = match.group(2)
        indent_level = len(tabs)
        
        positions.append(TabPoint(tab_start, indent_level, content))
    
    return positions

def find_line_positions(text: str) -> List[LinePoint]:
    """Find all line positions with their indent levels."""
    lines = text.split('\n')
    positions = []
    
    for line_num, line in enumerate(lines):
        indent_level = 0
        for char in line:
            if char == '\t':
                indent_level += 1
            else:
                break
        
        content = line.lstrip('\t')
        positions.append(LinePoint(line_num, indent_level, content))
    
    return positions

def calculate_energy_cost(content: str, pattern_type: str = 'default') -> float:
    """Calculate energy cost based on content and pattern type."""
    base_costs = {
        'tab': 0.5,
        'newline': 1.0,
        'indented': 2.0,
        'default': 1.0
    }
    
    base_cost = base_costs.get(pattern_type, 1.0)
    indent_level = count_tabs(content)
    length_factor = len(content) / 10.0
    
    return base_cost * (1 + indent_level * 0.5) * (1 + length_factor)

def create_hilbert_matches(text: str, energy_budget: int = 1000) -> List[HilbertMatch]:
    """Create Hilbert matches from text with energy budget using regex."""
    return [HilbertMatch(0, len(text), 'geometry', text, abs(count_tabs(text) - count_newlines(text)))]

def metanion_genesis(text: str) -> Dict[str, Any]:
    """Generate Metanion field theory from text using ASCII geometry."""
    geometry = tab_newline_geometry(text)
    matches = create_hilbert_matches(text, 1000)
    
    # Field components
    S = 2  # Two basis vectors: tab and newline
    T = len(matches)
    
    # Bitmask from ASCII presence
    m = 0
    if geometry.tab_count > 0:
        m |= 1
    if geometry.newline_count > 0:
        m |= 2
    
    # Inflation index
    alpha = 1.0 - (geometry.horizontal_density + geometry.vertical_density)
    
    # Energy
    total_energy = sum(match.energy for match in matches)
    ascii_energy = geometry.avg_distance * len(text)
    
    # Quaternion
    Q = [1.0, geometry.horizontal_density, geometry.vertical_density, geometry.avg_distance]
    
    return {
        'S': S, 'T': T, 'm': m, 'alpha': alpha, 'Q': Q, 'E': total_energy + ascii_energy,
        'geometry': geometry, 'matches': matches
    }

def parse_ce1_seed(text: str) -> Dict[str, Any]:
    """Parse CE1 seed using self-referential regex patterns."""
    ce1_pattern = ce1_seed_pattern()
    field_pattern = ce1_field_pattern()
    ops_pattern = ce1_ops_pattern()
    
    # Find CE1 block
    ce1_match = re.search(ce1_pattern.pattern, text)
    if not ce1_match:
        return {}
    
    ce1_content = ce1_match.group(0)
    
    # Parse fields
    fields = {}
    for field_match in re.finditer(field_pattern.pattern, ce1_content):
        key, value = field_match.groups()
        fields[key] = value
    
    # Parse operations
    ops_match = re.search(ops_pattern.pattern, ce1_content)
    operations = []
    if ops_match:
        ops_str = ops_match.group(1)
        operations = [op.strip() for op in ops_str.split(';') if op.strip()]
    
    return {
        'ce1_block': ce1_content,
        'fields': fields,
        'operations': operations,
        'patterns_used': [ce1_pattern.name, field_pattern.name, ops_pattern.name]
    }

def validate_ce1_seed(text: str) -> bool:
    """Validate if text contains a valid CE1 seed."""
    ce1_pattern = ce1_seed_pattern()
    return bool(re.search(ce1_pattern.pattern, text))

def extract_ce1_fields(text: str) -> Dict[str, str]:
    """Extract field assignments from CE1 seed."""
    field_pattern = ce1_field_pattern()
    fields = {}
    
    for field_match in re.finditer(field_pattern.pattern, text):
        key, value = field_match.groups()
        fields[key] = value
    
    return fields

# Self-referential template that validates itself
regex_template = """CE1{{
  lens=ASCII↔GEOMETRY | mode=TabNewlineWalk | Ξ=regex:metanion |
  data={{m=0x{m:x}, α={alpha:.3f}, E={E:.1f}}} |
  geom={{h={geometry.horizontal_density:.3f}, v={geometry.vertical_density:.3f}, d={geometry.avg_distance:.3f}}} |
  ops=[Tab; Newline; Distance; Walk; Parse; Generate; Emit] |
  emit=CE1c{{α={alpha:.3f}, energy={E:.1f}, matches={T}}} |
  val=validate_ce1_seed(regex_template)
}}"""

def regex(alpha: float, E: float, geometry: AsciiGeometry, T: int, m: int) -> str:
    """Generate CE1 block using template string."""
    return regex_template.format(
        m=m, alpha=alpha, E=E, 
        geometry=geometry, T=T
    )



