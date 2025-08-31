import re
from typing import List
import numpy as np
from genome import Grammar

class Tokenizer:
    def __init__(self, grammar: Grammar):
        self.grammar = grammar

    def from_text(self, text: str) -> List[str]:
        """Tokenizes a string into a list of alphanumeric tokens."""
        return re.findall(r"[A-Za-z0-9_]+", text.lower())

    def from_bytes(self, name: str, data: bytes) -> List[str]:
        """Decodes bytes to text if possible, otherwise tokenizes hex representation."""
        try:
            text_content = data.decode('utf-8', errors='ignore')
            return self.from_text(text_content)
        except Exception:
            hexpairs = ' '.join(f"{b:02x}" for b in data[:4096])
            return hexpairs.split()

    def ingest(self, name: str, data: bytes):
        """Processes a byte asset, tokenizes it, and updates the grammar."""
        if not data or len(data) < 8: return
        # Heuristic: ensure it's mostly printable text before processing
        printable = sum(1 for b in data if 32 <= b <= 126 or b in (10,13,9))
        if printable / max(len(data), 1) < 0.6: return
        
        tokens = self.from_bytes(name, data)
        if len(tokens) < 2: return
        
        # Update unigrams and bigrams
        for t in tokens:
            self.grammar.unigram_counts[t] += 1
            if t not in self.grammar.semantic_alleles:
                self.grammar.semantic_alleles[t] = np.array([1.0, 0.0, 0.0, 0.0])
        for a, b in zip(tokens[:-1], tokens[1:]):
            self.grammar.bigram_counts[a][b] += 1
