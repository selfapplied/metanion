from typing import List, Dict, Optional, Any
import re
from collections import Counter

# Expanded word shape classification
WORD_SHAPE_RE = {
    'SPACE': re.compile(r'^\s+$'),
    'PUNCT': re.compile(r'^[^\w\s]+$'),
    'CAMEL': re.compile(r'^[A-Z][a-z0-9]+([A-Z][a-z0-9]+)+$'),
    'SNAKE': re.compile(r'^[a-z][a-z0-9]*(_[a-z0-9]+)+$'),
    'TITLE': re.compile(r'^[A-Z][a-z0-9]+$'),
    'UPPER': re.compile(r'^[A-Z][A-Z0-9]+$'),
    'LOWER': re.compile(r'^[a-z][a-z0-9]+$'),
    'NUMERIC': re.compile(r'^[0-9]+$'),
    'ALPHANUM': re.compile(r'^[a-zA-Z0-9]+$'),
    'MIXED': re.compile(r'.*'),
}


def get_word_shape(token: str) -> str:
    """Classifies a token into a word shape category."""
    ordered_shapes = ['SPACE', 'PUNCT', 'CAMEL', 'SNAKE',
                      'TITLE', 'UPPER', 'LOWER', 'NUMERIC', 'ALPHANUM', 'MIXED']
    for shape in ordered_shapes:
        if WORD_SHAPE_RE[shape].match(token):
            return shape
    return 'MIXED'


def get_word_properties(token: str) -> Dict[str, Any]:
    """Classifies a token into a word shape and estimates its POS tag."""
    # Simple heuristic-based Part-of-Speech tagging
    POS_RULES = [
        (re.compile(r'^\d+$'), 'NUM'),
        (re.compile(r'^(the|a|an|this|that|these|those)$'), 'DET'),
        (re.compile(r'.*(ing|ed|es|s)$'), 'VERB'),
        (re.compile(r'.*(able|ible|al|ful|ic|ive|less|ous)$'), 'ADJ'),
        (re.compile(r'.*(ion|tion|sion|ment|ness|ity|ty)$'), 'NOUN'),
        (re.compile(r'.*ly$'), 'ADV'),
    ]
    shape = 'MIXED'
    ordered_shapes = ['SPACE', 'PUNCT', 'CAMEL', 'SNAKE',
                      'TITLE', 'UPPER', 'LOWER', 'NUMERIC', 'ALPHANUM', 'MIXED']
    for s in ordered_shapes:
        if WORD_SHAPE_RE[s].match(token):
            shape = s
            break

    pos = 'NOUN'  # Default to NOUN
    if shape in ['LOWER', 'TITLE']:
        for rx, tag in POS_RULES:
            if rx.match(token):
                pos = tag
                break
    elif shape == 'PUNCT':
        pos = 'PUNCT'
    elif shape == 'SPACE':
        pos = 'SPACE'

    return {'shape': shape, 'length': len(token), 'pos': pos}


def basic_tok(text: str) -> List[str]:
    """
    Tokenizes text by splitting camelCase, and preserving structure like snake_case,
    punctuation, and spaces.
    """
    # Add space before uppercase letters in camelCase for later splitting
    s1 = re.sub(r'([a-z0-9])([A-Z])', r'\1 \2', text)
    # Tokenize into words, punctuation, and spaces
    tokens = re.findall(r'\w+|[^\w\s]+|\s+', s1)

    # Lowercase only purely alphabetic tokens, preserving case elsewhere
    return [t.lower() if t.isalpha() else t for t in tokens if t]


def apply_learned_tokenizer(engine, text: str) -> List[str]:
    """Apply learned tokenization patterns"""
    tokens = []
    i = 0
    while i < len(text):
        found = False
        for token_len in range(min(engine.max_token_len, len(text) - i), 0, -1):
            candidate = text[i:i + token_len]
            if engine.learned_tokenizer and candidate in engine.learned_tokenizer:
                tokens.append(candidate)
                i += token_len
                found = True
                break
        if not found:
            tokens.append(text[i])
            i += 1
    return tokens

def tok(engine, text: str) -> List[str]:
    """Adaptive tokenization - learn tokens from data patterns"""
    if hasattr(engine, 'learned_tokenizer') and engine.learned_tokenizer:
        return apply_learned_tokenizer(engine, text)
    else:
        return basic_tok(text)

def learn_tokenization_excavation(engine, training_texts: List[str], iteration: int,
                                  assessment_context: Optional[Dict[str, Any]] = None) -> None:
    """Unified tokenization learning with assessment-guided strategy"""
    all_tokens = []
    for text in training_texts:
        all_tokens.extend(basic_tok(text))
    
    if not all_tokens:
        return

    # Keep top 2000 most common tokens as the vocab for pheno generation
    counts = Counter(all_tokens)
    engine.vocab = [t for t, c in counts.most_common(2000)]

    # Analyze and store token properties (shape, length, and POS)
    shape_counts = Counter(get_word_properties(t)
                               ['shape'] for t in all_tokens)
    total_shapes = sum(shape_counts.values())
    engine.shape_distribution = {
        shape: count / total_shapes for shape, count in shape_counts.items()
    }

    engine.token_properties = {
        t: get_word_properties(t) for t in all_tokens
    }

    if iteration < 3:
        # This is a placeholder for a more advanced tokenizer learning step
        pass
    else:
        if assessment_context and assessment_context.get('token_coverage', 0) < 0.8:
            # This is a placeholder for a more advanced tokenizer learning step
            pass
