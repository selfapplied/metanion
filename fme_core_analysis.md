# Analysis of `fme_core.py`: Valuable Concepts for Porting

This document summarizes the key architectural and conceptual ideas discovered in the older `eonyx/fme_core.py` that are worth considering for integration into the current `CE1Core` engine.

## 1. Learned Tokenization: Adaptive Perception

The most significant concept is a sophisticated, multi-stage process for learning a tokenizer directly from the source material. This allows the engine to develop an adaptive perception of its input, moving beyond simple whitespace or regex splitting to discover a vocabulary that is intrinsic to the "language" of the genome's assets.

### How it Works

1.  **Assessment Phase:** The engine first performs a statistical analysis of the text (e.g., analyzing character trigrams) to identify candidate substrings that are likely to be meaningful tokens.
2.  **Pattern Discovery:** It looks for high-frequency, statistically significant patterns that can serve as delimiters or content chunks.
3.  **Vocabulary Refinement:** The discovered patterns are used to build and refine a vocabulary (`self.learned_tokenizer`). This vocabulary is then used to tokenize the input texts for the main learning process.
4.  **Fallback Mechanism:** The system is robust. If the enhanced learning process fails to discover a sufficient number of useful patterns, it gracefully falls back to a simpler, basic tokenization strategy.

### Why it's Valuable

*   **Context-Awareness:** The engine adapts its "worldview" to the specific data it is processing. It can learn the syntax of Python code, the structure of Markdown, or the cadence of natural language prose.
*   **Richer Grammar:** A learned vocabulary leads to a more nuanced and representative grammar (`unigram_counts`, `bigram_counts`), which in turn can lead to more coherent and interesting generative output.
*   **Emergent Perception:** This is a form of unsupervised learning that embodies the core philosophical theme of the project: the system isn't just processing data; it's learning *how* to perceive it.

## 2. Incremental Learning & File Hashing

The `fme_core.py` engine treats learning as a persistent, ongoing process rather than a one-shot analysis. It achieves this through an intelligent caching and file-monitoring mechanism.

### How it Works

1.  **File Hashing:** The engine computes and stores a hash of every source file it processes. This cache of hashes represents the "known state" of the world.
2.  **Change Detection:** On subsequent runs, it re-hashes the files and compares them to the cached versions.
3.  **Targeted Processing:** The core "excavation" (learning) loop is only run on the subset of files that are new or have changed. Unchanged files are skipped.

### Why it's Valuable

*   **Efficiency:** This is a powerful optimization for large genomes or for workflows that involve frequent small changes. It dramatically speeds up the feedback loop for evolving a genome.
*   **Persistent Evolution:** It frames the engine's learning not as a single event, but as a continuous "life" where it adapts to changes in its environment (the source files) over time.

## 3. Color Phase Alleles: Learned Aesthetics

This is a system for moving beyond a static, hard-coded mapping of internal state to visual output. The engine learns a palette of characteristic visual styles, or "color alleles," and learns when to apply them.

### How it Works

1.  **Sample Collection:** During the final phase of its learning loop, the engine collects thousands of data points, each a tuple of its internal state (`delta`) and the resulting color (`(h, s, v)`).
2.  **Clustering:** It performs clustering on this dataset to find recurring patterns—regions where a certain range of `delta` values consistently produces a certain kind of color. Each discovered cluster is a "color phase allele."
3.  **Allele as a Style:** Each allele stores the statistical properties of its cluster (e.g., the average delta, average hue, prominence). It represents a learned "style" or a characteristic mode of visual expression.
4.  **Dynamic Application:** When rendering, the engine finds the learned allele that is most "similar" to the current internal state. It then uses this allele to modulate the final color, blending the default color with the allele's characteristic color.

### Why it's Valuable

*   **Emergent Aesthetics:** The visual style of the output is not designed; it emerges from the engine's interaction with the data. The system develops its own aesthetic sense.
*   **Context-Dependent Visuals:** The system can learn to render the same internal state in different ways depending on the context (referred to as "geometries" in the code), making its visual language richer and more expressive.
*   **Deepens the Metaphor:** This concept powerfully reinforces the biological metaphor of the "genome." The engine doesn't just have a fixed phenotype; it has a set of learned "alleles" that allow for varied expression of its underlying "genetic" state.
