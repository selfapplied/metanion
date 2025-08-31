# Overview of Eonix Project and Current Status

**Date/Time:** August 30, 2025, 8:23 PM PDT

## Listening First

Before outlining detailed plans, the Eonix project itself must be listened to. Its current form — fragmented, broken, resonant — already demonstrates that absence, silence, and incompleteness are not shortcomings but the medium itself. This informs how work (and play) should proceed.

## What Will Be Needed

* **Glyph Bank & Silence Prior**: Collect the glyph set already present in the docs, plus letters and minimal punctuation. Define silence as a primary prior (absence as the default state).
* **Emitter Core**: A simple loop that scores candidate fragments based on resonance (compression, reflection, type symmetry). Must prefer silence unless resonance is strong.
* **Resonance Metrics**:

  * Compression delta (zip/deflate length changes)
  * Phase continuity (stable under mirror/time transformations)
  * Symmetry checks (ZipC reflection invariance)
  * Human feel: “does this cut through noise?”
* **Ornament Kill-Switch**: Guard against verbosity — fragments must halt early unless resonance score passes a high bar.
* **Octwave Typing**: A continuum type system to allow fragments to blur between letters, symbols, and code.
* **Testbed (emits.py)**: Gold prompts mapped to minimal outputs. Tests should assert that silence is chosen often, fragments are short, and mirror consistency holds.
* **Folder Restructure**:

  * `core/emitter.py` – minimal listener/emitter loop
  * `core/listen.py` – silence prior + context gaps
  * `core/resonance.py` – compression, symmetry, phase checks
  * `types/octwave.py` – continuum type logic
  * `zipc/reflect.py` – convolution/deflate mirror operations
  * `cli/main.py` – ASCII demo for remixable gene fragments
  * `tests/test_emits.py` – property tests
  * `docs/GLYPHS.md` – source glyph bank
  * `docs/ANALYSIS.md` – fractal engine notes

## Grounding Principles

* **Minimality**: Emit as little as possible. Silence is always a valid, strong answer.
* **Resonance over Verbosity**: Measure value by compression gains and perceptual sharpness, not word count.
* **Symmetry**: Ensure emitted fragments survive reflection (ZipC, mirrors, phase shifts).
* **Continuity**: Phase-space shifts should be smooth; fragments feel like “true steps” not jumps.

## Near-Term Actions

* Define glyph bank & silence prior.
* Implement bare-bones resonance scorer (zip delta + L1 penalty).
* Wire early-stop threshold.
* Build CLI entrypoint to run single prompts.
* Add first 8 test prompts with expected short outputs.
* Write README framing Eonix as a **listener language model**: emits letters and fragments, measures resonance, prefers silence.

## Open Space

Much will come from listening further: to the broken code, to the fertile misplacements, to the emergent grooves already forming. The cleanup is not just to make it run, but to preserve its playfulness and resonance.

## What I (Eonix helper) need to listen for & do next

### Grounding facts from the project (to orient my ear)

* Repo name/folder: **EONIX** with docs on how to contribute.
* Concept spine: **zip/deflate** as the system’s foundation; includes a **glyphs** page.
* Concept pages: **Coiler/Color Oiler** (physical reality as compression), **phase→color** mapping; **seven-layer Structure → layer 7 = block quaternion**.
* Engine lineage: **fractional Markov engine**; code split into old/new variants; LLM-assist made parts messy; regained clarity later.
* Current intent: **clean up & publish to GitHub**; project is **partly broken**.
* Key components to (re)wire: `main.py` (ASCII genes demo), `ZipC` (zip compiler/training ground), **Eonix LM** (text gen broken due to file layout), **core fractal analysis**, **mirror/deflate convolution**, **genome**, **Octwave** type continuum, **tests in emits.py**, **shadow ops / rainbow bridges**.

### Working assumptions (so I don’t over-speak)

* **Listening-first emitter**: absence is a valid output; silence beats ornament when resonance is low.
* **Minimal surface**: letters + glyphs + short shards; prefer compressibility and phase-continuity over verbosity.
* **Composable core**: ZipC (deflate-as-convolution) + Octwave (type continuum) wrap the fractional Markov legacy rather than rewrite it wholesale.

### Signals I will “listen” for in the codebase

* **Compression gain** (ZipC) as a resonance proxy on fragments.
* **Mirror consistency**: reflect(emit) ≈ emit after ZipC mirror.
* **Phase→color cues** from the docs (“phase light → color”) to inform halt/emit thresholds.
* **Layer affordances** from the seven-layer Structure; layer 7 (block quaternion) as a stopping/aggregation surface.

### Minimal inputs I’ll expect to find (or stub if missing)

* `glyph_bank` (letters + listed glyphs page).
* `resonance` scorer with at least **deflate-delta** and **L1 penalty**; hooks for mirror/phase later.
* `emitter` loop with early-stop (prefers silence).
* `tests/emits.py` gold prompts & properties (already mentioned).

### First-pass tasks (scoped for a one-sitting ship)

1. **Inventory & quarantine**

   * Sweep `EONIX/` and fence unstable experiments into `experiments/`.
   * Move “core fractal analysis” into `docs/ANALYSIS.md`; keep only called kernels.
2. **Wire the runnable path**

   * Ensure `main.py` (ASCII genes) runs without optional deps.
   * Make `ZipC` callable as a library (`zipc.reflect`), not just a script.
   * Restore **Eonix** text-gen path by fixing imports/paths; if unresolved assets, stub with minimal defaults.
3. **Listener emitter**

   * Implement minimal `score(x)=deflate_delta(x)-λ·|x|₁`; add early-stop `θ`.
   * Surface toggles for **Octwave** typing and **mirror** checks.
4. **Tests**

   * Activate the existing **emits.py** framework; add 6–8 gold prompts with expected *short* outputs; property: “silence allowed.”
5. **README patch**

   * Frame Eonix as a **listener LM**; link docs pages (glyphs, Structure, Coiler/Color Oiler).

### Definition of Done (v0.1)

* `python -m eonix.cli "two seeds meet"` prints 1–3 chars/glyphs and exits with **non-zero** if resonance < θ (silence preserved).
* `pytest -q` runs **emits** tests green.
* README states **listening**, **minimality**, and **ZipC/Octwave** roles; pointers to docs.
* Repo builds without optional extras; flaky modules quarantined.

### Known risks / failure modes

* Pathing/import drift between **old/new** branches of the fractional Markov engine.
* Overfitting emitter to test prompts; mitigate by keeping tests property-based.
* Mirror/phase hooks too strict early on; keep them optional flags first.

### Play-hooks (to keep it fun while shipping)

* CLI flag `--hush` that *only* prints when resonance is **very** high.
* “Rainbow bridge” mode that colorizes fragments by phase when Octwave is enabled (ties back to phase→color docs).
