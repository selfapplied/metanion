### Eonyx Operator Spec Table

Legend. Primitives: 𝕄 Measure, 𝕌 Transport, ⋆ Convolve, ρ Route, Π Match, ℛ Reconcile, 𝕀 Inflate.
Painted ops: Ψ Paint/steer, Θ Stencil/bandpass, Ω Portal/teleport, Λ Afford/adapter, Σ Blend.
Domains: seq (tokens/bytes), tree (AST/scope), S³ (block-energy quats), mixed (cross-domain).

| File | Role | Domains | Primitives | Painted Ops | Signals (In→Out) | Laws | CE1c Emits | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fme_engine.py | Orchestrator (listen→walk→decide) | mixed | ρ, ℛ, (calls 𝕄,𝕌,⋆,Π,𝕀) | Σ, Λ | {Q_blocks}→w; matches→Δ | conservation, gauge(π) | w_route,Δ,roundtrip_err | stable |
| deflate.py | DEFLATE measurement + block quats | mixed,S³ | 𝕄, 𝕌 | Θ | payload → {hist,codebooks,Q_blocks} | gauge(π), unitary_commute | entropy_trace,Q_blocks | stable |
| qconv.py | Quaternion conv primitives | mixed | 𝕌, ⋆, Π | Θ | (K,X,B;D) → R | conv_theorem, unitary_commute | R_hash | stable |

| tokenizer.py | Tokenizer & SEQ conv surface | seq | ⋆, Π | Θ, Ψ, Ω | {K_seq,X_tokens,buffer}→R_seq→matches | conv_theorem, gauge(π) | R_seq_hash,seq_peaks | stable |
| fme_core.py | Core pipeline glue | mixed | ρ, ℛ | Σ | matches → Δ | conservation, shadow_monotone | Δ,proof_refs | stable |
| ce1_shadow_ops.py | Dual grammar (demand space) | mixed | Π, ℛ | Λ | responses → demands; supplies+demands → Δ | shadow_monotone, GetPut/PutGet | demands,Δ_reason | stable |
| ce1.py | CE1 seed parser/emitter | mixed | — | — | CE1 text ↔ objects | gauge(π) | seed_norm,seed_hash | stable |
| emits.py | CE1c ledger writer | mixed | — | — | op calls → CE1c rows | deterministic | block,ops[],invariants | stable |
| loader.py | IO/asset loading | mixed | 𝕄 (helper) | — | files → payloads/config | deterministic | load_report | stable |
| zip.py | Zip/deflate wrapper | mixed | 𝕄 (calls) | — | files → deflate stats | deterministic | zip_meta | stable |
| zipc.py | Zip compressor driver | mixed | 𝕄 (calls) | — | payload → compressed | deterministic | zipc_meta | stable |
| quaternion.py | Quaternion math utils | S³ | 𝕌 (math) | — | (U,K/X) → rotated | unitary | U_meta | stable |
| kern.py | Kernel shapes & factories | mixed | — | Σ, Θ | params → K | renorm_safe | kernel_id | stable |
| __init__.py | Package init & exports | mixed | — | — | modules → registry | deterministic | spec_hash | stable |
| main.py | CLI/entrypoint | mixed | — | — | argv → pipeline invocation | — | run_id | stable |
| eonyx.py | High-level API façade | mixed | ρ (calls), ℛ (calls) | Σ | payload/config → pipeline result | conservation | session_id | proposed |
| resonance.py | Pattern resonance scoring | mixed | Π | Θ | responses → resonance scores | gauge(π) | res_score | proposed |
| genome.py | Kernel genome registry | mixed | — | Σ | genes → kernel set | renorm_safe | genome_id | proposed |
| gene_style.py | Style genes → kernel edits | mixed | — | Ψ, Σ | style genes → K edits | renorm_safe | style_diff | proposed |
| style.py | Styling API | mixed | — | Ψ, Σ | style→kernel reweights | renorm_safe | style_apply | proposed |
| style_algebra.py | Algebra of styles | mixed | — | Ψ, Σ | styles → algebra ops | renorm_safe | style_ops | proposed |
| phases.py | Phase/state helpers | mixed | — | Ψ | features → phase tags | unitary | phase_tags | proposed |
| sprixel2.py | Visual primitives v2 | mixed | — | Ψ | signals → sprites v2 | unitary | sprite2_refs | proposed |
| sprixel.py | Visual primitives | mixed | — | Ψ | signals → sprites | unitary | sprite_refs | proposed |
| fme_text_generation.py | Text emission from signals | seq,tree | 𝕀 (calls) | Ψ | latents → tokens/snippets | adjoint_roundtrip | gen_snips | proposed |
| fme_training.py | Kernel fitting/learning | mixed | — | Σ, Θ, Ψ | data → updated kernels | renorm_safe | kernel_update | proposed |
| fme_color.py | Energy→color mapping | S³ | — | Ψ | q_path → color path | unitary | s3_viz | proposed |
| color.py | Color/phase mapping for visuals | mixed | — | Ψ, Θ | quats/features → color fields | unitary | viz_palette | proposed |
| branch.py | Flow control / routing variants | mixed | ρ | Σ | {Q_blocks} → route plan | gauge(π) | w_route | proposed |
| loom.py | Batch/conductor (many payloads) | mixed | ρ (calls) | Σ | batch → schedule/routes | gauge(π) | batch_plan | proposed |
| octwave.py | Wave/phase experiments | seq,S³ | ⋆, Π | Θ, Ψ | {K,X} → responses | conv_theorem | oct_peaks | proposed |
| deflate_genes.py | Learned motifs from DEFLATE | seq | 𝕄 | Θ, Σ | codebooks → motif kernels | conv_theorem | motif_ids | proposed |
| dena.py | DNA/sequence experiments | seq | ⋆, Π | Θ | {K_seq,X} → R_seq → peaks | conv_theorem | seq_peaks | proposed |
| twinz.py | Dual/paired runs | mixed | ρ (calls) | Σ | A/B configs → routes | gauge(π) | twin_plan | proposed |
| family.py | Type/trait families | mixed | — | Λ | traits → adapter hints | conservation | family_refs | proposed |
| vaez.py | VAE-ish latent experiments | mixed | 𝕀 | Θ, Σ | z → signals (seq/tree/S³) | adjoint_roundtrip | roundtrip_err | proposed |
| rich_grammar.py | High-level grammar skins | tree | ⋆, Π (calls) | Ψ | AST skin → responses | conv_theorem | rg_peaks | proposed |
| fme_analysis.py | Analysis reports | mixed | Π (calls) | Θ | responses → summaries | gauge(π) | report_refs | proposed |
| aspire.py | Style/intent presets | mixed | — | Ψ, Σ | style knobs → kernel/style mix | renorm_safe | style_id | proposed |
| vmg.py | Vector/metric geometry helpers | mixed | — | Θ | data → spectra/metrics | renorm_safe | metric_refs | proposed |

Notes
• “stable” = I’m confident it should own that contract now; “proposed” = clear fit but confirm placement.
• Laws are what each file should explicitly assert in its CE1 header comment.

⸻

CE1 header one-liners (paste at top of the files)

Use these to make the table enforceable:
	•	deflate.py

```text
CE1{lens=QCONV|mode=HilbertWalk|Ξ=eonyx:deflate|ops=[𝕄;𝕌]|emit=CE1c{entropy_trace,Q_blocks}}
```

	•	tokenizer.py

```text
CE1{Ξ=eonyx:seq|ops=[⋆;Π]|domain=seq|law=conv_theorem∧gauge(π)|emit=CE1c{R_seq_hash,seq_peaks}}
```

	•	fme_engine.py

```text
CE1{lens=QCONV|mode=HilbertWalk|Ξ=eonyx:fme|ops=[ρ;ℛ;𝕌;⋆;Π;𝕀]|emit=CE1c{w_route,Δ,roundtrip_err}}
```



	•	ce1_shadow_ops.py

```text
CE1{lens=Dual|mode=Shadow|Ξ=eonyx:shadow|ops=[Π;ℛ]|invariant=shadow_monotone|emit=CE1c{demands,Δ_reason}}
```

