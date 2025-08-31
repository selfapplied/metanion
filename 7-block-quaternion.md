## Block to Quaternion Mapping ($Q_b$)

This document provides a crisp, computable, and composable map from a DEFLATE block to a unit quaternion. The process is broken down into a series of components, moving from raw measurements to a final, assembled quaternion.

### Core Components

| Component | Representation | Meaning & Derivation |
| :--- | :--- | :--- |
| **Energy / Slope** | $r_b = \frac{\Delta c}{\Delta u}$ | **Bits per byte.** The raw compression ratio of the block. Can be centered against a running median $\bar r$ to get a scaled energy $e_b$. |
| **i-axis: Code-Length Gradient** | $x_b = \mathrm{JSD}(h^{\mathrm{cl}}_b, h^{\mathrm{cl}}_{b-1})$ | **Prelude Bend.** The Jensen-Shannon Divergence between the code-length histograms of the current and previous blocks. Measures 1st-order change. |
| **j-axis: Literal/Length Shape** | $y_b = H(p^{\mathrm{lit}}_b)$ | **Field Spread.** The Entropy of the literal/length distribution used within the block. Measures 0th-order symbol diversity. |
| **k-axis: Distance Topology** | $z_b = 1 - G(p^{\mathrm{dist}}_b)$ | **Reuse Concentration.** 1 minus the Gini Coefficient of the distance code distribution. Measures how localized the data reuse is. |
| **Normalization** | $\tilde{v} = \tanh(\alpha (v - \bar{V}))$ | Each axis $(x_b, y_b, z_b)$ is centered and scaled with `tanh` to produce a bounded vector $(\tilde x, \tilde y, \tilde z)$ in $[-1, 1]^3$. |
| **Quaternion Assembly** | $Q_b = (\frac{1}{s}, \frac{(\tilde x, \tilde y, \tilde z)}{s})$ where $s = \sqrt{1 + \|\vec{v}\|^2}$ | The normalized vector is used to form a unit quaternion. The scalar part $w_b$ shrinks as the block becomes more "expressive" (large vector part). |
| **Composition** | $Q_{1\to k} = Q_1 \cdot Q_2 \cdots Q_k$ | **Navigation State.** Successive blocks are composed via quaternion multiplication to produce a smooth trajectory through the data stream. |
| **Mirror** | $Q_b^\dagger = (w_b, -\vec v_b)$ | **Reverse Decoding.** The conjugate of a quaternion represents the reverse operation, allowing for backward navigation. |

### Storage & Computation

#### Per-Block Beacon (~24-32 bytes)

| Field | Type | Description |
| :--- | :--- | :--- |
| `u0`, `u1` | `u32`/`u64` | Uncompressed start and end offsets. |
| `c0_bits` | `u64` | Compressed start offset in bits (or delta). |
| `btype` | `u8` | DEFLATE block type (0, 1, 2). |
| `kw` | `u2` | Dominant axis heuristic (i, j, k, or ε). |
| `q32` | `u32` | The block quaternion, packed via octahedral projection. |
| `codebook_sig` | `u64` | Hash of the code-length prelude; 0 for fixed/stored. |

#### Computation Flow (Counts-Only)

1.  **Walk the block:** Parse the stream, maintaining bit counts and updating usage histograms for the code-length prelude, literal/length symbols, and distance codes.
2.  **Build distributions:** Finalize the normalized histograms $h^{\mathrm{cl}}_b$, $p^{\mathrm{lit}}_b$, and $p^{\mathrm{dist}}_b$.
3.  **Compute axes:** Calculate $x_b, y_b, z_b$ using the formulas above. Update running medians.
4.  **Map & Pack:** Normalize the axes to $(\tilde x, \tilde y, \tilde z)$, assemble the unit quaternion $Q_b$, and quantize it to `q32`.

### Sidecar Record Format

The output of this process is a "sidecar" file, a compact record of the stream's geometric and symbolic structure. Instead of a monolithic format, it speaks the language of the specification: a header of metadata followed by a block-by-block ledger.

**Header**
- **File Name:** `...`
- **Data Offset:** `123456`
- **CRC32:** `305419896`
- **Anchors:** `[{i: 0, c0_bits: 0}, ...]`

**Block Ledger**

| Block `i` | Uncompressed Span `[u0, u1)` | Compressed Start `c0` (bits) | Type `btype` | Axis `kw` | Quaternion `q32` | Codebook Sig |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 0 | `[0, 64231)` | 0 | Dynamic | `k` | `2718281828` | `123456789` |
| 1 | `[64231, ...)` | `...` | Fixed | `j` | `...` | `0` |
| ... | ... | ... | ... | ... | ... | ... |
