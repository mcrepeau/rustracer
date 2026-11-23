# rustracer

A physically-based path tracer built in Rust. It simulates how light actually behaves — bouncing, refracting, and scattering through a scene — to produce photorealistic images. It runs interactively in a window and can also render to PNG in headless mode.

**Scenes are TOML files.** There are no built-in scenes: the viewer, `--render`, and `--bench` all load a scene file (`scenes/scene.toml` / `scenes/bench.toml` by default, or anything you pass with `--scene`).

## Build & Run

```sh
cargo run --release                        # interactive viewer (scenes/scene.toml)
cargo run --release -- --scene scenes/sample.toml         # interactive viewer, that scene file
cargo run --release -- --render [options]  # headless PNG render (--scene <path.toml>, default scenes/scene.toml)
cargo run --release -- --bench             # performance benchmark (scenes/bench.toml; pass another file to bench it)
```

Optional `denoise` cargo feature (OIDN, bundles its own binaries — no system install needed):

```sh
cargo build --release --features denoise
```

### Headless render options

```
--scene <path.toml>  scene file (default: scenes/scene.toml)
--samples <n>        samples per pixel (default: scene maximum; with --adaptive, the hard SPP cap)
--preset <name>      measured recipes: preview = 16 spp + OIDN 0.6, final = 64 spp + OIDN 1.0,
                     reference = adaptive, cap 4096 spp, raw.  Explicit flags override the preset.
--width/--height     output size in pixels (default: 1200x800)
--exposure <f>       exposure multiplier (default: 1.0)
--output <path>      output PNG path, or a directory in animation mode (default: auto-generated)
--tonemapper <name>  agx or aces (default: agx)
--adaptive           adaptive sampling: run until the convergence target is reached
--convergence <pct>  convergence target in % (default: 99.0; requires --adaptive)
--min-samples <n>    minimum SPP before any pixel may be marked converged (default: 128; requires --adaptive)
--frames <n>         render only the first N frames of a scene animation; 1 = single still (default: all)
--no-photon-map      disable caustic photon map (faster, no caustics)
--denoise [strength] run OIDN after render; optional blend in [0.0, 1.0] (default: 1.0; requires the denoise feature)
```

Recommended workflow: render once with `--preset reference`, iterate with `--preset preview`, ship with
`--preset final`, and check the result with `render_qa.py` (see Tools below).

## Controls

| Key | Action |
|-----|--------|
| F | Toggle free camera (WASD + mouse look) |
| C | Reset camera |
| R | Reload the scene file (edit the TOML, press R) |
| Space / Shift | Move up / down (free cam) |
| , / . | Decrease / increase FOV |
| I / O | Decrease / increase aperture |
| - / = | Decrease / increase exposure |
| Arrows | Rotate sun (physical-sky scenes) |
| T | Toggle tonemapper (AgX / ACES) |
| V | Toggle adaptive sampling |
| M | Toggle photon map (caustics on/off) |
| L | Print current camera position to terminal (TOML format) |
| P | Save PNG |
| Enter | Pause / resume rendering |
| N | Toggle OIDN denoising (`denoise` feature) |
| J / K | Decrease / increase denoise blend |
| ? | Reprint controls |
| Esc | Release mouse / quit |

## Tools

- **`render_qa.py`** — render QA: artifact detection (black dots, fireflies, hue outliers), sanity stats
  (crushed/cliped %, noise σ, degenerate-image fail), and optional reference comparison:

  ```sh
  py render_qa.py render.png                          # artifacts + grade
  py render_qa.py a.png b.png                         # comparison table
  py render_qa.py render.png --reference ref.png      # + linear-space RMSE/PSNR vs a high-spp render
  py render_qa.py render.png --json                   # machine-readable (always a JSON list)
  py render_qa.py render.png --annotate               # save render_qa.png with circles
  ```

- **`make_studio_softbox.py`** — regenerates the procedural studio env maps: `py make_studio_softbox.py`
  writes the bright white-tent `scenes/assets/studio_softbox.hdr`, and
  `py make_studio_softbox.py --dark` writes `scenes/assets/studio_dark.hdr` — same softbox panels
  over a near-black room, used by gem scenes generated with `--dark` so the stone pops against a
  surround (gradient base + soft glows + warm `TINT`).

- **`report_to_toml.py`** — turns an IGI/GIA grading-report PDF into a renderable `scenes/<report>.toml`
  (stdlib only; see "From a grading-report PDF" below):
  `py report_to_toml.py 625469899.pdf [--material X] [--cut Y] [--gap-mm 15] [--no-caustics] [--no-animation] [--dark]`

## Features

### Rendering

Path tracing works by sending many rays from the camera into the scene. Each ray bounces around — hitting surfaces, scattering, sometimes hitting a light — and the average of thousands of these paths gives you the final pixel colour. More samples per pixel means less noise.

- **Unidirectional path tracing** with full global illumination (indirect lighting, colour bleeding, soft shadows — all emerge naturally from the simulation)
- **Next-event estimation (NEE) with MIS** — at each diffuse bounce, shadow rays are cast directly toward area lights, the sun, and the env map. Contributions are combined with the MIS **power heuristic (β = 2)** over all strategies (`wᵢ = pᵢ² / Σpⱼ²`), which partitions unity exactly; emission is never double-counted with NEE.
- **Cosine-weighted hemisphere sampling** for diffuse indirect bounces — samples are biased toward directions that contribute more energy, reducing variance without introducing bias
- **Depth of field** — rays are scattered across a simulated lens aperture, producing realistic out-of-focus blur. Bokeh shape is controlled by a configurable blade count.
- **Motion blur** — in animated scenes, each frame's rays are jittered over that frame's own time slice `[t, t+1/fps)`

### Spectral rendering

Most path tracers work entirely in RGB — red, green, and blue channels. This tracer optionally goes further by simulating actual wavelengths of light, which enables physically correct phenomena that RGB cannot reproduce.

- **Hero-wavelength spectral rendering** — one wavelength λ (between 380 and 700 nm) is sampled per camera ray and tracked through all bounces. The final colour is reconstructed using the CIE 1931 colour matching functions.
- **SpectralDielectric** — glass whose refractive index depends on wavelength (Cauchy equation: `n(λ) = B + C/λ²`). Different colours bend by different amounts, and at the first dispersive *refraction* the luminance-weighted CMF tint `spectral_fire_rgb(λ)` is folded into the throughput (applied once per path; external Fresnel reflection stays white), producing chromatic aberration, diamond fire, and rainbow caustics. The tint is weighted by ȳ(λ)^0.5 so violet fire is dimmer than green/yellow — dispersed white light stays white on average but reads as a balanced rainbow, not purple. Gem presets use physical constants (diamond nD 2.417, nF−nC ≈ 0.044).
- **SpectralMetal** — conductor Fresnel reflectance computed from measured IOR data (Johnson & Christy 1972) for gold, copper, and silver.

### Acceleration

- **QBVH (8-wide Bounding Volume Hierarchy)** — the scene is wrapped in a tree of axis-aligned bounding boxes. Ray traversal tests 8 child nodes simultaneously using AVX2 SIMD instructions, with a 4-wide SSE2/NEON fallback on other hardware. Shadow rays use a faster `any_hit()` variant that exits at the first hit.
- **Parallelised row rendering** via Rayon — image rows are rendered by separate CPU threads simultaneously
- **Adaptive sampling** — per-pixel per-channel variance (Welford online algorithm) tracks convergence; converged pixels are skipped in later passes. A hard SPP cap (`--samples` or the scene's `max_samples`) guarantees termination, and non-finite samples are dropped instead of poisoning the accumulator.

### Caustics (Photon Mapping)

Caustics are the bright, curved light patterns formed when light focuses through a curved transparent surface. This tracer uses a **photon map**: a separate forward-tracing pass shoots photons from the light source, stores where they land after specular bounces, and builds a kd-tree over them. During the main render, the local photon density estimates caustic radiance.

- Balanced kd-tree (median-split, widest-axis) built once per scene/frame
- Epanechnikov kernel density estimation
- Hemisphere normal filtering rejects photons whose stored normal opposes the shading normal — no caustic leakage through opaque geometry
- Photons are stored after any specular bounce (plain glass included, not just spectral materials)
- Emitter: the sun for `physical` backgrounds, a luminance-sampled bright direction for `env_map` backgrounds, or the first `[[lights]]` quad for `solid` backgrounds
- The sun/env emission disk is sized from the *real* geometry only (`Hittable::photon_bounds()` — infinite planes are excluded), so plane-floor scenes get a tight disk and a populated caustic map instead of wasting ~all photons; the photon storage luminance clamp scales with the disk area to stay sun-proportional at any scene scale

### Backgrounds

Three background modes, configured per scene:

```toml
[background]
type = "physical"          # Preetham sky + sun disk
sun_azimuth   = -55.0      # degrees clockwise from +Z
sun_elevation = 32.0       # degrees above the horizon
turbidity     = 3.0        # must be >= 1.7
```

```toml
[background]
type = "env_map"           # equirectangular HDR (.hdr Radiance RGBE or .exr), v=0 = zenith
path = "assets/sunset_puresky_4k.hdr"
```

```toml
[background]
type  = "solid"
color = [0.002, 0.002, 0.005]
```

Environment maps are importance-sampled with a 2D CDF built from luminance × sin θ, and NEE samples them directly at diffuse bounces. Files of any size load (no memory cap).

### Materials

TOML material types (see `scenes/scene.toml` for a full annotated example):

| TOML type | Description |
|-----------|-------------|
| `lambertian` | Cosine-weighted diffuse with a solid `color` |
| `checker` | 3-D checker, `even`/`odd` colours, `scale` = tile size in world units |
| `image` | Lambertian with an sRGB image texture (`path`) |
| `metal` | `color` + `fuzz`, maps to a metallic PBR |
| `dielectric` | Glass with exact Fresnel (Rs/Rp) and TIR, configurable `ior` |
| `spectral_dielectric` | Dispersive glass, `ior` at the d-line + `dispersion` (Cauchy) |
| `spectral_metal` | `variant` = gold/copper/silver (J&C 1972), `roughness` |
| `pbr` | Full GGX microfacet BRDF — anisotropic VNDF sampling (Heitz 2018), metallic/roughness, clearcoat, sheen |
| `textured_pbr` | PBR driven by image maps (albedo/roughness/metallic/ao/normal) |
| `sss` | Subsurface scattering: `albedo`, `sigma_a`, `ior`, `density`, `g` (Beer–Lambert + Henyey–Greenstein) |
| `pearl` | Thin-film iridescence with a nacre LUT; colour shifts with view angle |
| `diffuse_light` | Flat-colour emitter; values > 1.0 are HDR-bright |

Emitters in `[[lights]]` are quads (`corner`, `u`, `v`, `color`) that also join the NEE light list; optionally `temp_k` + `intensity` make the emitter a Planck blackbody (spectrally weighted).

`textured_pbr` supports **tangent-space normal maps** (OpenGL convention; per-vertex tangents from UV derivatives, Gram-Schmidt at shade time, TBN transform into world space).

### Post-processing

- **AgX** tonemapper (default) — filmic response curve that handles very bright highlights gracefully (Troy Sobotka)
- **ACES** tonemapper — the full RRT + ODT film pipeline
- Exact IEC 61966-2-1 sRGB encode (256-entry LUT) on save
- Optional **OIDN denoising** — runs on a background thread, updates every 32 samples, blend adjustable with J/K

## Custom scenes (TOML)

The viewer loads `scenes/scene.toml` by default, or `--scene path/to/file.toml`;
press **R** to hot-reload after editing. `--render --scene <path.toml>` renders headless.
Asset paths inside a scene file (env maps, textures, OBJ meshes) resolve relative to the
scene file's own folder first, then the working directory — that is why `scenes/` carries
its own `assets/` subfolder.

**`scenes/sample.toml` is the full parameter reference** — every section (`[camera]`,
`[background]`, `[[objects]]` shapes + transforms + materials, `[[lights]]`, `[[gems]]`,
`[caustics]`, `[animation]`) with every field commented. Copy it to start a new scene;
the other files in `scenes/` are working examples.

Gem cuts: `diamond` is the brilliant family (round, or oval via `oval_ratio > 1`) and
`step_cut` the step-cut family (polygon girdle outline, stepped rows, optional chevrons);
`[[gems]]` `cut = "..."` presets cover brilliant / step (emerald) / princess / asscher /
radiant / baguette / cushion / trillion / marquise / pear. All cuts are analytic convex
polyhedra (half-space intersection, ray cost O(planes)); non-convex cuts (hearts,
clover) need an OBJ mesh.

## From a grading-report PDF

`py report_to_toml.py REPORT.pdf` (Python stdlib only) pulls the IGI/GIA fields straight out of the
PDF's FlateDecode text streams — report number/date, description (-> material preset), cutting
style (-> cut preset), measurements, carat weight, color/clarity grades and the proportions
diagram (crown% / table% / pavilion% / depth%) — and writes a ready-to-render scene TOML into
`scenes/` (next to this script):

```
py report_to_toml.py 625469899.pdf        # writes scenes/625469899.toml
cargo run --release -- --scene scenes/625469899.toml
```

The stone is staged in bright product lighting — seamless light-gray backdrop (which also acts
as ambient fill) plus one overhead softbox panel sized relative to the stone, Blue Nile style
(`--dark` switches to the dark-room studio HDR) — with caustics on;
camera distance and photon gather radius scale with the measured size. The scene also ships a
ready-to-render `[animation]` / `[animation.orbit]` block whose `radius` / `height` /
`focus_dist` derive from the camera distance (3.2 × measured length, vfov 30°), so every stone
gets identical framing — a full 360° orbit, 96 frames at 24 fps. Overrides: `--material`,
`--cut`, `--gap-mm`, `--no-caustics`, `--no-animation`, `--dark`. Unparseable fields produce warnings,
never a broken file.
The report PDFs and the per-stone scenes they generate stay local (git-ignored) — only
`scenes/scene.toml`, `scenes/sample.toml` and `scenes/bench.toml` are tracked.

## Implementation notes

- Release builds use `target-cpu=native`, fat LTO, and a single codegen unit
- QBVH node width (8 vs 4) is selected at compile time via `#[cfg(target_feature = "avx2")]`; SAH-binned build; traversal uses an inline stack with a Vec overflow
- Spectral CMF weight is applied at the first spectral bounce via a `spectral_weighted` flag; later bounces multiply by a scalar reflectance so the weight is never compounded
- MIS power heuristic (β=2) with the sum of squares over area/sun/env/BRDF strategies — a true partition of unity; emission gets weight 1 on camera rays and after specular bounces, MIS weight after diffuse ones
- Env map importance sampling: 2D CDF (row marginal × per-row conditional) from luminance × sin θ, two-level binary search with sub-pixel jitter; solid-angle PDF `p_ω = L·W·H·inv_total / (2π²)` feeds the MIS weights
- Shading normals from normal maps feed all PDF/BRDF paths (`scattering_pdf`, `specular_brdf_cos`, `specular_sampling_pdf`, `CosinePdf`), keeping MIS consistent with the actual scattering geometry
- Photon kd-tree: `select_nth_unstable_by` median partition, split-plane pruned queries
- Adaptive sampling: per-channel Welford variance, relative threshold `ADAPTIVE_THRESHOLD`, min-samples floor, hard SPP cap, non-finite samples dropped; converged pixels are skipped in later passes
- OIDN runs on a background thread and blends into the display without blocking the render loop
- Normal-map tangents: per-face tangents accumulated per-vertex at OBJ load, normalised + Gram-Schmidt at shade time

### Compile-time tuning knobs

| Constant | File | Default | Effect |
|---|---|---|---|
| `MAX_DEPTH` | `src/renderer.rs` | `50` | Maximum ray bounce depth. Lower cuts render time but loses energy in glass/mirror-heavy scenes. |
| `ADAPTIVE_THRESHOLD` | `src/main.rs` | `0.05` | Relative standard error (σ/μ, per channel) below which a pixel is converged and skipped. |
| `FIREFLY_CLAMP` | `src/main.rs` | `8.0` | Samples above `max(FIREFLY_CLAMP × running pixel mean, FIREFLY_ABS_FLOOR)` luminance are scaled down to that threshold. Downward bias on high-variance pixels — use `--denoise` for bias-free firefly suppression in finals. |
| `FIREFLY_MIN_SAMPLES` | `src/main.rs` | `4` | Minimum samples before per-pixel firefly clamping activates. |
| `FIREFLY_ABS_FLOOR` | `src/main.rs` | `5.0` | Absolute floor (linear luminance) for the clamp threshold — without it, a pixel whose running mean is still ~0 clamps its first legitimate bright sample to ~0 and renders black. |
| `FIRE_LUMA_POW` | `src/spectrum.rs` | `0.5` | Exponent applied to the photopic luminance ȳ(λ) in the `spectral_fire_rgb` dispersion tint. Higher (→1.0) dims violet/blue fire harder so the gem reads warmer/whiter; lower (→0.0) gives full rainbow fire with more blue/violet sparkle. The tint stays per-channel normalised, so white light is white on average at any value. |
| `FIRE_LUMA_FLOOR` | `src/spectrum.rs` | `0.08` | Floor for the ȳ(λ)^`FIRE_LUMA_POW` weight so deep violet (380–420 nm) is dimmed, not erased, in the fire tint. |
