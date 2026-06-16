# rustracer

A physically-based spectral path tracer with interactive camera controls. Built in Rust with an 8-wide QBVH accelerator (AVX2), hero-wavelength spectral rendering, explicit NEE with MIS, anisotropic GGX, and a kd-tree caustic photon map.

## Build & Run

```sh
cargo run --release                        # interactive viewer
cargo run --release -- --render [options]  # headless PNG render
cargo run --release -- --bench             # performance benchmark
```

### Headless render options

```
--scene <name>       random | cornell | nextweek | <path.toml>
--samples <n>        samples per pixel (default: scene maximum)
--width  <n>         output width  in pixels (default: 1280)
--height <n>         output height in pixels (default: 720)
--exposure <f>       exposure multiplier (default: 1.0)
--output <path>      output PNG path (default: auto-generated)
--tonemapper <name>  agx or aces (default: agx)
--adaptive           stop per-pixel when converged
--denoise            run OIDN after render (requires denoise feature)
```

### OIDN denoising (optional)

Install [Intel Open Image Denoise](https://openimagedenoise.github.io) via Homebrew:

```sh
brew install open-image-denoise
cargo run --release --features denoise
```

Press **N** in-app to toggle denoising. The denoiser runs on a background thread and updates every 32 samples.

## Controls

| Key | Action |
|-----|--------|
| 1–4 | Switch scene |
| F | Toggle free camera (WASD + mouse look) |
| C | Reset camera |
| R | Restart / reload scene |
| Space / Shift | Move up / down (free cam) |
| , / . | Decrease / increase FOV |
| I / O | Decrease / increase aperture |
| - / = | Decrease / increase exposure |
| Arrows | Rotate sun (Physical sky scenes) |
| T | Toggle tonemapper (AgX / ACES) |
| V | Toggle adaptive sampling |
| P | Save PNG |
| Enter | Pause / resume rendering |
| N | Toggle OIDN denoising (`denoise` feature) |
| J / K | Decrease / increase denoise blend |
| ? | Reprint controls |
| Esc | Release mouse / quit |

## Features

### Rendering
- Unidirectional path tracing with full global illumination
- Explicit next-event estimation (NEE) with MIS balance heuristic — shadow ray cast toward area lights at every diffuse bounce; direct and BRDF sampling weights balanced via `p_nee / (p_nee + p_brdf)`
- `any_hit()` BVH traversal for shadow rays — exits at first hit, skips child sorting
- Cosine-weighted hemisphere sampling for indirect diffuse
- Depth of field with polygonal bokeh (configurable blade count)
- Motion blur

### Spectral rendering
- Hero-wavelength spectral path tracing: one wavelength λ sampled per camera ray, accumulated with CIE 1931 colour matching functions
- **SpectralDielectric** — Cauchy dispersion equation `n(λ) = B + C/λ²`; correct chromatic aberration and rainbow caustics from glass
- **SpectralMetal** — exact unpolarised conductor Fresnel from Johnson & Christy (1972) tabulated IOR data for Gold, Copper, and Silver (13 samples, 25 nm grid, 380–680 nm)
- `spectral_weighted` flag ensures the CMF weight is applied exactly once per path regardless of bounce count

### Acceleration
- QBVH: 8-wide with AVX2, 4-wide fallback with SSE2/NEON — selected at compile time
- Parallelised tile rendering, accumulation, and tone-mapping via Rayon
- Adaptive sampling with perceptual luminance-variance threshold per tile

### Photon mapping
- Balanced kd-tree (median-split, widest-axis) built once at scene construction
- Epanechnikov kernel density estimate pre-divided by π; caller multiplies by surface albedo
- Hemisphere normal filtering: photons whose stored surface normal opposes the shading normal are rejected, preventing caustic leakage through opaque geometry
- Only paths through spectral materials (`SpectralDielectric`, `SpectralMetal`) contribute photons — ordinary glass balls are excluded to keep the ground clean
- Per-scene gather radius calibrated to ~5% of coordinate scale

### Materials
| Material | Description |
|----------|-------------|
| `Lambertian` | Cosine-weighted diffuse with solid colour or texture |
| `PbrMaterial` | Full GGX microfacet BRDF — anisotropic VNDF sampling (Heitz 2018), metallic workflow, clearcoat lobe |
| `Dielectric` | Schlick-Fresnel glass with configurable IOR |
| `SpectralDielectric` | Dispersive glass via Cauchy equation; hero-wavelength CMF weighting |
| `SpectralMetal` | Conductor Fresnel from J&C 1972 IOR tables (Au, Cu, Ag) |
| `SssMaterial` | Subsurface scattering with Beer–Lambert absorption |
| `PearlMaterial` | Thin-film iridescence with nacre LUT; orientation-varying colour shift |
| `BlackbodyLight` | Planck emission spectrum at configurable temperature and brightness |
| `DiffuseLight` | Flat-colour emissive surface |

### Post-processing
- **AgX** tonemapper (default) — filmic response with pleasant highlight roll-off
- **ACES** tonemapper — full RRT + ODT pipeline with input/output transforms
- Optional OIDN denoising with adjustable blend ratio

### Scene system
- TOML scene file loading (`--scene path/to/scene.toml`)
- Procedural and image textures (Perlin noise, solid colour, checker, file-loaded)
- Shapes: Sphere (static and motion-blur), Quad, Box, Cylinder, Cone, Disk

## Scenes

1. **Random Spheres** — Physical sky, large checkerboard ground, eight hero spheres (gold, copper, silver SpectralMetal; SpectralDielectric diamond; PBR; SSS; pearl; iridescent), ~350 small random diffuse/glass/SSS balls. Spectral-material caustics projected onto ground by the sun.

2. **Cornell Box** — 6500 K BlackbodyLight, classic coloured walls, two stacked boxes, SpectralDielectric dense-flint glass sphere (rainbow caustics), SpectralMetal gold sphere. Photon-map caustics enabled.

3. **Next Week** — Area light, moving sphere, Dielectric glass sphere, metallic PBR sphere, blue volumetric fog sphere, earth-texture sphere, Perlin-noise sphere, SpectralDielectric diamond (radius 100) below the light, pearl sphere, cluster of 1000 small white spheres, global mist medium. Photon-map caustics enabled.

## Implementation notes

- Release builds use `target-cpu=native`, fat LTO, and a single codegen unit
- QBVH node width (8 vs 4) selected at compile time via `#[cfg(target_feature = "avx2")]`
- Spectral CMF weight applied at first spectral bounce via `spectral_weighted` flag; subsequent bounces multiply by a scalar Fresnel/transmittance so the weight is never compounded
- MIS weight: `w_nee = p_nee / (p_nee + p_brdf)` (balance heuristic); emitted radiance accumulated only on camera rays and after specular bounces to avoid double-counting with NEE
- Photon kd-tree built with `select_nth_unstable_by` (in-place median partition, O(N log N)); query recurses with split-plane pruning (O(√N) average for range queries)
- Adaptive sampling tracks per-tile luminance variance; tiles below threshold are skipped in subsequent passes
- OIDN runs on a background thread; result is blended into the display buffer without blocking the render loop
