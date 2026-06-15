# rustracer

A physically-based path tracer with interactive camera controls. Built with Rust, featuring an 8-wide QBVH accelerator (AVX2), SIMD intersection tests, importance sampling, and adaptive rendering.

## Build & Run

```sh
cargo run --release
```

Use `--bench` to run performance benchmarks:
```sh
cargo run --release -- --bench
```

### OIDN denoising (optional)

Install [Intel Open Image Denoise](https://openimagedenoise.github.io) via Homebrew:

```sh
brew install open-image-denoise
```

Then build with the `denoise` feature:

```sh
cargo run --release --features denoise
```

Press **N** in-app to toggle denoising. The denoiser runs in a background thread and updates the display every 32 samples.

## Controls

| Key | Action |
|-----|--------|
| 1-3 | Switch scene |
| F | Toggle free camera (WASD + mouse) |
| Esc | Exit (or release mouse from free cam) |
| P | Save PNG |
| T | Toggle adaptive sampling |
| N | Toggle OIDN denoising (`denoise` feature) |
| J / K | Decrease/increase denoise blend (default 80%) |
| Enter | Pause/resume physics |
| R | Restart random spheres scene |
| C | Reset camera |
| [ / ] | Decrease/increase aperture |
| , / . | Decrease/increase FOV |
| - / = | Decrease/increase exposure |
| Arrows | Adjust sun direction (Physical background) |
| Space / Shift | Move up/down |

## Features

- Path tracing with full global illumination
- QBVH accelerator: 8-wide with AVX2, 4-wide with SSE2/NEON — compile-time selected
- Parallelised rendering, accumulation, and tone-mapping via Rayon
- Depth of field and motion blur
- Importance sampling: cosine-weighted diffuse + area light NEE
- Adaptive sampling with perceptual variance threshold
- Optional OIDN denoising with adjustable blend (`--features denoise`)
- Physics simulation with sphere collisions and uniform-grid broad-phase
- Materials: Lambertian, Metal, Dielectric, SpectralDielectric, PBR, Pearl, DiffuseLight
- Shapes: Sphere, Quad, Box, Cylinder, Cone, Disk, InfinitePlane (with wave animation)
- Procedural and image textures (Perlin noise, solid colour, checker)
- Physical sky background with configurable sun direction
- Caustics via photon mapping (optional per-scene)
- TOML scene file loading (`cargo run --release -- path/to/scene.toml`)

## Scenes

1. **Random Spheres** — Ground plane, 3 showcase spheres, ~400 falling balls with gravity and collision
2. **Cornell Box** — Classic test scene with coloured walls, boxes, and a bouncing glass sphere
3. **Next Week** — Complex scene with ground boxes, motion blur, glass sphere, metal, fog, and earth texture

Custom scenes can be loaded from TOML files; see `scenes/` for examples.

## Implementation Notes

- Release builds use `target-cpu=native`, fat LTO, and single codegen unit
- The QBVH node width (8 vs 4) is selected at compile time via `#[cfg(target_feature = "avx2")]`
- BVH rebuilds automatically each tick when physics are active; static geometry is cached
- Adaptive sampling uses relative luminance variance per tile for consistent quality
- The denoiser runs on a background thread and blends with the raw accumulation buffer
