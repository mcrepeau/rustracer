# rustracer

A physically-based path tracer with interactive camera controls. Built with Rust, featuring a QBVH accelerator, SIMD intersection tests, importance sampling, and adaptive rendering.

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
| 1-4 | Switch scene |
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
- QBVH (4-wide BVH) with SSE2/NEON SIMD acceleration
- Motion blur (for static moving spheres)
- Depth of field
- Importance sampling (cosine + area lights)
- Adaptive sampling with perceptual variance threshold
- Optional OIDN denoising with adjustable blend (`--features denoise`)
- Participating media (fog, mist)
- OBJ mesh loading
- Physics simulation with sphere collisions
- Perlin noise textures

## Scenes

1. **Random Spheres** — Ground sphere, 3 feature spheres, ~400 falling balls with gravity
2. **Cornell Box** — Classic test scene with colored walls, boxes, and a bouncing glass sphere
3. **Mesh** — Loads OBJ files from `assets/model.obj` (auto-fits camera)
4. **Next Week** — Complex scene with ground boxes, motion blur, glass, metal, fog, earth texture

## Implementation Notes

- Release builds use LTO and single codegen unit for maximum performance
- BVH rebuilds automatically when scenes change (e.g., physics simulation)
- Adaptive sampling uses relative luminance variance for consistent quality
