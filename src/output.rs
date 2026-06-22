use crate::vec3::Color;
use image::{ImageBuffer, Rgb};

#[derive(Copy, Clone, PartialEq, Eq)]
pub enum ToneMapper { Aces, AgX }

// ── ACES RRT+ODT ─────────────────────────────────────────────────────────────
// Narkowicz 2015 fitted curve, input/output in ACES AP1 linear.
// Input matrix: sRGB → ACES AP1 (Stephen Hill / Baking Lab, matches Unreal).
// Output matrix: ACES AP1 → display-linear sRGB.
#[inline]
fn aces_curve(x: f32) -> f32 {
    let x = x.max(0.0);
    (x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14)
}

#[inline]
fn aces_tonemap(c: Color) -> Color {
    let r = c.x * 0.59719 + c.y * 0.35458 + c.z * 0.04823;
    let g = c.x * 0.07600 + c.y * 0.90834 + c.z * 0.01566;
    let b = c.x * 0.02840 + c.y * 0.13383 + c.z * 0.83777;
    let r = aces_curve(r);
    let g = aces_curve(g);
    let b = aces_curve(b);
    Color::new(
         r *  1.60475 + g * -0.53108 + b * -0.07367,
         r * -0.10208 + g *  1.10813 + b * -0.00605,
         r * -0.00327 + g * -0.07276 + b *  1.07602,
    )
}

// ── AgX (Troy Sobotka / Blender 4.0) ─────────────────────────────────────────
// Input matrix:  linear sRGB → AgX working space
// Log encoding:  log2(x) mapped from [-10, +6.5] EV → [0, 1]
// Curve:         per-channel S-curve in log space (bwrensch degree-6 poly)
// Output matrix: AgX working space → display-linear sRGB
#[inline]
fn agx_curve(x: f32) -> f32 {
    let x2 = x * x;
    let x4 = x2 * x2;
    15.5 * x4 * x2 - 40.14 * x4 * x + 31.96 * x4 - 6.868 * x2 * x + 0.4298 * x2 + 0.1191 * x - 0.00232
}

#[inline]
#[allow(clippy::excessive_precision)]
fn agx_tonemap(c: Color) -> Color {
    // Input transform: linear sRGB → AgX working space
    let r = c.x * 0.842479062253094  + c.y * 0.0784335999999992 + c.z * 0.0792237451477643;
    let g = c.x * 0.0423282422610123 + c.y * 0.878468636469772  + c.z * 0.0791661274605434;
    let b = c.x * 0.0423756549057051 + c.y * 0.0784336           + c.z * 0.879142973793104;

    // Log2 encoding: Blender's [-12.47393, +4.026069] EV range → [0, 1].
    // The curve polynomial was fitted for this specific range; using [-10, +6.5]
    // would place standard white at the 60% curve position instead of 75%,
    // causing the washed-out appearance.
    const MIN_EV: f32 = -12.47393;
    const RANGE:  f32 =  16.5;   // 4.026069 − (−12.47393)
    let r = ((r.max(1e-10_f32).log2() - MIN_EV) / RANGE).clamp(0.0, 1.0);
    let g = ((g.max(1e-10_f32).log2() - MIN_EV) / RANGE).clamp(0.0, 1.0);
    let b = ((b.max(1e-10_f32).log2() - MIN_EV) / RANGE).clamp(0.0, 1.0);

    let r = agx_curve(r);
    let g = agx_curve(g);
    let b = agx_curve(b);

    // Punchy look (Blender AgX "Punchy"): ASC CDL power 1.35 + 1.4× saturation.
    // Applied in curve-output space, before the outset matrix.
    let luma = r * 0.2126 + g * 0.7152 + b * 0.0722;
    let r = luma + 1.4 * (r.powf(1.35) - luma);
    let g = luma + 1.4 * (g.powf(1.35) - luma);
    let b = luma + 1.4 * (b.powf(1.35) - luma);

    // Output transform: AgX → display-linear sRGB
    Color::new(
         r *  1.19687900512017   + g * -0.0980208811401368 + b * -0.0990297440797205,
         r * -0.0528968517574562 + g *  1.15190312990417   + b * -0.0989611768448433,
         r * -0.0529716355144234 + g * -0.0980434501171241 + b *  1.15107567775604,
    )
}

// ── Shared ────────────────────────────────────────────────────────────────────

#[inline]
fn srgb_encode(x: f32) -> f32 {
    if x <= 0.0031308 { 12.92 * x } else { 1.055 * x.powf(1.0 / 2.4) - 0.055 }
}

#[inline]
pub fn tone_map(c: Color, scale: f32, tm: ToneMapper) -> [u8; 3] {
    let mapped = match tm {
        ToneMapper::Aces => aces_tonemap(c * scale),
        ToneMapper::AgX  => agx_tonemap(c * scale),
    };
    let enc = |x: f32| (srgb_encode(x.max(0.0)).clamp(0.0, 0.999) * 256.0) as u8;
    [enc(mapped.x), enc(mapped.y), enc(mapped.z)]
}

pub fn to_rgb_u32(c: Color, scale: f32, tm: ToneMapper) -> u32 {
    let [r, g, b] = tone_map(c, scale, tm);
    (r as u32) << 16 | (g as u32) << 8 | (b as u32)
}

/// `pixel_samples`: per-pixel sample counts used when adaptive sampling is active.
/// Pass `None` to use the uniform `samples` count for every pixel.
/// `spp_label`: overrides the spp value in the auto-generated filename.
/// `output_path`: explicit output path; overrides the auto-generated filename entirely.
#[allow(clippy::too_many_arguments)]
pub fn save_png(
    accumulator:   &[Color],
    samples:       u32,
    pixel_samples: Option<&[u32]>,
    scene_name:    &str,
    width:         u32,
    height:        u32,
    exposure:      f32,
    tm:            ToneMapper,
    spp_label:     Option<u32>,
    output_path:   Option<&str>,
) {
    if samples == 0 { return; }
    let path = if let Some(p) = output_path {
        p.to_string()
    } else {
        let slug  = scene_name.to_lowercase().replace(' ', "_");
        let label = spp_label.unwrap_or(samples);
        format!("render_{}_{:04}spp.png", slug, label)
    };
    let img = ImageBuffer::from_fn(width, height, |x, y| {
        let i = (y * width + x) as usize;
        let n = pixel_samples.map_or(samples, |ps| ps[i]).max(1);
        let [r, g, b] = tone_map(accumulator[i], exposure / n as f32, tm);
        Rgb([r, g, b])
    });
    match img.save(&path) {
        Ok(_)  => println!("Saved {path}"),
        Err(e) => eprintln!("Save failed: {e}"),
    }
}
