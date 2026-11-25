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
    let luma = Color::new(r, g, b).luminance();
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

/// 256-entry sRGB→linear LUT (exact IEC 61966-2-1 curve), built once — the powf
/// version cost a transcendental per texel per fetch in the render hot path.
static SRGB_DECODE_LUT: std::sync::LazyLock<[f32; 256]> = std::sync::LazyLock::new(|| {
    std::array::from_fn(|c| {
        let s = c as f32 / 255.0;
        if s <= 0.04045 { s / 12.92 } else { ((s + 0.055) / 1.055).powf(2.4) }
    })
});

#[inline]
pub fn srgb_decode(c: u8) -> f32 { SRGB_DECODE_LUT[c as usize] }

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

/// Post optics applied in linear light before tonemapping:
/// - vignette: multiply by `1 - vignette * r^2` with r normalised so the image
///   corners sit at r = 1 (vignette 0.3 -> corners at 70% brightness).
/// - distortion: faint barrel distortion; each output pixel samples the source
///   at `c + (p - c) * (1 - k * r^2)` (bilinear), k > 0 bows straight lines
///   outward like a real wide/macro lens.
/// Both 0 returns an untouched copy.
pub fn apply_optics(img: &[Color], w: u32, h: u32, vignette: f32, distortion: f32) -> Vec<Color> {
    if vignette == 0.0 && distortion == 0.0 { return img.to_vec(); }
    let cx = (w as f32 - 1.0) * 0.5;
    let cy = (h as f32 - 1.0) * 0.5;
    let rcorner = (cx * cx + cy * cy).sqrt().max(1e-6);
    let mut out = vec![Color::default(); img.len()];
    for y in 0..h as usize {
        for x in 0..w as usize {
            let dx = (x as f32 - cx) / rcorner;
            let dy = (y as f32 - cy) / rcorner;
            let r2 = dx * dx + dy * dy;
            let s = 1.0 - distortion * r2;
            let sx = (cx + dx * rcorner * s).clamp(0.0, w as f32 - 1.0);
            let sy = (cy + dy * rcorner * s).clamp(0.0, h as f32 - 1.0);
            let x0 = sx as usize;
            let y0 = sy as usize;
            let x1 = (x0 + 1).min(w as usize - 1);
            let y1 = (y0 + 1).min(h as usize - 1);
            let fx = sx - x0 as f32;
            let fy = sy - y0 as f32;
            let c00 = img[y0 * w as usize + x0];
            let c10 = img[y0 * w as usize + x1];
            let c01 = img[y1 * w as usize + x0];
            let c11 = img[y1 * w as usize + x1];
            let c = (c00 * (1.0 - fx) + c10 * fx) * (1.0 - fy)
                  + (c01 * (1.0 - fx) + c11 * fx) * fy;
            let vig = (1.0 - vignette * r2).max(0.0);
            out[y * w as usize + x] = c * vig;
        }
    }
    out
}

#[cfg(test)]
mod optics_tests {
    use super::*;

    #[test]
    fn optics_identity_when_off() {
        let img: Vec<Color> = (0..16).map(|i| Color::new(i as f32, 0.0, 0.0)).collect();
        let out = apply_optics(&img, 4, 4, 0.0, 0.0);
        assert_eq!(out.len(), img.len());
        for (a, b) in out.iter().zip(img.iter()) {
            assert!((a.x - b.x).abs() < 1e-7 && (a.y - b.y).abs() < 1e-7 && (a.z - b.z).abs() < 1e-7);
        }
    }

    #[test]
    fn vignette_darkens_corners_not_center() {
        let img = vec![Color::new(1.0, 1.0, 1.0); 16];
        let out = apply_optics(&img, 4, 4, 0.4, 0.0);
        let center = out[1 * 4 + 1];
        assert!(center.x > 0.9 && center.x < 1.0, "near-center pixel only mildly darkened, got {}", center.x);
        let corner = out[0];
        assert!((corner.x - 0.6).abs() < 1e-6, "corner r^2 = 1 -> 1 - 0.4 = 0.6, got {}", corner.x);
    }

    #[test]
    fn barrel_distortion_pulls_corners_inward() {
        // horizontal gradient: value = x
        let w = 9u32;   // odd size so pixel (4,4) is the exact centre
        let h = 9u32;
        let img: Vec<Color> = (0..(w * h) as usize)
            .map(|i| Color::new((i % w as usize) as f32, 0.0, 0.0))
            .collect();
        let out = apply_optics(&img, w, h, 0.0, 0.2);
        let corner_in = img[0].x;                       // 0.0
        let corner_out = out[0].x;
        // source moves inward -> samples x > 0 -> brighter than 0
        assert!(corner_out > corner_in, "barrel corner should sample inward: {corner_out} vs {corner_in}");
        // center unchanged
        let c_in = img[(4 * w + 4) as usize].x;
        let c_out = out[(4 * w + 4) as usize].x;
        assert!((c_out - c_in).abs() < 1e-6);
    }
}
