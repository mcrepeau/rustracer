use crate::vec3::Color;
use image::{ImageBuffer, Rgb};

// ACES RRT+ODT fitted curve (Narkowicz 2015). Input/output in ACES AP1 linear.
#[inline]
fn aces_curve(x: f32) -> f32 {
    let x = x.max(0.0);
    (x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14)
}

// Full ACES pipeline with proper input/output transforms.
// Input:  scene-linear sRGB (D65)
// Output: display-linear sRGB, ready for sRGB gamma encode.
//
// Input matrix: sRGB → ACES AP1 (RRT_SAT * XYZ_2_AP1 * D65_2_D60 * sRGB_2_XYZ)
// Output matrix: ACES AP1 → sRGB (sRGB_2_XYZ^-1 * D60_2_D65 * AP1_2_XYZ * ODT_SAT)
// Matrices from Stephen Hill / Baking Lab, matches Unreal Engine's ACES path.
#[inline]
fn aces_tonemap(c: Color) -> Color {
    // Input transform: scene-linear sRGB → ACES AP1
    let r = c.x * 0.59719 + c.y * 0.35458 + c.z * 0.04823;
    let g = c.x * 0.07600 + c.y * 0.90834 + c.z * 0.01566;
    let b = c.x * 0.02840 + c.y * 0.13383 + c.z * 0.83777;

    // RRT+ODT fitted curve in AP1 space
    let r = aces_curve(r);
    let g = aces_curve(g);
    let b = aces_curve(b);

    // Output transform: ACES AP1 → display-linear sRGB
    Color::new(
         r *  1.60475 + g * -0.53108 + b * -0.07367,
         r * -0.10208 + g *  1.10813 + b * -0.00605,
         r * -0.00327 + g * -0.07276 + b *  1.07602,
    )
}

#[inline]
fn srgb_encode(x: f32) -> f32 {
    if x <= 0.0031308 { 12.92 * x } else { 1.055 * x.powf(1.0 / 2.4) - 0.055 }
}

#[inline]
pub fn tone_map(c: Color, scale: f32) -> [u8; 3] {
    let mapped = aces_tonemap(c * scale);
    let enc    = |x: f32| (srgb_encode(x.max(0.0)).clamp(0.0, 0.999) * 256.0) as u8;
    [enc(mapped.x), enc(mapped.y), enc(mapped.z)]
}

pub fn to_rgb_u32(c: Color, scale: f32) -> u32 {
    let [r, g, b] = tone_map(c, scale);
    (r as u32) << 16 | (g as u32) << 8 | (b as u32)
}

pub fn save_png(accumulator: &[Color], samples: u32, scene_name: &str, width: u32, height: u32, exposure: f32) {
    if samples == 0 { return; }
    let slug = scene_name.to_lowercase().replace(' ', "_");
    let path = format!("render_{}_{:04}spp.png", slug, samples);
    let scale = exposure / samples as f32;
    let img = ImageBuffer::from_fn(width, height, |x, y| {
        let i = (y * width + x) as usize;
        let [r, g, b] = tone_map(accumulator[i], scale);
        Rgb([r, g, b])
    });
    match img.save(&path) {
        Ok(_)  => println!("Saved {path}"),
        Err(e) => eprintln!("Save failed: {e}"),
    }
}
