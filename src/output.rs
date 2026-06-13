use crate::vec3::Color;
use image::{ImageBuffer, Rgb};

fn aces(x: f32) -> f32 {
    let x = x.max(0.0);
    (x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14)
}

#[inline]
pub fn tone_map(c: Color, scale: f32) -> [u8; 3] {
    let f = |x: f32| (aces(x * scale).sqrt().clamp(0.0, 0.999) * 256.0) as u8;
    [f(c.x), f(c.y), f(c.z)]
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
