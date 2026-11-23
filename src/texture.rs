use std::sync::Arc;
use image::RgbImage;
use crate::output::srgb_decode;
use crate::perlin::Perlin;
use crate::vec3::{Color, Point3};

#[derive(Clone)]
pub enum Texture {
    Solid(Color),
    Checker { scale: f32, even: Color, odd: Color },
    Image(Arc<RgbImage>),
    /// Marble-like Perlin noise: 0.5*(1 + sin(scale·p.z + 10·turb(p)))
    #[allow(dead_code)]
    Noise { perlin: Arc<Perlin>, scale: f32 },
    /// Soft Perlin mottle between two colours (seamless studio-floor look):
    /// lerp(lo, hi, clamp(0.5 * turb(p * scale, 7), 0, 1))
    Mottle { perlin: Arc<Perlin>, scale: f32, lo: Color, hi: Color },
}


impl Texture {
    pub fn load(path: &str) -> Result<Self, image::ImageError> {
        Ok(Self::Image(Arc::new(image::open(path)?.into_rgb8())))
    }

    pub fn value(&self, u: f32, v: f32, p: Point3) -> Color {
        match self {
            Texture::Solid(c) => *c,
            Texture::Checker { scale, even, odd } => {
                let x = (scale * p.x).floor() as i32;
                let y = (scale * p.y).floor() as i32;
                let z = (scale * p.z).floor() as i32;
                if (x + y + z) % 2 == 0 { *even } else { *odd }
            }
            Texture::Image(img) => {
                let u = u.clamp(0.0, 1.0);
                let v = 1.0 - v.clamp(0.0, 1.0); // flip V: images are top-down, UV is bottom-up
                let x = ((u * img.width()  as f32) as u32).min(img.width()  - 1);
                let y = ((v * img.height() as f32) as u32).min(img.height() - 1);
                let px = img.get_pixel(x, y);
                // sRGB → linear
                let lin = srgb_decode;
                Color::new(lin(px[0]), lin(px[1]), lin(px[2]))
            }
            Texture::Noise { perlin, scale } => {
                Color::new(0.5, 0.5, 0.5)
                    * (1.0 + (scale * p.z + 10.0 * perlin.turb(p, 7)).sin())
            }
            Texture::Mottle { perlin, scale, lo, hi } => {
                let m = (0.5 * perlin.turb(Point3::new(p.x * scale, p.y * scale, p.z * scale), 7)).clamp(0.0, 1.0);
                *lo * (1.0 - m) + *hi * m
            }

        }
    }
}

impl From<Color> for Texture {
    fn from(c: Color) -> Self { Texture::Solid(c) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mottle_stays_within_lo_hi() {
        let lo = Color::new(0.40, 0.40, 0.42);
        let hi = Color::new(0.45, 0.45, 0.47);
        let tex = Texture::Mottle { perlin: Arc::new(Perlin::new(&mut { use rand::SeedableRng; rand::rngs::StdRng::seed_from_u64(0x5EED) })), scale: 3.0, lo, hi };
        for i in 0..2000 {
            let p = Point3::new(i as f32 * 0.371, 0.0, i as f32 * 0.113);
            let c = tex.value(0.0, 0.0, p);
            for k in 0..3 {
                let (a, b, v) = (lo[k], hi[k], c[k]);
                assert!(v >= a.min(b) - 1e-6 && v <= a.max(b) + 1e-6, "i={i} k={k} v={v}");
            }
        }
    }
}
