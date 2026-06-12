use std::sync::Arc;
use image::RgbImage;
use crate::perlin::Perlin;
use crate::vec3::{Color, Point3};

#[derive(Clone)]
pub enum Texture {
    Solid(Color),
    Checker { scale: f32, even: Color, odd: Color },
    #[allow(dead_code)]
    Image(Arc<RgbImage>),
    /// Marble-like Perlin noise: 0.5*(1 + sin(scale·p.z + 10·turb(p)))
    Noise { perlin: Arc<Perlin>, scale: f32 },
    /// Turbulence-based two-color blend for stellar surface granulation.
    SolarNoise { perlin: Arc<Perlin>, scale: f32, hot: Color, cool: Color },
    /// Step-function radial bands keyed on the UV `u` coordinate [0, 1].
    /// Each entry is `(u_threshold, color)`; the first entry whose threshold
    /// >= u wins.  Used for planetary ring systems (u = radial position).
    RingBands(Arc<Vec<(f32, Color)>>),
}

impl Texture {
    #[allow(dead_code)]
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
                let lin = |c: u8| (c as f32 / 255.0).powf(2.2);
                Color::new(lin(px[0]), lin(px[1]), lin(px[2]))
            }
            Texture::Noise { perlin, scale } => {
                Color::new(0.5, 0.5, 0.5)
                    * (1.0 + (scale * p.z + 10.0 * perlin.turb(p, 7)).sin())
            }
            Texture::SolarNoise { perlin, scale, hot, cool } => {
                let t = 0.5 * (1.0 + (10.0 * perlin.turb(
                    Point3::new(p.x * scale, p.y * scale, p.z * scale), 7,
                )).sin());
                *hot * t + *cool * (1.0 - t)
            }
            Texture::RingBands(bands) => {
                for &(threshold, color) in bands.as_slice() {
                    if u <= threshold { return color; }
                }
                bands.last().map(|&(_, c)| c).unwrap_or_default()
            }
        }
    }
}

impl From<Color> for Texture {
    fn from(c: Color) -> Self { Texture::Solid(c) }
}
