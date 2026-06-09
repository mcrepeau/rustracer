use crate::vec3::{Color, Point3};

#[derive(Clone, Copy)]
pub enum Texture {
    Solid(Color),
    Checker { scale: f64, even: Color, odd: Color },
}

impl Texture {
    pub fn value(&self, _u: f64, _v: f64, p: Point3) -> Color {
        match self {
            Texture::Solid(c) => *c,
            Texture::Checker { scale, even, odd } => {
                let x = (scale * p.x).floor() as i32;
                let y = (scale * p.y).floor() as i32;
                let z = (scale * p.z).floor() as i32;
                if (x + y + z) % 2 == 0 { *even } else { *odd }
            }
        }
    }
}

impl From<Color> for Texture {
    fn from(c: Color) -> Self { Texture::Solid(c) }
}
