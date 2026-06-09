use std::ops::*;
use rand::Rng;

#[derive(Clone, Copy, Debug, Default)]
pub struct Vec3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

pub type Color = Vec3;
pub type Point3 = Vec3;

impl Vec3 {
    #[inline] pub fn new(x: f64, y: f64, z: f64) -> Self { Self { x, y, z } }

    #[inline] pub fn length_squared(self) -> f64 { self.x*self.x + self.y*self.y + self.z*self.z }
    #[inline] pub fn length(self) -> f64 { self.length_squared().sqrt() }

    #[inline] pub fn dot(self, rhs: Self) -> f64 { self.x*rhs.x + self.y*rhs.y + self.z*rhs.z }

    #[inline]
    pub fn cross(self, rhs: Self) -> Self {
        Self::new(
            self.y*rhs.z - self.z*rhs.y,
            self.z*rhs.x - self.x*rhs.z,
            self.x*rhs.y - self.y*rhs.x,
        )
    }

    #[inline] pub fn unit(self) -> Self { self / self.length() }

    pub fn near_zero(self) -> bool {
        const S: f64 = 1e-8;
        self.x.abs() < S && self.y.abs() < S && self.z.abs() < S
    }

    pub fn reflect(self, n: Self) -> Self { self - 2.0 * self.dot(n) * n }

    pub fn refract(self, n: Self, etai_over_etat: f64) -> Self {
        let cos_theta = (-self).dot(n).min(1.0);
        let perp = etai_over_etat * (self + cos_theta * n);
        let parallel = -(1.0 - perp.length_squared()).abs().sqrt() * n;
        perp + parallel
    }

    pub fn random() -> Self {
        let mut rng = rand::thread_rng();
        Self::new(rng.gen(), rng.gen(), rng.gen())
    }

    pub fn random_range(min: f64, max: f64) -> Self {
        let mut rng = rand::thread_rng();
        Self::new(rng.gen_range(min..max), rng.gen_range(min..max), rng.gen_range(min..max))
    }

    pub fn random_in_unit_sphere() -> Self {
        loop {
            let p = Self::random_range(-1.0, 1.0);
            if p.length_squared() < 1.0 { return p; }
        }
    }

    pub fn random_unit_vector() -> Self { Self::random_in_unit_sphere().unit() }

    pub fn random_in_unit_disk() -> Self {
        let mut rng = rand::thread_rng();
        loop {
            let p = Self::new(rng.gen_range(-1.0..1.0), rng.gen_range(-1.0..1.0), 0.0);
            if p.length_squared() < 1.0 { return p; }
        }
    }
}

impl Add    for Vec3 { type Output = Self; fn add(self, r: Self) -> Self { Self::new(self.x+r.x, self.y+r.y, self.z+r.z) } }
impl Sub    for Vec3 { type Output = Self; fn sub(self, r: Self) -> Self { Self::new(self.x-r.x, self.y-r.y, self.z-r.z) } }
impl Neg    for Vec3 { type Output = Self; fn neg(self)          -> Self { Self::new(-self.x, -self.y, -self.z) } }
impl Mul    for Vec3 { type Output = Self; fn mul(self, r: Self) -> Self { Self::new(self.x*r.x, self.y*r.y, self.z*r.z) } }

impl Mul<f64> for Vec3 { type Output = Self; fn mul(self, t: f64) -> Self { Self::new(self.x*t, self.y*t, self.z*t) } }
impl Mul<Vec3> for f64 { type Output = Vec3; fn mul(self, v: Vec3) -> Vec3 { v * self } }
impl Div<f64>  for Vec3 { type Output = Self; fn div(self, t: f64) -> Self { self * (1.0/t) } }

impl AddAssign for Vec3 {
    fn add_assign(&mut self, rhs: Self) { self.x += rhs.x; self.y += rhs.y; self.z += rhs.z; }
}

impl SubAssign for Vec3 {
    fn sub_assign(&mut self, rhs: Self) { self.x -= rhs.x; self.y -= rhs.y; self.z -= rhs.z; }
}

impl std::ops::Index<usize> for Vec3 {
    type Output = f64;
    fn index(&self, i: usize) -> &f64 {
        match i { 0 => &self.x, 1 => &self.y, 2 => &self.z, _ => panic!("Vec3 index out of range") }
    }
}
