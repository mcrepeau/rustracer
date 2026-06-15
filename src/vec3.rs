use std::ops::*;
use rand::Rng;

#[derive(Clone, Copy, Debug, Default)]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

pub type Color = Vec3;
pub type Point3 = Vec3;

impl Vec3 {
    #[inline] pub fn new(x: f32, y: f32, z: f32) -> Self { Self { x, y, z } }

    #[inline] pub fn length_squared(self) -> f32 { self.x*self.x + self.y*self.y + self.z*self.z }
    #[inline] pub fn length(self) -> f32 { self.length_squared().sqrt() }

    #[inline] pub fn dot(self, rhs: Self) -> f32 { self.x*rhs.x + self.y*rhs.y + self.z*rhs.z }

    #[inline]
    pub fn cross(self, rhs: Self) -> Self {
        Self::new(
            self.y*rhs.z - self.z*rhs.y,
            self.z*rhs.x - self.x*rhs.z,
            self.x*rhs.y - self.y*rhs.x,
        )
    }

    #[inline] pub fn unit(self) -> Self { self / self.length() }

    /// Build a tangent-space ONB for a surface normal (must already be unit).
    /// Returns `(u_axis, v_axis)`, both perpendicular to `self` and each other.
    pub fn onb(self) -> (Self, Self) {
        let up = if self.y.abs() < 0.9 { Self::new(0.0, 1.0, 0.0) } else { Self::new(1.0, 0.0, 0.0) };
        let u  = self.cross(up).unit();
        let v  = self.cross(u);
        (u, v)
    }

    pub fn near_zero(self) -> bool {
        const S: f32 = 1e-6;
        self.x.abs() < S && self.y.abs() < S && self.z.abs() < S
    }

    pub fn reflect(self, n: Self) -> Self { self - 2.0 * self.dot(n) * n }

    pub fn refract(self, n: Self, etai_over_etat: f32) -> Self {
        let cos_theta = (-self).dot(n).min(1.0);
        let perp = etai_over_etat * (self + cos_theta * n);
        let parallel = -(1.0 - perp.length_squared()).abs().sqrt() * n;
        perp + parallel
    }

    pub fn random(rng: &mut (impl Rng + ?Sized)) -> Self {
        Self::new(rng.gen(), rng.gen(), rng.gen())
    }

    pub fn random_range(min: f32, max: f32, rng: &mut (impl Rng + ?Sized)) -> Self {
        Self::new(rng.gen_range(min..max), rng.gen_range(min..max), rng.gen_range(min..max))
    }

    pub fn random_in_unit_sphere(rng: &mut (impl Rng + ?Sized)) -> Self {
        loop {
            let p = Self::random_range(-1.0, 1.0, rng);
            if p.length_squared() < 1.0 { return p; }
        }
    }

    pub fn random_unit_vector(rng: &mut (impl Rng + ?Sized)) -> Self { Self::random_in_unit_sphere(rng).unit() }

    pub fn random_in_unit_disk(rng: &mut (impl Rng + ?Sized)) -> Self {
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

impl Mul<f32> for Vec3 { type Output = Self; fn mul(self, t: f32) -> Self { Self::new(self.x*t, self.y*t, self.z*t) } }
impl Mul<Vec3> for f32 { type Output = Vec3; fn mul(self, v: Vec3) -> Vec3 { v * self } }
impl Div<f32>  for Vec3 { type Output = Self; fn div(self, t: f32) -> Self { self * (1.0/t) } }

impl AddAssign for Vec3 {
    fn add_assign(&mut self, rhs: Self) { self.x += rhs.x; self.y += rhs.y; self.z += rhs.z; }
}

impl SubAssign for Vec3 {
    fn sub_assign(&mut self, rhs: Self) { self.x -= rhs.x; self.y -= rhs.y; self.z -= rhs.z; }
}

impl MulAssign for Vec3 {
    fn mul_assign(&mut self, rhs: Self) { self.x *= rhs.x; self.y *= rhs.y; self.z *= rhs.z; }
}
impl MulAssign<f32> for Vec3 {
    fn mul_assign(&mut self, t: f32) { self.x *= t; self.y *= t; self.z *= t; }
}
impl DivAssign<f32> for Vec3 {
    fn div_assign(&mut self, t: f32) { let inv = 1.0 / t; self.x *= inv; self.y *= inv; self.z *= inv; }
}

impl std::ops::Index<usize> for Vec3 {
    type Output = f32;
    #[inline]
    fn index(&self, i: usize) -> &f32 {
        match i {
            0 => &self.x,
            1 => &self.y,
            2 => &self.z,
            _ => panic!("Vec3 index out of bounds: {i}"),
        }
    }
}
