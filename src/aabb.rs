use crate::vec3::Point3;
use crate::ray::Ray;

#[derive(Clone, Copy, Default)]
pub struct Aabb {
    pub min: Point3,
    pub max: Point3,
}

impl Aabb {
    pub fn new(min: Point3, max: Point3) -> Self { Self { min, max } }

    pub fn surrounding(a: &Self, b: &Self) -> Self {
        Self {
            min: Point3::new(a.min.x.min(b.min.x), a.min.y.min(b.min.y), a.min.z.min(b.min.z)),
            max: Point3::new(a.max.x.max(b.max.x), a.max.y.max(b.max.y), a.max.z.max(b.max.z)),
        }
    }

    // Expand any axis-aligned slab thinner than DELTA (needed for flat quads).
    pub fn pad(self) -> Self {
        const D: f32 = 0.0001;
        let px = if self.max.x - self.min.x < D { D } else { 0.0 };
        let py = if self.max.y - self.min.y < D { D } else { 0.0 };
        let pz = if self.max.z - self.min.z < D { D } else { 0.0 };
        Self {
            min: Point3::new(self.min.x - px, self.min.y - py, self.min.z - pz),
            max: Point3::new(self.max.x + px, self.max.y + py, self.max.z + pz),
        }
    }

    pub fn hit(&self, r: &Ray, mut t_min: f32, mut t_max: f32) -> bool {
        for axis in 0..3 {
            let inv_d = 1.0 / r.direction[axis];
            let orig  = r.origin[axis];
            let mut t0 = (self.min[axis] - orig) * inv_d;
            let mut t1 = (self.max[axis] - orig) * inv_d;
            if inv_d < 0.0 { std::mem::swap(&mut t0, &mut t1); }
            if t0 > t_min { t_min = t0; }
            if t1 < t_max { t_max = t1; }
            if t_max <= t_min { return false; }
        }
        true
    }
}
