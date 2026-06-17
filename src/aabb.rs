use crate::vec3::Point3;

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

}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vec3::Point3;

    #[test]
    fn surrounding_is_tight_union() {
        let a = Aabb::new(Point3::new(-1.0, -2.0, -3.0), Point3::new(0.0, 0.0, 0.0));
        let b = Aabb::new(Point3::new( 0.5,  0.5,  0.5), Point3::new(2.0, 3.0, 4.0));
        let s = Aabb::surrounding(&a, &b);
        assert_eq!(s.min.x, -1.0);
        assert_eq!(s.min.y, -2.0);
        assert_eq!(s.min.z, -3.0);
        assert_eq!(s.max.x,  2.0);
        assert_eq!(s.max.y,  3.0);
        assert_eq!(s.max.z,  4.0);
    }

    #[test]
    fn pad_expands_zero_thickness_axis() {
        // A quad flat in Z: max.z == min.z → pad must give positive thickness.
        let flat = Aabb::new(Point3::new(0.0, 0.0, 1.0), Point3::new(1.0, 1.0, 1.0));
        let p    = flat.pad();
        assert!(p.max.z - p.min.z > 0.0, "padded Z thickness must be > 0");
        assert!(p.max.x - p.min.x > 0.0, "padded X thickness must be > 0");
        assert!(p.max.y - p.min.y > 0.0, "padded Y thickness must be > 0");
    }

    #[test]
    fn pad_preserves_already_thick_dimensions() {
        let thick = Aabb::new(Point3::new(0.0, 0.0, 0.0), Point3::new(1.0, 1.0, 1.0));
        let p     = thick.pad();
        assert_eq!(p.min.x, thick.min.x);
        assert_eq!(p.max.x, thick.max.x);
        assert_eq!(p.min.y, thick.min.y);
        assert_eq!(p.max.y, thick.max.y);
        assert_eq!(p.min.z, thick.min.z);
        assert_eq!(p.max.z, thick.max.z);
    }
}
