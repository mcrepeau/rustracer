use crate::vec3::{Point3, Vec3};

#[derive(Clone, Copy)]
pub struct Ray {
    pub origin:    Point3,
    pub direction: Vec3,
    pub inv_dir:   Vec3,
}

impl Ray {
    pub fn new(origin: Point3, direction: Vec3) -> Self {
        let inv_dir = Vec3::new(1.0 / direction.x, 1.0 / direction.y, 1.0 / direction.z);
        Self { origin, direction, inv_dir }
    }
    pub fn at(self, t: f32) -> Point3 { self.origin + t * self.direction }
}
