use std::sync::Arc;

use crate::aabb::Aabb;
use crate::hittable::{HitRecord, Hittable, Material};
use crate::ray::Ray;
use crate::vec3::{Point3, Vec3};

/// Flat disk: `center`, outward `normal` (auto-normalised), `radius`.
pub struct Disk {
    pub center: Point3,
    pub normal: Vec3,
    pub radius: f32,
    pub mat:    Arc<dyn Material>,
    u_axis:     Vec3,
    v_axis:     Vec3,
}

impl Disk {
    pub fn new(center: Point3, normal: Vec3, radius: f32, mat: Arc<dyn Material>) -> Self {
        let n = normal.unit();
        let (u_axis, v_axis) = n.onb();
        Self { center, normal: n, radius, mat, u_axis, v_axis }
    }
}

impl Hittable for Disk {
    fn hit(&self, r: &Ray, t_min: f32, t_max: f32) -> Option<HitRecord<'_>> {
        let denom = self.normal.dot(r.direction);
        if denom.abs() < 1e-8 { return None; }
        let t = self.normal.dot(self.center - r.origin) / denom;
        if t < t_min || t > t_max { return None; }
        let p   = r.at(t);
        let off = p - self.center;
        if off.length_squared() > self.radius * self.radius { return None; }
        let mut rec = HitRecord::new(p, t, &*self.mat, r, self.normal);
        // UV: [0,1]² mapped across the disk diameter
        rec.u = (off.dot(self.u_axis) / self.radius + 1.0) * 0.5;
        rec.v = (off.dot(self.v_axis) / self.radius + 1.0) * 0.5;
        Some(rec)
    }

    fn bounding_box(&self) -> Option<Aabb> {
        let n = self.normal;
        // Extent along axis i = R * sqrt(1 − N_i²)
        let ext = Vec3::new(
            self.radius * (1.0 - n.x * n.x).max(0.0).sqrt(),
            self.radius * (1.0 - n.y * n.y).max(0.0).sqrt(),
            self.radius * (1.0 - n.z * n.z).max(0.0).sqrt(),
        );
        Some(Aabb::new(self.center - ext, self.center + ext).pad())
    }
}
