use std::sync::Arc;

use crate::aabb::Aabb;
use crate::hittable::{HitRecord, Hittable, Material};
use crate::ray::Ray;
use crate::vec3::{Point3, Vec3};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hittable::ScatterRecord;

    struct DummyMat;
    impl Material for DummyMat {
        fn scatter(&self, _: &Ray, _: &HitRecord<'_>, _: &mut dyn rand::RngCore) -> Option<ScatterRecord> { None }
    }

    // Disk at origin, facing +z, radius 1.
    fn unit_disk() -> Disk {
        Disk::new(
            Point3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            1.0,
            Arc::new(DummyMat),
        )
    }

    fn shoot(d: &Disk, ox: f32, oy: f32, oz: f32, dx: f32, dy: f32, dz: f32) -> Option<HitRecord<'_>> {
        let r = Ray::new(Point3::new(ox, oy, oz), Vec3::new(dx, dy, dz));
        d.hit(&r, 0.001, f32::INFINITY)
    }

    #[test]
    fn disk_center_hit_t_value() {
        let d = unit_disk();
        // Ray aimed at center from z=2 — should hit at t=2.
        let rec = shoot(&d, 0.0, 0.0, 2.0, 0.0, 0.0, -1.0).expect("should hit");
        assert!((rec.t - 2.0).abs() < 1e-5, "t = {}", rec.t);
    }

    #[test]
    fn disk_center_uv_is_midpoint() {
        let d = unit_disk();
        let rec = shoot(&d, 0.0, 0.0, 2.0, 0.0, 0.0, -1.0).expect("should hit");
        assert!((rec.u - 0.5).abs() < 1e-5, "u = {}", rec.u);
        assert!((rec.v - 0.5).abs() < 1e-5, "v = {}", rec.v);
    }

    #[test]
    fn disk_hit_just_inside_edge() {
        let d = unit_disk();
        assert!(shoot(&d, 0.99, 0.0, 1.0, 0.0, 0.0, -1.0).is_some());
    }

    #[test]
    fn disk_miss_just_outside_radius() {
        let d = unit_disk();
        assert!(shoot(&d, 1.01, 0.0, 1.0, 0.0, 0.0, -1.0).is_none());
    }

    #[test]
    fn disk_miss_parallel_ray() {
        let d = unit_disk();
        // Ray in the XY plane — parallel to the disk, denom ≈ 0.
        assert!(shoot(&d, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0).is_none());
    }

    #[test]
    fn disk_front_face_orientation() {
        // Ray from +z hits the front of the disk.
        let d = unit_disk();
        let rec = shoot(&d, 0.0, 0.0, 2.0, 0.0, 0.0, -1.0).expect("should hit");
        assert!(rec.front_face, "ray from normal side should be front face");

        // Ray from −z hits the back.
        let d2 = unit_disk();
        let rec2 = shoot(&d2, 0.0, 0.0, -2.0, 0.0, 0.0, 1.0).expect("should hit");
        assert!(!rec2.front_face, "ray from back side should not be front face");
    }
}

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
