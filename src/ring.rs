use std::f32::consts::PI;
use std::sync::Arc;
use crate::aabb::Aabb;
use crate::hittable::{HitRecord, Hittable, Material};
use crate::ray::Ray;
use crate::vec3::{Point3, Vec3};

/// Flat annular disc (ring) lying in a plane defined by `normal`.
/// UV: u = normalised radial position [0,1], v = angle around ring [0,1].
pub struct Ring {
    center:  Point3,
    inner_r: f32,
    outer_r: f32,
    normal:  Vec3,
    u_axis:  Vec3,
    v_axis:  Vec3,
    mat:     Arc<dyn Material>,
}

impl Ring {
    pub fn new(
        center:  Point3,
        inner_r: f32,
        outer_r: f32,
        normal:  Vec3,
        mat:     Arc<dyn Material>,
    ) -> Self {
        let n = normal.unit();
        // Build an orthonormal frame in the ring plane.
        let up = if n.y.abs() < 0.9 { Vec3::new(0.0, 1.0, 0.0) } else { Vec3::new(1.0, 0.0, 0.0) };
        let u_axis = n.cross(up).unit();
        let v_axis = n.cross(u_axis);
        Self { center, inner_r, outer_r, normal: n, u_axis, v_axis, mat }
    }
}

impl Hittable for Ring {
    fn hit(&self, r: &Ray, t_min: f32, t_max: f32) -> Option<HitRecord<'_>> {
        let denom = r.direction.dot(self.normal);
        if denom.abs() < 1e-8 { return None; }

        let t = (self.center - r.origin).dot(self.normal) / denom;
        if t < t_min || t > t_max { return None; }

        let p    = r.at(t);
        let diff = p - self.center;
        let d2   = diff.length_squared();

        if d2 < self.inner_r * self.inner_r || d2 > self.outer_r * self.outer_r {
            return None;
        }

        let dist = d2.sqrt();
        let u    = (dist - self.inner_r) / (self.outer_r - self.inner_r);
        let angle = diff.dot(self.v_axis).atan2(diff.dot(self.u_axis));
        let v    = (angle / (2.0 * PI) + 0.5).rem_euclid(1.0);

        let mut rec = HitRecord::new(p, t, &*self.mat, r, self.normal);
        rec.u = u;
        rec.v = v;
        Some(rec)
    }

    fn bounding_box(&self) -> Option<Aabb> {
        let r   = self.outer_r;
        // Tight AABB: extent in each axis = r * sqrt(1 - (n·eᵢ)²),
        // plus a small epsilon along the normal for numerical robustness.
        let eps = 0.5;
        let ex  = r * (1.0 - self.normal.x * self.normal.x).sqrt() + eps * self.normal.x.abs();
        let ey  = r * (1.0 - self.normal.y * self.normal.y).sqrt() + eps * self.normal.y.abs();
        let ez  = r * (1.0 - self.normal.z * self.normal.z).sqrt() + eps * self.normal.z.abs();
        let half = Vec3::new(ex, ey, ez);
        Some(Aabb::new(self.center - half, self.center + half))
    }
}
