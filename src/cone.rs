use std::sync::Arc;
use std::f32::consts::TAU;

use crate::aabb::Aabb;
use crate::hittable::{HitRecord, Hittable, Material};
use crate::ray::Ray;
use crate::vec3::{Point3, Vec3};

/// Y-axis-aligned finite cone with a closed base disk.
/// `center` is the centre of the base disk; the apex is at `center.y + height`.
pub struct Cone {
    pub center: Point3,
    pub radius: f32,
    pub height: f32,
    pub mat:    Arc<dyn Material>,
}

impl Hittable for Cone {
    fn hit(&self, r: &Ray, t_min: f32, t_max: f32) -> Option<HitRecord<'_>> {
        let oc = r.origin - self.center;
        let h  = self.height;
        // k = (r/h)²: maps lateral distance to (h-y) scale
        let k  = (self.radius / h) * (self.radius / h);

        let dx = r.direction.x;
        let dy = r.direction.y;
        let dz = r.direction.z;
        // Height of the apex above the ray origin in local frame
        let hy = h - oc.y;

        let mut best_t = t_max;
        let mut best: Option<HitRecord<'_>> = None;

        // ── Lateral surface: x² + z² = k·(h−y)² ─────────────────────────────
        let a      = dx * dx + dz * dz - k * dy * dy;
        let half_b = oc.x * dx + oc.z * dz + k * hy * dy;
        let c      = oc.x * oc.x + oc.z * oc.z - k * hy * hy;

        if a.abs() > 1e-8 {
            let disc = half_b * half_b - a * c;
            if disc >= 0.0 {
                let sqrtd = disc.sqrt();
                for root in [(-half_b - sqrtd) / a, (-half_b + sqrtd) / a] {
                    if root < t_min || root >= best_t { continue; }
                    let p       = r.at(root);
                    let local_y = p.y - self.center.y;
                    if local_y < 0.0 || local_y > h { continue; }
                    let lx = p.x - self.center.x;
                    let lz = p.z - self.center.z;
                    // Gradient of F = x²+z² − k·(h−y)², gives outward normal
                    let outward = Vec3::new(lx, k * (h - local_y), lz).unit();
                    let mut rec = HitRecord::new(p, root, &*self.mat, r, outward);
                    let phi = lz.atan2(lx);
                    rec.u = (phi / TAU + 0.5).fract();
                    rec.v = local_y / h;
                    best_t = root;
                    best   = Some(rec);
                }
            }
        }

        // ── Base disk at y = 0 ────────────────────────────────────────────────
        if dy.abs() > 1e-8 {
            let t = -oc.y / dy;
            if t >= t_min && t < best_t {
                let p  = r.at(t);
                let lx = p.x - self.center.x;
                let lz = p.z - self.center.z;
                if lx * lx + lz * lz <= self.radius * self.radius {
                    let mut rec = HitRecord::new(p, t, &*self.mat, r, Vec3::new(0.0, -1.0, 0.0));
                    rec.u = (lx / self.radius + 1.0) * 0.5;
                    rec.v = (lz / self.radius + 1.0) * 0.5;
                    best   = Some(rec);
                }
            }
        }

        best
    }

    fn bounding_box(&self) -> Option<Aabb> {
        let r = Vec3::new(self.radius, 0.0, self.radius);
        Some(Aabb::new(
            self.center - r,
            self.center + r + Vec3::new(0.0, self.height, 0.0),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hittable::ScatterRecord;

    struct DummyMat;
    impl Material for DummyMat {
        fn scatter(&self, _: &Ray, _: &HitRecord<'_>, _: &mut dyn rand::RngCore) -> Option<ScatterRecord> { None }
    }

    // Cone: center=(0,0,0), radius=1, height=2.
    // k = (r/h)² = 0.25.  Apex at y=2.  Base disk at y=0, radius=1.
    // At height y=1 the cone radius = 0.5.
    fn unit_cone() -> Cone {
        Cone { center: Point3::new(0.0, 0.0, 0.0), radius: 1.0, height: 2.0, mat: Arc::new(DummyMat) }
    }

    fn shoot(c: &Cone, ox: f32, oy: f32, oz: f32, dx: f32, dy: f32, dz: f32) -> Option<HitRecord<'_>> {
        let r = Ray::new(Point3::new(ox, oy, oz), Vec3::new(dx, dy, dz));
        c.hit(&r, 0.001, f32::INFINITY)
    }

    #[test]
    fn cone_lateral_hit_t_value() {
        // Ray along +z at y=1, starting at z=−5.
        // a=1, half_b=−5, c=24.75, disc=0.25 → t=4.5.
        let c = unit_cone();
        let rec = shoot(&c, 0.0, 1.0, -5.0, 0.0, 0.0, 1.0).expect("should hit lateral surface");
        assert!((rec.t - 4.5).abs() < 1e-5, "lateral t = {}", rec.t);
    }

    #[test]
    fn cone_lateral_normal_has_positive_y() {
        // Gradient of x²+z²=k(h−y)² gives outward_y = k*(h−local_y) > 0 — normal tilts toward apex.
        let c = unit_cone();
        let rec = shoot(&c, 0.0, 1.0, -5.0, 0.0, 0.0, 1.0).expect("should hit");
        let outward = if rec.front_face { rec.normal } else { -rec.normal };
        assert!(outward.y > 0.0, "lateral outward normal y should be > 0, got {}", outward.y);
        assert!((rec.normal.length() - 1.0).abs() < 1e-5, "normal must be unit length");
    }

    #[test]
    fn cone_base_cap_is_returned_when_closer_than_lateral() {
        // Ray going +y from below: lateral also intersects but at t=6; disk at t=5 wins.
        let c = unit_cone();
        let rec = shoot(&c, 0.5, -5.0, 0.0, 0.0, 1.0, 0.0).expect("should hit base cap");
        assert!((rec.t - 5.0).abs() < 1e-5, "base cap t = {}", rec.t);
    }

    #[test]
    fn cone_base_cap_normal_points_downward() {
        // Outward normal for the base disk is (0,−1,0).
        let c = unit_cone();
        let rec = shoot(&c, 0.5, -5.0, 0.0, 0.0, 1.0, 0.0).expect("should hit");
        let outward = if rec.front_face { rec.normal } else { -rec.normal };
        assert!(outward.y < -0.99, "base cap outward normal should point −y, got {}", outward.y);
    }

    #[test]
    fn cone_miss_outside_radius() {
        // Ray aimed 2 units from axis at y=1: discriminant < 0 (cone radius at y=1 is only 0.5).
        let c = unit_cone();
        assert!(shoot(&c, 2.0, 1.0, -5.0, 0.0, 0.0, 1.0).is_none());
    }

    #[test]
    fn cone_miss_above_height() {
        // Lateral intersections exist geometrically but local_y=3 > height=2 → both skipped.
        let c = unit_cone();
        assert!(shoot(&c, 0.0, 3.0, -5.0, 0.0, 0.0, 1.0).is_none());
    }
}
