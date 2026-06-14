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
