use std::sync::Arc;
use std::f32::consts::TAU;

use crate::aabb::Aabb;
use crate::hittable::{HitRecord, Hittable, Material};
use crate::ray::Ray;
use crate::vec3::{Point3, Vec3};

/// Y-axis-aligned finite cylinder with closed caps.
/// `center` is the centre of the bottom disk; the top is at `center.y + height`.
pub struct Cylinder {
    pub center: Point3,
    pub radius: f32,
    pub height: f32,
    pub mat:    Arc<dyn Material>,
}

impl Hittable for Cylinder {
    fn hit(&self, r: &Ray, t_min: f32, t_max: f32) -> Option<HitRecord<'_>> {
        let oc = r.origin - self.center;
        let dx = r.direction.x;
        let dz = r.direction.z;
        let dy = r.direction.y;

        let mut best_t = t_max;
        let mut best: Option<HitRecord<'_>> = None;

        // ── Lateral surface: x² + z² = r² ────────────────────────────────────
        let a = dx * dx + dz * dz;
        if a > 1e-8 {
            let half_b = oc.x * dx + oc.z * dz;
            let c      = oc.x * oc.x + oc.z * oc.z - self.radius * self.radius;
            let disc   = half_b * half_b - a * c;
            if disc >= 0.0 {
                let sqrtd = disc.sqrt();
                for root in [(-half_b - sqrtd) / a, (-half_b + sqrtd) / a] {
                    if root < t_min || root >= best_t { continue; }
                    let p       = r.at(root);
                    let local_y = p.y - self.center.y;
                    if local_y < 0.0 || local_y > self.height { continue; }
                    let outward = Vec3::new(
                        (p.x - self.center.x) / self.radius,
                        0.0,
                        (p.z - self.center.z) / self.radius,
                    );
                    let mut rec = HitRecord::new(p, root, &*self.mat, r, outward);
                    let phi = (p.z - self.center.z).atan2(p.x - self.center.x);
                    rec.u = (phi / TAU + 0.5).fract();
                    rec.v = local_y / self.height;
                    best_t = root;
                    best   = Some(rec);
                }
            }
        }

        // ── Caps ──────────────────────────────────────────────────────────────
        if dy.abs() > 1e-8 {
            for (cap_local_y, ny) in [(0.0_f32, -1.0_f32), (self.height, 1.0_f32)] {
                let t = (cap_local_y - oc.y) / dy;
                if t < t_min || t >= best_t { continue; }
                let p  = r.at(t);
                let lx = p.x - self.center.x;
                let lz = p.z - self.center.z;
                if lx * lx + lz * lz > self.radius * self.radius { continue; }
                let mut rec = HitRecord::new(p, t, &*self.mat, r, Vec3::new(0.0, ny, 0.0));
                rec.u = (lx / self.radius + 1.0) * 0.5;
                rec.v = (lz / self.radius + 1.0) * 0.5;
                best_t = t;
                best   = Some(rec);
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
