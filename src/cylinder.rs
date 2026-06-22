use std::sync::Arc;
use std::f32::consts::TAU;

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

    // Cylinder at origin, radius=1, height=2.  Bottom cap at y=0, top cap at y=2.
    fn unit_cylinder() -> Cylinder {
        Cylinder { center: Point3::new(0.0, 0.0, 0.0), radius: 1.0, height: 2.0, mat: Arc::new(DummyMat) }
    }

    fn shoot(c: &Cylinder, ox: f32, oy: f32, oz: f32, dx: f32, dy: f32, dz: f32) -> Option<HitRecord<'_>> {
        let r = Ray::new(Point3::new(ox, oy, oz), Vec3::new(dx, dy, dz));
        c.hit(&r, 0.001, f32::INFINITY)
    }

    #[test]
    fn cylinder_lateral_hit_t_value() {
        // Ray along +z at y=1, starting at z=−5.
        // Discriminant: disc=1, roots t=4 and t=6.  t=4 hits front face.
        let c = unit_cylinder();
        let rec = shoot(&c, 0.0, 1.0, -5.0, 0.0, 0.0, 1.0).expect("should hit");
        assert!((rec.t - 4.0).abs() < 1e-5, "lateral t = {}", rec.t);
    }

    #[test]
    fn cylinder_lateral_normal_is_radial() {
        // The lateral surface normal must have y=0 and be a unit vector.
        let c = unit_cylinder();
        let rec = shoot(&c, 0.0, 1.0, -5.0, 0.0, 0.0, 1.0).expect("should hit");
        assert!(rec.normal.y.abs() < 1e-5, "lateral normal y should be 0, got {}", rec.normal.y);
        assert!((rec.normal.length() - 1.0).abs() < 1e-5, "normal should be unit length");
    }

    #[test]
    fn cylinder_lateral_normal_points_outward() {
        let c = unit_cylinder();
        let rec = shoot(&c, 0.0, 1.0, -5.0, 0.0, 0.0, 1.0).expect("should hit");
        let p = rec.p;
        // Outward direction from cylinder axis to hit point (ignoring y).
        let expected = Vec3::new(p.x, 0.0, p.z).unit();
        let outward  = if rec.front_face { rec.normal } else { -rec.normal };
        assert!((outward.dot(expected) - 1.0).abs() < 1e-4,
            "lateral normal should point radially outward");
    }

    #[test]
    fn cylinder_bottom_cap_hit_t_value() {
        // Ray along +y from below — hits bottom cap at y=0.  t = (0−(−5))/1 = 5.
        let c = unit_cylinder();
        let rec = shoot(&c, 0.0, -5.0, 0.0, 0.0, 1.0, 0.0).expect("should hit");
        assert!((rec.t - 5.0).abs() < 1e-5, "bottom cap t = {}", rec.t);
    }

    #[test]
    fn cylinder_top_cap_hit_t_value() {
        // Ray along −y from above — hits top cap (y=2) at t=3, bottom cap (y=0) at t=5.
        let c = unit_cylinder();
        let rec = shoot(&c, 0.0, 5.0, 0.0, 0.0, -1.0, 0.0).expect("should hit");
        assert!((rec.t - 3.0).abs() < 1e-5, "top cap t = {}", rec.t);
    }

    #[test]
    fn cylinder_cap_normals_are_axial() {
        // Bottom cap outward normal should point in −y; top cap in +y.
        let cb = unit_cylinder();
        let bot = shoot(&cb, 0.0, -5.0, 0.0, 0.0,  1.0, 0.0).expect("bottom hit");
        let bot_out = if bot.front_face { bot.normal } else { -bot.normal };
        assert!(bot_out.y < -0.99, "bottom cap outward normal should point −y, got {}", bot_out.y);

        let ct = unit_cylinder();
        let top = shoot(&ct, 0.0, 5.0, 0.0, 0.0, -1.0, 0.0).expect("top hit");
        let top_out = if top.front_face { top.normal } else { -top.normal };
        assert!(top_out.y >  0.99, "top cap outward normal should point +y, got {}",    top_out.y);
    }

    #[test]
    fn cylinder_miss_outside_radius() {
        let c = unit_cylinder();
        // Ray aimed 2 units from axis — clears the radius-1 cylinder.
        assert!(shoot(&c, 2.0, 1.0, -5.0, 0.0, 0.0, 1.0).is_none());
    }

    #[test]
    fn cylinder_miss_above_height() {
        let c = unit_cylinder();
        // Lateral intersection exists geometrically but the hit point is at y=3 > height=2.
        assert!(shoot(&c, 0.0, 3.0, -5.0, 0.0, 0.0, 1.0).is_none());
    }
}

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
