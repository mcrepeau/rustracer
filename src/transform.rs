use std::sync::Arc;
use crate::aabb::Aabb;
use crate::hittable::{HitRecord, Hittable};
use crate::ray::Ray;
use crate::vec3::{Point3, Vec3};

// ── Translate ─────────────────────────────────────────────────────────────────

pub struct Translate {
    object: Arc<dyn Hittable>,
    offset: Vec3,
}

impl Translate {
    pub fn new(object: Arc<dyn Hittable>, offset: Vec3) -> Self {
        Self { object, offset }
    }
}

impl Hittable for Translate {
    fn hit(&self, r: &Ray, t_min: f32, t_max: f32) -> Option<HitRecord> {
        let moved = Ray::new(r.origin - self.offset, r.direction);
        let mut rec = self.object.hit(&moved, t_min, t_max)?;
        rec.p += self.offset;
        Some(rec)
    }

    fn bounding_box(&self) -> Option<Aabb> {
        self.object.bounding_box().map(|bb| {
            Aabb::new(bb.min + self.offset, bb.max + self.offset)
        })
    }
}

// ── RotateY ───────────────────────────────────────────────────────────────────

pub struct RotateY {
    object:    Arc<dyn Hittable>,
    sin_theta: f32,
    cos_theta: f32,
    bbox:      Option<Aabb>,
}

impl RotateY {
    pub fn new(object: Arc<dyn Hittable>, angle_deg: f32) -> Self {
        let theta     = angle_deg.to_radians();
        let sin_theta = theta.sin();
        let cos_theta = theta.cos();

        let bbox = object.bounding_box().map(|bb| {
            let mut min = Point3::new( f32::INFINITY,  f32::INFINITY,  f32::INFINITY);
            let mut max = Point3::new(f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY);
            for i in 0..2usize {
                for j in 0..2usize {
                    for k in 0..2usize {
                        let x = if i == 0 { bb.min.x } else { bb.max.x };
                        let y = if j == 0 { bb.min.y } else { bb.max.y };
                        let z = if k == 0 { bb.min.z } else { bb.max.z };
                        // Apply forward rotation (+theta) to get world-space corners
                        let rx =  cos_theta * x + sin_theta * z;
                        let rz = -sin_theta * x + cos_theta * z;
                        min.x = min.x.min(rx); max.x = max.x.max(rx);
                        min.y = min.y.min(y);  max.y = max.y.max(y);
                        min.z = min.z.min(rz); max.z = max.z.max(rz);
                    }
                }
            }
            Aabb::new(min, max)
        });

        Self { object, sin_theta, cos_theta, bbox }
    }
}

impl Hittable for RotateY {
    fn hit(&self, r: &Ray, t_min: f32, t_max: f32) -> Option<HitRecord> {
        // Rotate ray into object space (inverse: −theta)
        let rotated = Ray::new(
            Point3::new(
                self.cos_theta * r.origin.x - self.sin_theta * r.origin.z,
                r.origin.y,
                self.sin_theta * r.origin.x + self.cos_theta * r.origin.z,
            ),
            Vec3::new(
                self.cos_theta * r.direction.x - self.sin_theta * r.direction.z,
                r.direction.y,
                self.sin_theta * r.direction.x + self.cos_theta * r.direction.z,
            ),
        );

        let mut rec = self.object.hit(&rotated, t_min, t_max)?;

        // Rotate hit point and normal back to world space (+theta)
        let px = rec.p.x;
        rec.p.x =  self.cos_theta * px + self.sin_theta * rec.p.z;
        rec.p.z = -self.sin_theta * px + self.cos_theta * rec.p.z;

        let nx = rec.normal.x;
        rec.normal.x =  self.cos_theta * nx + self.sin_theta * rec.normal.z;
        rec.normal.z = -self.sin_theta * nx + self.cos_theta * rec.normal.z;

        Some(rec)
    }

    fn bounding_box(&self) -> Option<Aabb> { self.bbox }
}
