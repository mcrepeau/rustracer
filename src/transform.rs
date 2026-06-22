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
    fn hit(&self, r: &Ray, t_min: f32, t_max: f32) -> Option<HitRecord<'_>> {
        let moved = Ray::scatter_from(r.origin - self.offset, r.direction, r);
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

// ── Rotate ────────────────────────────────────────────────────────────────────

pub struct Rotate {
    object: Arc<dyn Hittable>,
    /// World-from-object rotation matrix (forward transform).
    /// Inverse is the transpose (rotation matrices are orthogonal).
    fwd:  [[f32; 3]; 3],
    bbox: Option<Aabb>,
}

fn mat_apply(m: &[[f32; 3]; 3], v: Vec3) -> Vec3 {
    Vec3::new(
        m[0][0] * v.x + m[0][1] * v.y + m[0][2] * v.z,
        m[1][0] * v.x + m[1][1] * v.y + m[1][2] * v.z,
        m[2][0] * v.x + m[2][1] * v.y + m[2][2] * v.z,
    )
}

fn mat_apply_t(m: &[[f32; 3]; 3], v: Vec3) -> Vec3 {
    Vec3::new(
        m[0][0] * v.x + m[1][0] * v.y + m[2][0] * v.z,
        m[0][1] * v.x + m[1][1] * v.y + m[2][1] * v.z,
        m[0][2] * v.x + m[1][2] * v.y + m[2][2] * v.z,
    )
}

impl Rotate {
    fn with_matrix(object: Arc<dyn Hittable>, fwd: [[f32; 3]; 3]) -> Self {
        let bbox = object.bounding_box().map(|bb| {
            let mut min = Point3::new( f32::INFINITY,  f32::INFINITY,  f32::INFINITY);
            let mut max = Point3::new(f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY);
            for i in 0..2usize {
                for j in 0..2usize {
                    for k in 0..2usize {
                        let corner = Vec3::new(
                            if i == 0 { bb.min.x } else { bb.max.x },
                            if j == 0 { bb.min.y } else { bb.max.y },
                            if k == 0 { bb.min.z } else { bb.max.z },
                        );
                        let r = mat_apply(&fwd, corner);
                        min.x = min.x.min(r.x); max.x = max.x.max(r.x);
                        min.y = min.y.min(r.y); max.y = max.y.max(r.y);
                        min.z = min.z.min(r.z); max.z = max.z.max(r.z);
                    }
                }
            }
            Aabb::new(min, max)
        });
        Self { object, fwd, bbox }
    }

    pub fn around_y(object: Arc<dyn Hittable>, angle_deg: f32) -> Self {
        let (s, c) = angle_deg.to_radians().sin_cos();
        Self::with_matrix(object, [
            [  c, 0.0,   s],
            [0.0, 1.0, 0.0],
            [ -s, 0.0,   c],
        ])
    }

}

impl Hittable for Rotate {
    fn hit(&self, r: &Ray, t_min: f32, t_max: f32) -> Option<HitRecord<'_>> {
        // Transform ray into object space using the inverse rotation (= transpose).
        let rotated = Ray::scatter_from(
            mat_apply_t(&self.fwd, r.origin),
            mat_apply_t(&self.fwd, r.direction),
            r,
        );
        let mut rec = self.object.hit(&rotated, t_min, t_max)?;
        // Transform hit point and normal back to world space.
        rec.p      = mat_apply(&self.fwd, rec.p);
        rec.normal = mat_apply(&self.fwd, rec.normal);
        Some(rec)
    }

    fn bounding_box(&self) -> Option<Aabb> { self.bbox }
}

