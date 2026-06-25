use std::sync::Arc;
use crate::aabb::Aabb;
use crate::hittable::{HitRecord, Hittable};
use crate::ray::Ray;
use crate::vec3::{Point3, Vec3};

// ── UniformScale ──────────────────────────────────────────────────────────────

/// Uniform (isotropic) scale transform.  A single scalar factor is required
/// because non-uniform scale needs the inverse-transpose for correct normal
/// transformation; uniform scale's inverse-transpose is the identity, so
/// normals can be passed through unchanged.  `factor` must be positive —
/// negative values flip normals and break front-face detection.
pub struct UniformScale {
    object: Arc<dyn Hittable>,
    factor: f32,
}

impl UniformScale {
    pub fn new(object: Arc<dyn Hittable>, factor: f32) -> Self {
        debug_assert!(factor > 0.0, "UniformScale factor must be positive; negative scale flips normals");
        Self { object, factor }
    }
}

impl Hittable for UniformScale {
    fn hit(&self, r: &Ray, t_min: f32, t_max: f32) -> Option<HitRecord<'_>> {
        let inv = 1.0 / self.factor;
        // Transform ray to object space: scale both origin and direction by inv.
        // t is invariant: P(t) = O + t·D  →  P/s = (O/s) + t·(D/s).
        let scaled = Ray::scatter_from(inv * r.origin, inv * r.direction, r);
        let mut rec = self.object.hit(&scaled, t_min, t_max)?;
        rec.p = self.factor * rec.p;
        // Normals are unchanged: inverse-transpose of a uniform scale is the identity.
        Some(rec)
    }

    fn any_hit(&self, r: &Ray, t_min: f32, t_max: f32) -> bool {
        let inv = 1.0 / self.factor;
        let scaled = Ray::scatter_from(inv * r.origin, inv * r.direction, r);
        self.object.any_hit(&scaled, t_min, t_max)
    }

    fn bounding_box(&self) -> Option<Aabb> {
        self.object.bounding_box().map(|bb| {
            Aabb::new(self.factor * bb.min, self.factor * bb.max)
        })
    }
}


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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hittable::{Material, ScatterRecord};
    use crate::sphere::Sphere;

    struct DummyMat;
    impl Material for DummyMat {
        fn scatter(&self, _: &Ray, _: &HitRecord<'_>, _: &mut dyn rand::RngCore) -> Option<ScatterRecord> { None }
    }

    fn sphere_at(cx: f32, cy: f32, cz: f32, r: f32) -> Arc<Sphere> {
        Arc::new(Sphere::new(Point3::new(cx, cy, cz), r, Arc::new(DummyMat)))
    }

    fn shoot<H: Hittable>(h: &H, ox: f32, oy: f32, oz: f32, dx: f32, dy: f32, dz: f32) -> Option<HitRecord<'_>> {
        let r = Ray::new(Point3::new(ox, oy, oz), Vec3::new(dx, dy, dz));
        h.hit(&r, 0.001, f32::INFINITY)
    }

    // ── Translate ─────────────────────────────────────────────────────────────

    #[test]
    fn translate_hit_point_is_offset() {
        // Sphere at origin, radius=1. Translate by (5,0,0) → appears at (5,0,0).
        // Ray from (5,0,−5) going (0,0,1): in object space it's a centre shot → t=4.
        let t = Translate::new(sphere_at(0.0, 0.0, 0.0, 1.0), Vec3::new(5.0, 0.0, 0.0));
        let rec = shoot(&t, 5.0, 0.0, -5.0, 0.0, 0.0, 1.0).expect("should hit");
        // Near face of the sphere is at z=−1 in object space → world (5, 0, −1).
        assert!((rec.t - 4.0).abs() < 1e-5, "t = {}", rec.t);
        assert!((rec.p.x - 5.0).abs() < 1e-5, "p.x = {}", rec.p.x);
        assert!((rec.p.z + 1.0).abs() < 1e-5, "p.z = {}", rec.p.z);
    }

    #[test]
    fn translate_miss_at_original_position() {
        // Ray aimed at the pre-translation origin must miss the translated sphere.
        let t = Translate::new(sphere_at(0.0, 0.0, 0.0, 1.0), Vec3::new(5.0, 0.0, 0.0));
        assert!(shoot(&t, 0.0, 0.0, -5.0, 0.0, 0.0, 1.0).is_none());
    }

    #[test]
    fn translate_preserves_normal_direction() {
        // Normal is a direction — it must not be translated, only the hit point is.
        // A centre shot in -z gives outward normal (0,0,−1) and front_face=true.
        let t = Translate::new(sphere_at(0.0, 0.0, 0.0, 1.0), Vec3::new(5.0, 0.0, 0.0));
        let rec = shoot(&t, 5.0, 0.0, -5.0, 0.0, 0.0, 1.0).expect("should hit");
        assert!(rec.front_face, "should be front face");
        assert!(rec.normal.z < -0.99, "normal should point −z, got {}", rec.normal.z);
    }

    #[test]
    fn translate_bounding_box_is_shifted() {
        // AABB of unit sphere at origin is [−1,1]³; shift by (5,0,0) → [4,6]×[−1,1]×[−1,1].
        let t = Translate::new(sphere_at(0.0, 0.0, 0.0, 1.0), Vec3::new(5.0, 0.0, 0.0));
        let bb = t.bounding_box().expect("should have bbox");
        assert!((bb.min.x - 4.0).abs() < 1e-5, "min.x = {}", bb.min.x);
        assert!((bb.max.x - 6.0).abs() < 1e-5, "max.x = {}", bb.max.x);
        assert!((bb.min.y + 1.0).abs() < 1e-5, "min.y = {}", bb.min.y);
        assert!((bb.max.y - 1.0).abs() < 1e-5, "max.y = {}", bb.max.y);
    }

    // ── Rotate ────────────────────────────────────────────────────────────────

    #[test]
    fn rotate_y_90deg_moves_off_axis_sphere() {
        // Sphere at (1,0,0), radius=0.5. Rotate 90° around Y:
        // fwd = [[0,0,1],[0,1,0],[−1,0,0]] maps (1,0,0) → (0,0,−1).
        // Ray from (0,0,−5)→(0,0,1): hits the sphere now at (0,0,−1), t=3.5.
        let obj = Rotate::around_y(sphere_at(1.0, 0.0, 0.0, 0.5), 90.0);
        let rec = shoot(&obj, 0.0, 0.0, -5.0, 0.0, 0.0, 1.0).expect("should hit rotated sphere");
        assert!((rec.t - 3.5).abs() < 1e-4, "t = {}", rec.t);
    }

    #[test]
    fn rotate_y_90deg_hit_point_is_at_rotated_position() {
        // The hit point on the near face of the sphere (world z ≈ −1.5, x ≈ 0, y ≈ 0).
        let obj = Rotate::around_y(sphere_at(1.0, 0.0, 0.0, 0.5), 90.0);
        let rec = shoot(&obj, 0.0, 0.0, -5.0, 0.0, 0.0, 1.0).expect("should hit");
        assert!(rec.p.x.abs()          < 1e-4, "p.x = {}", rec.p.x);
        assert!(rec.p.y.abs()          < 1e-4, "p.y = {}", rec.p.y);
        assert!((rec.p.z + 1.5).abs()  < 1e-4, "p.z = {}", rec.p.z);
    }

    #[test]
    fn rotate_y_90deg_normal_is_rotated() {
        // Object-space outward normal at the near face is (1,0,0) (pointing away from sphere centre).
        // After 90° Y rotation: mat_apply([[0,0,1],[0,1,0],[−1,0,0]], (1,0,0)) = (0,0,−1).
        let obj = Rotate::around_y(sphere_at(1.0, 0.0, 0.0, 0.5), 90.0);
        let rec = shoot(&obj, 0.0, 0.0, -5.0, 0.0, 0.0, 1.0).expect("should hit");
        let outward = if rec.front_face { rec.normal } else { -rec.normal };
        assert!(outward.x.abs()         < 1e-4, "outward.x = {}", outward.x);
        assert!(outward.y.abs()         < 1e-4, "outward.y = {}", outward.y);
        assert!((outward.z + 1.0).abs() < 1e-4, "outward.z = {}", outward.z);
    }

    #[test]
    fn rotate_y_zero_degrees_is_identity() {
        // 0° rotation: sphere at (1,0,0), ray from (5,0,0)→(−1,0,0) hits at t=3.5.
        let obj = Rotate::around_y(sphere_at(1.0, 0.0, 0.0, 0.5), 0.0);
        let rec = shoot(&obj, 5.0, 0.0, 0.0, -1.0, 0.0, 0.0).expect("should hit");
        assert!((rec.t - 3.5).abs() < 1e-4, "t = {}", rec.t);
    }
}

