use std::sync::Arc;
use rand::RngCore;
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
        // Hard assert (not debug_assert): in release, factor <= 0 gives NaN rays and
        // an inverted AABB, silently breaking every hit on the object.
        assert!(factor > 0.0, "UniformScale factor must be positive; negative scale flips normals");
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

    fn transmittance(&self, r: &Ray, t_min: f32, t_max: f32) -> f32 {
        let inv = 1.0 / self.factor;
        let scaled = Ray::scatter_from(inv * r.origin, inv * r.direction, r);
        // t is invariant under uniform scale, but a medium's physical path length
        // scales by `factor`, so Beer-Lambert in object space underestimates
        // attenuation by that factor: T_world = T_obj^factor (surfaces are 0/1,
        // for which the powf is the identity).
        self.object.transmittance(&scaled, t_min, t_max).powf(self.factor)
    }

    fn light_area(&self) -> f32 { self.object.light_area() * self.factor * self.factor }

    fn pdf_value(&self, origin: Point3, dir: Vec3, time: f32) -> f32 {
        // Solid angle is invariant under uniform scale when the origin scales too:
        // dist^2/(cos*area) is unchanged, and CDF weights keep their ratios.
        self.object.pdf_value((1.0 / self.factor) * origin, dir, time)
    }

    fn pdf_generate(&self, origin: Point3, rng: &mut dyn RngCore, time: f32) -> Vec3 {
        // World direction to a scaled sample point equals the local direction.
        self.object.pdf_generate((1.0 / self.factor) * origin, rng, time)
    }

    fn bounding_box(&self) -> Option<Aabb> {
        self.object.bounding_box().map(|bb| {
            Aabb::new(self.factor * bb.min, self.factor * bb.max)
        })
    }

    fn photon_bounds(&self) -> Option<Aabb> {
        self.object.photon_bounds().map(|bb| {
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

    fn any_hit(&self, r: &Ray, t_min: f32, t_max: f32) -> bool {
        let moved = Ray::scatter_from(r.origin - self.offset, r.direction, r);
        self.object.any_hit(&moved, t_min, t_max)
    }

    fn transmittance(&self, r: &Ray, t_min: f32, t_max: f32) -> f32 {
        let moved = Ray::scatter_from(r.origin - self.offset, r.direction, r);
        self.object.transmittance(&moved, t_min, t_max)
    }

    fn light_area(&self) -> f32 { self.object.light_area() }

    fn pdf_value(&self, origin: Point3, dir: Vec3, time: f32) -> f32 {
        self.object.pdf_value(origin - self.offset, dir, time)
    }

    fn pdf_generate(&self, origin: Point3, rng: &mut dyn RngCore, time: f32) -> Vec3 {
        self.object.pdf_generate(origin - self.offset, rng, time)
    }

    fn bounding_box(&self) -> Option<Aabb> {
        self.object.bounding_box().map(|bb| {
            Aabb::new(bb.min + self.offset, bb.max + self.offset)
        })
    }

    fn photon_bounds(&self) -> Option<Aabb> {
        self.object.photon_bounds().map(|bb| {
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
    /// Same as `bbox` but from the child's photon_bounds() (proxy children -> None).
    photon_bbox: Option<Aabb>,
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

fn mat_mul(a: &[[f32; 3]; 3], b: &[[f32; 3]; 3]) -> [[f32; 3]; 3] {
    let mut m = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            m[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
        }
    }
    m
}

fn rotated_aabb(m: &[[f32; 3]; 3], bb: &Aabb) -> Aabb {
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
                let r = mat_apply(m, corner);
                min.x = min.x.min(r.x); max.x = max.x.max(r.x);
                min.y = min.y.min(r.y); max.y = max.y.max(r.y);
                min.z = min.z.min(r.z); max.z = max.z.max(r.z);
            }
        }
    }
    Aabb::new(min, max)
}

impl Rotate {
    fn with_matrix(object: Arc<dyn Hittable>, fwd: [[f32; 3]; 3]) -> Self {
        let bbox        = object.bounding_box().map(|bb| rotated_aabb(&fwd, &bb));
        let photon_bbox = object.photon_bounds().map(|bb| rotated_aabb(&fwd, &bb));
        Self { object, fwd, bbox, photon_bbox }
    }

    /// Euler rotation in degrees [rx, ry, rz] about the local origin,
    /// applied as fwd = Ry * Rx * Rz (Rz first, column-vector convention).
    pub fn euler(object: Arc<dyn Hittable>, rx_deg: f32, ry_deg: f32, rz_deg: f32) -> Self {
        let (sx, cx) = rx_deg.to_radians().sin_cos();
        let (sy, cy) = ry_deg.to_radians().sin_cos();
        let (sz, cz) = rz_deg.to_radians().sin_cos();
        let rx = [[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]];
        let ry = [[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]];
        let rz = [[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]];
        Self::with_matrix(object, mat_mul(&ry, &mat_mul(&rx, &rz)))
    }

    /// Euler rotation about a pivot point: world = R * (local - pivot) + pivot.
    /// Returns the object unchanged for all-zero angles.
    pub fn euler_about(object: Arc<dyn Hittable>, pivot: Point3, angles_deg: [f32; 3]) -> Arc<dyn Hittable> {
        let [rx, ry, rz] = angles_deg;
        if rx == 0.0 && ry == 0.0 && rz == 0.0 { return object; }
        let h = Translate::new(object, -pivot);
        let h = Rotate::euler(Arc::new(h), rx, ry, rz);
        Arc::new(Translate::new(Arc::new(h), pivot))
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
        // Transform hit point, normal, and tangent back to world space.
        rec.p       = mat_apply(&self.fwd, rec.p);
        rec.normal  = mat_apply(&self.fwd, rec.normal);
        rec.tangent = mat_apply(&self.fwd, rec.tangent);
        Some(rec)
    }

    fn any_hit(&self, r: &Ray, t_min: f32, t_max: f32) -> bool {
        let rotated = Ray::scatter_from(
            mat_apply_t(&self.fwd, r.origin),
            mat_apply_t(&self.fwd, r.direction),
            r,
        );
        self.object.any_hit(&rotated, t_min, t_max)
    }

    fn transmittance(&self, r: &Ray, t_min: f32, t_max: f32) -> f32 {
        let rotated = Ray::scatter_from(
            mat_apply_t(&self.fwd, r.origin),
            mat_apply_t(&self.fwd, r.direction),
            r,
        );
        self.object.transmittance(&rotated, t_min, t_max)
    }

    fn light_area(&self) -> f32 { self.object.light_area() }

    fn pdf_value(&self, origin: Point3, dir: Vec3, time: f32) -> f32 {
        // Rotation preserves solid angles; map origin/dir into object space.
        self.object.pdf_value(mat_apply_t(&self.fwd, origin), mat_apply_t(&self.fwd, dir), time)
    }

    fn pdf_generate(&self, origin: Point3, rng: &mut dyn RngCore, time: f32) -> Vec3 {
        let d_local = self.object.pdf_generate(mat_apply_t(&self.fwd, origin), rng, time);
        mat_apply(&self.fwd, d_local)
    }

    fn bounding_box(&self) -> Option<Aabb> { self.bbox }

    fn photon_bounds(&self) -> Option<Aabb> { self.photon_bbox }
}

// ── SpinY ─────────────────────────────────────────────────────────────────────

/// Time-driven turntable rotation about the world Y axis through `center`:
/// angle(t) = start + (end - start) * clamp(t / duration, 0, 1), evaluated from
/// each ray's `time`.  Used by `[animation.spin]` to turn gems in place while the
/// camera stays fixed.  Spin frames get a degenerate ray-time window (every ray
/// at the exact frame time), so each rendered frame is a crisp still at a stepped
/// angle.  Stateless: the same object serves every frame; a still render (ray
/// time 0) shows `start`.
pub struct SpinY {
    object:      Arc<dyn Hittable>,
    center:      Point3,
    start:       f32,   // radians
    end:         f32,   // radians
    duration:    f32,   // seconds
    bbox:        Option<Aabb>,
    photon_bbox: Option<Aabb>,
}

impl SpinY {
    pub fn new(object: Arc<dyn Hittable>, center: Point3, start_deg: f32, end_deg: f32, duration: f32) -> Self {
        let start = start_deg.to_radians();
        let end   = end_deg.to_radians();
        // Swept bbox: a child-bbox corner at horizontal distance r from the spin
        // axis sweeps a circle of radius r under a full turn, so the exact x/z extent
        // is center +/- r (conservative for partial ranges).  Never smaller than the
        // true swept volume, so BVH culling stays safe.
        let sweep = |bb: &Aabb| -> Aabb {
            let mut min = Point3::new(f32::INFINITY, bb.min.y, f32::INFINITY);
            let mut max = Point3::new(f32::NEG_INFINITY, bb.max.y, f32::NEG_INFINITY);
            for i in 0..2usize {
                for k in 0..2usize {
                    let x = if i == 0 { bb.min.x } else { bb.max.x };
                    let z = if k == 0 { bb.min.z } else { bb.max.z };
                    let r = ((x - center.x).powi(2) + (z - center.z).powi(2)).sqrt();
                    min.x = min.x.min(center.x - r); max.x = max.x.max(center.x + r);
                    min.z = min.z.min(center.z - r); max.z = max.z.max(center.z + r);
                }
            }
            Aabb::new(min, max)
        };
        let bbox        = object.bounding_box().map(|bb| sweep(&bb));
        let photon_bbox = object.photon_bounds().map(|bb| sweep(&bb));
        Self { object, center, start, end, duration, bbox, photon_bbox }
    }

    fn angle_at(&self, t: f32) -> f32 {
        let frac = if self.duration > 1e-8 { (t / self.duration).clamp(0.0, 1.0) } else { 0.0 };
        self.start + (self.end - self.start) * frac
    }

    /// World-from-local rotation matrix at the given ray time.
    fn fwd_at(&self, t: f32) -> [[f32; 3]; 3] {
        let (s, c) = self.angle_at(t).sin_cos();
        [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]]
    }
}

impl Hittable for SpinY {
    fn hit(&self, r: &Ray, t_min: f32, t_max: f32) -> Option<HitRecord<'_>> {
        let fwd = self.fwd_at(r.time);
        // world -> local:  local = R^T (world - center) + center
        let local = Ray::scatter_from(
            mat_apply_t(&fwd, r.origin - self.center) + self.center,
            mat_apply_t(&fwd, r.direction),
            r,
        );
        let mut rec = self.object.hit(&local, t_min, t_max)?;
        // local -> world:  world = R (local - center) + center
        rec.p       = mat_apply(&fwd, rec.p - self.center) + self.center;
        rec.normal  = mat_apply(&fwd, rec.normal);
        rec.tangent = mat_apply(&fwd, rec.tangent);
        Some(rec)
    }

    fn any_hit(&self, r: &Ray, t_min: f32, t_max: f32) -> bool {
        let fwd = self.fwd_at(r.time);
        let local = Ray::scatter_from(
            mat_apply_t(&fwd, r.origin - self.center) + self.center,
            mat_apply_t(&fwd, r.direction),
            r,
        );
        self.object.any_hit(&local, t_min, t_max)
    }

    fn transmittance(&self, r: &Ray, t_min: f32, t_max: f32) -> f32 {
        let fwd = self.fwd_at(r.time);
        let local = Ray::scatter_from(
            mat_apply_t(&fwd, r.origin - self.center) + self.center,
            mat_apply_t(&fwd, r.direction),
            r,
        );
        self.object.transmittance(&local, t_min, t_max)
    }

    fn light_area(&self) -> f32 { self.object.light_area() }

    fn pdf_value(&self, origin: Point3, dir: Vec3, time: f32) -> f32 {
        let fwd = self.fwd_at(time);
        self.object.pdf_value(
            mat_apply_t(&fwd, origin - self.center) + self.center,
            mat_apply_t(&fwd, dir),
            time,
        )
    }

    fn pdf_generate(&self, origin: Point3, rng: &mut dyn RngCore, time: f32) -> Vec3 {
        let fwd = self.fwd_at(time);
        let d_local = self.object.pdf_generate(mat_apply_t(&fwd, origin - self.center) + self.center, rng, time);
        mat_apply(&fwd, d_local)
    }

    fn bounding_box(&self) -> Option<Aabb> { self.bbox }
    fn photon_bounds(&self) -> Option<Aabb> { self.photon_bbox }
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
        let obj = Rotate::euler(sphere_at(1.0, 0.0, 0.0, 0.5), 0.0, 90.0, 0.0);
        let rec = shoot(&obj, 0.0, 0.0, -5.0, 0.0, 0.0, 1.0).expect("should hit rotated sphere");
        assert!((rec.t - 3.5).abs() < 1e-4, "t = {}", rec.t);
    }

    #[test]
    fn rotate_y_90deg_hit_point_is_at_rotated_position() {
        // The hit point on the near face of the sphere (world z ≈ −1.5, x ≈ 0, y ≈ 0).
        let obj = Rotate::euler(sphere_at(1.0, 0.0, 0.0, 0.5), 0.0, 90.0, 0.0);
        let rec = shoot(&obj, 0.0, 0.0, -5.0, 0.0, 0.0, 1.0).expect("should hit");
        assert!(rec.p.x.abs()          < 1e-4, "p.x = {}", rec.p.x);
        assert!(rec.p.y.abs()          < 1e-4, "p.y = {}", rec.p.y);
        assert!((rec.p.z + 1.5).abs()  < 1e-4, "p.z = {}", rec.p.z);
    }

    #[test]
    fn rotate_y_90deg_normal_is_rotated() {
        // Object-space outward normal at the near face is (1,0,0) (pointing away from sphere centre).
        // After 90° Y rotation: mat_apply([[0,0,1],[0,1,0],[−1,0,0]], (1,0,0)) = (0,0,−1).
        let obj = Rotate::euler(sphere_at(1.0, 0.0, 0.0, 0.5), 0.0, 90.0, 0.0);
        let rec = shoot(&obj, 0.0, 0.0, -5.0, 0.0, 0.0, 1.0).expect("should hit");
        let outward = if rec.front_face { rec.normal } else { -rec.normal };
        assert!(outward.x.abs()         < 1e-4, "outward.x = {}", outward.x);
        assert!(outward.y.abs()         < 1e-4, "outward.y = {}", outward.y);
        assert!((outward.z + 1.0).abs() < 1e-4, "outward.z = {}", outward.z);
    }

    // ── SpinY ─────────────────────────────────────────────────────────────────

    fn spin_sphere(start_deg: f32, end_deg: f32, dur: f32) -> SpinY {
        SpinY::new(sphere_at(1.0, 0.0, 0.0, 0.5), Point3::new(0.0, 0.0, 0.0), start_deg, end_deg, dur)
    }

    #[test]
    fn spin_y_time_zero_is_start_angle() {
        // t=0 (angle 0): ray from (1,0,-5) dir +z hits sphere at (1,0,0) r=0.5 -> p=(1,0,-0.5), t=4.5
        let s = spin_sphere(0.0, 180.0, 4.0);
        let r = Ray::new_at_time(Point3::new(1.0, 0.0, -5.0), Vec3::new(0.0, 0.0, 1.0), 0.0);
        let rec = s.hit(&r, 0.001, f32::INFINITY).expect("should hit");
        assert!((rec.t - 4.5).abs() < 1e-4, "t = {}", rec.t);
        assert!((rec.p.x - 1.0).abs() < 1e-4 && (rec.p.z + 0.5).abs() < 1e-4, "p.x = {}, p.z = {}", rec.p.x, rec.p.z);
    }

    #[test]
    fn spin_y_at_duration_rotates_hit_180() {
        // t=4 (=duration, angle 180): world sphere at (-1,0,0); ray from (-1,0,-5) -> p=(-1,0,-0.5)
        let s = spin_sphere(0.0, 180.0, 4.0);
        let r = Ray::new_at_time(Point3::new(-1.0, 0.0, -5.0), Vec3::new(0.0, 0.0, 1.0), 4.0);
        let rec = s.hit(&r, 0.001, f32::INFINITY).expect("should hit");
        assert!((rec.t - 4.5).abs() < 1e-4, "t = {}", rec.t);
        assert!((rec.p.x + 1.0).abs() < 1e-4 && (rec.p.z + 0.5).abs() < 1e-4, "p.x = {}, p.z = {}", rec.p.x, rec.p.z);
    }

    #[test]
    fn spin_y_time_is_clamped_to_duration() {
        // t=100 >> duration behaves exactly like t=duration
        let s = spin_sphere(0.0, 180.0, 4.0);
        let r = Ray::new_at_time(Point3::new(-1.0, 0.0, -5.0), Vec3::new(0.0, 0.0, 1.0), 100.0);
        let rec = s.hit(&r, 0.001, f32::INFINITY).expect("should hit");
        assert!((rec.p.x + 1.0).abs() < 1e-4, "p.x = {}", rec.p.x);
    }

    #[test]
    fn spin_y_bbox_covers_full_sweep() {
        // sphere at (1,0,0) r=0.5 has bbox corners up to (1.5, +-0.5); each corner sweeps a
        // circle of radius hypot(1.5, 0.5) = sqrt(2.5) about the axis, so x,z in [-sqrt(2.5), sqrt(2.5)]
        let s = spin_sphere(0.0, 360.0, 4.0);
        let bb = s.bounding_box().expect("bbox");
        let r = 2.5_f32.sqrt();
        assert!((bb.min.x + r).abs() < 1e-4 && (bb.max.x - r).abs() < 1e-4, "x: {}..{}", bb.min.x, bb.max.x);
        assert!((bb.min.z + r).abs() < 1e-4 && (bb.max.z - r).abs() < 1e-4, "z: {}..{}", bb.min.z, bb.max.z);
        assert!((bb.min.y + 0.5).abs() < 1e-4 && (bb.max.y - 0.5).abs() < 1e-4, "y: {}..{}", bb.min.y, bb.max.y);
    }

    #[test]
    fn spin_y_angle_step_per_frame_is_uniform() {
        // 24 fps x 4 s = 96 frames over 360 deg -> exactly 3.75 deg per frame
        let s = spin_sphere(0.0, 360.0, 4.0);
        let step = (360.0_f32 / 96.0).to_radians();
        let d0 = s.angle_at(1.0 / 24.0) - s.angle_at(0.0);
        let d1 = s.angle_at(2.0 / 24.0) - s.angle_at(1.0 / 24.0);
        assert!((d0 - step).abs() < 1e-6, "d0 = {}", d0);
        assert!((d0 - d1).abs() < 1e-6, "steps differ: {} vs {}", d0, d1);
    }

    // ── Euler ─────────────────────────────────────────────────────────────────

    #[test]
    fn euler_y90_hit_point_and_normal() {
        // euler(0,90,0) must equal around_y(90): sphere (1,0,0) r=0.5 -> (0,0,-1), t=3.5
        let obj = Rotate::euler(sphere_at(1.0, 0.0, 0.0, 0.5), 0.0, 90.0, 0.0);
        let rec = shoot(&obj, 0.0, 0.0, -5.0, 0.0, 0.0, 1.0).expect("should hit");
        assert!((rec.t - 3.5).abs() < 1e-4, "t = {}", rec.t);
        assert!((rec.p.z + 1.5).abs() < 1e-4, "p.z = {}", rec.p.z);
    }

    #[test]
    fn euler_x90_rotates_about_x() {
        // Rx(90) maps (0,0,1) -> (0,-1,0): ray from (0,-5,0) dir +y hits at y=-1.5, t=3.5
        let obj = Rotate::euler(sphere_at(0.0, 0.0, 1.0, 0.5), 90.0, 0.0, 0.0);
        let rec = shoot(&obj, 0.0, -5.0, 0.0, 0.0, 1.0, 0.0).expect("should hit");
        assert!((rec.t - 3.5).abs() < 1e-4, "t = {}", rec.t);
        assert!((rec.p.y + 1.5).abs() < 1e-4, "p.y = {}", rec.p.y);
    }

    #[test]
    fn euler_about_pivot_rotates_in_place() {
        // sphere at (2,0,0), pivot (1,0,0), ry=180 -> sphere moves to (0,0,0);
        // ray from (0,0,-5) dir +z hits p=(0,0,-0.5), t=4.5
        let obj = Rotate::euler_about(sphere_at(2.0, 0.0, 0.0, 0.5), Point3::new(1.0, 0.0, 0.0), [0.0, 180.0, 0.0]);
        let r = Ray::new(Point3::new(0.0, 0.0, -5.0), Vec3::new(0.0, 0.0, 1.0));
        let rec = obj.hit(&r, 0.001, f32::INFINITY).expect("should hit");
        assert!((rec.t - 4.5).abs() < 1e-4, "t = {}", rec.t);
        assert!(rec.p.x.abs() < 1e-4 && (rec.p.z + 0.5).abs() < 1e-4, "p.x = {}, p.z = {}", rec.p.x, rec.p.z);
    }

    #[test]
    fn euler_about_zero_angles_returns_same_object() {
        let s: Arc<dyn Hittable> = sphere_at(1.0, 0.0, 0.0, 0.5);
        let h = Rotate::euler_about(Arc::clone(&s), Point3::new(1.0, 0.0, 0.0), [0.0, 0.0, 0.0]);
        // identity: ray from (5,0,0) dir -x hits at t=3.5 (same as unrotated)
        let r = Ray::new(Point3::new(5.0, 0.0, 0.0), Vec3::new(-1.0, 0.0, 0.0));
        let rec = h.hit(&r, 0.001, f32::INFINITY).expect("should hit");
        assert!((rec.t - 3.5).abs() < 1e-4, "t = {}", rec.t);
    }

    #[test]
    fn rotate_y_zero_degrees_is_identity() {
        // 0° rotation: sphere at (1,0,0), ray from (5,0,0)→(−1,0,0) hits at t=3.5.
        let obj = Rotate::euler(sphere_at(1.0, 0.0, 0.0, 0.5), 0.0, 0.0, 0.0);
        let rec = shoot(&obj, 5.0, 0.0, 0.0, -1.0, 0.0, 0.0).expect("should hit");
        assert!((rec.t - 3.5).abs() < 1e-4, "t = {}", rec.t);
    }
}

