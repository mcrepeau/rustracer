use std::sync::Arc;

use crate::aabb::Aabb;
use crate::hittable::{HitRecord, Hittable, Material};
use crate::ray::Ray;
use crate::vec3::{Point3, Vec3};

/// Single triangle with per-vertex shading normals and UV coordinates.
/// Intersection uses the Möller–Trumbore algorithm.
pub struct Triangle {
    v0:  Point3,
    v1:  Point3,
    v2:  Point3,
    n0:  Vec3,
    n1:  Vec3,
    n2:  Vec3,
    uv0: (f32, f32),
    uv1: (f32, f32),
    uv2: (f32, f32),
    pub mat: Arc<dyn Material>,
}

impl Triangle {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        v0: Point3, v1: Point3, v2: Point3,
        n0: Vec3,   n1: Vec3,   n2: Vec3,
        uv0: (f32, f32), uv1: (f32, f32), uv2: (f32, f32),
        mat: Arc<dyn Material>,
    ) -> Self {
        Self { v0, v1, v2, n0, n1, n2, uv0, uv1, uv2, mat }
    }

    /// Triangle with a flat (face) normal and no meaningful UV coordinates.
    #[cfg(test)]
    fn flat(v0: Point3, v1: Point3, v2: Point3, mat: Arc<dyn Material>) -> Self {
        let n = (v1 - v0).cross(v2 - v0).unit();
        Self::new(v0, v1, v2, n, n, n, (0.0, 0.0), (1.0, 0.0), (0.5, 1.0), mat)
    }
}

impl Hittable for Triangle {
    fn hit(&self, r: &Ray, t_min: f32, t_max: f32) -> Option<HitRecord<'_>> {
        let e1 = self.v1 - self.v0;
        let e2 = self.v2 - self.v0;
        let h  = r.direction.cross(e2);
        let a  = e1.dot(h);
        if a.abs() < 1e-8 { return None; }  // ray parallel to triangle
        let f  = 1.0 / a;
        let s  = r.origin - self.v0;
        let b1 = f * s.dot(h);
        if !(0.0..=1.0).contains(&b1) { return None; }
        let q  = s.cross(e1);
        let b2 = f * r.direction.dot(q);
        if b2 < 0.0 || b1 + b2 > 1.0 { return None; }
        let t  = f * e2.dot(q);
        if t < t_min || t > t_max { return None; }

        let b0 = 1.0 - b1 - b2;
        let p  = r.at(t);
        // Interpolate shading normal and UV using barycentric coordinates.
        let sn = (b0 * self.n0 + b1 * self.n1 + b2 * self.n2).unit();
        let u  = b0 * self.uv0.0 + b1 * self.uv1.0 + b2 * self.uv2.0;
        let v  = b0 * self.uv0.1 + b1 * self.uv1.1 + b2 * self.uv2.1;
        let mut rec = HitRecord::new(p, t, &*self.mat, r, sn);
        rec.u = u;
        rec.v = v;
        Some(rec)
    }

    fn any_hit(&self, r: &Ray, t_min: f32, t_max: f32) -> bool {
        let e1 = self.v1 - self.v0;
        let e2 = self.v2 - self.v0;
        let h  = r.direction.cross(e2);
        let a  = e1.dot(h);
        if a.abs() < 1e-8 { return false; }
        let f  = 1.0 / a;
        let s  = r.origin - self.v0;
        let b1 = f * s.dot(h);
        if !(0.0..=1.0).contains(&b1) { return false; }
        let q  = s.cross(e1);
        let b2 = f * r.direction.dot(q);
        if b2 < 0.0 || b1 + b2 > 1.0 { return false; }
        let t  = f * e2.dot(q);
        t >= t_min && t <= t_max
    }

    fn bounding_box(&self) -> Option<Aabb> {
        let min = Point3::new(
            self.v0.x.min(self.v1.x).min(self.v2.x),
            self.v0.y.min(self.v1.y).min(self.v2.y),
            self.v0.z.min(self.v1.z).min(self.v2.z),
        );
        let max = Point3::new(
            self.v0.x.max(self.v1.x).max(self.v2.x),
            self.v0.y.max(self.v1.y).max(self.v2.y),
            self.v0.z.max(self.v1.z).max(self.v2.z),
        );
        Some(Aabb::new(min, max).pad())  // pad handles axis-aligned coplanar triangles
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

    // Triangle: v0=(0,0,0), v1=(2,0,0), v2=(0,2,0).
    // Face normal = (v1-v0)×(v2-v0) = (0,0,1), lies in the XY plane (z=0).
    // Centroid = (2/3, 2/3, 0).
    fn xy_triangle() -> Triangle {
        Triangle::flat(
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(2.0, 0.0, 0.0),
            Point3::new(0.0, 2.0, 0.0),
            Arc::new(DummyMat),
        )
    }

    fn shoot(tri: &Triangle, ox: f32, oy: f32, oz: f32, dx: f32, dy: f32, dz: f32) -> Option<HitRecord<'_>> {
        let r = Ray::new(Point3::new(ox, oy, oz), Vec3::new(dx, dy, dz));
        tri.hit(&r, 0.001, f32::INFINITY)
    }

    #[test]
    fn triangle_hit_centroid_correct_t() {
        // Ray from z=−5, aimed at the centroid (2/3, 2/3, 0).
        // Möller-Trumbore: a=−4, b1=b2=1/3, t=5.
        let tri = xy_triangle();
        let rec = shoot(&tri, 2.0/3.0, 2.0/3.0, -5.0, 0.0, 0.0, 1.0).expect("should hit");
        assert!((rec.t - 5.0).abs() < 1e-5, "t = {}", rec.t);
    }

    #[test]
    fn triangle_hit_vertex() {
        // Ray aimed directly at v0 = (0,0,0).  b1=b2=0, b0=1.
        let tri = xy_triangle();
        let rec = shoot(&tri, 0.0, 0.0, -3.0, 0.0, 0.0, 1.0).expect("v0 should hit");
        assert!((rec.t - 3.0).abs() < 1e-5, "t = {}", rec.t);
    }

    #[test]
    fn triangle_miss_b1_exceeds_one() {
        // x=3 places the ray outside v0–v1 edge: b1 ≈ 1.5 → miss.
        let tri = xy_triangle();
        assert!(shoot(&tri, 3.0, 0.0, -1.0, 0.0, 0.0, 1.0).is_none());
    }

    #[test]
    fn triangle_miss_b2_negative() {
        // y=−1 gives b2 < 0 → miss.
        let tri = xy_triangle();
        assert!(shoot(&tri, 0.0, -1.0, -1.0, 0.0, 0.0, 1.0).is_none());
    }

    #[test]
    fn triangle_miss_b1_plus_b2_exceeds_one() {
        // (1.5, 1.5) → b1+b2 ≈ 1.5 → miss (past the hypotenuse).
        let tri = xy_triangle();
        assert!(shoot(&tri, 1.5, 1.5, -1.0, 0.0, 0.0, 1.0).is_none());
    }

    #[test]
    fn triangle_miss_parallel_ray() {
        // Ray direction in the XY plane — a ≈ 0 → miss.
        let tri = xy_triangle();
        assert!(shoot(&tri, 0.5, 0.5, 0.0, 1.0, 0.0, 0.0).is_none());
    }

    #[test]
    fn triangle_uv_interpolated_at_centroid() {
        // Triangle with explicit UV: uv0=(0,0), uv1=(1,0), uv2=(0,1).
        // At centroid, UV = (1/3, 1/3).
        let tri = Triangle::new(
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(2.0, 0.0, 0.0),
            Point3::new(0.0, 2.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(0.0, 0.0, 1.0),
            (0.0, 0.0), (1.0, 0.0), (0.0, 1.0),
            Arc::new(DummyMat),
        );
        let r = Ray::new(Point3::new(2.0/3.0, 2.0/3.0, -1.0), Vec3::new(0.0, 0.0, 1.0));
        let rec = tri.hit(&r, 0.001, f32::INFINITY).expect("should hit");
        assert!((rec.u - 1.0/3.0).abs() < 1e-5, "u = {}", rec.u);
        assert!((rec.v - 1.0/3.0).abs() < 1e-5, "v = {}", rec.v);
    }

    #[test]
    fn triangle_shading_normal_interpolated() {
        // Vertex normals tilted away from (0,0,1) — centroid should give their average.
        let n0 = Vec3::new( 0.0, 0.0, 1.0);
        let n1 = Vec3::new( 1.0, 0.0, 1.0).unit();
        let n2 = Vec3::new(-1.0, 0.0, 1.0).unit();
        let tri = Triangle::new(
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(2.0, 0.0, 0.0),
            Point3::new(0.0, 2.0, 0.0),
            n0, n1, n2,
            (0.0, 0.0), (1.0, 0.0), (0.0, 1.0),
            Arc::new(DummyMat),
        );
        let r = Ray::new(Point3::new(2.0/3.0, 2.0/3.0, -1.0), Vec3::new(0.0, 0.0, 1.0));
        let rec = tri.hit(&r, 0.001, f32::INFINITY).expect("should hit");
        // Expected: (n0+n1+n2) / 3 normalized.
        let avg = (n0 + n1 + n2).unit();
        let got = if rec.front_face { rec.normal } else { -rec.normal };
        assert!((got.dot(avg) - 1.0).abs() < 1e-4,
            "shading normal should be interpolated: got {:?}, expected {:?}", got, avg);
    }

    #[test]
    fn triangle_bounding_box_encloses_vertices() {
        let tri = xy_triangle();
        let bb = tri.bounding_box().expect("should have bbox");
        assert!(bb.min.x <= 0.0 && bb.max.x >= 2.0);
        assert!(bb.min.y <= 0.0 && bb.max.y >= 2.0);
    }
}
