use std::sync::Arc;
use std::f32::consts::PI;

use crate::aabb::Aabb;
use crate::hittable::{HitRecord, Hittable, Material};
use crate::ray::Ray;
use crate::vec3::{Point3, Vec3};

/// Round brilliant diamond as an analytic convex polyhedron.
///
/// The girdle (widest circle) lies in the horizontal plane through `center`.
/// The table (flat top octagon) faces +Y; the culet (bottom point) faces −Y.
///
/// Tolkowsky ideal-cut proportions used to derive the half-space normals:
///   crown angle    = 34.5°  →  crown_h    = (r − table_r) · tan 34.5°
///   pavilion angle = 40.75° →  pavilion_h = r · tan 40.75°
///   table inscribed radius  = 0.53 r
///
/// The 17 half-spaces (1 table + 8 crown + 8 pavilion) are arranged so that
/// the crown and pavilion faces alternate every 22.5° around the girdle.  Their
/// pairwise intersections form a regular 16-gon with inscribed radius `radius`,
/// giving the girdle without any explicit vertical girdle planes.
///
/// Recommended material: `Dielectric { ir: 2.417 }`.
pub struct Diamond {
    planes:     Vec<(Vec3, f32)>,   // (outward unit normal n, offset d) — interior iff n·x ≤ d
    bbox:       Aabb,
    mat:        Arc<dyn Material>,
    center:     Point3,
    crown_h:    f32,
    pavilion_h: f32,
}

impl Diamond {
    pub fn new(center: Point3, radius: f32, mat: Arc<dyn Material>) -> Self {
        let r          = radius;
        let table_r    = 0.53 * r;
        let crown_h    = (r - table_r) * 34.5_f32.to_radians().tan();
        let pavilion_h = r             * 40.75_f32.to_radians().tan();

        let mut planes: Vec<(Vec3, f32)> = Vec::with_capacity(17);

        // ── Table ──────────────────────────────────────────────────────────────
        planes.push((Vec3::new(0.0, 1.0, 0.0), center.y + crown_h));

        // ── Crown main facets (8, bezel facets) ───────────────────────────────
        // At azimuth θ = k · 45°.  The face runs from the girdle edge (r, 0) to
        // the table edge (table_r, crown_h) in the radial-Y plane, so:
        //   unnorm outward normal = (cos θ · crown_h,  r − table_r,  sin θ · crown_h)
        for k in 0..8usize {
            let theta = k as f32 * PI / 4.0;
            let (s, c) = theta.sin_cos();
            let n = Vec3::new(c * crown_h, r - table_r, s * crown_h).unit();
            // Face passes through the girdle edge at this azimuth.
            let girdle_pt = Vec3::new(center.x + c * r, center.y, center.z + s * r);
            planes.push((n, n.dot(girdle_pt)));
        }

        // ── Pavilion main facets (8) ──────────────────────────────────────────
        // Offset 22.5° from crown so the girdle is a uniform 16-gon.
        // Face runs from girdle (r, 0) down to the culet (0, −pavilion_h):
        //   unnorm outward normal = (cos φ · pavilion_h,  −r,  sin φ · pavilion_h)
        for k in 0..8usize {
            let phi = k as f32 * PI / 4.0 + PI / 8.0;
            let (s, c) = phi.sin_cos();
            let n = Vec3::new(c * pavilion_h, -r, s * pavilion_h).unit();
            let girdle_pt = Vec3::new(center.x + c * r, center.y, center.z + s * r);
            planes.push((n, n.dot(girdle_pt)));
        }

        // ── Bounding box ───────────────────────────────────────────────────────
        // Circumradius of the 16-gon girdle = r / cos(π/16) ≈ 1.020 r.
        let circ_r = r / (PI / 16.0_f32).cos();
        let bbox = Aabb::new(
            center - Vec3::new(circ_r, pavilion_h, circ_r),
            center + Vec3::new(circ_r, crown_h,    circ_r),
        );

        Self { planes, bbox, mat, center, crown_h, pavilion_h }
    }

}

impl Hittable for Diamond {
    fn hit(&self, r: &Ray, t_min: f32, t_max: f32) -> Option<HitRecord<'_>> {
        // Convex polyhedron intersection: maintain the interval [t_enter, t_exit]
        // during which the ray is inside ALL half-spaces simultaneously.
        let mut t_enter              = t_min;
        let mut t_exit               = t_max;
        let mut enter_normal         = Vec3::new(0.0, 1.0, 0.0);
        let mut exit_outward_normal  = Vec3::new(0.0, 1.0, 0.0);
        let mut entry_found          = false;

        for &(n, d) in &self.planes {
            let nd     = n.dot(r.direction);
            let t_face = (d - n.dot(r.origin)) / nd;

            if nd.abs() < 1e-8 {
                // Parallel: if origin is outside this half-space, no hit.
                if n.dot(r.origin) > d { return None; }
                continue;
            }

            if nd < 0.0 {
                // Ray enters this half-space at t_face.
                if t_face > t_enter {
                    t_enter = t_face;
                    enter_normal = n;
                    entry_found = true;
                }
            } else {
                // Ray exits this half-space at t_face.
                if t_face < t_exit {
                    t_exit = t_face;
                    exit_outward_normal = n;
                }
            }

            if t_enter >= t_exit { return None; }
        }

        if t_enter >= t_exit { return None; }

        // entry_found  → exterior ray: hit at t_enter (entering the diamond).
        // !entry_found → interior ray (e.g. after refraction or TIR): hit at
        //                t_exit, which is the next internal surface crossing.
        let (t, outward_normal) = if entry_found {
            (t_enter, enter_normal)
        } else {
            if t_exit >= t_max { return None; }
            (t_exit, exit_outward_normal)
        };

        let p   = r.at(t);
        let rel = p - self.center;
        // Cylindrical UV: u = azimuth (0–1), v = height bottom-to-top (0–1).
        let u = ((-rel.z).atan2(rel.x) + PI) / (2.0 * PI);
        let v = ((rel.y + self.pavilion_h) / (self.crown_h + self.pavilion_h)).clamp(0.0, 1.0);

        let mut rec = HitRecord::new(p, t, &*self.mat, r, outward_normal);
        rec.u = u;
        rec.v = v;
        Some(rec)
    }

    fn bounding_box(&self) -> Option<Aabb> { Some(self.bbox) }
}
