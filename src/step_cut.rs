use std::sync::Arc;
use std::f32::consts::PI;

use crate::aabb::Aabb;
use crate::diamond::convex_hit;
use crate::hittable::{HitRecord, Hittable, Material};
use crate::ray::Ray;
use crate::vec3::{Point3, Vec3};

/// Proportion parameters for a rectangular step cut ("emerald cut").
///
/// The girdle outline is an octagon: a `2a x 2b` rectangle (length x width)
/// with the four corners chamfered at 45°.  Crown and pavilion each consist of
/// `crown_steps` / `pavilion_steps` planes per outline edge; the facet angles
/// spread linearly from `(1 + step_spread) * angle` at the girdle to
/// `(1 - step_spread) * angle` at the table/culet, which is what produces the
/// visible steps.  The table and culet are the outline inset by a constant
/// margin along each edge normal (flat caps).
#[derive(Clone, Debug)]
pub struct StepCutParams {
    /// Table width as a percentage of the gem width.  Emerald cuts: 60–75 %.
    pub table_pct:      f32,
    /// Nominal crown angle at the girdle, in degrees (emerald cuts ~ 30–40°).
    pub crown_angle:    f32,
    /// Nominal pavilion angle at the girdle, in degrees (~ 40–45°).
    pub pavilion_angle: f32,
    /// Corner chamfer length as a percentage of the smaller semi-axis.
    pub corner_pct:     f32,
    /// Facet rows per outline edge on the crown (>= 1).
    pub crown_steps:    u32,
    /// Facet rows per outline edge on the pavilion (>= 1).
    pub pavilion_steps: u32,
    /// Culet flat width as a percentage of the gem width (0 = pointed culet).
    pub culet_pct:      f32,
    /// Relative angle spread between the girdle row and the table row
    /// (0 = single slope, no visible steps).
    pub step_spread:    f32,
}

impl Default for StepCutParams {
    fn default() -> Self {
        Self {
            table_pct:      65.0,
            crown_angle:    35.0,
            pavilion_angle: 42.0,
            corner_pct:     12.0,
            crown_steps:    3,
            pavilion_steps: 3,
            culet_pct:      8.0,
            step_spread:    0.25,
        }
    }
}

/// Rectangular step cut ("emerald cut") as an analytic convex polyhedron.
///
/// The girdle (widest octagon) lies in the horizontal plane through `center`.
/// The table (flat top) faces +Y; the culet (flat bottom) faces −Y.
///
/// Recommended material: `SpectralDielectric` with the beryl constants
/// (cauchy_b ≈ 1.556, cauchy_c ≈ 0.0073) plus a Cr³⁺ absorption tint.
pub struct StepCut {
    planes:     Vec<(Vec3, f32)>,   // (outward unit normal n, offset d) — interior iff n·x ≤ d
    bbox:       Aabb,
    mat:        Arc<dyn Material>,
    center:     Point3,
    crown_h:    f32,
    pavilion_h: f32,
}

impl StepCut {
    /// `half_a` = semi-length along X, `half_b` = semi-width along Z.
    pub fn new(center: Point3, half_a: f32, half_b: f32, params: StepCutParams, mat: Arc<dyn Material>) -> Self {
        assert!(half_a > 0.0 && half_b > 0.0, "StepCut: half extents must be positive");
        let a = half_a;
        let b = half_b;
        let k    = (params.table_pct / 100.0).clamp(0.05, 0.95);        // table inset fraction of b
        let kc   = (params.culet_pct / 100.0).clamp(0.0, 0.5);          // culet inset fraction of b
        let c    = (params.corner_pct / 100.0 * a.min(b)).min(0.45 * a.min(b)); // chamfer length
        let s    = b * (1.0 - k);                                        // table inset (perpendicular)
        let sp   = b * (1.0 - kc);                                       // culet inset (perpendicular)
        let crown_h    = s  * params.crown_angle.to_radians().tan();
        let pavilion_h = sp * params.pavilion_angle.to_radians().tan();
        assert!(crown_h > 0.0 && pavilion_h > 0.0, "StepCut: angles must yield positive crown/pavilion height");

        // Girdle outline octagon, described per edge by (outward unit normal m
        // in the XZ plane, distance h of the edge line from the center).
        // Vertices (relative to center, XZ): (±(a−c), ±b) and (±a, ±(b−c)).
        let s2 = std::f32::consts::FRAC_1_SQRT_2;
        let hc = (a - c + b) * s2;   // chamfer edge distance
        let edges: [(Vec3, f32); 8] = [
            (Vec3::new( 1.0, 0.0,  0.0), a),
            (Vec3::new( 0.0, 0.0,  1.0), b),
            (Vec3::new(-1.0, 0.0,  0.0), a),
            (Vec3::new( 0.0, 0.0, -1.0), b),
            (Vec3::new( s2, 0.0,  s2), hc),
            (Vec3::new(-s2, 0.0,  s2), hc),
            (Vec3::new(-s2, 0.0, -s2), hc),
            (Vec3::new( s2, 0.0, -s2), hc),
        ];

        let cs = params.crown_steps.max(1);
        let ps = params.pavilion_steps.max(1);
        let spread = params.step_spread.clamp(0.0, 0.9);
        // Facet angle factor for row i (0 = at the girdle, steepest):
        // tan(theta_i) = tan(angle) * factor_i, factors symmetric about 1.
        let factor = |i: u32, rows: u32| -> f32 {
            if rows <= 1 { 1.0 } else { 1.0 + spread * (1.0 - 2.0 * i as f32 / (rows - 1) as f32) }
        };

        let mut planes: Vec<(Vec3, f32)> = Vec::with_capacity(2 + 8 * (cs + ps) as usize);

        for &(m, h) in &edges {
            // ── Crown steps: equal vertical rises, runs set by the row angle ──
            let dv = crown_h / cs as f32;
            let mut dist = h;                       // distance along m of the current profile point
            let mut y    = 0.0f32;
            for i in 0..cs {
                let dh = dv / (params.crown_angle.to_radians().tan() * factor(i, cs));
                let n = Vec3::new(m.x * dv, dh, m.z * dv).unit();
                let p = Vec3::new(center.x + m.x * dist, center.y + y, center.z + m.z * dist);
                planes.push((n, n.dot(p)));
                dist -= dh;
                y    += dv;
            }
            // ── Pavilion steps ──
            let dvp = pavilion_h / ps as f32;
            let mut dist = h;
            let mut y    = 0.0f32;
            for i in 0..ps {
                let dh = dvp / (params.pavilion_angle.to_radians().tan() * factor(i, ps));
                let n = Vec3::new(m.x * dvp, -dh, m.z * dvp).unit();
                let p = Vec3::new(center.x + m.x * dist, center.y + y, center.z + m.z * dist);
                planes.push((n, n.dot(p)));
                dist -= dh;
                y    -= dvp;
            }
        }

        // ── Table and culet caps ──────────────────────────────────────────────
        planes.push((Vec3::new(0.0,  1.0, 0.0),  center.y + crown_h));
        planes.push((Vec3::new(0.0, -1.0, 0.0), -(center.y - pavilion_h)));

        let bbox = Aabb::new(
            center - Vec3::new(a, pavilion_h, b),
            center + Vec3::new(a, crown_h,    b),
        );

        Self { planes, bbox, mat, center, crown_h, pavilion_h }
    }
}

impl Hittable for StepCut {

    fn hit(&self, r: &Ray, t_min: f32, t_max: f32) -> Option<HitRecord<'_>> {
        let (t, outward_normal) = convex_hit(&self.planes, r, t_min, t_max)?;

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::material::Dielectric;

    fn mat() -> Arc<dyn Material> { Arc::new(Dielectric { ir: 1.5 }) }

    fn params() -> StepCutParams {
        StepCutParams { table_pct: 50.0, crown_angle: 45.0, pavilion_angle: 45.0,
                        corner_pct: 20.0, ..Default::default() }   // steps 3+3, culet 8%, spread 0.25
    }

    #[test]
    fn plane_count_is_two_plus_eight_times_steps() {
        let s = StepCut::new(Point3::new(0.0, 0.0, 0.0), 2.0, 1.0, params(), mat());
        assert_eq!(s.planes.len(), 2 + 8 * (3 + 3));
        let s = StepCut::new(Point3::new(0.0, 0.0, 0.0), 2.0, 1.0,
            StepCutParams { crown_steps: 1, pavilion_steps: 2, ..params() }, mat());
        assert_eq!(s.planes.len(), 2 + 8 * (1 + 2));
    }

    #[test]
    fn bbox_matches_outline_and_heights() {
        // a=2, b=1, table 50% -> s = 0.5, crown 45° -> crown_h = 0.5
        // culet 8% -> sp = 0.92, pavilion 45° -> pavilion_h = 0.92
        let s = StepCut::new(Point3::new(0.0, 0.0, 0.0), 2.0, 1.0, params(), mat());
        let bb = s.bounding_box().unwrap();
        assert!((bb.min.x + 2.0).abs() < 1e-6 && (bb.max.x - 2.0).abs() < 1e-6);
        assert!((bb.min.z + 1.0).abs() < 1e-6 && (bb.max.z - 1.0).abs() < 1e-6);
        assert!((bb.max.y - 0.5).abs() < 1e-6 && (bb.min.y + 0.92).abs() < 1e-6);
    }

    #[test]
    fn horizontal_ray_hits_girdle_edge_at_semi_axis() {
        let s = StepCut::new(Point3::new(0.0, 0.0, 0.0), 2.0, 1.0, params(), mat());
        let r = Ray::new(Point3::new(10.0, 0.0, 0.0), Vec3::new(-1.0, 0.0, 0.0));
        let rec = s.hit(&r, 0.001, 100.0).expect("hit");
        assert!((rec.p.x - 2.0).abs() < 1e-4, "x = {}", rec.p.x);
        assert!(rec.p.y.abs() < 1e-4);
    }

    #[test]
    fn diagonal_ray_hits_chamfer_not_the_sharp_corner() {
        // a=2, b=1, corner 20% of b -> c = 0.2; chamfer plane: x + z = a - c + b = 2.8
        let s = StepCut::new(Point3::new(0.0, 0.0, 0.0), 2.0, 1.0, params(), mat());
        // Aim at the chamfer face midpoint (1.9, 0, 0.9).
        let target = Vec3::new(1.9, 0.0, 0.9);
        let r = Ray::new(Point3::new(10.0, 0.0, 5.0), (target - Vec3::new(10.0, 0.0, 5.0)).unit());
        let rec = s.hit(&r, 0.001, 100.0).expect("hit");
        assert!((rec.p.x + rec.p.z - 2.8).abs() < 1e-4, "x+z = {}", rec.p.x + rec.p.z);
        // The uncut corner (2, 1) would give x+z = 3 — must NOT be reached.
        assert!(rec.p.x + rec.p.z < 2.9);
        // A ray along the exact diagonal enters through the +Z side, not the chamfer.
        let r2 = Ray::new(Point3::new(10.0, 0.0, 10.0), Vec3::new(-1.0, 0.0, -1.0).unit());
        let rec2 = s.hit(&r2, 0.001, 100.0).expect("hit");
        assert!((rec2.p.z - 1.0).abs() < 1e-4, "z = {}", rec2.p.z);
    }

    #[test]
    fn ray_from_above_hits_table_with_up_normal() {
        let s = StepCut::new(Point3::new(0.0, 0.0, 0.0), 2.0, 1.0, params(), mat());
        let r = Ray::new(Point3::new(0.0, 10.0, 0.0), Vec3::new(0.0, -1.0, 0.0));
        let rec = s.hit(&r, 0.001, 100.0).expect("hit");
        assert!((rec.p.y - 0.5).abs() < 1e-5, "y = {}", rec.p.y);
        assert!((rec.normal.y - 1.0).abs() < 1e-5);
    }

    #[test]
    fn interior_ray_exits_through_side() {
        let s = StepCut::new(Point3::new(0.0, 0.0, 0.0), 2.0, 1.0, params(), mat());
        let r = Ray::new(Point3::new(0.0, 0.25, 0.0), Vec3::new(1.0, 0.0, 0.0));
        let rec = s.hit(&r, 0.001, 100.0).expect("exit hit");
        assert!(rec.p.x > 1.0, "exit x = {}", rec.p.x);
        // Exiting through the +X crown: back-face, so rec.normal points -X (toward the ray).
        assert!(!rec.front_face && rec.normal.x < 0.0, "got {:?} front={}", rec.normal, rec.front_face);
    }

    #[test]
    fn steps_make_visibly_different_planes() {
        // With spread > 0 the crown rows of one edge must have distinct normals.
        let s = StepCut::new(Point3::new(0.0, 0.0, 0.0), 2.0, 1.0, params(), mat());
        let crown_planes: Vec<&(Vec3, f32)> = s.planes.iter()
            .filter(|(n, _)| n.y > 0.3 && n.x > 0.5).collect();
        assert!(crown_planes.len() >= 3);
        let y0 = crown_planes[0].0.y;
        let y2 = crown_planes[2].0.y;
        assert!((y0 - y2).abs() > 0.05, "row normals should differ: {y0} vs {y2}");
    }
}
