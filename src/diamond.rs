use std::sync::Arc;
use std::f32::consts::PI;

use crate::aabb::Aabb;
use crate::hittable::{HitRecord, Hittable, Material};
use crate::ray::Ray;
use crate::vec3::{Point3, Vec3};

/// GIA-reportable proportion parameters for a round brilliant cut.
///
/// The three fields directly control the half-space normals of the analytic
/// polyhedron built by [`Diamond::new`].
#[derive(Clone, Debug)]
pub struct DiamondParams {
    /// Table facet diameter as a percentage of the girdle diameter.
    /// GIA ideal: 53–57 %. Tolkowsky: 53 %.
    pub table_pct:      f32,
    /// Crown main (bezel) facet angle from the girdle plane, in degrees.
    /// GIA ideal: 34–35 °. Tolkowsky: 34.5 °.
    pub crown_angle:    f32,
    /// Pavilion main facet angle from the girdle plane, in degrees.
    /// GIA ideal: 40.6–41.0 °. Tolkowsky: 40.75 °.
    pub pavilion_angle: f32,
    /// Girdle stretch along X (length / width).  1.0 = round brilliant;
    /// e.g. 1.40 for an 8.99 x 6.42 mm oval.  `radius` is the semi-MINOR axis.
    pub oval_ratio: f32,
    /// Total girdle segments N (rounded to even, min 4).  16 = classic round
    /// brilliant (8 crown + 8 pavilion mains).  Ovals read as polygons at 16:
    /// the girdle vertices lie exactly ON the ellipse (q*r*cos t, r*sin t),
    /// so the outline deviation is ~r*(pi/N)^2/2 — 1.9% of the radius at
    /// N=16, 0.12% at N=64, i.e. a visually perfect oval.
    pub girdle_facets: u32,
}

impl Default for DiamondParams {
    fn default() -> Self {
        Self { table_pct: 53.0, crown_angle: 34.5, pavilion_angle: 40.75, oval_ratio: 1.0, girdle_facets: 16 }
    }
}

/// Round brilliant diamond as an analytic convex polyhedron.
///
/// The girdle (widest circle) lies in the horizontal plane through `center`.
/// The table (flat top octagon) faces +Y; the culet (bottom point) faces −Y.
///
/// The (N+1) half-spaces (1 table + N/2 crown bezel + N/2 pavilion main, N =
/// `girdle_facets`, 16 by default) are arranged so crown and pavilion faces
/// alternate every 180°/N around the girdle.  Their pairwise intersections
/// form a regular N-gon with apothem `radius`, giving the girdle without any
/// explicit vertical girdle planes.
///
/// Recommended material: `SpectralDielectric { cauchy_b: 2.395, cauchy_c: 0.00585, .. }`.
pub struct Diamond {
    planes:     Vec<(Vec3, f32)>,   // (outward unit normal n, offset d) — interior iff n·x ≤ d
    bbox:       Aabb,
    mat:        Arc<dyn Material>,
    center:     Point3,
    crown_h:    f32,
    pavilion_h: f32,
}

impl Diamond {
    pub fn new(center: Point3, radius: f32, params: DiamondParams, mat: Arc<dyn Material>) -> Self {
        let r          = radius;
        let q          = params.oval_ratio.max(0.05);   // girdle stretch along X
        let table_r    = (params.table_pct / 100.0) * r;
        let crown_h    = (r - table_r) * params.crown_angle.to_radians().tan();
        let pavilion_h = r             * params.pavilion_angle.to_radians().tan();

        let n = (params.girdle_facets.max(4) / 2) * 2;  // even, >= 4
        let m = n / 2;                                   // main facets per side
        let mut planes: Vec<(Vec3, f32)> = Vec::with_capacity((n + 1) as usize);

        // Push a facet plane given its ROUND-space unit normal `n` and a point
        // `g` on it: the oval shape is the round shape scaled by S = diag(q,1,1),
        // which maps plane (n, d) to (unit(S^-1 n), unit(S^-1 n) . (S g)).
        let mut push_plane = |n: Vec3, g: Vec3| {
            let nw = Vec3::new(n.x / q, n.y, n.z).unit();
            let gw = Vec3::new(q * g.x, g.y, g.z);
            planes.push((nw, nw.dot(gw)));
        };

        // ── Table ──────────────────────────────────────────────────────────────
        push_plane(Vec3::new(0.0, 1.0, 0.0), Vec3::new(center.x, center.y + crown_h, center.z));

        // ── Crown main facets (8, bezel facets) ───────────────────────────────
        // At azimuth θ = k · 45°.  The face runs from the girdle edge (r, 0) to
        // the table edge (table_r, crown_h) in the radial-Y plane, so:
        //   unnorm outward normal = (cos θ · crown_h,  r − table_r,  sin θ · crown_h)
        for k in 0..m {
            let theta = k as f32 * 2.0 * PI / m as f32;
            let (s, c) = theta.sin_cos();
            let n = Vec3::new(c * crown_h, r - table_r, s * crown_h).unit();
            // Face passes through the girdle edge at this azimuth.
            let girdle_pt = Vec3::new(center.x + c * r, center.y, center.z + s * r);
            push_plane(n, girdle_pt);
        }

        // ── Pavilion main facets (8) ──────────────────────────────────────────
        // Offset half a segment from crown so the girdle is a uniform N-gon.
        // Face runs from girdle (r, 0) down to the culet (0, −pavilion_h):
        //   unnorm outward normal = (cos φ · pavilion_h,  −r,  sin φ · pavilion_h)
        for k in 0..m {
            let phi = k as f32 * 2.0 * PI / m as f32 + PI / m as f32;
            let (s, c) = phi.sin_cos();
            let n = Vec3::new(c * pavilion_h, -r, s * pavilion_h).unit();
            let girdle_pt = Vec3::new(center.x + c * r, center.y, center.z + s * r);
            push_plane(n, girdle_pt);
        }

        // ── Bounding box ───────────────────────────────────────────────────────
        // Circumradius of the N-gon girdle = r / cos(π/N).
        let circ_r = r / (PI / n as f32).cos();
        let bbox = Aabb::new(
            center - Vec3::new(circ_r * q, pavilion_h, circ_r),
            center + Vec3::new(circ_r * q, crown_h,    circ_r),
        );

        Self { planes, bbox, mat, center, crown_h, pavilion_h }
    }

}

/// Slab-test a ray against a convex polyhedron given as outward half-spaces
/// `(n, d)` (interior iff `n.x <= d`).  Returns `(t, outward_normal)` of the
/// surface crossing: `t_enter` for exterior rays, `t_exit` for interior rays
/// (e.g. after refraction or TIR inside the solid).
pub(crate) fn convex_hit(planes: &[(Vec3, f32)], r: &Ray, t_min: f32, t_max: f32) -> Option<(f32, Vec3)> {
        // Convex polyhedron intersection: maintain the interval [t_enter, t_exit]
        // during which the ray is inside ALL half-spaces simultaneously.
        let mut t_enter              = t_min;
        let mut t_exit               = t_max;
        let mut enter_normal         = Vec3::new(0.0, 1.0, 0.0);
        let mut exit_outward_normal  = Vec3::new(0.0, 1.0, 0.0);
        let mut entry_found          = false;

        for &(n, d) in planes {
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

            if t_enter > t_exit { return None; }
        }

        if t_enter > t_exit { return None; }
        // Zero-chord interval (t_enter == t_exit): reject interior tangents — in
        // f32 they are rounding slivers around edges/vertices, not real hits, and
        // accepting them measurably slows gem-heavy scenes — but accept the closed
        // endpoint t == t_max per the Hittable convention.
        if t_enter == t_exit && t_enter < t_max { return None; }

        if entry_found {
            Some((t_enter, enter_normal))
        } else {
            if t_exit > t_max { return None; }
            Some((t_exit, exit_outward_normal))
        }
}

impl Hittable for Diamond {

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
    fn any_hit(&self, r: &Ray, t_min: f32, t_max: f32) -> bool {
        // Occlusion test: the slab test alone decides; skip HitRecord + UVs.
        convex_hit(&self.planes, r, t_min, t_max).is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::material::Dielectric;

    fn mat() -> Arc<dyn Material> { Arc::new(Dielectric { ir: 1.5 }) }

    #[test]
    fn convex_hit_accepts_entry_at_exactly_t_max() {
        // Closed-interval convention: entering the solid at t == t_max is a
        // hit (slab x in [-1, 1]: ray along +x from x=-5 enters at t=4).
        let planes = [(Vec3::new(1.0, 0.0, 0.0), 1.0), (Vec3::new(-1.0, 0.0, 0.0), 1.0)];
        let r = Ray::new(Point3::new(-5.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0));
        let hit = convex_hit(&planes, &r, 0.001, 4.0).expect("entry at t == t_max must be accepted");
        assert!((hit.0 - 4.0).abs() < 1e-6, "t = {}", hit.0);
        assert!(convex_hit(&planes, &r, 0.001, 4.0 - 1e-4).is_none(), "entry just above t_max must miss");
    }

    #[test]
    fn girdle_facets_scales_plane_count() {
        let d = Diamond::new(Point3::new(0.0, 0.0, 0.0), 1.0,
            DiamondParams { girdle_facets: 64, ..Default::default() }, mat());
        assert_eq!(d.planes.len(), 65);   // 1 table + 32 crown + 32 pavilion
        let d = Diamond::new(Point3::new(0.0, 0.0, 0.0), 1.0, Default::default(), mat());
        assert_eq!(d.planes.len(), 17);   // classic: 1 + 8 + 8
    }

    #[test]
    fn odd_facet_count_rounds_down_to_even() {
        let d = Diamond::new(Point3::new(0.0, 0.0, 0.0), 1.0,
            DiamondParams { girdle_facets: 17, ..Default::default() }, mat());
        assert_eq!(d.planes.len(), 17);   // 17 -> 16 -> 1 + 8 + 8
    }

    #[test]
    fn oval_girdle_extreme_x_is_q_times_radius() {
        // Horizontal ray from +X hits the girdle where x = q*r (the girdle
        // polygon touches the ellipse x^2/(q r)^2 + z^2/r^2 = 1 at theta=0).
        let q = 1.4;
        for n in [16u32, 64] {
            let d = Diamond::new(Point3::new(0.0, 0.0, 0.0), 1.0,
                DiamondParams { oval_ratio: q, girdle_facets: n, ..Default::default() }, mat());
            let r = Ray::new(Point3::new(10.0, 0.0, 0.0), Vec3::new(-1.0, 0.0, 0.0));
            let rec = d.hit(&r, 0.001, 100.0).expect("hit");
            assert!((rec.p.x - q).abs() < 1e-4, "n={n} x={}", rec.p.x);
        }
    }

    #[test]
    fn any_hit_agrees_with_hit() {
        let d = Diamond::new(Point3::new(0.0, 0.0, 0.0), 1.0,
            DiamondParams { oval_ratio: 1.4, girdle_facets: 64, ..Default::default() }, mat());
        let rays = [
            (0.0, 10.0, 0.0, 0.0, -1.0, 0.0),    // table from above
            (10.0, 0.0, 0.0, -1.0, 0.0, 0.0),    // girdle edge +X
            (10.0, 0.0, 10.0, -1.0, 0.0, -1.0),  // diagonal
            (0.0, 0.3, 0.0, 1.0, 0.0, 0.0),      // interior exit
            (0.0, 10.0, 3.0, 0.0, -1.0, 0.0),    // miss (outside girdle)
            (0.0, -10.0, 0.0, 0.0, 1.0, 0.0),    // culet from below
        ];
        for (ox, oy, oz, dx, dy, dz) in rays {
            let r = Ray::new(Point3::new(ox, oy, oz), Vec3::new(dx, dy, dz));
            assert_eq!(
                d.hit(&r, 0.001, 100.0).is_some(),
                d.any_hit(&r, 0.001, 100.0),
                "mismatch for ray ({ox},{oy},{oz}) -> ({dx},{dy},{dz})"
            );
        }
    }

    #[test]
    fn round_default_still_hits_table_from_above() {
        let d = Diamond::new(Point3::new(0.0, 0.0, 0.0), 1.0, Default::default(), mat());
        let r = Ray::new(Point3::new(0.0, 10.0, 0.0), Vec3::new(0.0, -1.0, 0.0));
        let rec = d.hit(&r, 0.001, 100.0).expect("hit");
        assert!(rec.p.y > 0.0 && rec.p.y < 10.0);
        assert!((rec.normal.y - 1.0).abs() < 1e-5, "table normal should be +Y");
    }
}
