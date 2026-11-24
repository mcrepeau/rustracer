use std::sync::Arc;
use std::f32::consts::PI;

use crate::aabb::Aabb;
use crate::diamond::convex_hit;
use crate::hittable::{HitRecord, Hittable, Material};
use crate::ray::Ray;
use crate::vec3::{Point3, Vec3};

/// Girdle outline of a step cut, in the XZ plane relative to the gem centre.
/// Vertices are generated from the semi-extents `a` (half length, X) and
/// `b` (half width, Z); table/culet percentages are relative to the width `b`.
#[derive(Clone, Debug)]
pub enum StepOutline {
    /// Rectangle, optionally with 45° corner chamfers (emerald / princess /
    /// baguette / asscher family).  `corner_pct` = 0 gives sharp corners.
    Rect { corner_pct: f32 },
    /// Rectangle with rounded corners (cushion cut): each corner is a quarter
    /// arc of radius `corner_pct`% of the smaller semi-axis, `segs` segments.
    RoundedRect { corner_pct: f32, segs: u32 },
    /// Triangle (trillion cut).
    Triangle,
    /// Hexagon.
    Hexagon,
    /// Pointed oval (marquise / pear): an ellipse pinched at the ends via a
    /// |cos|^p profile (p < 1 -> pointed tips).  `pear` keeps one end round
    /// (p = 1) and pinches the other.
    PointedOval { segs: u32, pear: bool },
    /// Explicit convex polygon, XZ coords relative to the centre.
    Polygon(Vec<[f32; 2]>),
}

impl Default for StepOutline {
    fn default() -> Self { StepOutline::Rect { corner_pct: 12.0 } }
}

impl StepOutline {
    /// Outline vertices (XZ, relative to centre), sorted counterclockwise.
    fn vertices(&self, a: f32, b: f32) -> Vec<(f32, f32)> {
        let mut v: Vec<(f32, f32)> = match *self {
            StepOutline::Rect { corner_pct } => {
                let c = (corner_pct / 100.0 * a.min(b)).min(0.45 * a.min(b));
                if c <= 1e-6 {
                    vec![(a, b), (a, -b), (-a, -b), (-a, b)]
                } else {
                    vec![
                        (a - c, b), (a, b - c), (a, -(b - c)), (a - c, -b),
                        (-(a - c), -b), (-a, -(b - c)), (-a, b - c), (-(a - c), b),
                    ]
                }
            }
            StepOutline::RoundedRect { corner_pct, segs } => {
                let r = (corner_pct / 100.0 * a.min(b)).min(0.45 * a.min(b));
                let segs = segs.max(1);
                let mut v = Vec::with_capacity(4 * segs as usize);
                for &sx in &[1.0f32, -1.0] {
                    for &sz in &[1.0f32, -1.0] {
                        let ccx = sx * (a - r);
                        let ccz = sz * (b - r);
                        for j in 0..=segs {
                            let ang = (j as f32 / segs as f32) * std::f32::consts::FRAC_PI_2;
                            v.push((ccx + sx * r * ang.cos(), ccz + sz * r * ang.sin()));
                        }
                    }
                }
                v
            }
            StepOutline::Triangle => (0..3)
                .map(|k| {
                    let th = std::f32::consts::FRAC_PI_2 + k as f32 * 2.0 * PI / 3.0;
                    (a * th.cos(), b * th.sin())
                })
                .collect(),
            StepOutline::Hexagon => (0..6)
                .map(|k| {
                    let th = k as f32 * PI / 3.0;
                    (a * th.cos(), b * th.sin())
                })
                .collect(),
            StepOutline::PointedOval { segs, pear } => {
                let n = (segs.max(4) * 2) as usize;
                (0..n)
                    .map(|i| {
                        let th = 2.0 * PI * i as f32 / n as f32;
                        let c = th.cos();
                        let s = th.sin();
                        let p = if pear {
                            if c >= 0.0 { 0.6 } else { 1.0 }   // pointed +X end, round -X end
                        } else { 0.65 };
                        (a * c.signum() * c.abs().powf(p), b * s)
                    })
                    .collect()
            }
            StepOutline::Polygon(ref pts) => pts.iter().map(|&[x, z]| (x, z)).collect(),
        };
        // Sort by angle around the centroid -> convex outlines end up CCW.
        let cx = v.iter().map(|p| p.0).sum::<f32>() / v.len() as f32;
        let cz = v.iter().map(|p| p.1).sum::<f32>() / v.len() as f32;
        v.sort_by(|p, q| {
            let ap = (p.1 - cz).atan2(p.0 - cx);
            let aq = (q.1 - cz).atan2(q.0 - cx);
            ap.partial_cmp(&aq).unwrap_or(std::cmp::Ordering::Equal)
        });
        v
    }
}

/// Proportion parameters for a step-family cut.
///
/// Crown and pavilion each consist of `crown_steps` / `pavilion_steps` planes
/// per outline edge; the facet angles spread linearly from
/// `(1 + step_spread) * angle` at the girdle to `(1 - step_spread) * angle` at
/// the table/culet, which is what produces the visible steps.  `chevrons`
/// adds that many diagonal planes per outline VERTEX, along the supporting
/// line at the vertex (the angle-bisector perpendicular; for a square corner
/// (+a, +a) that is x + z = 2a), rising toward the
/// table/culet — the princess-cut chevron facets.  The table and culet are the
/// outline inset by a constant margin along each edge normal (flat caps).
#[derive(Clone, Debug)]
pub struct StepCutParams {
    /// Table width as a percentage of the gem width.  Emerald cuts: 60–75 %.
    /// Exact: row runs are normalized so the table inset equals b*(1 - table_pct/100).
    pub table_pct:      f32,
    /// Nominal crown angle at the girdle, in degrees (emerald cuts ~ 30–40°).
    pub crown_angle:    f32,
    /// Nominal pavilion angle at the girdle, in degrees (~ 40–45°).
    pub pavilion_angle: f32,
    /// Girdle outline (see [`StepOutline`]).
    pub outline:        StepOutline,
    /// Facet rows per outline edge on the crown (>= 1).
    pub crown_steps:    u32,
    /// Facet rows per outline edge on the pavilion (>= 1).
    pub pavilion_steps: u32,
    /// Culet flat width as a percentage of the gem width (0 = pointed culet).
    pub culet_pct:      f32,
    /// Per-facet micro-tilt amplitude in radians (polish marks); 0 = optical-perfect planes.
    pub polish:         f32,
    /// Relative angle spread between the girdle row and the table row
    /// (0 = single slope, no visible steps).
    pub step_spread:    f32,
    /// Diagonal chevron rows per outline vertex (0 = none).
    pub chevrons:       u32,
    /// Girdle band thickness as a percentage of the gem width (minor axis);
    /// GIA-style: thin ~1.5%, medium ~3%.  0 = knife edge.
    pub girdle_pct:     f32,
}

impl Default for StepCutParams {
    fn default() -> Self {
        Self {
            table_pct:      65.0,
            crown_angle:    35.0,
            pavilion_angle: 42.0,
            outline:        StepOutline::default(),
            crown_steps:    3,
            pavilion_steps: 3,
            culet_pct:      8.0,
            step_spread:    0.25,
            chevrons:       0,
            girdle_pct:     0.0,
            polish:         0.0,
        }
    }
}

/// Step-family cut (emerald / princess / cushion / trillion / marquise / ...)
/// as an analytic convex polyhedron.
///
/// The girdle (widest outline) lies in the horizontal plane through `center`.
/// The table (flat top) faces +Y; the culet (flat bottom) faces −Y.
///
/// Recommended material: `SpectralDielectric` with a gem preset
/// (diamond/sapphire/ruby/emerald constants plus a body-colour absorption).
pub struct StepCut {
    planes:     Vec<(Vec3, f32)>,   // (outward unit normal n, offset d) — interior iff n·x ≤ d
    bbox:       Aabb,
    mat:        Arc<dyn Material>,
    center:     Point3,
    crown_h:    f32,
    pavilion_h: f32,
    girdle_h:   f32,
    polish:      f32,
}

#[inline]
fn norm2(x: f32, z: f32) -> (f32, f32) {
    let l = (x * x + z * z).sqrt();
    if l > 1e-12 { (x / l, z / l) } else { (0.0, 0.0) }
}

/// Dedupe adjacent (incl. wrap-around) vertices, then require >= 3 distinct
/// vertices, a minimum edge length, non-zero area and convexity (all
/// consecutive edge cross products share the sign of the signed area).
fn validate_outline(verts: &mut Vec<(f32, f32)>, scale: f32) -> Result<(), String> {
    let eps = 1e-6 * scale;
    let mut out: Vec<(f32, f32)> = Vec::with_capacity(verts.len());
    for &p in verts.iter() {
        if let Some(&last) = out.last() {
            if (p.0 - last.0).hypot(p.1 - last.1) < eps { continue; }
        }
        out.push(p);
    }
    if out.len() > 1 {
        let first = out[0];
        let last = *out.last().unwrap();
        if (first.0 - last.0).hypot(first.1 - last.1) < eps { out.pop(); }
    }
    *verts = out;
    let n = verts.len();
    if n < 3 { return Err("outline needs at least 3 distinct vertices".into()); }
    for i in 0..n {
        let p = verts[i];
        let q = verts[(i + 1) % n];
        if (p.0 - q.0).hypot(p.1 - q.1) < eps {
            return Err("outline has a zero-length edge".into());
        }
    }
    let area2: f32 = (0..n)
        .map(|i| { let p = verts[i]; let q = verts[(i + 1) % n]; p.0 * q.1 - q.0 * p.1 })
        .sum();
    if area2.abs() < eps * eps { return Err("outline has zero area".into()); }
    for i in 0..n {
        let p = verts[i];
        let q = verts[(i + 1) % n];
        let r = verts[(i + 2) % n];
        let cross = (q.0 - p.0) * (r.1 - q.1) - (q.1 - p.1) * (r.0 - q.0);
        if cross * area2 < 0.0 { return Err("polygon outline must be convex".into()); }
    }
    Ok(())
}

impl StepCut {
    /// `half_a` = semi-length along X, `half_b` = semi-width along Z.
    /// Returns Err (instead of panicking) on degenerate input so TOML errors
    /// surface cleanly.
    pub fn new(center: Point3, half_a: f32, half_b: f32, params: StepCutParams, mat: Arc<dyn Material>)
        -> Result<Self, String>
    {
        if !(half_a > 0.0 && half_b > 0.0) || !half_a.is_finite() || !half_b.is_finite() {
            return Err("StepCut: half extents must be positive and finite".into());
        }
        let a = half_a;
        let b = half_b;
        let k    = (params.table_pct / 100.0).clamp(0.05, 0.95);        // table inset fraction of b
        let kc   = (params.culet_pct / 100.0).clamp(0.0, 0.5);          // culet inset fraction of b
        let s    = b * (1.0 - k);                                        // table inset (perpendicular)
        let sp   = b * (1.0 - kc);                                       // culet inset (perpendicular)
        let crown_h    = s  * params.crown_angle.to_radians().tan();
        let pavilion_h = sp * params.pavilion_angle.to_radians().tan();
        // Girdle band height: pct of the width (2b), GIA-style.
        let girdle_h   = (params.girdle_pct / 100.0 * 2.0 * b).clamp(0.0, 0.3 * b);

        if !(params.girdle_pct.is_finite() && params.girdle_pct >= 0.0 && params.girdle_pct < 100.0) {
            return Err(format!("StepCut: girdle_pct must be in [0, 100), got {}", params.girdle_pct));
        }

        if !(crown_h > 0.0 && pavilion_h > 0.0) {
            return Err(format!("StepCut: crown/pavilion angles must be in (0, 90) degrees, got {} / {}",
                               params.crown_angle, params.pavilion_angle));
        }

        // ── Girdle outline -> edges (outward unit normal m in XZ, distance h) ─
        let mut verts = params.outline.vertices(a, b);
        validate_outline(&mut verts, a.max(b))?;
        let nv = verts.len();
        let gcx = verts.iter().map(|p| p.0).sum::<f32>() / nv as f32;
        let gcz = verts.iter().map(|p| p.1).sum::<f32>() / nv as f32;
        let mut edges: Vec<(Vec3, f32)> = Vec::with_capacity(nv);
        for i in 0..nv {
            let p = verts[i];
            let q = verts[(i + 1) % nv];
            let e = (q.0 - p.0, q.1 - p.1);
            let mut m = norm2(e.1, -e.0);
            // Orient outward: away from the outline centroid.
            if m.0 * (p.0 - gcx) + m.1 * (p.1 - gcz) < 0.0 { m = (-m.0, -m.1); }
            edges.push((Vec3::new(m.0, 0.0, m.1), m.0 * p.0 + m.1 * p.1));
        }

        let cs = params.crown_steps.max(1);
        let ps = params.pavilion_steps.max(1);
        let spread = params.step_spread.clamp(0.0, 0.9);
        // Facet angle factor for row i (0 = at the girdle, steepest):
        // tan(theta_i) = tan(angle) * factor_i, factors symmetric about 1.
        let factor = |i: u32, rows: u32| -> f32 {
            if rows <= 1 { 1.0 } else { 1.0 + spread * (1.0 - 2.0 * i as f32 / (rows - 1) as f32) }
        };

        let ch = params.chevrons;
        let mut planes: Vec<(Vec3, f32)> =
            Vec::with_capacity(2 + edges.len() * (cs + ps) as usize + 2 * nv * ch as usize + nv);

        // Row runs are normalized so the total inset is exactly s (table) / sp
        // (culet): with step_spread > 0 the raw sum of dh differs from s, which
        // used to make the table ~4% smaller than table_pct requested.
        let tan_c = params.crown_angle.to_radians().tan();
        let tan_p = params.pavilion_angle.to_radians().tan();
        let dv  = crown_h / cs as f32;
        let dvp = pavilion_h / ps as f32;
        let sum_c: f32 = (0..cs).map(|i| dv  / (tan_c * factor(i, cs))).sum();
        let sum_p: f32 = (0..ps).map(|i| dvp / (tan_p * factor(i, ps))).sum();
        let norm_c = s / sum_c;
        let norm_p = sp / sum_p;

        for &(m, h) in &edges {
            // ── Crown steps: equal vertical rises, runs set by the row angle ──
            if girdle_h > 0.0 {
                // vertical girdle band face through this outline edge
                planes.push((Vec3::new(m.x, 0.0, m.z), m.x * center.x + m.z * center.z + h));
            }
            let mut dist = h;                       // distance along m of the current profile point
            let mut y    = 0.5f32 * girdle_h;

            for i in 0..cs {
                let dh = dv / (tan_c * factor(i, cs)) * norm_c;
                let n = Vec3::new(m.x * dv, dh, m.z * dv).unit();
                let p = Vec3::new(center.x + m.x * dist, center.y + y, center.z + m.z * dist);
                planes.push((n, n.dot(p)));
                dist -= dh;
                y    += dv;
            }
            // ── Pavilion steps ──
            let mut dist = h;
            let mut y    = -0.5f32 * girdle_h;

            for i in 0..ps {
                let dh = dvp / (tan_p * factor(i, ps)) * norm_p;
                let n = Vec3::new(m.x * dvp, -dh, m.z * dvp).unit();
                let p = Vec3::new(center.x + m.x * dist, center.y + y, center.z + m.z * dist);
                planes.push((n, n.dot(p)));
                dist -= dh;
                y    -= dvp;
            }
        }

        // ── Chevron facets: diagonal planes tangent at each outline vertex ────
        for i in 0..nv {
            let v  = verts[i];
            let vp = verts[(i + nv - 1) % nv];
            let vn = verts[(i + 1) % nv];
            let u1 = norm2(vp.0 - v.0, vp.1 - v.1);
            let u2 = norm2(vn.0 - v.0, vn.1 - v.1);
            let bis = (u1.0 + u2.0, u1.1 + u2.1);
            if bis.0 * bis.0 + bis.1 * bis.1 < 1e-12 { continue; }   // collinear vertex
            // Supporting-line direction at v: perpendicular to the angle bisector.
            let t  = (-bis.1, bis.0);
            let c3 = Vec3::new(center.x + v.0, center.y, center.z + v.1);
            let d3 = Vec3::new(t.0, 0.0, t.1);
            let tc = Vec3::new(center.x + v.0 * k,  center.y + 0.5 * girdle_h + crown_h,    center.z + v.1 * k);
            let cc = Vec3::new(center.x + v.0 * kc, center.y - 0.5 * girdle_h - pavilion_h, center.z + v.1 * kc);
            for j in 0..ch {
                let f = (j + 1) as f32 / ch as f32;
                for p in [c3 + (tc - c3) * f, c3 + (cc - c3) * f] {
                    let mut n = d3.cross(p - c3).unit();
                    if n.dot(center) > n.dot(c3) { n = -n; }   // keep interior on the <= side
                    planes.push((n, n.dot(c3)));
                }
            }
        }

        // ── Table and culet caps ──────────────────────────────────────────────
        planes.push((Vec3::new(0.0,  1.0, 0.0),  center.y + 0.5 * girdle_h + crown_h));
        planes.push((Vec3::new(0.0, -1.0, 0.0), -(center.y - 0.5 * girdle_h - pavilion_h)));

        // Exact bbox in X/Z from the outline vertices.  In Y the cap planes at
        // +/- (girdle_h/2 + crown_h/pavilion_h) always bound the solid (they are
        // real planes), so the unclamped heights stay correct even when the
        // table/culet inset exceeds the outline apothem (pointed culet case).

        let minx = verts.iter().map(|v| v.0).fold(f32::INFINITY, f32::min);
        let maxx = verts.iter().map(|v| v.0).fold(f32::NEG_INFINITY, f32::max);
        let minz = verts.iter().map(|v| v.1).fold(f32::INFINITY, f32::min);
        let maxz = verts.iter().map(|v| v.1).fold(f32::NEG_INFINITY, f32::max);
        let top    = 0.5 * girdle_h + s  * tan_c;
        let bottom = 0.5 * girdle_h + sp * tan_p;

        let bbox = Aabb::new(
            Point3::new(center.x + minx, center.y - bottom, center.z + minz),
            Point3::new(center.x + maxx, center.y + top,    center.z + maxz),
        );

        let polish = if params.polish.is_finite() { params.polish.clamp(0.0, 0.25) } else { 0.0 };
        Ok(Self { planes, bbox, mat, center, crown_h, pavilion_h, girdle_h, polish })
    }
}

impl Hittable for StepCut {

    fn hit(&self, r: &Ray, t_min: f32, t_max: f32) -> Option<HitRecord<'_>> {
        let (t, outward_normal) = convex_hit(&self.planes, r, t_min, t_max)?;

        let p   = r.at(t);
        let rel = p - self.center;
        // Cylindrical UV: u = azimuth (0–1), v = height bottom-to-top (0–1).
        let u = ((-rel.z).atan2(rel.x) + PI) / (2.0 * PI);
        let v = ((rel.y + 0.5 * self.girdle_h + self.pavilion_h) / (self.crown_h + self.pavilion_h + self.girdle_h)).clamp(0.0, 1.0);

        let mut rec = HitRecord::new(p, t, &*self.mat, r, outward_normal);
        if self.polish > 0.0 {
            let facet = self.planes.iter().position(|(n, _)| (n.x - outward_normal.x).abs() < 1e-6 && (n.y - outward_normal.y).abs() < 1e-6 && (n.z - outward_normal.z).abs() < 1e-6).unwrap_or(0);
            rec.normal = crate::hittable::micro_tilt(facet, self.polish, rec.normal);
        }
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

    fn params() -> StepCutParams {
        StepCutParams { table_pct: 50.0, crown_angle: 45.0, pavilion_angle: 45.0,
                        outline: StepOutline::Rect { corner_pct: 20.0 },
                        ..Default::default() }   // steps 3+3, culet 8%, spread 0.25
    }

    #[test]
    fn plane_count_is_caps_plus_edges_times_steps_plus_vertex_chevrons() {
        let s = StepCut::new(Point3::new(0.0, 0.0, 0.0), 2.0, 1.0, params(), mat()).unwrap();
        assert_eq!(s.planes.len(), 2 + 8 * (3 + 3));               // octagon outline
        let s = StepCut::new(Point3::new(0.0, 0.0, 0.0), 2.0, 1.0,
            StepCutParams { crown_steps: 1, pavilion_steps: 2, ..params() }, mat()).unwrap();
        assert_eq!(s.planes.len(), 2 + 8 * (1 + 2));
        // triangle outline: 3 edges, 3 vertices
        let s = StepCut::new(Point3::new(0.0, 0.0, 0.0), 2.0, 1.0,
            StepCutParams { outline: StepOutline::Triangle, chevrons: 2, ..params() }, mat()).unwrap();
        assert_eq!(s.planes.len(), 2 + 3 * (3 + 3) + 2 * 3 * 2);
    }

    #[test]
    fn bbox_matches_outline_and_heights() {
        // a=2, b=1, table 50% -> s = 0.5, crown 45° -> crown_h = 0.5
        // culet 8% -> sp = 0.92, pavilion 45° -> pavilion_h = 0.92
        let s = StepCut::new(Point3::new(0.0, 0.0, 0.0), 2.0, 1.0, params(), mat()).unwrap();
        let bb = s.bounding_box().unwrap();
        assert!((bb.min.x + 2.0).abs() < 1e-6 && (bb.max.x - 2.0).abs() < 1e-6);
        assert!((bb.min.z + 1.0).abs() < 1e-6 && (bb.max.z - 1.0).abs() < 1e-6);
        assert!((bb.max.y - 0.5).abs() < 1e-6 && (bb.min.y + 0.92).abs() < 1e-6);
    }

    #[test]
    fn girdle_band_adds_vertical_faces_and_height() {
        // girdle_pct = 10% of width (2b = 2) -> band height 0.2
        let p = StepCutParams { girdle_pct: 10.0, ..params() };
        let s = StepCut::new(Point3::new(0.0, 0.0, 0.0), 2.0, 1.0, p, mat()).unwrap();
        assert_eq!(s.planes.len(), 2 + 8 + 8 * (3 + 3));   // +8 vertical band planes
        let r = Ray::new(Point3::new(10.0, 0.0, 0.0), Vec3::new(-1.0, 0.0, 0.0));
        let rec = s.hit(&r, 0.001, 100.0).expect("hit");
        assert!((rec.p.x - 2.0).abs() < 1e-4 && rec.normal.y.abs() < 1e-6,
                "band face: x={}, n.y={}", rec.p.x, rec.normal.y);
        let bb = s.bounding_box().unwrap();
        // crown_h = 0.5*tan45 = 0.5, pavilion_h = 0.92*tan45 = 0.92, band = 0.2
        assert!((bb.max.y - 0.6).abs() < 1e-4, "max.y = {}", bb.max.y);
        assert!((bb.min.y + 1.02).abs() < 1e-4, "min.y = {}", bb.min.y);
    }

    #[test]
    fn horizontal_ray_hits_girdle_edge_at_semi_axis() {

        let s = StepCut::new(Point3::new(0.0, 0.0, 0.0), 2.0, 1.0, params(), mat()).unwrap();
        let r = Ray::new(Point3::new(10.0, 0.0, 0.0), Vec3::new(-1.0, 0.0, 0.0));
        let rec = s.hit(&r, 0.001, 100.0).expect("hit");
        assert!((rec.p.x - 2.0).abs() < 1e-4, "x = {}", rec.p.x);
        assert!(rec.p.y.abs() < 1e-4);
    }

    #[test]
    fn diagonal_ray_hits_chamfer_not_the_sharp_corner() {
        // a=2, b=1, corner 20% of b -> c = 0.2; chamfer plane: x + z = a - c + b = 2.8
        let s = StepCut::new(Point3::new(0.0, 0.0, 0.0), 2.0, 1.0, params(), mat()).unwrap();
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
    fn zero_corner_pct_gives_sharp_square_corners() {
        // corner_pct = 0 -> 4 outline vertices (no chamfer planes)
        let p = StepCutParams { outline: StepOutline::Rect { corner_pct: 0.0 }, ..params() };
        let s = StepCut::new(Point3::new(0.0, 0.0, 0.0), 1.0, 1.0, p, mat()).unwrap();
        assert_eq!(s.planes.len(), 2 + 4 * (3 + 3));
        // Diagonal ray reaches the sharp corner (1, 1): x + z = a + b = 2.
        let r = Ray::new(Point3::new(10.0, 0.0, 10.0), Vec3::new(-1.0, 0.0, -1.0).unit());
        let rec = s.hit(&r, 0.001, 100.0).expect("hit");
        assert!((rec.p.x + rec.p.z - 2.0).abs() < 1e-4, "x+z = {}", rec.p.x + rec.p.z);
    }

    #[test]
    fn ray_from_above_hits_table_with_up_normal() {
        let s = StepCut::new(Point3::new(0.0, 0.0, 0.0), 2.0, 1.0, params(), mat()).unwrap();
        let r = Ray::new(Point3::new(0.0, 10.0, 0.0), Vec3::new(0.0, -1.0, 0.0));
        let rec = s.hit(&r, 0.001, 100.0).expect("hit");
        assert!((rec.p.y - 0.5).abs() < 1e-5, "y = {}", rec.p.y);
        assert!((rec.normal.y - 1.0).abs() < 1e-5);
    }

    #[test]
    fn interior_ray_exits_through_side() {
        let s = StepCut::new(Point3::new(0.0, 0.0, 0.0), 2.0, 1.0, params(), mat()).unwrap();
        let r = Ray::new(Point3::new(0.0, 0.25, 0.0), Vec3::new(1.0, 0.0, 0.0));
        let rec = s.hit(&r, 0.001, 100.0).expect("exit hit");
        assert!(rec.p.x > 1.0, "exit x = {}", rec.p.x);
        // Exiting through the +X crown: back-face, so rec.normal points -X (toward the ray).
        assert!(!rec.front_face && rec.normal.x < 0.0, "got {:?} front={}", rec.normal, rec.front_face);
    }

    #[test]
    fn chevron_planes_keep_outline_and_bbox() {
        // Square 1x1 (a=b=1), corner_pct=0, chevrons=2 -> 4 vertices x 2 rows x 2 sides.
        let p = StepCutParams { outline: StepOutline::Rect { corner_pct: 0.0 }, chevrons: 2, ..params() };
        let s = StepCut::new(Point3::new(0.0, 0.0, 0.0), 1.0, 1.0, p, mat()).unwrap();
        assert_eq!(s.planes.len(), 2 + 4 * (3 + 3) + 8 * 2);
        // Girdle outline unchanged: diagonal ray still reaches the sharp corner.
        let r = Ray::new(Point3::new(10.0, 0.0, 10.0), Vec3::new(-1.0, 0.0, -1.0).unit());
        let rec = s.hit(&r, 0.001, 100.0).expect("hit");
        assert!((rec.p.x + rec.p.z - 2.0).abs() < 1e-4, "x+z = {}", rec.p.x + rec.p.z);
        // bbox unchanged (chevron planes stay inside the outline box)
        let bb = s.bounding_box().unwrap();
        assert!((bb.max.x - 1.0).abs() < 1e-6 && (bb.max.z - 1.0).abs() < 1e-6);
    }

    #[test]
    fn chevron_plane_cuts_crown_near_corner() {
        // Ray straight down at (0.9, ., 0.9): the corner chevron plane is lower
        // than the side planes there, so the hit is below the table with a
        // diagonal normal (x ~ z).
        let p = StepCutParams { outline: StepOutline::Rect { corner_pct: 0.0 }, chevrons: 2, ..params() };
        let s = StepCut::new(Point3::new(0.0, 0.0, 0.0), 1.0, 1.0, p, mat()).unwrap();
        let r = Ray::new(Point3::new(0.9, 10.0, 0.9), Vec3::new(0.0, -1.0, 0.0));
        let rec = s.hit(&r, 0.001, 100.0).expect("hit");
        assert!(rec.p.y < 0.5 - 1e-3, "should be cut below the table (y={})", rec.p.y);
        assert!((rec.normal.x - rec.normal.z).abs() < 1e-4 && rec.normal.x > 0.1,
                "diagonal normal expected, got {:?}", rec.normal);
    }

    #[test]
    fn triangle_outline_is_pointed_and_bounded() {
        let p = StepCutParams { outline: StepOutline::Triangle, ..params() };
        let s = StepCut::new(Point3::new(0.0, 0.0, 0.0), 2.0, 1.0, p, mat()).unwrap();
        // +X ray misses (triangle tip is at +Z here); +Z ray hits the tip at z = b.
        let r = Ray::new(Point3::new(0.0, 0.0, 10.0), Vec3::new(0.0, 0.0, -1.0));
        let rec = s.hit(&r, 0.001, 100.0).expect("hit");
        assert!((rec.p.z - 1.0).abs() < 1e-4, "z = {}", rec.p.z);
        // +X ray at z=0 hits the slanted edge from (0, b) to (0.866a, -0.5b) at x = 2/sqrt(3).
        let r = Ray::new(Point3::new(10.0, 0.0, 0.0), Vec3::new(-1.0, 0.0, 0.0));
        let rec = s.hit(&r, 0.001, 100.0).expect("hit slanted edge");
        let expect = 2.0 / 3.0f32.sqrt();
        assert!((rec.p.x - expect).abs() < 1e-4, "x = {} vs {expect}", rec.p.x);
    }

    #[test]
    fn marquise_outline_has_pointed_tips() {
        let p = StepCutParams { outline: StepOutline::PointedOval { segs: 16, pear: false }, ..params() };
        let s = StepCut::new(Point3::new(0.0, 0.0, 0.0), 2.0, 0.5, p, mat()).unwrap();
        // +X ray hits the pointed tip at x = a = 2.
        let r = Ray::new(Point3::new(10.0, 0.0, 0.0), Vec3::new(-1.0, 0.0, 0.0));
        let rec = s.hit(&r, 0.001, 100.0).expect("hit");
        assert!((rec.p.x - 2.0).abs() < 1e-3, "x = {}", rec.p.x);
        // bbox is exact: x in [-2, 2], z in [-0.5, 0.5]
        let bb = s.bounding_box().unwrap();
        assert!((bb.max.x - 2.0).abs() < 1e-6 && (bb.max.z - 0.5).abs() < 1e-6);
    }

    #[test]
    fn rounded_rect_arc_reaches_full_extents() {
        // cushion-style outline: bbox must reach exactly ±a in X and ±b in Z
        let p = StepCutParams { outline: StepOutline::RoundedRect { corner_pct: 30.0, segs: 4 }, ..params() };
        let s = StepCut::new(Point3::new(0.0, 0.0, 0.0), 2.0, 1.0, p, mat()).unwrap();
        let bb = s.bounding_box().unwrap();
        assert!((bb.max.x - 2.0).abs() < 1e-6, "max.x = {}", bb.max.x);
        assert!((bb.min.x + 2.0).abs() < 1e-6, "min.x = {}", bb.min.x);
        assert!((bb.max.z - 1.0).abs() < 1e-6, "max.z = {}", bb.max.z);
        assert!((bb.min.z + 1.0).abs() < 1e-6, "min.z = {}", bb.min.z);
    }

    #[test]
    fn table_pct_is_exact_under_step_spread() {
        // With spread > 0 the rows' raw runs overshoot s; the normalized runs must
        // land the table edge exactly at k*b from the axis: ray straight down at
        // x slightly inside k*b must hit the table (y = crown_h), just outside
        // must hit a crown row (y < crown_h).
        let p = StepCutParams { table_pct: 50.0, crown_angle: 45.0, pavilion_angle: 45.0,
                                outline: StepOutline::Rect { corner_pct: 0.0 }, ..Default::default() };
        // a=b=1 -> s = 0.5, k = 0.5, crown_h = 0.5
        let s = StepCut::new(Point3::new(0.0, 0.0, 0.0), 1.0, 1.0, p, mat()).unwrap();
        let inside  = s.hit(&Ray::new(Point3::new(0.49, 10.0, 0.0), Vec3::new(0.0, -1.0, 0.0)), 0.001, 100.0).unwrap();
        let outside = s.hit(&Ray::new(Point3::new(0.51, 10.0, 0.0), Vec3::new(0.0, -1.0, 0.0)), 0.001, 100.0).unwrap();
        assert!((inside.p.y - 0.5).abs() < 1e-4, "table y = {}", inside.p.y);
        assert!(outside.p.y < 0.5 - 1e-3, "crown y = {}", outside.p.y);
    }

    #[test]
    fn polygon_outline_validation() {
        // non-convex polygon must error, not silently self-intersect
        let p = StepCutParams { outline: StepOutline::Polygon(vec![[1.0, 0.0], [0.0, 1.0], [0.2, 0.2], [-1.0, 0.0], [0.0, -1.0]]), ..params() };
        assert!(StepCut::new(Point3::new(0.0, 0.0, 0.0), 1.0, 1.0, p, mat()).is_err());
        // duplicate adjacent points are deduped, valid triangle ok
        let p = StepCutParams { outline: StepOutline::Polygon(vec![[0.0, 1.0], [0.0, 1.0], [-0.866, -0.5], [0.866, -0.5]]), ..params() };
        assert!(StepCut::new(Point3::new(0.0, 0.0, 0.0), 1.0, 1.0, p, mat()).is_ok());
        // fewer than 3 distinct points -> Err
        let p = StepCutParams { outline: StepOutline::Polygon(vec![[1.0, 0.0], [1.0, 0.0]]), ..params() };
        assert!(StepCut::new(Point3::new(0.0, 0.0, 0.0), 1.0, 1.0, p, mat()).is_err());
    }

    #[test]
    fn invalid_params_error_not_panic() {
        let p = StepCutParams { crown_angle: 0.0, ..params() };
        assert!(StepCut::new(Point3::new(0.0, 0.0, 0.0), 1.0, 1.0, p, mat()).is_err());
        let p = StepCutParams { crown_angle: 95.0, ..params() };
        assert!(StepCut::new(Point3::new(0.0, 0.0, 0.0), 1.0, 1.0, p, mat()).is_err());
        assert!(StepCut::new(Point3::new(0.0, 0.0, 0.0), 0.0, 1.0, params(), mat()).is_err());
        assert!(StepCut::new(Point3::new(0.0, 0.0, 0.0), -1.0, 1.0, params(), mat()).is_err());
    }

    #[test]
    fn any_hit_agrees_with_hit() {
        // princess-style params (chevrons on) + an emerald-style one
        for outline in [StepOutline::Rect { corner_pct: 0.0 }, StepOutline::RoundedRect { corner_pct: 30.0, segs: 4 }] {
            let p = StepCutParams { outline, chevrons: 2, ..params() };
            let s = StepCut::new(Point3::new(0.0, 0.0, 0.0), 2.0, 1.0, p, mat()).unwrap();
            let rays = [
                (0.0, 10.0, 0.0, 0.0, -1.0, 0.0),    // table from above
                (10.0, 0.0, 0.0, -1.0, 0.0, 0.0),    // girdle edge +X
                (10.0, 0.0, 10.0, -1.0, 0.0, -1.0),  // diagonal
                (0.0, 0.25, 0.0, 1.0, 0.0, 0.0),     // interior exit
                (0.0, 10.0, 5.0, 0.0, -1.0, 0.0),    // miss
                (0.0, -10.0, 0.0, 0.0, 1.0, 0.0),    // culet from below
            ];
            for (ox, oy, oz, dx, dy, dz) in rays {
                let r = Ray::new(Point3::new(ox, oy, oz), Vec3::new(dx, dy, dz));
                assert_eq!(
                    s.hit(&r, 0.001, 100.0).is_some(),
                    s.any_hit(&r, 0.001, 100.0),
                    "mismatch for ray ({ox},{oy},{oz}) -> ({dx},{dy},{dz})"
                );
            }
        }
    }

    #[test]
    fn steps_make_visibly_different_planes() {
        // With spread > 0 the crown rows of one edge must have distinct normals.
        let s = StepCut::new(Point3::new(0.0, 0.0, 0.0), 2.0, 1.0, params(), mat()).unwrap();
        // +X side rows only (m = (1,0,0) -> n.z == 0)
        let crown_planes: Vec<&(Vec3, f32)> = s.planes.iter()
            .filter(|(n, _)| n.y > 0.3 && n.x > 0.5 && n.z.abs() < 1e-6).collect();
        assert!(crown_planes.len() >= 3);
        let y0 = crown_planes[0].0.y;
        let y2 = crown_planes[2].0.y;
        assert!((y0 - y2).abs() > 0.05, "row normals should differ: {y0} vs {y2}");
    }
}
