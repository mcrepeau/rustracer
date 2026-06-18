use std::f32::consts::PI;
use std::ops::Range;
use hashbrown::HashMap;
use rand::Rng;
use rand::rngs::SmallRng;
use rand::SeedableRng;
use rayon::prelude::*;
use crate::hittable::Hittable;
use crate::ray::Ray;
use crate::vec3::{Color, Point3, Vec3};

#[derive(Clone, Copy)]
struct KdPhoton {
    pos:   [f32; 3],
    power: [f32; 3],
    nor:   [f32; 3],
    /// Split axis used when this node was inserted (x=0, y=1, z=2).
    /// Only meaningful for interior nodes; leaves keep the default 0.
    axis:  u8,
}

const DISK_R: f32 = 20.0;

/// Balanced-kd-tree caustic photon map with surface-normal hemisphere filtering.
///
/// Photons are emitted from a light source and stored only when they reach a
/// diffuse surface after at least one specular/transmissive bounce (caustic
/// path).  `irradiance()` returns the Epanechnikov-filtered estimate
/// pre-divided by π, so the caller multiplies by albedo to get radiance.
/// Photons whose stored surface normal opposes the shading normal are
/// rejected, preventing caustic "leakage" through opaque surfaces.
pub struct PhotonMap {
    nodes:     Vec<KdPhoton>,
    gather_r2: f32,
    /// 2 / (π² R²) — Epanechnikov kernel normaliser pre-divided by π.
    norm:      f32,
}

impl PhotonMap {
    /// Trace `num_photons` from a directional disk facing `sun_dir` and build
    /// the map.  `sun_color` should be `background.eval(sun_dir) * PI`.
    pub fn build(
        world:         &dyn Hittable,
        sun_dir:       Vec3,
        sun_color:     Color,
        num_photons:   u32,
        gather_radius: f32,
    ) -> Self {
        let sun_down    = (-sun_dir).unit();
        let up          = if sun_down.x.abs() < 0.999 { Vec3::new(1.0, 0.0, 0.0) } else { Vec3::new(0.0, 1.0, 0.0) };
        let t           = sun_down.cross(up).unit();
        let b           = sun_down.cross(t);
        let disk_center = sun_dir.unit() * 30.0;
        let disk_area   = PI * DISK_R * DISK_R;
        let power       = sun_color * (disk_area / (num_photons as f32 * PI));

        let mut photons: Vec<KdPhoton> = (0..num_photons)
            .into_par_iter()
            .filter_map(|i| {
                let mut rng = SmallRng::seed_from_u64(
                    (i as u64).wrapping_mul(6_364_136_223_846_793_005) ^ 0x9E3779B97F4A7C15,
                );
                let r   = DISK_R * rng.gen::<f32>().sqrt();
                let phi = 2.0 * PI * rng.gen::<f32>();
                let origin = disk_center + t * (r * phi.cos()) + b * (r * phi.sin());
                trace_photon(world, origin, sun_down, power, &mut rng)
            })
            .collect();

        kd_build(&mut photons);
        Self::from_nodes(photons, gather_radius)
    }

    /// Trace `num_photons` emitted as a Lambertian source from a quad and build.
    pub fn build_from_quad(
        world:         &dyn Hittable,
        light_origin:  Point3,
        light_u:       Vec3,
        light_v:       Vec3,
        light_color:   Color,
        num_photons:   u32,
        gather_radius: f32,
    ) -> Self {
        let quad_area = light_u.cross(light_v).length();
        let power     = light_color * (quad_area * PI / num_photons as f32);

        let mut photons: Vec<KdPhoton> = (0..num_photons)
            .into_par_iter()
            .filter_map(|i| {
                let mut rng = SmallRng::seed_from_u64(
                    (i as u64).wrapping_mul(6_364_136_223_846_793_005) ^ 0x9E3779B97F4A7C15,
                );
                let s = rng.gen::<f32>();
                let t = rng.gen::<f32>();
                let origin    = light_origin + light_u * s + light_v * t;
                let cos_theta = rng.gen::<f32>().sqrt();
                let sin_theta = (1.0 - cos_theta * cos_theta).sqrt();
                let phi       = 2.0 * PI * rng.gen::<f32>();
                let dir       = Vec3::new(sin_theta * phi.cos(), -cos_theta, sin_theta * phi.sin());
                trace_photon(world, origin, dir, power, &mut rng)
            })
            .collect();

        kd_build(&mut photons);
        Self::from_nodes(photons, gather_radius)
    }

    fn from_nodes(nodes: Vec<KdPhoton>, gather_radius: f32) -> Self {
        Self {
            nodes,
            gather_r2: gather_radius * gather_radius,
            norm:      2.0 / (PI * PI * gather_radius * gather_radius),
        }
    }

    /// Epanechnikov-filtered irradiance at `pos` on a surface with outward
    /// `normal`, pre-divided by π.  Photons on the opposite hemisphere
    /// (stored normal · shading normal ≤ 0) are silently skipped.
    pub fn irradiance(&self, pos: Point3, normal: Vec3) -> Color {
        if self.nodes.is_empty() {
            return Color::default();
        }
        let p = [pos.x, pos.y, pos.z];
        let n = [normal.x, normal.y, normal.z];
        let mut acc = Color::default();
        kd_gather(&self.nodes, &p, self.gather_r2, &n, &mut acc);
        acc * self.norm
    }
}

// ── kd-tree build ─────────────────────────────────────────────────────────────
// Balanced implicit kd-tree: the root of nodes[lo..=hi] is always at the
// median index mid = (lo + hi) / 2 after partitioning by that node's axis.
// Using subslices keeps the bookkeeping index-free.

fn kd_build(nodes: &mut [KdPhoton]) {
    let n = nodes.len();
    if n == 0 { return; }

    // Choose the axis with the widest spread across this subset.
    let mut min = [f32::INFINITY; 3];
    let mut max = [f32::NEG_INFINITY; 3];
    for nd in nodes.iter() {
        for k in 0..3 {
            if nd.pos[k] < min[k] { min[k] = nd.pos[k]; }
            if nd.pos[k] > max[k] { max[k] = nd.pos[k]; }
        }
    }
    let axis = (0..3)
        .max_by(|&a, &b| (max[a] - min[a]).partial_cmp(&(max[b] - min[b])).unwrap())
        .unwrap();

    // Partial-sort so the median lands at nodes[mid].
    let mid = n / 2;
    nodes.select_nth_unstable_by(mid, |a, b| {
        a.pos[axis].partial_cmp(&b.pos[axis]).unwrap_or(std::cmp::Ordering::Equal)
    });
    nodes[mid].axis = axis as u8;

    kd_build(&mut nodes[..mid]);
    kd_build(&mut nodes[mid + 1..]);
}

// ── kd-tree gather ────────────────────────────────────────────────────────────

fn kd_gather(nodes: &[KdPhoton], pos: &[f32; 3], r2: f32, normal: &[f32; 3], acc: &mut Color) {
    let n = nodes.len();
    if n == 0 { return; }

    let mid  = n / 2;
    let node = &nodes[mid];

    let dx = node.pos[0] - pos[0];
    let dy = node.pos[1] - pos[1];
    let dz = node.pos[2] - pos[2];
    let d2 = dx * dx + dy * dy + dz * dz;

    if d2 < r2 {
        // Hemisphere check: skip photons stored on the opposite side of the surface.
        let dot = node.nor[0] * normal[0] + node.nor[1] * normal[1] + node.nor[2] * normal[2];
        if dot > 0.0 {
            *acc += Color::new(node.power[0], node.power[1], node.power[2]) * (1.0 - d2 / r2);
        }
    }

    // Prune subtrees using the split plane distance.
    let axis = node.axis as usize;
    let sq   = pos[axis] - node.pos[axis]; // positive → query is right-of-split

    if sq <= 0.0 {
        // Query is on the left (low) side: always search left, only right if sphere extends past split.
        kd_gather(&nodes[..mid], pos, r2, normal, acc);
        if sq * sq < r2 { kd_gather(&nodes[mid + 1..], pos, r2, normal, acc); }
    } else {
        // Query is on the right (high) side.
        kd_gather(&nodes[mid + 1..], pos, r2, normal, acc);
        if sq * sq < r2 { kd_gather(&nodes[..mid], pos, r2, normal, acc); }
    }
}

// ── SPPM types ────────────────────────────────────────────────────────────────

/// Photon-emission source used by the SPPM photon pass.
pub enum PhotonSource {
    Sun  { dir: Vec3, color: Color },
    Quad { origin: Point3, u_axis: Vec3, v_axis: Vec3, color: Color },
}

impl PhotonSource {
    /// Emit one photon and trace it to its first caustic diffuse hit.
    /// `n_photons` is the total photons per iteration; divides into per-photon power.
    fn emit_one(&self, world: &dyn Hittable, n_photons: u32, rng: &mut SmallRng) -> Option<KdPhoton> {
        match self {
            PhotonSource::Sun { dir, color } => {
                let sun_down    = (-*dir).unit();
                let up          = if sun_down.x.abs() < 0.999 { Vec3::new(1.0, 0.0, 0.0) } else { Vec3::new(0.0, 1.0, 0.0) };
                let t           = sun_down.cross(up).unit();
                let b           = sun_down.cross(t);
                let disk_center = dir.unit() * 30.0;
                let disk_area   = PI * DISK_R * DISK_R;
                let power       = *color * (disk_area / (n_photons as f32 * PI));
                let r_disk      = DISK_R * rng.gen::<f32>().sqrt();
                let phi         = 2.0 * PI * rng.gen::<f32>();
                let origin      = disk_center + t * (r_disk * phi.cos()) + b * (r_disk * phi.sin());
                trace_photon(world, origin, sun_down, power, rng)
            }
            PhotonSource::Quad { origin, u_axis, v_axis, color } => {
                let quad_area = u_axis.cross(*v_axis).length();
                let power     = *color * (quad_area * PI / n_photons as f32);
                let s         = rng.gen::<f32>();
                let t         = rng.gen::<f32>();
                let org       = *origin + *u_axis * s + *v_axis * t;
                let cos_theta = rng.gen::<f32>().sqrt();
                let sin_theta = (1.0 - cos_theta * cos_theta).sqrt();
                let phi       = 2.0 * PI * rng.gen::<f32>();
                let dir       = Vec3::new(sin_theta * phi.cos(), -cos_theta, sin_theta * phi.sin());
                trace_photon(world, org, dir, power, rng)
            }
        }
    }
}

/// A visible point: the first caustic-eligible diffuse hit along a camera path.
pub struct VisiblePoint {
    pub pos:    Point3,
    pub normal: Vec3,
    pub albedo: Color,
    pub beta:   Color,  // path throughput from camera to this hit
    pub radius: f32,    // current SPPM search radius for this pixel
    pub pixel:  usize,  // flat pixel index in the output image
}

/// Per-pixel SPPM accumulation state.
#[derive(Clone)]
pub struct SppmPixel {
    pub radius: f32,
    pub flux:   Color,
    pub n:      f32,    // fractional accumulated photon count
}

impl SppmPixel {
    fn new(radius: f32) -> Self {
        Self { radius, flux: Color::default(), n: 0.0 }
    }

    // SPPM update rule — called once per photon pass for each VP that received hits.
    fn apply_update(&mut self, d_n: f32, d_flux: Color, alpha: f32) {
        let n_new     = self.n + alpha * d_n;
        let scale     = n_new / (self.n + d_n);
        self.radius   = self.radius * scale.sqrt();
        self.flux     = (self.flux + d_flux) * scale;
        self.n        = n_new;
    }
}

pub struct SppmState {
    pub pixels:           Vec<SppmPixel>,
    /// Caustic radiance estimate per pixel, updated after each photon pass.
    pub caustic_buf:      Vec<Color>,
    pub total_photons:    u64,
    pub alpha:            f32,            // shrinkage parameter; 2/3 is optimal
    pub photons_per_iter: u32,
    pub initial_radius:   f32,
}

impl SppmState {
    pub fn new(n_pixels: usize, initial_radius: f32, alpha: f32, photons_per_iter: u32) -> Self {
        Self {
            pixels:           (0..n_pixels).map(|_| SppmPixel::new(initial_radius)).collect(),
            caustic_buf:      vec![Color::default(); n_pixels],
            total_photons:    0,
            alpha,
            photons_per_iter,
            initial_radius,
        }
    }

    pub fn reset(&mut self) {
        let r = self.initial_radius;
        for p in &mut self.pixels { p.radius = r; p.flux = Color::default(); p.n = 0.0; }
        self.caustic_buf.fill(Color::default());
        self.total_photons = 0;
    }

    pub fn resize(&mut self, n_pixels: usize) {
        self.pixels.resize_with(n_pixels, || SppmPixel::new(self.initial_radius));
        self.caustic_buf.resize(n_pixels, Color::default());
        self.reset();
    }

    #[cfg(test)]
    fn caustic_at(&self, i: usize) -> Color {
        if self.total_photons == 0 { return Color::default(); }
        let p      = &self.pixels[i];
        if p.n == 0.0 { return Color::default(); }
        let n_iter = (self.total_photons / self.photons_per_iter as u64) as f32;
        p.flux / (PI * p.radius * p.radius * n_iter)
    }

    fn refresh_caustic_buf(&mut self) {
        if self.total_photons == 0 {
            self.caustic_buf.fill(Color::default());
            return;
        }
        let n_iter = (self.total_photons / self.photons_per_iter as u64) as f32;
        self.caustic_buf.par_iter_mut()
            .zip(self.pixels.par_iter())
            .for_each(|(c, px)| {
                *c = if px.n == 0.0 {
                    Color::default()
                } else {
                    px.flux / (PI * px.radius * px.radius * n_iter)
                };
            });
    }
}

// ── SPPM hash grid ────────────────────────────────────────────────────────────
// Sorted-array grid: each VP is mapped to a (cell_coord, vp_index) pair,
// the array is parallel-sorted by cell coord, and lookups use two
// partition_point binary searches to find the range of VPs in a cell.
//
// Compared to HashMap<coord, Vec<usize>>:
//   Build:  parallel par_sort instead of sequential insertions + ~n_cells
//           individual Vec allocations (the main prior bottleneck).
//   Lookup: 27 × 2 binary searches (O(log n)) instead of 27 hash probes.
//   Memory: one flat Vec<(coord, usize)> instead of a HashMap with many
//           small heap-allocated bucket Vecs.

#[cfg(test)]
struct HashGrid {
    sorted:    Vec<((i32, i32, i32), usize)>,
    cell_size: f32,
}

#[cfg(test)]
impl HashGrid {
    fn build(vps: &[VisiblePoint]) -> Self {
        let max_r     = vps.par_iter().map(|v| v.radius).reduce(|| 0.0f32, f32::max);
        let cell_size = max_r.max(1e-6);
        let mut sorted: Vec<((i32, i32, i32), usize)> = vps.par_iter()
            .enumerate()
            .map(|(i, vp)| (Self::coord(vp.pos, cell_size), i))
            .collect();
        sorted.par_sort_unstable_by_key(|&(coord, _)| coord);
        Self { sorted, cell_size }
    }

    fn coord(p: Point3, s: f32) -> (i32, i32, i32) {
        ((p.x / s).floor() as i32, (p.y / s).floor() as i32, (p.z / s).floor() as i32)
    }

    fn for_neighbors(&self, pos: Point3, mut f: impl FnMut(usize)) {
        let (cx, cy, cz) = Self::coord(pos, self.cell_size);
        for dx in -1i32..=1 {
            for dy in -1i32..=1 {
                for dz in -1i32..=1 {
                    let target = (cx + dx, cy + dy, cz + dz);
                    let start  = self.sorted.partition_point(|&(k, _)| k < target);
                    let end    = self.sorted.partition_point(|&(k, _)| k <= target);
                    for &(_, vi) in &self.sorted[start..end] { f(vi); }
                }
            }
        }
    }
}

// ── SPPM photon grid ──────────────────────────────────────────────────────────
// Compact hash map over surviving photons.  The VP-parallel gather queries
// 27 neighbour cells per VP; a HashMap lookup is O(1) with no branch
// misprediction, which is dramatically faster than the two partition_point
// binary searches that a sorted-array approach would require.

struct PhotonGrid {
    photons:   Vec<KdPhoton>,
    cells:     HashMap<(i32, i32, i32), Range<usize>>,
    cell_size: f32,
}

impl PhotonGrid {
    fn build(photons: Vec<KdPhoton>, cell_size: f32) -> Self {
        let ph_cell = |ph: &KdPhoton| -> (i32, i32, i32) {
            (
                (ph.pos[0] / cell_size).floor() as i32,
                (ph.pos[1] / cell_size).floor() as i32,
                (ph.pos[2] / cell_size).floor() as i32,
            )
        };

        // Sort photons by cell coord (sequential — n is small, ~few thousand).
        let mut order: Vec<usize> = (0..photons.len()).collect();
        order.sort_unstable_by_key(|&i| ph_cell(&photons[i]));
        let sorted: Vec<KdPhoton> = order.iter().map(|&i| photons[i]).collect();

        // Build cell → contiguous range map from the sorted photons.
        let mut cells: HashMap<(i32, i32, i32), Range<usize>> = HashMap::new();
        let mut i = 0;
        while i < sorted.len() {
            let start = i;
            let c = ph_cell(&sorted[i]);
            while i < sorted.len() && ph_cell(&sorted[i]) == c { i += 1; }
            cells.insert(c, start..i);
        }

        Self { photons: sorted, cells, cell_size }
    }

    #[inline]
    fn for_neighbors(&self, pos: Point3, mut f: impl FnMut(&KdPhoton)) {
        let cx = (pos.x / self.cell_size).floor() as i32;
        let cy = (pos.y / self.cell_size).floor() as i32;
        let cz = (pos.z / self.cell_size).floor() as i32;
        for dx in -1i32..=1 {
            for dy in -1i32..=1 {
                for dz in -1i32..=1 {
                    if let Some(range) = self.cells.get(&(cx + dx, cy + dy, cz + dz)) {
                        for ph in &self.photons[range.clone()] { f(ph); }
                    }
                }
            }
        }
    }
}

/// Run one SPPM photon pass: emit photons, update visible points, shrink radii.
///
/// Strategy: VP-parallel gather.
///   1. Trace all n photons in parallel → collect only caustic survivors.
///   2. Build a compact hash grid over the surviving photons (~few thousand
///      entries, fits in L2 cache).
///   3. For each VP in parallel, query the photon grid and accumulate flux
///      directly into state.pixels[vp.pixel] — no per-thread delta buffer,
///      no reduce step.
///
/// This avoids the num_threads × n_px delta buffer (≈300 MB of zeroing +
/// tree-reduce) that the photon-parallel approach required.
pub fn sppm_photon_pass(
    world:  &dyn Hittable,
    source: &PhotonSource,
    state:  &mut SppmState,
    vps:    &[VisiblePoint],
) {
    if vps.is_empty() { return; }
    let n     = state.photons_per_iter;
    let epoch = state.total_photons;
    let alpha = state.alpha;

    // 1. Trace photons in parallel; keep only caustic survivors.
    let photons: Vec<KdPhoton> = (0..n as usize)
        .into_par_iter()
        .filter_map(|i| {
            let mut rng = SmallRng::seed_from_u64(
                (i as u64).wrapping_mul(6_364_136_223_846_793_005)
                    ^ epoch.wrapping_mul(0x9E3779B97F4A7C15),
            );
            source.emit_one(world, n, &mut rng)
        })
        .collect();

    if !photons.is_empty() {
        // 2. Build a compact photon grid.  Cell size = max VP radius ensures
        //    every photon within any VP's radius is in one of the 27 neighbour cells.
        let max_r     = vps.par_iter().map(|v| v.radius).reduce(|| 0.0f32, f32::max);
        let cell_size = max_r.max(1e-6);
        let grid      = PhotonGrid::build(photons, cell_size);

        // 3. VP-parallel gather with direct write to state.pixels.
        //
        // SAFETY: collect_visible_points maps each pixel index i (from the unique
        // range 0..n_px) to at most one VP with vp.pixel = i, so all VP pixel
        // indices in `vps` are distinct.  Concurrent writes to different
        // state.pixels[vp.pixel] elements therefore never alias.
        let px_ptr = state.pixels.as_mut_ptr() as usize;
        vps.par_iter().for_each(|vp| {
            let mut d_flux = Color::default();
            let mut d_n    = 0.0f32;
            grid.for_neighbors(vp.pos, |ph| {
                let dx   = ph.pos[0] - vp.pos.x;
                let dy   = ph.pos[1] - vp.pos.y;
                let dz   = ph.pos[2] - vp.pos.z;
                let ph_n = Vec3::new(ph.nor[0], ph.nor[1], ph.nor[2]);
                if dx * dx + dy * dy + dz * dz < vp.radius * vp.radius
                    && ph_n.dot(vp.normal) > 0.0
                {
                    d_flux += Color::new(ph.power[0], ph.power[1], ph.power[2])
                              * vp.beta * vp.albedo;
                    d_n    += 1.0;
                }
            });
            if d_n > 0.0 {
                let base = px_ptr as *mut SppmPixel;
                unsafe { (*base.add(vp.pixel)).apply_update(d_n, d_flux, alpha); }
            }
        });
    }

    state.total_photons += n as u64;
    state.refresh_caustic_buf();
}

// ── photon tracing ────────────────────────────────────────────────────────────

fn trace_photon(
    world:  &dyn Hittable,
    origin: Point3,
    dir:    Vec3,
    power:  Color,
    rng:    &mut SmallRng,
) -> Option<KdPhoton> {
    let mut pos          = origin;
    let mut dir          = dir;
    let mut power        = power;
    let mut spec_depth   = 0u32;
    let mut hit_spectral = false;

    for _ in 0..12 {
        let mut ray = Ray::new_at_time(pos, dir, 0.0);
        ray.wavelength = rng.gen_range(380.0_f32..700.0);
        let rec = world.hit(&ray, 0.001, f32::INFINITY)?;
        let sr  = rec.mat.scatter(&ray, &rec, rng)?;

        if sr.skip_pdf {
            if rec.mat.is_spectral() { hit_spectral = true; }
            power     *= sr.attenuation;
            pos        = sr.ray.origin;
            dir        = sr.ray.direction;
            spec_depth += 1;

            // Clamp spectral spikes from compounding single-channel bounces.
            if rec.mat.is_spectral() {
                let lum = 0.2126 * power.x + 0.7152 * power.y + 0.0722 * power.z;
                if lum > 15.0 { power *= 15.0 / lum; }
            }
        } else {
            if spec_depth > 0 && hit_spectral && rec.mat.can_receive_caustics() {
                let nor = rec.normal;
                return Some(KdPhoton {
                    pos:   [rec.p.x, rec.p.y, rec.p.z],
                    power: [power.x, power.y, power.z],
                    nor:   [nor.x,   nor.y,   nor.z],
                    axis:  0,
                });
            }
            return None;
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vp(x: f32, y: f32, z: f32, radius: f32) -> VisiblePoint {
        VisiblePoint {
            pos:    Point3::new(x, y, z),
            normal: Vec3::new(0.0, 1.0, 0.0),
            albedo: Color::new(1.0, 1.0, 1.0),
            beta:   Color::new(1.0, 1.0, 1.0),
            radius,
            pixel:  0,
        }
    }

    fn near(a: f32, b: f32) -> bool { (a - b).abs() < 1e-5 }

    // ── HashGrid ──────────────────────────────────────────────────────────────

    #[test]
    fn hash_grid_same_cell() {
        let vps  = vec![vp(1.0, 1.0, 1.0, 0.5)];
        let grid = HashGrid::build(&vps);
        let mut hits = vec![];
        grid.for_neighbors(Point3::new(1.0, 1.0, 1.0), |i| hits.push(i));
        assert_eq!(hits, vec![0]);
    }

    #[test]
    fn hash_grid_adjacent_cell_reached() {
        // VP at origin; photon query just past the cell boundary.
        // The 3×3×3 neighbourhood must include the VP's cell.
        let vps  = vec![vp(0.0, 0.0, 0.0, 0.5)];
        let grid = HashGrid::build(&vps);
        let mut hits = vec![];
        grid.for_neighbors(Point3::new(0.51, 0.0, 0.0), |i| hits.push(i));
        assert_eq!(hits, vec![0]);
    }

    #[test]
    fn hash_grid_distant_point_not_found() {
        let vps  = vec![vp(0.0, 0.0, 0.0, 0.5)];
        let grid = HashGrid::build(&vps);
        let mut hits = vec![];
        // More than one cell away — outside the 3×3×3 neighbourhood.
        grid.for_neighbors(Point3::new(1.1, 0.0, 0.0), |i| hits.push(i));
        assert!(hits.is_empty());
    }

    #[test]
    fn hash_grid_multiple_vps_same_cell() {
        let vps  = vec![
            { let mut v = vp(0.0, 0.0, 0.0, 0.5); v.pixel = 0; v },
            { let mut v = vp(0.0, 0.0, 0.0, 0.5); v.pixel = 1; v },
            { let mut v = vp(5.0, 5.0, 5.0, 0.5); v.pixel = 2; v },
        ];
        let grid = HashGrid::build(&vps);
        let mut hits = vec![];
        grid.for_neighbors(Point3::new(0.0, 0.0, 0.0), |i| hits.push(i));
        hits.sort();
        // VP 2 is far away; only VPs 0 and 1 are found.
        assert_eq!(hits, vec![0, 1]);
    }

    // ── SppmPixel::apply_update (SPPM update rule) ────────────────────────────

    #[test]
    fn update_rule_first_hit() {
        // N_old = 0, d_n = 3, α = 2/3:
        //   N_new = 0 + 2/3·3 = 2          scale = 2/3
        //   R_new = R₀ · √(2/3)
        //   Φ_new = (0 + d_flux) · 2/3
        let alpha  = 2.0_f32 / 3.0;
        let mut px = SppmPixel::new(1.0);
        px.apply_update(3.0, Color::new(6.0, 0.0, 0.0), alpha);

        assert!(near(px.n,        2.0),               "n={}",        px.n);
        assert!(near(px.radius,   (2.0_f32/3.0).sqrt()), "r={}",     px.radius);
        assert!(near(px.flux.x,   4.0),               "flux.x={}",   px.flux.x);
        assert!(near(px.flux.y,   0.0),               "flux.y={}",   px.flux.y);
    }

    #[test]
    fn update_rule_radius_shrinks_monotonically() {
        let alpha  = 2.0_f32 / 3.0;
        let mut px = SppmPixel::new(1.0);
        let mut prev_r = px.radius;
        for _ in 0..20 {
            px.apply_update(1.0, Color::new(1.0, 1.0, 1.0), alpha);
            assert!(px.radius < prev_r, "radius did not shrink: {} >= {}", px.radius, prev_r);
            prev_r = px.radius;
        }
    }

    #[test]
    fn update_rule_alpha_one_keeps_all_photons() {
        // α=1 → no downweighting of new photons; N_new = N_old + d_n exactly.
        let mut px = SppmPixel::new(1.0);
        px.apply_update(5.0, Color::new(0.0, 0.0, 0.0), 1.0);
        assert!(near(px.n, 5.0), "n={}", px.n);
    }

    // ── SppmState ─────────────────────────────────────────────────────────────

    #[test]
    fn state_reset_zeroes_all_fields() {
        let mut s = SppmState::new(4, 1.0, 2.0 / 3.0, 1000);
        s.pixels[0].n      = 5.0;
        s.pixels[0].flux   = Color::new(1.0, 2.0, 3.0);
        s.pixels[0].radius = 0.1;
        s.total_photons    = 99;
        s.caustic_buf[0]   = Color::new(0.5, 0.5, 0.5);

        s.reset();

        assert_eq!(s.total_photons, 0);
        for px in &s.pixels {
            assert!(near(px.n,      0.0));
            assert!(near(px.flux.x, 0.0) && near(px.flux.y, 0.0) && near(px.flux.z, 0.0));
            assert!(near(px.radius, 1.0));
        }
        for c in &s.caustic_buf {
            assert!(near(c.x, 0.0) && near(c.y, 0.0) && near(c.z, 0.0));
        }
    }

    #[test]
    fn state_resize_adjusts_pixel_count() {
        let mut s = SppmState::new(4, 1.0, 2.0 / 3.0, 1000);
        s.resize(8);
        assert_eq!(s.pixels.len(), 8);
        assert_eq!(s.caustic_buf.len(), 8);
        s.resize(2);
        assert_eq!(s.pixels.len(), 2);
        assert_eq!(s.caustic_buf.len(), 2);
    }

    // ── caustic_at ────────────────────────────────────────────────────────────

    #[test]
    fn caustic_at_zero_before_any_photons() {
        let s = SppmState::new(4, 1.0, 2.0 / 3.0, 1000);
        let l = s.caustic_at(0);
        assert!(near(l.x, 0.0) && near(l.y, 0.0) && near(l.z, 0.0));
    }

    #[test]
    fn caustic_at_matches_formula() {
        // L = Φ / (π R² · n_iters)   where n_iters = total_photons / photons_per_iter
        // With Φ.x = π, R = 1, n_iters = 5000/1000 = 5: L.x = π / (π · 1 · 5) = 0.2
        let mut s           = SppmState::new(1, 1.0, 2.0 / 3.0, 1000);
        s.total_photons     = 5000;
        s.pixels[0].flux    = Color::new(PI, 0.0, 0.0);
        s.pixels[0].n       = 1.0;
        s.pixels[0].radius  = 1.0;

        let l = s.caustic_at(0);
        assert!(near(l.x, 0.2), "L.x = {} (expected 0.2)", l.x);
        assert!(near(l.y, 0.0));
        assert!(near(l.z, 0.0));
    }
}
