use std::f32::consts::PI;
use rand::Rng;
use rand::rngs::SmallRng;
use rand::SeedableRng;
use rayon::prelude::*;
use crate::hittable::Hittable;
use crate::ray::Ray;
use crate::vec3::{Color, Point3, Vec3};

struct KdPhoton {
    pos:   [f32; 3],
    power: [f32; 3],
    nor:   [f32; 3],
    /// Split axis used when this node was inserted (x=0, y=1, z=2).
    /// Only meaningful for interior nodes; leaves keep the default 0.
    axis:  u8,
}

const DISK_R: f32 = 20.0;
pub const PHOTON_MAX_DEPTH: i32 = 12;

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

        photons.shrink_to_fit();
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
        let cross     = light_u.cross(light_v);
        let quad_area = cross.length();
        let normal    = cross / quad_area; // unit normal, emission side
        let (t_axis, b_axis) = normal.onb();
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
                // Cosine-weighted hemisphere sample in the quad's own frame.
                let dir = sin_theta * phi.cos() * t_axis
                        + sin_theta * phi.sin() * b_axis
                        + cos_theta * normal;
                trace_photon(world, origin, dir, power, &mut rng)
            })
            .collect();

        photons.shrink_to_fit();
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
    let wavelength       = rng.gen_range(380.0_f32..700.0);

    for _ in 0..PHOTON_MAX_DEPTH {
        let mut ray = Ray::new_at_time(pos, dir, 0.0);
        ray.wavelength = wavelength;
        let rec = world.hit(&ray, 0.001, f32::INFINITY)?;
        let sr  = rec.mat.scatter(&ray, &rec, rng)?;

        if sr.skip_pdf {
            if rec.mat.is_spectral() { hit_spectral = true; }
            power     *= sr.attenuation;
            pos        = sr.ray.origin;
            dir        = sr.ray.direction;
            spec_depth += 1;
        } else {
            if spec_depth > 0 && hit_spectral && rec.mat.can_receive_caustics() {
                // Clamp spectral spikes once at storage, not per bounce.
                let lum = 0.2126 * power.x + 0.7152 * power.y + 0.0722 * power.z;
                if lum > 15.0 { power *= 15.0 / lum; }
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
