use std::f32::consts::PI;
use rand::Rng;
use rand::rngs::SmallRng;
use rand::SeedableRng;
use rayon::prelude::*;
use crate::hittable::Hittable;
use crate::ray::Ray;
use crate::vec3::{Color, Point3, Vec3};

struct RawPhoton { x: f32, y: f32, z: f32, r: f32, g: f32, b: f32 }

/// Flat 3D grid for O(1) photon cell lookup without hash overhead.
/// Dimensions are derived from the actual photon bounding box so memory
/// usage scales with the lit region, not the full scene extent.
struct PhotonGrid {
    cells: Vec<Vec<u32>>,
    nx:    usize,
    ny:    usize,
    nz:    usize,
    ox:    i32,   // minimum cell coordinate on each axis
    oy:    i32,
    oz:    i32,
}

impl PhotonGrid {
    fn build(photons: &[RawPhoton], cell_size: f32) -> Self {
        if photons.is_empty() {
            return Self { cells: Vec::new(), nx: 0, ny: 0, nz: 0, ox: 0, oy: 0, oz: 0 };
        }
        let (mut lx, mut hx) = (i32::MAX, i32::MIN);
        let (mut ly, mut hy) = (i32::MAX, i32::MIN);
        let (mut lz, mut hz) = (i32::MAX, i32::MIN);
        for p in photons {
            let (cx, cy, cz) = (cell_coord(p.x, cell_size), cell_coord(p.y, cell_size), cell_coord(p.z, cell_size));
            lx = lx.min(cx); hx = hx.max(cx);
            ly = ly.min(cy); hy = hy.max(cy);
            lz = lz.min(cz); hz = hz.max(cz);
        }
        let (nx, ny, nz) = ((hx-lx+1) as usize, (hy-ly+1) as usize, (hz-lz+1) as usize);
        let mut cells = vec![Vec::<u32>::new(); nx * ny * nz];
        for (idx, p) in photons.iter().enumerate() {
            let ix = (cell_coord(p.x, cell_size) - lx) as usize;
            let iy = (cell_coord(p.y, cell_size) - ly) as usize;
            let iz = (cell_coord(p.z, cell_size) - lz) as usize;
            cells[iz * ny * nx + iy * nx + ix].push(idx as u32);
        }
        Self { cells, nx, ny, nz, ox: lx, oy: ly, oz: lz }
    }

    /// Returns the photon index slice for cell `(cx, cy, cz)`, or `&[]` if out of bounds.
    #[inline]
    fn get(&self, cx: i32, cy: i32, cz: i32) -> &[u32] {
        let ix = cx - self.ox;
        let iy = cy - self.oy;
        let iz = cz - self.oz;
        if ix < 0 || iy < 0 || iz < 0
            || ix >= self.nx as i32
            || iy >= self.ny as i32
            || iz >= self.nz as i32
        { return &[]; }
        &self.cells[iz as usize * self.ny * self.nx + iy as usize * self.nx + ix as usize]
    }
}

/// Grid-accelerated caustic photon map.
///
/// Photons are emitted from a disk above the scene in the sun direction.  Only
/// photons that reach a diffuse surface **after at least one specular or
/// transmissive bounce** (i.e., through or off glass/metal) are stored —
/// these are the caustic contributors that unidirectional path tracing cannot
/// efficiently sample.
///
/// During rendering, `irradiance()` returns the Epanechnikov-filtered estimate
/// of the caustic irradiance at a surface point, pre-divided by π so the
/// caller only needs to multiply by the surface albedo to get radiance.
pub struct PhotonMap {
    photons:       Vec<RawPhoton>,
    grid:          PhotonGrid,
    gather_radius: f32,
    gather_r2:     f32,
    /// Epanechnikov kernel normaliser ÷ π: 2/(π² R²).
    /// Pre-folding the Lambertian 1/π lets callers write `albedo * irradiance`.
    norm:          f32,
}

impl PhotonMap {
    /// Trace `num_photons` from a disk facing `sun_dir` and build the map.
    /// `sun_color` should be `background.eval(sun_dir) * PI` so the caustic
    /// brightness is automatically calibrated to the sky model.
    pub fn build(world: &dyn Hittable, sun_dir: Vec3, sun_color: Color, num_photons: u32) -> Self {
        const GATHER_RADIUS: f32 = 0.15;
        const DISK_R:        f32 = 20.0;

        let sun_down = (-sun_dir).unit();
        let up = if sun_down.x.abs() < 0.999 {
            Vec3::new(1.0, 0.0, 0.0)
        } else {
            Vec3::new(0.0, 1.0, 0.0)
        };
        let t = sun_down.cross(up).unit();
        let b = sun_down.cross(t);
        let disk_center = sun_dir.unit() * 30.0;

        // Power per photon: sun_color × disk_area / (num_photons × π).
        // The ×1/π pre-folds the Lambertian BRDF normaliser.
        let disk_area    = PI * DISK_R * DISK_R;
        let photon_power = sun_color * (disk_area / (num_photons as f32 * PI));

        let photons: Vec<RawPhoton> = (0..num_photons)
            .into_par_iter()
            .filter_map(|i| {
                let mut rng = SmallRng::seed_from_u64(
                    (i as u64).wrapping_mul(6_364_136_223_846_793_005)
                        ^ 0x9E3779B97F4A7C15,
                );
                let r2  = DISK_R * rng.gen::<f32>().sqrt();
                let phi = 2.0 * PI * rng.gen::<f32>();
                let origin = disk_center + t * (r2 * phi.cos()) + b * (r2 * phi.sin());
                trace_photon(world, origin, sun_down, photon_power, &mut rng)
            })
            .collect();

        let grid = PhotonGrid::build(&photons, GATHER_RADIUS);

        Self {
            photons,
            grid,
            gather_radius: GATHER_RADIUS,
            gather_r2:     GATHER_RADIUS * GATHER_RADIUS,
            norm:          2.0 / (PI * PI * GATHER_RADIUS * GATHER_RADIUS),
        }
    }

    /// Epanechnikov-filtered irradiance estimate at `pos`, already divided by π
    /// so the caller multiplies by albedo to get reflected radiance.
    pub fn irradiance(&self, pos: Point3) -> Color {
        let cx = cell_coord(pos.x, self.gather_radius);
        let cy = cell_coord(pos.y, self.gather_radius);
        let cz = cell_coord(pos.z, self.gather_radius);
        let mut acc = Color::new(0.0, 0.0, 0.0);

        for dz in -1i32..=1 {
            for dy in -1i32..=1 {
                for dx in -1i32..=1 {
                    for &id in self.grid.get(cx + dx, cy + dy, cz + dz) {
                        let p  = &self.photons[id as usize];
                        let d2 = (p.x - pos.x).powi(2)
                               + (p.y - pos.y).powi(2)
                               + (p.z - pos.z).powi(2);
                        if d2 < self.gather_r2 {
                            // Epanechnikov weight: w = 1 − d²/R²
                            acc += Color::new(p.r, p.g, p.b) * (1.0 - d2 / self.gather_r2);
                        }
                    }
                }
            }
        }
        acc * self.norm
    }

    #[allow(dead_code)]
    pub fn stored_count(&self) -> usize { self.photons.len() }
}

#[inline] fn cell_coord(x: f32, size: f32) -> i32 { (x / size).floor() as i32 }

/// Trace one photon from `origin` in `dir`.
/// Returns `Some(photon)` only when the photon hits a diffuse surface
/// **after** at least one specular/transmissive bounce (caustic path).
fn trace_photon(
    world:  &dyn Hittable,
    origin: Point3,
    dir:    Vec3,
    power:  Color,
    rng:    &mut SmallRng,
) -> Option<RawPhoton> {
    let mut pos        = origin;
    let mut dir        = dir;
    let mut power      = power;
    let mut spec_depth = 0u32;

    for _ in 0..12 {
        let ray = Ray::new_at_time(pos, dir, 0.0);
        let rec = world.hit(&ray, 0.001, f32::INFINITY)?;
        let sr  = rec.mat.scatter(&ray, &rec, rng)?;

        if sr.skip_pdf {
            // Specular or transmissive bounce — continue tracing.
            // Use sr.ray.origin (not rec.p) so SSS interior scatters start
            // from the correct volumetric point rather than the surface.
            power     *= sr.attenuation;
            pos        = sr.ray.origin;
            dir        = sr.ray.direction;
            spec_depth += 1;

            // Clamp spectral spikes: SpectralDielectric emits 3× single-channel
            // attenuation; multiple internal reflections can compound this into
            // extreme values.  Apply only after a spectral bounce so legitimate
            // metallic mirror concentrations are not silently stolen.
            if rec.mat.is_spectral() {
                let lum = 0.2126 * power.x + 0.7152 * power.y + 0.0722 * power.z;
                if lum > 15.0 { power *= 15.0 / lum; }
            }
        } else {
            // First diffuse hit: store only if a caustic path (spec_depth > 0)
            // reached a material that actually receives caustics.  Using the
            // material trait avoids the previous hard-coded rec.normal.y > 0.7
            // threshold, which would have killed caustics on any non-horizontal
            // surface (walls, sloped terrain, etc.).
            if spec_depth > 0 && rec.mat.can_receive_caustics() {
                return Some(RawPhoton {
                    x: rec.p.x, y: rec.p.y, z: rec.p.z,
                    r: power.x, g: power.y, b: power.z,
                });
            }
            return None;
        }
    }
    None
}
