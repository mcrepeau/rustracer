use std::collections::HashMap;
use std::f32::consts::PI;
use rand::Rng;
use rand::rngs::SmallRng;
use rand::SeedableRng;
use rayon::prelude::*;
use crate::hittable::Hittable;
use crate::ray::Ray;
use crate::vec3::{Color, Point3, Vec3};

struct RawPhoton { x: f32, y: f32, z: f32, r: f32, g: f32, b: f32 }

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
    grid:          HashMap<(i32, i32, i32), Vec<u32>>,
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

        let mut grid: HashMap<(i32, i32, i32), Vec<u32>> = HashMap::new();
        for (idx, p) in photons.iter().enumerate() {
            let key = cell_key(p.x, p.y, p.z, GATHER_RADIUS);
            grid.entry(key).or_default().push(idx as u32);
        }

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
                    if let Some(ids) = self.grid.get(&(cx + dx, cy + dy, cz + dz)) {
                        for &id in ids {
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
        }
        acc * self.norm
    }

    #[allow(dead_code)]
    pub fn stored_count(&self) -> usize { self.photons.len() }
}

#[inline] fn cell_coord(x: f32, size: f32) -> i32 { (x / size).floor() as i32 }
#[inline] fn cell_key(x: f32, y: f32, z: f32, size: f32) -> (i32, i32, i32) {
    (cell_coord(x, size), cell_coord(y, size), cell_coord(z, size))
}

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
            // Specular or transmissive bounce — continue tracing
            power     *= sr.attenuation;
            pos        = rec.p;
            dir        = sr.ray.direction;
            spec_depth += 1;

            // Guard against SpectralDielectric 3× channel spikes accumulating
            // over multiple bounces inside the diamond.
            let lum = 0.2126 * power.x + 0.7152 * power.y + 0.0722 * power.z;
            if lum > 15.0 { power *= 15.0 / lum; }
        } else {
            // First diffuse hit: store only if caustic path (spec_depth > 0)
            // and the surface is roughly upward-facing (ground plane, not a
            // sphere's side face).  This prevents photons that exit a marble
            // sideways from lighting up neighbouring sphere surfaces, which
            // produces an unnatural "marble-as-lamp" glow.
            if spec_depth > 0 && rec.normal.y > 0.7 {
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
