use std::f32::consts::PI;
use hashbrown::HashMap;
use rand::Rng;
use rand::rngs::SmallRng;
use rand::SeedableRng;
use rayon::prelude::*;
use crate::hittable::Hittable;
use crate::ray::Ray;
use crate::vec3::{Color, Point3, Vec3};

struct RawPhoton { x: f32, y: f32, z: f32, r: f32, g: f32, b: f32 }

const DISK_R: f32 = 20.0;

/// Grid-accelerated caustic photon map.
///
/// Photons are emitted either from a directional disk (sun) or from a
/// rectangular area light, and only stored when they reach a diffuse surface
/// after at least one specular or transmissive bounce.
///
/// During rendering, `irradiance()` returns the Epanechnikov-filtered estimate
/// of the caustic irradiance at a surface point, pre-divided by π so the
/// caller only needs to multiply by the surface albedo to get radiance.
pub struct PhotonMap {
    photons:   Vec<RawPhoton>,
    grid:      HashMap<(i32, i32, i32), Vec<u32>>,
    gather_r2: f32,
    /// Epanechnikov kernel normaliser ÷ π: 2/(π² R²).
    /// Pre-folding the Lambertian 1/π lets callers write `albedo * irradiance`.
    norm:      f32,
    /// Grid cell size = gather radius.  Stored so `irradiance` can compute
    /// cell coordinates without a global constant.
    cell_r:    f32,
}

impl PhotonMap {
    /// Trace `num_photons` from a disk facing `sun_dir` and build the map.
    /// `sun_color` should be `background.eval(sun_dir) * PI` so the caustic
    /// brightness is automatically calibrated to the sky model.
    /// `gather_radius` must match the spatial scale of the scene.
    pub fn build(world: &dyn Hittable, sun_dir: Vec3, sun_color: Color, num_photons: u32, gather_radius: f32) -> Self {
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

        Self::from_photons(photons, gather_radius)
    }

    /// Trace `num_photons` emitted as a Lambertian source from the rectangular
    /// area light defined by `origin`, `u`, `v` and build the map.
    /// `gather_radius` must match the spatial scale of the scene.
    pub fn build_from_quad(
        world:        &dyn Hittable,
        light_origin: Point3,
        light_u:      Vec3,
        light_v:      Vec3,
        light_color:  Color,
        num_photons:  u32,
        gather_radius: f32,
    ) -> Self {
        // Total Lambertian flux = L × π × area; power per photon = flux / N.
        let quad_area    = light_u.cross(light_v).length();
        let photon_power = light_color * (quad_area * PI / num_photons as f32);

        let photons: Vec<RawPhoton> = (0..num_photons)
            .into_par_iter()
            .filter_map(|i| {
                let mut rng = SmallRng::seed_from_u64(
                    (i as u64).wrapping_mul(6_364_136_223_846_793_005)
                        ^ 0x9E3779B97F4A7C15,
                );
                // Uniform random point on the quad.
                let s = rng.gen::<f32>();
                let t = rng.gen::<f32>();
                let origin = light_origin + light_u * s + light_v * t;
                // Cosine-weighted direction in the lower hemisphere (-Y dominant).
                // pdf = cos θ / π, importance weight = 1 (cancels with flux formula).
                let cos_theta = rng.gen::<f32>().sqrt();
                let sin_theta = (1.0 - cos_theta * cos_theta).sqrt();
                let phi = 2.0 * PI * rng.gen::<f32>();
                let dir = Vec3::new(sin_theta * phi.cos(), -cos_theta, sin_theta * phi.sin());
                trace_photon(world, origin, dir, photon_power, &mut rng)
            })
            .collect();

        Self::from_photons(photons, gather_radius)
    }

    fn from_photons(photons: Vec<RawPhoton>, gather_radius: f32) -> Self {
        let cell = |x: f32| (x / gather_radius).floor() as i32;
        let mut grid: HashMap<(i32, i32, i32), Vec<u32>> = HashMap::new();
        for (idx, p) in photons.iter().enumerate() {
            grid.entry((cell(p.x), cell(p.y), cell(p.z))).or_default().push(idx as u32);
        }
        Self {
            photons,
            grid,
            gather_r2: gather_radius * gather_radius,
            norm:      2.0 / (PI * PI * gather_radius * gather_radius),
            cell_r:    gather_radius,
        }
    }

    /// Epanechnikov-filtered irradiance estimate at `pos`, already divided by π
    /// so the caller multiplies by albedo to get reflected radiance.
    pub fn irradiance(&self, pos: Point3) -> Color {
        let cell = |x: f32| (x / self.cell_r).floor() as i32;
        let cx = cell(pos.x);
        let cy = cell(pos.y);
        let cz = cell(pos.z);
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
        let mut ray = Ray::new_at_time(pos, dir, 0.0);
        ray.wavelength = rng.gen_range(380.0_f32..700.0);
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
