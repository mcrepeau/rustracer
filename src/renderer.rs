use crate::camera::Camera;
use crate::hittable::{Hittable, HittableList};
use crate::material::{clear_pearl_sun_dir, set_pearl_sun_dir};
use crate::pdf::{CosinePdf, HittablePdf, MixturePdf, Pdf};
use crate::photon::PhotonMap;
use crate::ray::Ray;
use crate::vec3::{Color, Vec3};
use rand::Rng;
use rand::rngs::SmallRng;
use rand::SeedableRng;
use rayon::prelude::*;

const MAX_DEPTH:      i32 = 50;
const MAX_LUMINANCE:  f32 = 10.0;
// Fraction of scatter samples drawn toward explicit lights vs. cosine lobe.
// Higher values reduce noise in scenes dominated by a single area light;
// lower values help when indirect illumination dominates.
const LIGHT_PDF_WEIGHT: f32 = 0.7;

// ── Background ────────────────────────────────────────────────────────────────

#[derive(Clone, Copy)]
pub enum Background {
    Solid(Color),
    Physical { sun_dir: Vec3 },
}

impl Background {
    pub fn eval(self, dir: Vec3) -> Color {
        match self {
            Background::Solid(c) => c,
            Background::Physical { sun_dir } => sky_color(dir, sun_dir),
        }
    }
}

/// Physically-inspired sky model: blue zenith shading to warm horizon, soft Mie glow.
fn sky_color(dir: Vec3, sun_dir: Vec3) -> Color {
    let d        = dir.unit();
    let sun      = sun_dir.unit();
    let sun_elev = sun.y.clamp(0.0, 1.0);
    let cos_a    = d.dot(sun).max(0.0);
    let t        = d.y.max(0.0).powf(0.4);

    let zenith  = Color::new(0.08, 0.22, 0.75) * (0.4 + 0.6 * sun_elev);
    let horizon = Color::new(0.70, 0.55, 0.35) * (1.0 - sun_elev)
                + Color::new(0.65, 0.78, 0.92) *  sun_elev;
    let sky = zenith * t + horizon * (1.0 - t);

    let mie = Color::new(1.0, 0.85, 0.60) * cos_a.powf(8.0) * 0.8 * sun_elev;

    if d.y < 0.0 {
        sky * (1.0 + d.y * 5.0).max(0.0)
    } else {
        sky + mie
    }
}

// ── Path tracer ───────────────────────────────────────────────────────────────

/// `bg_scale` is multiplied into the background sample only (not scene hits).
/// Pass `1.0 / exposure` to keep the star field at constant apparent brightness
/// regardless of the scene exposure setting.
pub fn ray_color(r: &Ray, world: &dyn Hittable, background: Background, lights: &HittableList, bg_scale: f32, photon_map: Option<&PhotonMap>, rng: &mut impl Rng) -> Color {
    let mut throughput = Color::new(1.0, 1.0, 1.0);
    let mut color      = Color::default();
    let mut ray        = *r;

    for depth in 0..MAX_DEPTH {
        match world.hit(&ray, 0.001, f32::INFINITY) {
            None => {
                color += throughput * background.eval(ray.direction) * bg_scale;
                break;
            }
            Some(rec) => {
                color += throughput * rec.mat.emitted(rec.u, rec.v, rec.p);

                let Some(sr) = rec.mat.scatter(&ray, &rec, rng) else { break; };

                if sr.skip_pdf {
                    throughput *= sr.attenuation;
                    ray = sr.ray;
                } else {
                    // Caustic injection: add photon-map irradiance at this
                    // diffuse surface.  The photon map stores only paths that
                    // went through at least one specular bounce (caustics),
                    // so there is no double-counting with direct NEE.
                    // `irradiance()` already divides by π, so we only need
                    // to multiply by the surface albedo (Lambertian: L=albedo×E/π).
                    if let Some(pm) = photon_map {
                        let irr = pm.irradiance(rec.p);
                        if irr.x > 0.0 || irr.y > 0.0 || irr.z > 0.0 {
                            let alb = rec.mat.albedo_hint(rec.u, rec.v, rec.p);
                            color += throughput * alb * irr;
                        }
                    }

                    let scattered_dir;
                    let pdf_val;

                    if lights.objects.is_empty() {
                        let cpdf = CosinePdf::new(rec.normal);
                        scattered_dir = cpdf.generate(rng);
                        pdf_val = cpdf.value(scattered_dir);
                    } else {
                        let cpdf  = CosinePdf::new(rec.normal);
                        let lpdf  = HittablePdf::new(lights, rec.p, ray.time);
                        let mpdf  = MixturePdf::new(&cpdf, &lpdf, LIGHT_PDF_WEIGHT);
                        scattered_dir = mpdf.generate(rng);
                        pdf_val       = mpdf.value(scattered_dir);
                    };

                    if pdf_val <= 0.0 { break; }
                    let scattered = Ray::new_at_time(rec.p, scattered_dir, ray.time);
                    let scat_pdf  = rec.mat.scattering_pdf(&ray, &rec, &scattered);
                    if scat_pdf <= 0.0 { break; }

                    throughput *= sr.attenuation * (scat_pdf / pdf_val);
                    ray = scattered;
                }

                if depth >= 2 {
                    let survive = throughput.x.max(throughput.y).max(throughput.z);
                    let p = survive.min(1.0);
                    if survive <= 0.0 || rng.gen::<f32>() >= p { break; }
                    throughput /= p;
                }
            }
        }
    }

    let lum = color.x * 0.2126 + color.y * 0.7152 + color.z * 0.0722;
    if lum > MAX_LUMINANCE { color *= MAX_LUMINANCE / lum; }
    color
}

// ── Auxiliary pass (albedo + normal) for OIDN ────────────────────────────────

/// Render a single first-hit pass and return flat `f32` buffers suitable for
/// passing to OIDN as the albedo and normal auxiliary inputs.
///
/// Each buffer has `width * height * 3` elements.  The albedo is the material's
/// unlit base colour (clamped to [0, 1]); the normal is the world-space shading
/// normal.  Background pixels carry the sky colour as albedo and a zero normal.
/// Using pixel centres (no jitter) keeps the buffers essentially noise-free so
/// OIDN can apply `clean_aux` quality internally.
#[cfg(feature = "denoise")]
pub fn render_aux_pass(
    width:      u32,
    height:     u32,
    camera:     &Camera,
    world:      &dyn Hittable,
    background: Background,
) -> (Vec<f32>, Vec<f32>) {
    let w       = width  as usize;
    let n       = w * height as usize;
    let w_denom = (width  - 1).max(1) as f32;
    let h_denom = (height - 1).max(1) as f32;

    let pairs: Vec<([f32; 3], [f32; 3])> = (0..n)
        .into_par_iter()
        .map(|i| {
            let row   = i / w;
            let col   = i % w;
            let mut rng = SmallRng::seed_from_u64(i as u64 ^ 0x9E3779B97F4A7C15);
            // Pixel centre — no jitter — keeps albedo/normal clean.
            let u = (col as f32 + 0.5) / w_denom;
            let v = ((height - 1 - row as u32) as f32 + 0.5) / h_denom;
            let ray = camera.get_ray(u, v, &mut rng);

            match world.hit(&ray, 0.001, f32::INFINITY) {
                None => {
                    let bg = background.eval(ray.direction);
                    ([bg.x, bg.y, bg.z], [0.0f32; 3])
                }
                Some(rec) => {
                    let alb = rec.mat.albedo_hint(rec.u, rec.v, rec.p);
                    let n   = rec.normal;
                    (
                        [alb.x.clamp(0.0, 1.0), alb.y.clamp(0.0, 1.0), alb.z.clamp(0.0, 1.0)],
                        [n.x, n.y, n.z],
                    )
                }
            }
        })
        .collect();

    let mut albedo = Vec::with_capacity(n * 3);
    let mut normal = Vec::with_capacity(n * 3);
    for (a, n) in pairs {
        albedo.extend_from_slice(&a);
        normal.extend_from_slice(&n);
    }
    (albedo, normal)
}

// ── Tile renderer ─────────────────────────────────────────────────────────────

/// Render one sample pass into `scratch` in parallel.
/// `strata` = floor(sqrt(max_samples)); controls the stratified-sampling grid size.
#[allow(clippy::too_many_arguments)]
pub fn render_tiles(
    scratch:     &mut [Color],
    sample_idx:  u32,
    strata:      u32,
    width:       u32,
    height:      u32,
    camera:      &Camera,
    world:       &dyn Hittable,
    background:  Background,
    lights:      &HittableList,
    bg_scale:    f32,
    photon_map:  Option<&PhotonMap>,
) {
    let w        = width  as usize;
    let w_denom  = (width  - 1).max(1) as f32;
    let h_denom  = (height - 1).max(1) as f32;
    let strata2  = strata * strata;
    let strata_f = strata as f32;

    match background {
        Background::Physical { sun_dir } => set_pearl_sun_dir(sun_dir),
        _                                => clear_pearl_sun_dir(),
    }

    scratch.par_iter_mut().enumerate().for_each(|(i, out)| {
        let row = i / w;
        let col = i % w;
            let mut rng = SmallRng::seed_from_u64(
                (i as u64).wrapping_mul(6364136223846793005)
                    ^ (sample_idx as u64).wrapping_mul(0x9E3779B97F4A7C15),
            );
            let ray_y = height - 1 - row as u32;

            // Stratified pixel sampling: map sample_idx into a strata×strata grid.
            // A per-pixel cyclic offset (Fibonacci hash) ensures neighboring pixels
            // visit strata in different orders, avoiding spatial correlation.
            let (u_jitter, v_jitter) = if strata2 > 0 && sample_idx < strata2 {
                let offset = (i as u32).wrapping_mul(0x9E3779B9) % strata2;
                let s  = (sample_idx + offset) % strata2;
                let sx = s % strata;
                let sy = s / strata;
                (
                    (sx as f32 + rng.gen::<f32>()) / strata_f,
                    (sy as f32 + rng.gen::<f32>()) / strata_f,
                )
            } else {
                (rng.gen::<f32>(), rng.gen::<f32>())
            };

            let u = (col as f32 + u_jitter) / w_denom;
            let v = (ray_y as f32 + v_jitter) / h_denom;
        *out = ray_color(&camera.get_ray(u, v, &mut rng), world, background, lights, bg_scale, photon_map, &mut rng);
    });
}
