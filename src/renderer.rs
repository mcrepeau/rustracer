use crate::camera::Camera;
use crate::hittable::{Hittable, HittableList};
use crate::pdf::{CosinePdf, HittablePdf, MixturePdf, Pdf};
use crate::ray::Ray;
use crate::vec3::{Color, Vec3};
use rand::Rng;
use rand::rngs::SmallRng;
use rand::SeedableRng;
use rayon::prelude::*;

const MAX_DEPTH:    i32 = 50;
const MAX_LUMINANCE: f32 = 10.0;

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

pub fn ray_color(r: &Ray, world: &dyn Hittable, background: Background, lights: &HittableList, rng: &mut impl Rng) -> Color {
    let mut throughput = Color::new(1.0, 1.0, 1.0);
    let mut color      = Color::default();
    let mut ray        = *r;

    for depth in 0..MAX_DEPTH {
        match world.hit(&ray, 0.001, f32::INFINITY) {
            None => {
                color += throughput * background.eval(ray.direction);
                break;
            }
            Some(rec) => {
                color += throughput * rec.mat.emitted();

                let Some(sr) = rec.mat.scatter(&ray, &rec, rng) else { break; };

                if sr.skip_pdf {
                    throughput *= sr.attenuation;
                    ray = sr.ray;
                } else {
                    let scattered_dir;
                    let pdf_val;

                    if lights.objects.is_empty() {
                        let cpdf = CosinePdf::new(rec.normal);
                        scattered_dir = cpdf.generate(rng);
                        pdf_val = cpdf.value(scattered_dir);
                    } else {
                        let cpdf  = CosinePdf::new(rec.normal);
                        let lpdf  = HittablePdf::new(lights, rec.p);
                        let mpdf  = MixturePdf::new(&cpdf, &lpdf);
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
                    if survive <= 0.0 || rng.gen::<f32>() >= survive { break; }
                    throughput /= survive;
                }
            }
        }
    }

    let lum = color.x * 0.2126 + color.y * 0.7152 + color.z * 0.0722;
    if lum > MAX_LUMINANCE { color *= MAX_LUMINANCE / lum; }
    color
}

// ── Tile renderer ─────────────────────────────────────────────────────────────

/// Render one sample pass into `scratch` in parallel.
/// `strata` = floor(sqrt(max_samples)); controls the stratified-sampling grid size.
/// Pass an empty slice for `conv` to disable adaptive-sampling skipping.
#[allow(clippy::too_many_arguments)]
pub fn render_tiles(
    scratch:    &mut [Color],
    sample_idx: u32,
    strata:     u32,
    width:      u32,
    height:     u32,
    camera:     &Camera,
    world:      &dyn Hittable,
    background: Background,
    lights:     &HittableList,
    conv:       &[bool],
) {
    let w        = width  as usize;
    let w_denom  = (width  - 1).max(1) as f32;
    let h_denom  = (height - 1).max(1) as f32;
    let adaptive = !conv.is_empty();
    let strata2  = strata * strata;
    let strata_f = strata as f32;

    scratch.par_iter_mut().enumerate().for_each(|(i, out)| {
        *out = if adaptive && conv[i] {
            Color::default()
        } else {
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
            ray_color(&camera.get_ray(u, v, &mut rng), world, background, lights, &mut rng)
        };
    });
}
