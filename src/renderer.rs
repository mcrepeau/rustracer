use crate::camera::Camera;
use crate::hittable::{Hittable, HittableList};
use crate::pdf::{CosinePdf, HittablePdf, MixturePdf, Pdf};
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
    /// Procedural star field: hash-based pseudo-random stars on the unit sphere.
    Stars,
}

impl Background {
    pub fn eval(self, dir: Vec3) -> Color {
        match self {
            Background::Solid(c) => c,
            Background::Physical { sun_dir } => sky_color(dir, sun_dir),
            Background::Stars => star_field(dir),
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

/// Procedural star field with spectral colour variation, power-law magnitude
/// distribution, and a Milky Way band.
///
/// Stars: hash-based ~0.002-radian cells, ~0.4 % base density boosted up to
/// 5× near the galactic equator and centre.  Each star draws independent
/// hashes for magnitude (t² power law → many dim, few bright) and spectral
/// class (O/B through M, weighted toward a realistic visible-sky colour mix).
///
/// Milky Way: a cool blue-grey diffuse glow along the galactic band, with a
/// warm orange concentration toward the galactic centre (old stellar
/// populations, interstellar reddening).
///
/// Galactic frame: pole and centre directions are derived from J2000
/// equatorial coordinates via the standard ecliptic rotation (ε = 23.44°),
/// then expressed in scene space (Y-up, XZ = ecliptic plane).
fn star_field(dir: Vec3) -> Color {
    let d = dir.unit();

    // ── Galactic frame ────────────────────────────────────────────────────────
    // Galactic north pole: equatorial RA 12h51.4m, Dec +27.1°  → scene space.
    // Galactic centre:     equatorial RA 17h45.7m, Dec −29.0°  → scene space.
    // The two vectors are perpendicular by construction (b = 90° vs b = 0°).
    let gal_pole   = Vec3::new(-0.8679,  0.4977, -0.0009);
    let gal_center = Vec3::new(-0.0561, -0.0977, -0.9935);

    let sin_lat       = d.dot(gal_pole);           // signed galactic latitude sine
    let toward_center = d.dot(gal_center).max(0.0); // 0 → 1 toward galactic centre

    // band_t: 1.0 on the galactic equator, 0.0 at the poles.
    let band_t = 1.0 - sin_lat * sin_lat;

    // ── Milky Way diffuse glow ────────────────────────────────────────────────
    let band_glow   = band_t.powf(4.5);
    let center_glow = (toward_center * toward_center * band_t).powf(1.5);
    let milky = Color::new(0.016, 0.014, 0.028) * band_glow       // cool blue-grey haze
              + Color::new(0.060, 0.030, 0.020) * center_glow;    // warm centre glow

    // ── Star cells ────────────────────────────────────────────────────────────
    // Quantise direction to ~0.002-radian cells; offset keeps negatives positive.
    let ix = (d.x * 500.0 + 500.5) as u32;
    let iy = (d.y * 500.0 + 500.5) as u32;
    let iz = (d.z * 500.0 + 500.5) as u32;
    let h  = ix.wrapping_mul(2654435761)
           ^ iy.wrapping_mul(2246822519)
           ^ iz.wrapping_mul(3266489917);
    // Two independent LCG steps for magnitude and spectral class.
    let h2 = h .wrapping_mul(1664525).wrapping_add(1013904223);
    let h3 = h2.wrapping_mul(1664525).wrapping_add(1013904223);

    // Density: ~0.4 % base, up to 5× higher near galactic equator / centre.
    let density_boost = 1.0
        + 2.5 * band_t.powf(2.0)
        + 1.5 * (toward_center * band_t).powf(2.0);
    let thresh = (17_179_869_u32 as f32 * density_boost) as u32;

    let star = if h < thresh {
        // t² power law: most stars are dim, a few are very bright.
        let mag_t      = h2 as f32 / u32::MAX as f32;
        let brightness = 0.10 + mag_t * mag_t * 2.5; // 0.10 … 2.60

        // Spectral class — weights approximate real visible-sky colour mix.
        let col_t = h3 as f32 / u32::MAX as f32;
        let hue = if col_t < 0.04 {
            Color::new(0.70, 0.83, 1.00) // O/B  blue-white     4 %
        } else if col_t < 0.13 {
            Color::new(0.93, 0.96, 1.00) // A    white           9 %
        } else if col_t < 0.30 {
            Color::new(1.00, 0.97, 0.87) // F    yellow-white   17 %
        } else if col_t < 0.55 {
            Color::new(1.00, 0.92, 0.68) // G    yellow         25 %
        } else if col_t < 0.80 {
            Color::new(1.00, 0.76, 0.48) // K    orange         25 %
        } else {
            Color::new(1.00, 0.55, 0.32) // M    orange-red     20 %
        };

        hue * brightness
    } else {
        Color::default()
    };

    star + milky
}

// ── Path tracer ───────────────────────────────────────────────────────────────

/// `bg_scale` is multiplied into the background sample only (not scene hits).
/// Pass `1.0 / exposure` to keep the star field at constant apparent brightness
/// regardless of the scene exposure setting.
pub fn ray_color(r: &Ray, world: &dyn Hittable, background: Background, lights: &HittableList, bg_scale: f32, rng: &mut impl Rng) -> Color {
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
                    let scattered_dir;
                    let pdf_val;

                    if lights.objects.is_empty() {
                        let cpdf = CosinePdf::new(rec.normal);
                        scattered_dir = cpdf.generate(rng);
                        pdf_val = cpdf.value(scattered_dir);
                    } else {
                        let cpdf  = CosinePdf::new(rec.normal);
                        let lpdf  = HittablePdf::new(lights, rec.p);
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
    bg_scale:   f32,
) {
    let w        = width  as usize;
    let w_denom  = (width  - 1).max(1) as f32;
    let h_denom  = (height - 1).max(1) as f32;
    let strata2  = strata * strata;
    let strata_f = strata as f32;

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
        *out = ray_color(&camera.get_ray(u, v, &mut rng), world, background, lights, bg_scale, &mut rng);
    });
}
