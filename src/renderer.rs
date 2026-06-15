use crate::camera::Camera;
use crate::hittable::{Hittable, HittableList};
use crate::material::{clear_pearl_sun_dir, set_pearl_sun_dir};
use crate::pdf::{CosinePdf, HittablePdf, Pdf};
use crate::photon::PhotonMap;
use crate::ray::Ray;
use crate::vec3::{Color, Vec3};
use rand::Rng;
use rand::rngs::SmallRng;
use rand::SeedableRng;
use rayon::prelude::*;

const MAX_DEPTH:     i32 = 50;
const MAX_LUMINANCE: f32 = 10.0;

// ── Background ────────────────────────────────────────────────────────────────

#[derive(Clone, Copy)]
pub enum Background {
    Solid(Color),
    Physical { sun_dir: Vec3, turbidity: f32 },
}

impl Background {
    pub fn eval(self, dir: Vec3) -> Color {
        match self {
            Background::Solid(c) => c,
            Background::Physical { sun_dir, turbidity } => preetham_sky(dir, sun_dir, turbidity),
        }
    }
}

/// Preetham (1999) analytic sky model.
///
/// Computes CIE Yxy from the Perez luminance distribution, converts to
/// linear sRGB.  Turbidity T ∈ [1, 20]: 2 = very clear, 3 = clear,
/// 5 = light haze, 10 = heavy haze.  Output is scaled so a midday
/// zenith sits around 0.7 in linear before ACES tone mapping.
fn preetham_sky(dir: Vec3, sun_dir: Vec3, turbidity: f32) -> Color {
    use std::f32::consts::PI;

    let d   = dir.unit();
    let sun = sun_dir.unit();

    // Angle between sky sample and sun
    let cos_gamma = d.dot(sun).clamp(-1.0, 1.0);
    let gamma     = cos_gamma.acos();

    // Below ground — return black
    if d.y < 0.0 { return Color::default(); }

    let t = turbidity.clamp(1.0, 20.0);

    // Sun zenith angle (floor at 1° above horizon; model invalid for θ_s > 90°)
    let cos_theta_s = sun.y.max(0.0175);   // sin(1°) ≈ 0.0175
    let theta_s     = cos_theta_s.acos();

    // Perez distribution F(θ,γ; A–E) = (1+A·exp(B/cosθ))·(1+C·exp(D·γ)+E·cos²γ)
    let perez = |a: f32, b: f32, c: f32, dp: f32, e: f32,
                 ct: f32, g: f32, cg: f32| -> f32 {
        (1.0 + a * (b / ct).exp()) * (1.0 + c * (dp * g).exp() + e * cg * cg)
    };

    // Coefficients A–E (each linearly dependent on turbidity)
    let (ay, by_, cy, dy, ey) = (
         0.1787*t - 1.4630, -0.3554*t + 0.4275,
         0.0227*t + 5.3251,  0.1206*t - 2.5771, -0.0670*t + 0.3703,
    );
    let (ax, bx, cx, dx, ex) = (
        -0.0193*t - 0.2592, -0.0665*t + 0.0008,
        -0.0004*t + 0.2125, -0.0641*t - 0.8989, -0.0033*t + 0.0452,
    );
    let (acy, bcy, ccy, dcy, ecy) = (
        -0.0167*t - 0.2608, -0.0950*t + 0.0092,
        -0.0079*t + 0.2102, -0.0441*t - 1.6537, -0.0109*t + 0.0529,
    );

    // Normalization: F at zenith (cosθ=1) looking toward sun (γ=θ_s)
    let fn_y  = perez(ay,  by_,  cy,  dy,  ey,  1.0, theta_s, cos_theta_s);
    let fn_x  = perez(ax,  bx,   cx,  dx,  ex,  1.0, theta_s, cos_theta_s);
    let fn_cy = perez(acy, bcy,  ccy, dcy, ecy, 1.0, theta_s, cos_theta_s);

    // Zenith luminance Y_z (kcd/m²), Preetham eq. 7
    let yz = ((4.0453*t - 4.9710) * ((4.0/9.0 - t/120.0) * (PI - 2.0*theta_s)).tan()
             - 0.2155*t + 2.4192).max(0.0);

    // Solar disc: rays within ~1.3° of sun get a bright warm-white return
    if cos_gamma > 0.9997 && sun.y > 0.0 {
        let disc = yz * 80.0 * 0.05;
        return Color::new(disc, disc * 0.95, disc * 0.85);
    }

    // Zenith chromaticity x_z, y_z (Preetham table 2 polynomial, θ_s in radians)
    let ts  = theta_s;
    let ts2 = ts * ts;
    let ts3 = ts2 * ts;
    let xz = t*t * ( 0.00166*ts3 - 0.00375*ts2 + 0.00209*ts)
           + t   * (-0.02903*ts3 + 0.06377*ts2 - 0.03202*ts + 0.00394)
           +       ( 0.11693*ts3 - 0.21196*ts2 + 0.06052*ts + 0.25886);
    let yzc = t*t * ( 0.00275*ts3 - 0.00610*ts2 + 0.00317*ts)
            + t   * (-0.04214*ts3 + 0.08970*ts2 - 0.04153*ts + 0.00516)
            +       ( 0.15346*ts3 - 0.26756*ts2 + 0.06670*ts + 0.26688);

    // Evaluate Perez distribution at sky sample (clamp cosθ to avoid horizon singularity)
    let ct  = d.y.max(0.001);
    let fy  = perez(ay,  by_,  cy,  dy,  ey,  ct, gamma, cos_gamma);
    let fx  = perez(ax,  bx,   cx,  dx,  ex,  ct, gamma, cos_gamma);
    let fcy = perez(acy, bcy,  ccy, dcy, ecy, ct, gamma, cos_gamma);

    // Sky Yxy
    let cap_y  = (yz  * fy  / fn_y).max(0.0);
    let cap_x  = (xz  * fx  / fn_x).clamp(0.001, 0.998);
    let cap_yc = (yzc * fcy / fn_cy).clamp(0.001, 0.998);

    // xyY → XYZ
    // Scale kcd/m² → renderer linear units (0.05 maps midday zenith to ~0.7 linear)
    let lum   = cap_y * 0.05;
    let big_x = cap_x / cap_yc * lum;
    let big_y = lum;
    let big_z = ((1.0 - cap_x - cap_yc) / cap_yc * lum).max(0.0);

    // XYZ → linear sRGB (D65)
    let r = ( 3.2406*big_x - 1.5372*big_y - 0.4986*big_z).max(0.0);
    let g = (-0.9689*big_x + 1.8758*big_y + 0.0415*big_z).max(0.0);
    let b = ( 0.0557*big_x - 0.2040*big_y + 1.0570*big_z).max(0.0);

    Color::new(r, g, b)
}

// ── Path tracer ───────────────────────────────────────────────────────────────

/// `bg_scale` is multiplied into the background sample only (not scene hits).
pub fn ray_color(r: &Ray, world: &dyn Hittable, background: Background, lights: &HittableList, bg_scale: f32, photon_map: Option<&PhotonMap>, rng: &mut impl Rng) -> Color {
    let mut throughput      = Color::new(1.0, 1.0, 1.0);
    let mut color           = Color::default();
    let mut ray             = *r;
    let mut prev_specular   = true; // camera ray: always add full emission on first hit
    let mut prev_mis_w_brdf = 1.0f32; // MIS weight for emission from the previous BRDF sample

    for depth in 0..MAX_DEPTH {
        match world.hit(&ray, 0.001, f32::INFINITY) {
            None => {
                color += throughput * background.eval(ray.direction) * bg_scale;
                break;
            }
            Some(rec) => {
                // Emission weight:
                //   1.0  — camera ray or after a specular bounce (no NEE was done, no
                //          double-counting risk).
                //   MIS  — after a diffuse bounce: the previous iteration already added
                //          a NEE estimate for direct lighting, so the BRDF-sample path
                //          that lands on the light gets the complementary MIS weight
                //          w_brdf = p_brdf / (p_brdf + p_nee) to avoid double-counting
                //          while still capturing unsampled directions.
                //   1.0  — no area lights in the scene (no NEE at all).
                let emit_w = if prev_specular || lights.objects.is_empty() { 1.0 }
                             else { prev_mis_w_brdf };
                color += throughput * rec.mat.emitted(rec.u, rec.v, rec.p) * emit_w;

                let Some(sr) = rec.mat.scatter(&ray, &rec, rng) else { break; };

                if sr.skip_pdf {
                    throughput      *= sr.attenuation;
                    ray              = sr.ray;
                    prev_specular    = true;
                    prev_mis_w_brdf  = 1.0;
                } else {
                    // Caustic injection: add photon-map irradiance at this diffuse
                    // surface. The photon map stores only caustic paths (at least one
                    // specular bounce), so there is no double-counting with NEE below.
                    if let Some(pm) = photon_map {
                        let irr = pm.irradiance(rec.p);
                        if irr.x > 0.0 || irr.y > 0.0 || irr.z > 0.0 {
                            let alb = rec.mat.albedo_hint(rec.u, rec.v, rec.p);
                            color += throughput * alb * irr;
                        }
                    }

                    // ── Direct lighting: explicit NEE with MIS balance heuristic ──
                    // w_nee = p_nee / (p_nee + p_brdf)
                    // The estimator simplifies to: attenuation * brdf * L_e / (p_nee + p_brdf)
                    if !lights.objects.is_empty() {
                        let lpdf      = HittablePdf::new(lights, rec.p, ray.time);
                        let light_dir = lpdf.generate(rng);
                        let shadow    = Ray::new_at_time(rec.p, light_dir, ray.time);
                        if let Some(lrec) = lights.hit(&shadow, 0.001, f32::INFINITY) {
                            if !world.any_hit(&shadow, 0.001, lrec.t * (1.0 - 1e-4)) {
                                let l_pdf = lpdf.value(light_dir);
                                let brdf  = rec.mat.scattering_pdf(&ray, &rec, &shadow);
                                let mis_d = l_pdf + brdf; // balance heuristic denominator
                                if mis_d > 0.0 && brdf > 0.0 {
                                    let nee_emit = lrec.mat.emitted(lrec.u, lrec.v, lrec.p);
                                    color += throughput * sr.attenuation * brdf * nee_emit / mis_d;
                                }
                            }
                        }
                    }

                    // ── Indirect lighting: cosine-weighted BRDF sample ────────────
                    // Also compute the MIS weight for the case where this ray hits a
                    // light next iteration: w_brdf = p_brdf / (p_brdf + p_nee).
                    let cpdf     = CosinePdf::new(rec.normal);
                    let ind_dir  = cpdf.generate(rng);
                    let pdf_val  = cpdf.value(ind_dir);
                    if pdf_val <= 0.0 { break; }
                    let scattered = Ray::scatter_from(rec.p, ind_dir, &ray);
                    let scat_pdf  = rec.mat.scattering_pdf(&ray, &rec, &scattered);
                    if scat_pdf <= 0.0 { break; }

                    let nee_pdf_for_ind = if lights.objects.is_empty() { 0.0 }
                                         else { lights.pdf_value(rec.p, ind_dir, ray.time) };
                    prev_mis_w_brdf = pdf_val / (pdf_val + nee_pdf_for_ind).max(1e-8);

                    throughput    *= sr.attenuation * (scat_pdf / pdf_val);
                    ray            = scattered;
                    prev_specular  = false;
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
        Background::Physical { sun_dir, .. } => set_pearl_sun_dir(sun_dir),
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
        let mut cam_ray = camera.get_ray(u, v, &mut rng);
        // Sample a hero wavelength uniformly over the visible spectrum.
        // Spectral materials (dispersive glass, thin-film pearl) read this
        // from the ray; non-spectral materials ignore it.
        cam_ray.wavelength = rng.gen_range(380.0_f32..700.0);
        *out = ray_color(&cam_ray, world, background, lights, bg_scale, photon_map, &mut rng);
    });
}
