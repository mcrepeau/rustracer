use std::sync::Arc;
use image::Rgb32FImage;
use crate::camera::Camera;
use crate::hittable::{Hittable, HittableList};
use crate::material::{clear_pearl_sun_dir, set_pearl_sun_dir};
use crate::pdf::{CosinePdf, HittablePdf, Pdf};
use crate::photon::{PhotonMap, PHOTON_MAX_DEPTH, SppmPixel, VisiblePoint};
use crate::ray::Ray;
use crate::vec3::{Color, Vec3};
use rand::Rng;
use rand::rngs::SmallRng;
use rand::SeedableRng;
use rayon::prelude::*;


const MAX_DEPTH:   i32 = 50;
/// True solar angular radius ≈ 0.265° → cos(0.265° * π/180) ≈ 0.9999892.
const COS_SUN_MAX: f32 = 0.9999892;

// ── Background ────────────────────────────────────────────────────────────────

/// Equirectangular HDR environment map loaded from an EXR file.
pub struct EnvMapData {
    image:        Rgb32FImage,
    marginal_cdf: Vec<f32>,  // len = H; CDF over image rows
    cond_cdf:     Vec<f32>,  // len = H*W, row-major; conditional CDF over columns per row
    inv_total:    f32,       // reciprocal of Σ(luminance × sinθ), used in PDF evaluation
}

impl EnvMapData {
    pub fn new(image: Rgb32FImage) -> Self {
        use std::f32::consts::PI;
        let w = image.width()  as usize;
        let h = image.height() as usize;
        let lum = |r: f32, g: f32, b: f32| 0.2126 * r + 0.7152 * g + 0.0722 * b;

        // Build per-row conditional CDFs and collect the row marginal weights.
        // Each pixel is weighted by luminance × sin(θ) — the solid-angle element
        // for an equirectangular map — so bright, equator-facing pixels are sampled
        // more often than equally bright pixels near the poles.
        let mut row_weights = vec![0.0f32; h];
        let mut cond_cdf    = vec![0.0f32; h * w];
        for j in 0..h {
            let sin_j   = (PI * (j as f32 + 0.5) / h as f32).sin();
            let mut sum = 0.0f32;
            for i in 0..w {
                let px = image.get_pixel(i as u32, j as u32);
                sum += lum(px[0], px[1], px[2]) * sin_j;
                cond_cdf[j * w + i] = sum;
            }
            if sum > 0.0 {
                for i in 0..w { cond_cdf[j * w + i] /= sum; }
            } else {
                for i in 0..w { cond_cdf[j * w + i] = (i + 1) as f32 / w as f32; }
            }
            row_weights[j] = sum;
        }

        // Build the 1-D marginal CDF over rows.
        let total: f32 = row_weights.iter().sum();
        let mut marginal_cdf = vec![0.0f32; h];
        let mut cum = 0.0f32;
        for j in 0..h {
            cum += row_weights[j];
            marginal_cdf[j] = if total > 0.0 { cum / total } else { (j + 1) as f32 / h as f32 };
        }

        Self {
            inv_total: if total > 0.0 { 1.0 / total } else { 0.0 },
            image,
            marginal_cdf,
            cond_cdf,
        }
    }

    fn sample(&self, dir: Vec3) -> Color {
        use std::f32::consts::PI;
        let d  = dir.unit();
        let u  = (0.5 + d.x.atan2(d.z) / (2.0 * PI)).rem_euclid(1.0);
        let v  = d.y.clamp(-1.0, 1.0).acos() / PI;
        let w  = self.image.width();
        let h  = self.image.height();
        let px = ((u * w as f32) as u32).min(w - 1);
        let py = ((v * h as f32) as u32).min(h - 1);
        let p  = self.image.get_pixel(px, py);
        Color::new(p[0], p[1], p[2])
    }

    /// Importance-sample a world-space direction proportional to luminance × solid angle.
    /// Returns (direction, solid-angle PDF).
    pub fn sample_dir(&self, rng: &mut impl rand::Rng) -> (Vec3, f32) {
        use std::f32::consts::PI;
        let w = self.image.width()  as usize;
        let h = self.image.height() as usize;

        // Two-stage inversion: first pick a row, then a column within that row.
        let j = {
            let xi: f32 = rng.gen();
            self.marginal_cdf.partition_point(|&v| v < xi).min(h - 1)
        };
        let i = {
            let xi: f32 = rng.gen();
            self.cond_cdf[j * w .. (j + 1) * w].partition_point(|&v| v < xi).min(w - 1)
        };

        // Sub-pixel jitter for continuous (u,v) sampling.
        let u = (i as f32 + rng.gen::<f32>()) / w as f32;
        let v = (j as f32 + rng.gen::<f32>()) / h as f32;

        // Invert the equirectangular projection used in sample():
        //   u = 0.5 + atan2(x, z) / (2π)  →  φ = 2π(u − 0.5)
        //   v = acos(y) / π                →  θ = πv
        let phi   = 2.0 * PI * (u - 0.5);
        let theta = PI * v;
        let sin_t = theta.sin();
        let dir   = Vec3::new(sin_t * phi.sin(), theta.cos(), sin_t * phi.cos());

        (dir, self.dir_pdf_at(i, j))
    }

    /// Solid-angle PDF for a given world-space direction.
    pub fn dir_pdf(&self, dir: Vec3) -> f32 {
        use std::f32::consts::PI;
        let w = self.image.width()  as usize;
        let h = self.image.height() as usize;
        let d = dir.unit();
        let u = (0.5 + d.x.atan2(d.z) / (2.0 * PI)).rem_euclid(1.0);
        let v = d.y.clamp(-1.0, 1.0).acos() / PI;
        let i = ((u * w as f32) as usize).min(w - 1);
        let j = ((v * h as f32) as usize).min(h - 1);
        self.dir_pdf_at(i, j)
    }

    fn dir_pdf_at(&self, i: usize, j: usize) -> f32 {
        use std::f32::consts::PI;
        let w   = self.image.width()  as usize;
        let h   = self.image.height() as usize;
        let px  = self.image.get_pixel(i as u32, j as u32);
        let lum = 0.2126 * px[0] + 0.7152 * px[1] + 0.0722 * px[2];
        // The sinθ weighting used during CDF construction cancels exactly with the
        // equirectangular→solid-angle Jacobian (|dω/d(u,v)| = 2π² sinθ), giving:
        //   p_ω = L(i,j) × W × H / (Z × 2π²)
        lum * (w * h) as f32 * self.inv_total / (2.0 * PI * PI)
    }
}

#[derive(Clone)]
pub enum Background {
    Solid(Color),
    Physical { sun_dir: Vec3, turbidity: f32 },
    EnvMap(Arc<EnvMapData>),
}

impl Background {
    pub fn eval(&self, dir: Vec3) -> Color {
        match self {
            Background::Solid(c)                           => *c,
            Background::Physical { sun_dir, turbidity }    => preetham_sky(dir, *sun_dir, *turbidity),
            Background::EnvMap(em)                         => em.sample(dir),
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

    // Solar disc: rays within ~1.3° of sun and above horizon get a bright warm-white return
    if cos_gamma > COS_SUN_MAX && sun.y > 0.0 && d.y > 0.0 {
        // Brightness scaled proportionally to the smaller solid angle so total
        // solar irradiance is preserved: old_Ω/new_Ω = 0.0003/0.0000108 ≈ 27.8.
        let disc = yz * 111.0;
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

    let sky = Color::new(r, g, b);

    // Below horizon: blend from the Preetham horizon colour (ct was clamped to 0.001
    // above, so every below-horizon ray evaluates the horizon slice) toward a warm
    // dim ground tone.  This ensures scenes whose ground geometry misses near-
    // horizontal rays (e.g. a finite ground sphere) never show a black band.
    if d.y < 0.0 {
        let t = (-d.y * 8.0).min(1.0);
        let ground = Color::new(0.07, 0.06, 0.05);
        sky * (1.0 - t) + ground * t
    } else {
        sky
    }
}

// ── Sun sampling helpers ──────────────────────────────────────────────────────

/// Uniform solid-angle PDF for the solar disc cone; returns 0 outside the disc.
/// The disc half-angle matches the `cos_gamma > 0.9997` threshold in preetham_sky.
fn sun_pdf_value(dir: Vec3, background: &Background) -> f32 {
    use std::f32::consts::PI;
    if let Background::Physical { sun_dir, .. } = background {
        if sun_dir.y > 0.0 && dir.unit().dot(sun_dir.unit()) > COS_SUN_MAX {
            return 1.0 / (2.0 * PI * (1.0 - COS_SUN_MAX));
        }
    }
    0.0
}

/// Sample a direction uniformly within the solar disc cone around `axis` (unit).
fn sample_sun_cone(axis: Vec3, cos_theta_max: f32, rng: &mut impl Rng) -> Vec3 {
    use std::f32::consts::PI;
    let r1: f32 = rng.gen();
    let r2: f32 = rng.gen();
    let cos_theta = 1.0 - r1 * (1.0 - cos_theta_max);
    let sin_theta = (1.0 - cos_theta * cos_theta).max(0.0).sqrt();
    let phi = 2.0 * PI * r2;
    let up = if axis.x.abs() < 0.999 { Vec3::new(1.0, 0.0, 0.0) } else { Vec3::new(0.0, 1.0, 0.0) };
    let t  = axis.cross(up).unit();
    let b  = axis.cross(t);
    t * (sin_theta * phi.cos()) + b * (sin_theta * phi.sin()) + axis * cos_theta
}

fn env_pdf_value(dir: Vec3, background: &Background) -> f32 {
    if let Background::EnvMap(em) = background { em.dir_pdf(dir) } else { 0.0 }
}

// ── Path tracer ───────────────────────────────────────────────────────────────

/// `bg_scale` is multiplied into the background sample only (not scene hits).
pub fn ray_color(r: &Ray, world: &dyn Hittable, background: &Background, lights: &HittableList, bg_scale: f32, photon_map: Option<&PhotonMap>, rng: &mut impl Rng) -> Color {
    let mut throughput           = Color::new(1.0, 1.0, 1.0);
    let mut color                = Color::default();
    let mut ray                  = *r;
    let mut prev_specular        = true;  // camera ray: always add full emission on first hit
    let mut prev_mis_w_brdf      = 1.0f32; // MIS weight for diffuse-bounce emission
    let mut prev_spec_sun_weight = 1.0f32; // MIS weight for specular-bounce sun disc hit

    for depth in 0..MAX_DEPTH {
        match world.hit(&ray, 0.001, f32::INFINITY) {
            None => {
                // MIS weight for the background:
                // - Not hitting sun disc (sun_pdf == 0): always full weight.
                // - After a diffuse bounce (prev_specular = false): use prev_mis_w_brdf
                //   (accounts for both area-light and sun NEE done on that bounce).
                // - After a specular bounce (prev_specular = true): use prev_spec_sun_weight
                //   (1.0 if no specular sun NEE was done, else p_vndf/(p_vndf+p_sun)).
                let sun_pdf = sun_pdf_value(ray.direction, background);
                let env_pdf = env_pdf_value(ray.direction, background);
                let emit_w  = if sun_pdf + env_pdf == 0.0 { 1.0 }
                              else if prev_specular { prev_spec_sun_weight }
                              else                  { prev_mis_w_brdf };
                color += throughput * background.eval(ray.direction) * bg_scale * emit_w;
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
                color += throughput * rec.mat.emitted_at(rec.u, rec.v, rec.p, ray.wavelength) * emit_w;

                let Some(sr) = rec.mat.scatter(&ray, &rec, rng) else { break; };

                if sr.skip_pdf {
                    // ── Specular sun NEE ─────────────────────────────────────────
                    // For smooth specular surfaces the BRDF-sampled ray almost never
                    // hits the tiny solar disc.  We importance-sample the sun directly,
                    // evaluate the specular BRDF in that direction, and MIS-weight both
                    // the NEE contribution and the continuing BRDF-sample path.
                    prev_spec_sun_weight = 1.0; // reset; updated below if NEE fires
                    if let Background::Physical { sun_dir, .. } = background {
                        if sun_dir.y > 0.0 {
                            use std::f32::consts::PI;
                            let p_sun      = 1.0 / (2.0 * PI * (1.0 - COS_SUN_MAX));
                            let sun_u      = sun_dir.unit();
                            let sun_sample = sample_sun_cone(sun_u, COS_SUN_MAX, rng);
                            let sun_shadow = Ray::new_at_time(rec.p, sun_sample, ray.time);
                            if !world.any_hit(&sun_shadow, 0.001, f32::INFINITY) {
                                let brdf_cos = rec.mat.specular_brdf_cos(&ray, &rec, sun_sample);
                                if brdf_cos.x > 0.0 || brdf_cos.y > 0.0 || brdf_cos.z > 0.0 {
                                    let p_mat     = rec.mat.specular_sampling_pdf(&ray, &rec, sun_sample);
                                    let sun_color = background.eval(sun_sample);
                                    color += throughput * brdf_cos * sun_color / (p_sun + p_mat);
                                }
                            }
                            let p_mat_brdf = rec.mat.specular_sampling_pdf(&ray, &rec, sr.ray.direction);
                            if p_mat_brdf > 0.0 {
                                prev_spec_sun_weight = p_mat_brdf / (p_mat_brdf + p_sun);
                            }
                        }
                    }
                    throughput      *= sr.attenuation;
                    ray              = sr.ray;
                    prev_specular    = true;
                    prev_mis_w_brdf  = 1.0;
                } else {
                    // Caustic injection: add photon-map irradiance at this diffuse
                    // surface. The photon map stores only caustic paths (at least one
                    // specular bounce), so there is no double-counting with NEE below.
                    if let Some(pm) = photon_map {
                        let irr = pm.irradiance(rec.p, rec.normal);
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
                                    let nee_emit = lrec.mat.emitted_at(lrec.u, lrec.v, lrec.p, ray.wavelength);
                                    color += throughput * sr.attenuation * brdf * nee_emit / mis_d;
                                }
                            }
                        }
                    }

                    // ── Sun NEE: directly sample the solar disc ──────────────────
                    // Independent of area lights — the sun is a directional source
                    // at infinity, not part of `lights`.  Shadow test with t_max=∞
                    // correctly occludes finite geometry between the surface and sky.
                    if let Background::Physical { sun_dir, .. } = background {
                        if sun_dir.y > 0.0 {
                            use std::f32::consts::PI;
                            let sun_u      = sun_dir.unit();
                            let sun_sample = sample_sun_cone(sun_u, COS_SUN_MAX, rng);
                            let sun_shadow = Ray::new_at_time(rec.p, sun_sample, ray.time);
                            if !world.any_hit(&sun_shadow, 0.001, f32::INFINITY) {
                                let brdf = rec.mat.scattering_pdf(&ray, &rec, &sun_shadow);
                                if brdf > 0.0 {
                                    let sun_pdf   = 1.0 / (2.0 * PI * (1.0 - COS_SUN_MAX));
                                    let sun_color = background.eval(sun_sample);
                                    let area_pdf  = if lights.objects.is_empty() { 0.0 }
                                                    else { lights.pdf_value(rec.p, sun_sample, ray.time) };
                                    let mis_d     = sun_pdf + brdf + area_pdf;
                                    color += throughput * sr.attenuation * brdf * sun_color / mis_d;
                                }
                            }
                        }
                    }

                    // ── Env-map NEE: importance-sample bright env map regions ───────
                    // Mirrors the sun NEE block above.  A direction is drawn from the
                    // luminance-weighted 2-D CDF built at load time; if the shadow ray
                    // is unoccluded the env radiance is added with MIS weighting.
                    if let Background::EnvMap(em) = background {
                        let (env_dir, env_pdf) = em.sample_dir(rng);
                        if env_pdf > 0.0 {
                            let shadow = Ray::new_at_time(rec.p, env_dir, ray.time);
                            if !world.any_hit(&shadow, 0.001, f32::INFINITY) {
                                let brdf = rec.mat.scattering_pdf(&ray, &rec, &shadow);
                                if brdf > 0.0 {
                                    let area_pdf = if lights.objects.is_empty() { 0.0 }
                                                   else { lights.pdf_value(rec.p, env_dir, ray.time) };
                                    let env_color = background.eval(env_dir);
                                    color += throughput * sr.attenuation * brdf * env_color
                                        / (env_pdf + brdf + area_pdf);
                                }
                            }
                        }
                    }

                    // ── Indirect lighting: cosine-weighted BRDF sample ────────────
                    // Also compute the MIS weight for the case where this ray hits a
                    // light next iteration: w_brdf = p_brdf / (p_brdf + p_nee).
                    let cpdf     = CosinePdf::new(rec.mat.shading_normal(&rec));
                    let ind_dir  = cpdf.generate(rng);
                    let pdf_val  = cpdf.value(ind_dir);
                    if pdf_val <= 0.0 { break; }
                    let scattered = Ray::scatter_from(rec.p, ind_dir, &ray);
                    let scat_pdf  = rec.mat.scattering_pdf(&ray, &rec, &scattered);
                    if scat_pdf <= 0.0 { break; }

                    let nee_pdf_for_ind = {
                        let area = if lights.objects.is_empty() { 0.0 }
                                   else { lights.pdf_value(rec.p, ind_dir, ray.time) };
                        let sun  = sun_pdf_value(ind_dir, background);
                        let env  = env_pdf_value(ind_dir, background);
                        area + sun + env
                    };
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
    background: &Background,
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

// ── SPPM camera pass ──────────────────────────────────────────────────────────

/// Trace camera rays to their first caustic-eligible diffuse hit and return
/// the collected visible points.  Uses the same stratified-sample RNG as
/// `render_tiles` so the VP positions are consistent with the camera paths
/// already traced for the non-caustic contribution.
pub fn collect_visible_points(
    width:       u32,
    height:      u32,
    camera:      &Camera,
    world:       &dyn Hittable,
    sample_idx:  u32,
    strata:      u32,
    sppm_pixels: &[SppmPixel],
) -> Vec<VisiblePoint> {
    let w        = width  as usize;
    let h        = height as usize;
    let n        = w * h;
    let w_denom  = (width  - 1).max(1) as f32;
    let h_denom  = (height - 1).max(1) as f32;
    let strata2  = strata * strata;
    let strata_f = strata as f32;

    (0..n)
        .into_par_iter()
        .filter_map(|i| {
            let row = i / w;
            let col = i % w;
            let mut rng = SmallRng::seed_from_u64(
                (i as u64).wrapping_mul(6_364_136_223_846_793_005)
                    ^ (sample_idx as u64).wrapping_mul(0x9E3779B97F4A7C15),
            );
            let ray_y = height - 1 - row as u32;

            // Mirror the stratified jitter from render_tiles exactly so the
            // specular scatter decisions are seeded identically.
            let (u_jitter, v_jitter) = if strata2 > 0 {
                let offset = (i as u32).wrapping_mul(0x9E3779B9) % strata2;
                let s      = (sample_idx + offset) % strata2;
                let sx     = s % strata;
                let sy     = s / strata;
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
            cam_ray.wavelength = rng.gen_range(380.0_f32..700.0);

            trace_to_visible_point(&cam_ray, world, i, sppm_pixels[i].radius, &mut rng)
        })
        .collect()
}

/// Follow specular bounces until the first caustic-eligible diffuse hit.
fn trace_to_visible_point(
    r:      &Ray,
    world:  &dyn Hittable,
    pixel:  usize,
    radius: f32,
    rng:    &mut SmallRng,
) -> Option<VisiblePoint> {
    let mut throughput = Color::new(1.0, 1.0, 1.0);
    let mut ray        = *r;
    for _ in 0..PHOTON_MAX_DEPTH {
        let rec = world.hit(&ray, 0.001, f32::INFINITY)?;
        let sr  = rec.mat.scatter(&ray, &rec, rng)?;
        if sr.skip_pdf {
            throughput *= sr.attenuation;
            ray         = sr.ray;
        } else {
            if rec.mat.can_receive_caustics() {
                return Some(VisiblePoint {
                    pos:    rec.p,
                    normal: rec.mat.shading_normal(&rec),
                    albedo: rec.mat.albedo_hint(rec.u, rec.v, rec.p),
                    beta:   throughput,
                    radius,
                    pixel,
                });
            }
            return None;
        }
    }
    None
}

// ── Tile renderer ─────────────────────────────────────────────────────────────

/// Render one sample pass into `scratch` in parallel.
/// `strata` = floor(sqrt(max_samples)); controls the stratified-sampling grid size.
/// `converged`: optional per-pixel mask — `true` pixels are skipped (scratch written as black).
#[allow(clippy::too_many_arguments)]
pub fn render_tiles(
    scratch:     &mut [Color],
    converged:   Option<&[bool]>,
    sample_idx:  u32,
    strata:      u32,
    width:       u32,
    height:      u32,
    camera:      &Camera,
    world:       &dyn Hittable,
    background:  &Background,
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
        Background::Physical { sun_dir, .. } => set_pearl_sun_dir(*sun_dir),
        _                                    => clear_pearl_sun_dir(),
    }

    // Tile the image into 16×16 blocks so that rays within a tile share BVH
    // cache lines.  Tiles are non-overlapping, so parallel writes are safe.
    // The pointer is carried as usize (which is Send+Sync) to avoid the
    // raw-pointer Sync restriction, then cast back inside each closure call.
    const TILE: usize = 16;
    let h = height as usize;
    let tiles_x = w.div_ceil(TILE);
    let tiles_y = h.div_ceil(TILE);
    // SAFETY: tiles partition the image without overlap; each pixel is written
    // by exactly one tile iteration.
    let buf_ptr: usize = scratch.as_mut_ptr() as usize;

    (0..tiles_x * tiles_y).into_par_iter().for_each(move |tile_idx| {
        let base = buf_ptr as *mut Color;
        let tile_r = tile_idx / tiles_x;
        let tile_c = tile_idx % tiles_x;
        let col0   = tile_c * TILE;
        let row0   = tile_r * TILE;
        let col1   = (col0 + TILE).min(w);
        let row1   = (row0 + TILE).min(h);

        for row in row0..row1 {
            for col in col0..col1 {
                let i = row * w + col;
                if converged.is_some_and(|c| c[i]) {
                    // SAFETY: same non-overlapping guarantee as active pixels.
                    unsafe { *base.add(i) = Color::default(); }
                    continue;
                }
                let mut rng = SmallRng::seed_from_u64(
                    (i as u64).wrapping_mul(6364136223846793005)
                        ^ (sample_idx as u64).wrapping_mul(0x9E3779B97F4A7C15),
                );
                let ray_y = height - 1 - row as u32;

                // Stratified pixel sampling: map sample_idx into a strata×strata grid.
                // A per-pixel cyclic offset (Fibonacci hash) ensures neighboring pixels
                // visit strata in different orders, avoiding spatial correlation.
                let (u_jitter, v_jitter) = if strata2 > 0 {
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
                cam_ray.wavelength = rng.gen_range(380.0_f32..700.0);
                let color = ray_color(&cam_ray, world, background, lights, bg_scale, photon_map, &mut rng);
                // SAFETY: each (row, col) maps to a unique index; tiles are non-overlapping.
                unsafe { *base.add(i) = color; }
            }
        }
    });
}
