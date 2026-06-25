use std::f32::consts::PI;
use std::sync::{Arc, OnceLock};
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use image::RgbImage;
use rand::{Rng, RngCore};
use crate::output::srgb_decode;
use crate::vec3::{Color, Point3, Vec3};
use crate::ray::Ray;
use crate::hittable::{HitRecord, Material, ScatterRecord};
use crate::texture::Texture;
use crate::spectrum::{cauchy_ior, copper_ior, fresnel_conductor, gold_ior, planck_raw, silver_ior, spectral_to_rgb};
use crate::volume::hg_sample;

pub struct DiffuseLight {
    pub emit: Texture,
}

impl Material for DiffuseLight {
    fn scatter(&self, _r_in: &Ray, _rec: &HitRecord<'_>, _rng: &mut dyn RngCore) -> Option<ScatterRecord> { None }
    fn emitted(&self, u: f32, v: f32, p: Point3) -> Color { self.emit.value(u, v, p) }
    fn albedo_hint(&self, u: f32, v: f32, p: Point3) -> Color { self.emitted(u, v, p) }
}

/// Area light with a physically-based blackbody spectral power distribution.
///
/// Uses the hero-wavelength spectral framework: `emitted_at(λ)` returns
/// `spectral_to_rgb(λ) × planck_norm(λ, T) × intensity`, where
/// `planck_norm = planck_raw / mean(planck_raw over [380, 700])`.
///
/// For non-dispersive diffuse paths this time-averages to a warm/cool RGB
/// color matching the color temperature.  For dispersive paths (e.g. through
/// a glass sphere) the λ-dependent weight biases the rainbow: a 3000 K lamp
/// produces a warm red-heavy rainbow; 6500 K daylight produces a balanced one.
///
/// Common color temperatures: 2700 K = warm tungsten, 3000 K = halogen,
/// 5500 K ≈ sunlight, 6500 K = D65 daylight.
pub struct BlackbodyLight {
    pub temp_k:    f32,
    pub intensity: f32,
    norm:          f32,   // 1 / mean(planck_raw over [380, 700])
    avg_color:     Color, // E_λ[planck_norm(λ,T) × spectral_to_rgb(λ)] × intensity
}

impl BlackbodyLight {
    pub fn new(temp_k: f32, intensity: f32) -> Self {
        const N: usize = 65;
        let raw: [f32; N] = std::array::from_fn(|i| planck_raw(380.0 + i as f32 * 5.0, temp_k));
        let mean = raw.iter().sum::<f32>() / N as f32;
        let norm = 1.0 / mean.max(1e-30);
        let avg_color = {
            let mut acc = Color::default();
            for (i, &r) in raw.iter().enumerate() {
                acc += spectral_to_rgb(380.0 + i as f32 * 5.0) * (r * norm);
            }
            acc / N as f32 * intensity
        };
        Self { temp_k, intensity, norm, avg_color }
    }
}

impl Material for BlackbodyLight {
    fn scatter(&self, _r_in: &Ray, _rec: &HitRecord<'_>, _rng: &mut dyn RngCore) -> Option<ScatterRecord> { None }
    fn emitted(&self, _u: f32, _v: f32, _p: Point3) -> Color { self.avg_color }
    fn emitted_at(&self, _u: f32, _v: f32, _p: Point3, lambda: f32, spectral_weighted: bool) -> Color {
        if spectral_weighted {
            // CMF is already baked into the path throughput from the first spectral
            // bounce — return scalar power only to avoid double-counting it.
            let scalar = planck_raw(lambda, self.temp_k) * self.norm * self.intensity;
            Color::new(scalar, scalar, scalar)
        } else {
            // Non-spectral path: the hero wavelength λ is irrelevant (no dispersive
            // material was hit), so returning the per-λ value only adds variance
            // without physical benefit.  avg_color is the zero-variance unbiased
            // estimator — identical expectation, no firefly-clamping colour bias.
            self.avg_color
        }
    }
    fn albedo_hint(&self, _u: f32, _v: f32, _p: Point3) -> Color { self.avg_color }
}

pub struct Lambertian {
    pub texture: Texture,
}

impl Material for Lambertian {
    fn scatter(&self, r_in: &Ray, rec: &HitRecord<'_>, _rng: &mut dyn RngCore) -> Option<ScatterRecord> {
        let albedo = self.texture.value(rec.u, rec.v, rec.p);
        // Direction is overridden by the PDF in ray_color; normal is a harmless placeholder.
        Some(ScatterRecord { attenuation: albedo, ray: Ray::scatter_from(rec.p, rec.normal, r_in), skip_pdf: false })
    }

    fn scattering_pdf(&self, _r_in: &Ray, rec: &HitRecord<'_>, scattered: &Ray) -> f32 {
        let cosine = rec.normal.dot(scattered.direction.unit());
        (cosine / PI).max(0.0)
    }
    fn albedo_hint(&self, u: f32, v: f32, p: Point3) -> Color { self.texture.value(u, v, p) }
    fn can_receive_caustics(&self) -> bool { true }
}


#[inline]
fn schlick(cosine: f32, ref_idx: f32) -> f32 {
    let b  = (1.0 - ref_idx) / (1.0 + ref_idx);
    let r0 = b * b;
    let u  = 1.0 - cosine;
    r0 + (1.0 - r0) * (u * u * u * u * u)
}

/// Dielectric boundary scatter shared by all glass-like materials.
/// Returns `(exit_direction, reflected)` where `reflected` is true for TIR
/// or Fresnel reflection and false for refraction.
#[inline]
fn dielectric_boundary(ior: f32, r_in: &Ray, rec: &HitRecord<'_>, rng: &mut dyn RngCore) -> (Vec3, bool) {
    let ratio     = if rec.front_face { 1.0 / ior } else { ior };
    let unit      = r_in.direction.unit();
    let cos_theta = (-unit).dot(rec.normal).min(1.0);
    let sin_theta = (1.0 - cos_theta * cos_theta).sqrt();
    let reflected = ratio * sin_theta > 1.0 || schlick(cos_theta, ratio) > rng.gen::<f32>();
    let direction = if reflected {
        unit.reflect(rec.normal)
    } else {
        // TIR already ruled out above; unwrap_or_else handles floating-point
        // edge cases where sin_theta rounds just under the TIR threshold.
        unit.refract(rec.normal, ratio).unwrap_or_else(|| unit.reflect(rec.normal))
    };
    (direction, reflected)
}

// ── GGX / PBR helpers ─────────────────────────────────────────────────────────

/// Schlick Fresnel with a colored F0 (supports metal tints).
#[inline]
fn schlick_color(cos_theta: f32, f0: Color) -> Color {
    let u = (1.0 - cos_theta).max(0.0);
    let t = u * u * u * u * u;
    f0 + (Color::new(1.0, 1.0, 1.0) - f0) * t
}

/// Smith G1 term for isotropic GGX.
#[inline]
fn smith_g1(cos_theta: f32, alpha: f32) -> f32 {
    let a2 = alpha * alpha;
    let c2 = cos_theta * cos_theta;
    2.0 * cos_theta / (cos_theta + (a2 + (1.0 - a2) * c2).sqrt())
}

/// Smith G1 for anisotropic GGX.
/// `tx` / `ty` are dot(v, tangent) / dot(v, bitangent); `cos_theta` = dot(v, normal).
#[inline]
fn smith_g1_aniso(cos_theta: f32, tx: f32, ty: f32, ax: f32, ay: f32) -> f32 {
    let denom = cos_theta + (cos_theta*cos_theta + ax*ax*tx*tx + ay*ay*ty*ty).sqrt();
    (2.0 * cos_theta / denom.max(1e-6)).clamp(0.0, 1.0)
}

/// Sample a microfacet normal from the anisotropic GGX VNDF (Heitz 2018).
/// `wo_ts` is the outgoing direction in tangent space (z = dot(wo, n) > 0).
/// Returns the microfacet normal in tangent space.
#[inline]
fn vndf_sample_aniso(wo_ts: Vec3, ax: f32, ay: f32, xi1: f32, xi2: f32) -> Vec3 {
    // Stretch wo into an isotropic hemisphere
    let wh = Vec3::new(ax * wo_ts.x, ay * wo_ts.y, wo_ts.z).unit();

    // Orthonormal basis around wh (T1 perpendicular to wh in the xy-plane)
    let len = (wh.x * wh.x + wh.y * wh.y).sqrt();
    let t1 = if len > 1e-6 {
        Vec3::new(-wh.y / len, wh.x / len, 0.0)
    } else {
        Vec3::new(1.0, 0.0, 0.0)
    };
    let t2 = wh.cross(t1);

    // Sample the projected area of the hemisphere cap (Heitz 2018, Algorithm 2)
    let a  = 1.0 / (1.0 + wh.z);
    let r  = xi1.sqrt();
    let (p1, p2) = if xi2 < a {
        let phi = xi2 / a * PI;
        (r * phi.cos(), r * phi.sin())
    } else {
        let phi = PI + (xi2 - a) / (1.0 - a) * PI;
        (r * phi.cos(), r * phi.sin() * wh.z)
    };

    // Compose sample on the unit hemisphere
    let p3 = (1.0 - p1*p1 - p2*p2).max(0.0).sqrt();
    let nh  = t1 * p1 + t2 * p2 + wh * p3;

    // Unstretch → microfacet normal in tangent space
    Vec3::new(ax * nh.x, ay * nh.y, nh.z.max(0.0)).unit()
}


/// Isotropic GGX NDF: D(cos_h, α) = α² / (π · (1 + cos²_h · (α²−1))²).
#[inline]
fn ggx_ndf(cos_h: f32, alpha: f32) -> f32 {
    let a2 = alpha * alpha;
    let denom = 1.0 + cos_h * cos_h * (a2 - 1.0);
    a2 / (PI * denom * denom).max(1e-12)
}

/// Anisotropic GGX NDF in tangent space.
#[inline]
fn ggx_ndf_aniso(h_ts: Vec3, ax: f32, ay: f32) -> f32 {
    let hx = h_ts.x / ax;
    let hy = h_ts.y / ay;
    let d  = hx * hx + hy * hy + h_ts.z * h_ts.z;
    1.0 / (PI * ax * ay * d * d).max(1e-12)
}

pub struct Dielectric {
    pub ir: f32,
}

impl Material for Dielectric {
    fn scatter(&self, r_in: &Ray, rec: &HitRecord<'_>, rng: &mut dyn RngCore) -> Option<ScatterRecord> {
        let (direction, _) = dielectric_boundary(self.ir, r_in, rec, rng);
        Some(ScatterRecord { attenuation: Color::new(1.0, 1.0, 1.0), ray: Ray::scatter_from(rec.p, direction, r_in), skip_pdf: true })
    }
}

/// Dispersive dielectric using a continuous hero-wavelength model.
///
/// The ray carries a wavelength λ ∈ [380, 700] nm sampled once per path.
/// At each boundary event the IOR is evaluated via the Cauchy equation
/// n(λ) = B + C/λ² (λ in μm), and refracted rays are weighted by the CIE
/// colour-matching function for that wavelength — normalised so that
/// E_λ[weight] = (1,1,1).  This produces smooth spectral dispersion (rainbows,
/// coloured diamond fire) rather than the coarse R/G/B bands of the 3-channel
/// approach.  Reflections carry (1,1,1) since Fresnel is nearly achromatic.
///
/// Cauchy parameters for common materials (B, C in μm²):
/// - Crown glass:  cauchy_b ≈ 1.507, cauchy_c ≈ 0.00375
/// - Dense flint:  cauchy_b ≈ 1.612, cauchy_c ≈ 0.00950
/// - Diamond:      cauchy_b ≈ 2.395, cauchy_c ≈ 0.00585
///
/// `absorption` is a per-unit-length Beer-Lambert coefficient (RGB, scene units).
/// Applied once per interior segment — accumulates correctly across TIR bounces.
/// `[0,0,0]` = clear glass. Example: `[0.05, 0.01, 0.0]` gives an amber tint
/// that deepens with thickness.
pub struct SpectralDielectric {
    pub cauchy_b:   f32,
    pub cauchy_c:   f32,
    pub absorption: Color,
}

impl Default for SpectralDielectric {
    fn default() -> Self {
        Self { cauchy_b: 1.5, cauchy_c: 0.0, absorption: Color::new(0.0, 0.0, 0.0) }
    }
}

impl Material for SpectralDielectric {
    fn scatter(&self, r_in: &Ray, rec: &HitRecord<'_>, rng: &mut dyn RngCore) -> Option<ScatterRecord> {
        let ior = cauchy_ior(r_in.wavelength, self.cauchy_b, self.cauchy_c);
        let (direction, _reflected) = dielectric_boundary(ior, r_in, rec, rng);

        // Glass only affects direction (Cauchy dispersion), not amplitude.
        // The CMF weight is NOT applied here — that keeps diffuse surfaces and
        // photon-map caustics downstream of glass white instead of rainbow-tinted
        // by the camera sample's random wavelength.  spectral_weighted stays as-is
        // so SpectralMetal and BlackbodyLight on the same path are unaffected.
        let mut attenuation = Color::new(1.0, 1.0, 1.0);

        // Beer-Lambert absorption: applied on every interior segment (exit or TIR).
        // rec.t is the chord length since the last boundary event, so absorption
        // accumulates correctly across multiple TIR bounces inside the medium.
        if !rec.front_face {
            let d = rec.t;
            attenuation.x *= (-self.absorption.x * d).exp();
            attenuation.y *= (-self.absorption.y * d).exp();
            attenuation.z *= (-self.absorption.z * d).exp();
        }

        let scattered = Ray::scatter_from(rec.p, direction, r_in);
        Some(ScatterRecord { attenuation, ray: scattered, skip_pdf: true })
    }

    fn is_spectral(&self) -> bool { true }
}

/// Which metal to use for `SpectralMetal`.
#[derive(Clone, Copy, Debug)]
pub enum SpectralMetalVariant { Gold, Copper, Silver }

impl SpectralMetalVariant {
    fn ior_at(self, lambda_nm: f32) -> (f32, f32) {
        match self {
            Self::Gold   => gold_ior(lambda_nm),
            Self::Copper => copper_ior(lambda_nm),
            Self::Silver => silver_ior(lambda_nm),
        }
    }
}

/// Physically-based conductor using spectral complex IOR (n + ik) sampled at
/// the hero wavelength.
///
/// On the first spectral bounce `spectral_to_rgb(λ)` is folded in (exactly as
/// `SpectralDielectric` does) so brightness encodes the full spectral Fresnel.
/// Subsequent bounces scale the RGB triple by the scalar F(λ), which is
/// equivalent to multiplying spectral power by F at the same wavelength.
///
/// Data: Johnson & Christy (1972), 380–680 nm, 25 nm grid.
pub struct SpectralMetal {
    pub variant:   SpectralMetalVariant,
    /// 0 = perfect mirror; higher values add a diffuse-sphere perturbation.
    pub roughness: f32,
    avg_color:     Color,  // mean spectral F0 for OIDN albedo hint
}

impl SpectralMetal {
    pub fn new(variant: SpectralMetalVariant, roughness: f32) -> Self {
        let mut acc = Color::default();
        for i in 0..METAL_SAMPLES {
            let lam = 380.0 + i as f32 * 25.0;
            let (n, k) = variant.ior_at(lam);
            acc += spectral_to_rgb(lam) * fresnel_conductor(1.0, n, k);
        }
        Self { variant, roughness, avg_color: acc / METAL_SAMPLES as f32 }
    }
}

const METAL_SAMPLES: usize = 13;

impl Material for SpectralMetal {
    fn scatter(&self, r_in: &Ray, rec: &HitRecord<'_>, rng: &mut dyn RngCore) -> Option<ScatterRecord> {
        let n     = rec.normal;
        let wo    = (-r_in.direction).unit();
        let cos_o = wo.dot(n);
        if cos_o <= 0.0 { return None; }

        let lambda = r_in.wavelength;

        // Sample outgoing direction and record cos(wo, h) for Fresnel.
        let (dir, cos_h) = if self.roughness == 0.0 {
            // Perfect mirror: half-vector = normal, so cos_h = cos_o.
            let wi = r_in.direction.unit().reflect(n);
            (wi, cos_o)
        } else {
            // GGX VNDF sampling (isotropic).
            let alpha  = (self.roughness * self.roughness).max(1e-4);
            let (t, b) = n.onb();
            let wo_ts  = Vec3::new(wo.dot(t), wo.dot(b), cos_o);
            let m_ts   = vndf_sample_aniso(wo_ts, alpha, alpha, rng.gen(), rng.gen());
            let m      = (t * m_ts.x + b * m_ts.y + n * m_ts.z).unit();
            let vo_h   = wo.dot(m).max(0.0);
            let wi     = (2.0 * vo_h * m - wo).unit();
            if wi.dot(n) <= 0.0 { return None; }
            (wi, vo_h)
        };

        if dir.dot(n) <= 0.0 { return None; }

        let (ior_n, ior_k) = self.variant.ior_at(lambda);
        let fresnel = fresnel_conductor(cos_h, ior_n, ior_k);

        // VNDF weight is F × G1(wi).  For a perfect mirror G1 = 1.
        let g1_wi = if self.roughness == 0.0 {
            1.0
        } else {
            let alpha = (self.roughness * self.roughness).max(1e-4);
            smith_g1(dir.dot(n).max(0.0), alpha)
        };

        let attenuation = if r_in.spectral_weighted {
            Color::new(fresnel * g1_wi, fresnel * g1_wi, fresnel * g1_wi)
        } else {
            spectral_to_rgb(lambda) * (fresnel * g1_wi)
        };
        let mut scattered = Ray::scatter_from(rec.p, dir, r_in);
        if !r_in.spectral_weighted { scattered.spectral_weighted = true; }

        Some(ScatterRecord { attenuation, ray: scattered, skip_pdf: true })
    }

    /// GGX BRDF × cos_i at direction `wi`, with spectral conductor Fresnel.
    /// Returns zero for smooth (roughness = 0) since the BRDF is a delta function.
    fn specular_brdf_cos(&self, r_in: &Ray, rec: &HitRecord<'_>, wi: Vec3) -> Color {
        if self.roughness == 0.0 { return Color::default(); }

        let n     = rec.normal;
        let wo    = (-r_in.direction).unit();
        let cos_o = wo.dot(n);
        let cos_i = wi.dot(n);
        if cos_o <= 0.0 || cos_i <= 0.0 { return Color::default(); }

        let h    = (wo + wi).unit();
        let wo_h = wo.dot(h).max(0.0);
        let h_n  = h.dot(n).max(0.0);

        let alpha = (self.roughness * self.roughness).max(1e-4);
        let d     = ggx_ndf(h_n, alpha);
        let g1_o  = smith_g1(cos_o, alpha);
        let g1_i  = smith_g1(cos_i, alpha);

        let lambda = r_in.wavelength;
        let (ior_n, ior_k) = self.variant.ior_at(lambda);
        let fresnel = fresnel_conductor(wo_h, ior_n, ior_k);

        let val = d * fresnel * g1_o * g1_i / (4.0 * cos_o);

        if r_in.spectral_weighted {
            Color::new(val, val, val)
        } else {
            spectral_to_rgb(lambda) * val
        }
    }

    /// VNDF sampling PDF for the scattered direction.
    /// Returns zero for smooth metals (delta function).
    fn specular_sampling_pdf(&self, r_in: &Ray, rec: &HitRecord<'_>, wi: Vec3) -> f32 {
        if self.roughness == 0.0 { return 0.0; }

        let n     = rec.normal;
        let wo    = (-r_in.direction).unit();
        let cos_o = wo.dot(n);
        if cos_o <= 0.0 || wi.dot(n) <= 0.0 { return 0.0; }

        let h   = (wo + wi).unit();
        let h_n = h.dot(n).max(0.0);

        let alpha = (self.roughness * self.roughness).max(1e-4);
        ggx_ndf(h_n, alpha) * smith_g1(cos_o, alpha) / (4.0 * cos_o)
    }

    fn is_spectral(&self)                              -> bool  { true }
    fn albedo_hint(&self, _u: f32, _v: f32, _p: Point3) -> Color { self.avg_color }
}

/// Translucent marble: glass boundary with volumetric multiple scattering inside.
///
/// Entering rays refract normally (IOR).  While inside, each path segment samples
/// a free-path distance from `Exp(density)`.  If a scatter occurs before the exit
/// surface, the path deflects via the Henyey-Greenstein phase function and the
/// throughput is tinted by `albedo` (one event's worth of colour absorption).
/// Otherwise the ray refracts out through the far surface.
///
/// This produces the characteristic soft glow of jade and glass marbles: light
/// enters from a wide cone, diffuses over a few scattering lengths, and exits
/// smoothly from a spread of surface points.
///
/// - `albedo` — per-scatter albedo (single-scatter tint, also the RR survival colour).
/// - `sigma_a` — Beer-Lambert absorption coefficient (per unit length, per channel).
///   Applied exponentially to every free path: `T = exp(-σ_a · L)`.  Zero = no
///   absorption beyond per-scatter tinting.  For r=0.15 marbles a value around 2–4
///   gives T ≈ 0.4–0.6 over a full diameter, producing rich saturated glass colour.
/// - `density` — σ_t (events per unit length).  For radius-0.15 marbles,
///   `density ≈ 7` gives ≈2 scatters per diameter traversal.
/// - `g` — anisotropy (0 = isotropic; ~0.3 suits glass inclusions).
pub struct SSSMaterial {
    pub albedo:   Color,
    pub sigma_a:  Color,
    pub ior:      f32,
    pub density:  f32,
    pub g:        f32,
}

/// Per-channel Beer-Lambert transmittance over path length `t`.
#[inline]
fn beer_lambert(sigma_a: Color, t: f32) -> Color {
    Color::new(
        (-sigma_a.x * t).exp(),
        (-sigma_a.y * t).exp(),
        (-sigma_a.z * t).exp(),
    )
}

impl Material for SSSMaterial {
    fn scatter(&self, r_in: &Ray, rec: &HitRecord<'_>, rng: &mut dyn RngCore) -> Option<ScatterRecord> {
        let unit = r_in.direction.unit();

        // Inside the medium: check for scatter before the exit surface.
        if !rec.front_face && self.density > 0.0 {
            let path_length = (rec.p - r_in.origin).length();
            let t_scat      = -(rng.gen::<f32>().max(1e-9).ln()) / self.density;
            if t_scat < path_length {
                // Scatter event: tint by albedo + Beer-Lambert absorption up to scatter point.
                let p_scat  = r_in.origin + unit * t_scat;
                let new_dir = hg_sample(unit, self.g, rng);
                return Some(ScatterRecord {
                    attenuation: self.albedo * beer_lambert(self.sigma_a, t_scat),
                    ray:         Ray::scatter_from(p_scat, new_dir, r_in),
                    skip_pdf:    true,
                });
            }
            // Unscattered exit: apply Beer-Lambert over the full traversed path.
            let (direction, _) = dielectric_boundary(self.ior, r_in, rec, rng);
            return Some(ScatterRecord {
                attenuation: beer_lambert(self.sigma_a, path_length),
                ray:         Ray::scatter_from(rec.p, direction, r_in),
                skip_pdf:    true,
            });
        }

        // Entry boundary (or density == 0): standard Fresnel, no absorption yet.
        let (direction, _) = dielectric_boundary(self.ior, r_in, rec, rng);
        Some(ScatterRecord {
            attenuation: Color::new(1.0, 1.0, 1.0),
            ray:         Ray::scatter_from(rec.p, direction, r_in),
            skip_pdf:    true,
        })
    }

    fn albedo_hint(&self, _u: f32, _v: f32, _p: Point3) -> Color { self.albedo }
}

/// Returns the 2×2 bilinear neighbourhood for `(u, v)` in texel space.
/// Pixel centres are at half-integer positions; edges are clamped.
/// Output: `(x0, y0, x1, y1, tx, ty)` — blend weights `tx,ty ∈ [0,1]`.
fn bilinear_coords(img: &RgbImage, u: f32, v: f32) -> (u32, u32, u32, u32, f32, f32) {
    let u  = u.clamp(0.0, 1.0);
    let v  = 1.0 - v.clamp(0.0, 1.0);          // flip V: UV bottom-up, images top-down
    let fx = (u * img.width()  as f32 - 0.5).clamp(0.0, (img.width()  - 1) as f32);
    let fy = (v * img.height() as f32 - 0.5).clamp(0.0, (img.height() - 1) as f32);
    let x0 = fx as u32;
    let y0 = fy as u32;
    let x1 = (x0 + 1).min(img.width()  - 1);
    let y1 = (y0 + 1).min(img.height() - 1);
    (x0, y0, x1, y1, fx - x0 as f32, fy - y0 as f32)
}

fn sample_srgb(img: &RgbImage, u: f32, v: f32) -> Color {
    let (x0, y0, x1, y1, tx, ty) = bilinear_coords(img, u, v);
    let lin = srgb_decode;
    let px  = |x, y| { let p = img.get_pixel(x, y); Color::new(lin(p[0]), lin(p[1]), lin(p[2])) };
    let c0  = px(x0, y0) * (1.0 - tx) + px(x1, y0) * tx;
    let c1  = px(x0, y1) * (1.0 - tx) + px(x1, y1) * tx;
    c0 * (1.0 - ty) + c1 * ty
}

fn sample_linear(img: &RgbImage, u: f32, v: f32) -> f32 {
    let (x0, y0, x1, y1, tx, ty) = bilinear_coords(img, u, v);
    let px = |x, y| img.get_pixel(x, y)[0] as f32 / 255.0;
    let v0 = px(x0, y0) * (1.0 - tx) + px(x1, y0) * tx;
    let v1 = px(x0, y1) * (1.0 - tx) + px(x1, y1) * tx;
    v0 * (1.0 - ty) + v1 * ty
}

fn sample_rgb(img: &RgbImage, u: f32, v: f32) -> Color {
    let (x0, y0, x1, y1, tx, ty) = bilinear_coords(img, u, v);
    let px = |x, y| { let p = img.get_pixel(x, y); Color::new(p[0] as f32 / 255.0, p[1] as f32 / 255.0, p[2] as f32 / 255.0) };
    let c0 = px(x0, y0) * (1.0 - tx) + px(x1, y0) * tx;
    let c1 = px(x0, y1) * (1.0 - tx) + px(x1, y1) * tx;
    c0 * (1.0 - ty) + c1 * ty
}

/// Physically-based material using GGX microfacet specular + Lambertian diffuse.
///
/// All specular lobes are sampled from the Visible Normal Distribution Function
/// (VNDF, Heitz 2018).  At `anisotropy == 0`, ax = ay = α, which reduces exactly
/// to isotropic GGX — no branch needed.
///
/// Parameters:
/// - `albedo`: base colour — diffuse tint for dielectrics, F0 for metals.
/// - `roughness`: 0 = perfect mirror, 1 = fully diffuse specular (α = roughness²).
/// - `metallic`: 0 = dielectric (F0 ≈ 0.04, tinted diffuse), 1 = conductor (no diffuse).
/// - `anisotropy`: 0 = isotropic, 1 = maximum elongation (brushed-metal look).
/// - `anisotropy_angle`: rotates the highlight direction in the tangent plane (radians).
/// - `clearcoat`: weight of a smooth dielectric overcoat [0–1].
/// - `clearcoat_roughness`: roughness of the coat surface (default ~0.03).
/// - `film_thickness`: thin-film thickness in nm (0 = achromatic; 400–800 for vivid colours).
/// - `film_ior`: IOR of the thin film (default 1.5 for common dielectric coats).
/// - `sheen`: weight of the grazing-angle retroreflective lobe [0–1], for cloth and velvet.
/// - `sheen_tint`: 0 = white sheen, 1 = sheen tinted toward the base albedo (default 0.5).
/// - `emission`: RGB emission colour (linear); multiplied by `emission_strength`.
/// - `emission_strength`: scale factor for the emission (0 = dark, default).
/// - `albedo_tex / roughness_tex / metallic_tex / ao_tex`: optional image maps that override
///   the corresponding scalar fields when set. Albedo decoded sRGB→linear; others are linear.
pub struct PbrMaterial {
    pub albedo:              Color,
    pub roughness:           f32,
    pub metallic:            f32,
    pub anisotropy:          f32,
    pub anisotropy_angle:    f32,
    pub clearcoat:           f32,
    pub clearcoat_roughness: f32,
    pub film_thickness:      f32,
    pub film_ior:            f32,
    pub sheen:               f32,
    pub sheen_tint:          f32,
    pub emission:            Color,
    pub emission_strength:   f32,
    pub albedo_tex:          Option<Arc<RgbImage>>,
    pub roughness_tex:       Option<Arc<RgbImage>>,
    pub metallic_tex:        Option<Arc<RgbImage>>,
    pub ao_tex:              Option<Arc<RgbImage>>,
    pub normal_tex:          Option<Arc<RgbImage>>,
}

impl Default for PbrMaterial {
    fn default() -> Self {
        Self {
            albedo:              Color::new(0.8, 0.8, 0.8),
            roughness:           0.5,
            metallic:            0.0,
            anisotropy:          0.0,
            anisotropy_angle:    0.0,
            clearcoat:           0.0,
            clearcoat_roughness: 0.03,
            film_thickness:      0.0,
            film_ior:            1.5,
            sheen:               0.0,
            sheen_tint:          0.5,
            emission:            Color::default(),
            emission_strength:   0.0,
            albedo_tex:          None,
            roughness_tex:       None,
            metallic_tex:        None,
            ao_tex:              None,
            normal_tex:          None,
        }
    }
}

impl PbrMaterial {
    fn tbn_normal(&self, rec: &HitRecord<'_>) -> Vec3 {
        if let Some(tex) = &self.normal_tex {
            if rec.tangent.length_squared() > 1e-6 {
                let raw  = sample_rgb(tex, rec.u, rec.v);
                let ts_n = Vec3::new(raw.x * 2.0 - 1.0, raw.y * 2.0 - 1.0, raw.z * 2.0 - 1.0);
                let ng   = rec.normal;
                let t    = rec.tangent;
                let b    = ng.cross(t);
                return (t * ts_n.x + b * ts_n.y + ng * ts_n.z).unit();
            }
        }
        rec.normal
    }

    fn albedo_at(&self, u: f32, v: f32) -> Color {
        let c = self.albedo_tex.as_ref().map_or(self.albedo, |t| sample_srgb(t, u, v));
        match &self.ao_tex {
            Some(ao) => c * sample_linear(ao, u, v),
            None     => c,
        }
    }
    fn roughness_at(&self, u: f32, v: f32) -> f32 {
        self.roughness_tex.as_ref().map_or(self.roughness, |t| sample_linear(t, u, v))
    }
    fn metallic_at(&self, u: f32, v: f32) -> f32 {
        self.metallic_tex.as_ref().map_or(self.metallic, |t| sample_linear(t, u, v))
    }
}

impl Material for PbrMaterial {
    fn shading_normal(&self, rec: &HitRecord<'_>) -> Vec3 {
        self.tbn_normal(rec)
    }

    fn scatter(&self, r_in: &Ray, rec: &HitRecord<'_>, rng: &mut dyn RngCore) -> Option<ScatterRecord> {
        let n = self.tbn_normal(rec);
        let wo    = (-r_in.direction).unit();
        let cos_o = wo.dot(n);
        if cos_o <= 0.0 { return None; }

        let albedo    = self.albedo_at(rec.u, rec.v);
        let roughness = self.roughness_at(rec.u, rec.v);
        let metallic  = self.metallic_at(rec.u, rec.v);
        let alpha = (roughness * roughness).max(1e-4_f32);
        let f0    = Color::new(0.04, 0.04, 0.04) * (1.0 - metallic)
                  + albedo * metallic;

        // Clearcoat lobe: dielectric IOR 1.5 (F0 = 0.04), scaled by clearcoat weight.
        // The Fresnel at normal incidence is only ~4%, making the clearcoat very rarely
        // sampled and the individual contributions very bright — high variance.  We clamp
        // the sampling probability to a minimum of 12% × clearcoat while dividing the
        // weight by the same clamped value, so the mean is exactly preserved.
        let p_coat = if self.clearcoat > 1e-3 {
            (schlick(cos_o, 1.0 / 1.5_f32) * self.clearcoat).max(0.12 * self.clearcoat)
        } else {
            0.0
        };

        // Base specular probability from Fresnel at the view angle.
        let f_approx = schlick_color(cos_o, f0);
        let p_spec = if metallic > 0.999 {
            1.0_f32
        } else {
            f_approx.luminance().clamp(0.04, 0.9)
        };

        if rng.gen::<f32>() < p_coat {
            // ── Clearcoat specular (isotropic VNDF) ──────────────────────────
            let alpha_coat = (self.clearcoat_roughness * self.clearcoat_roughness).max(1e-4);
            let (t, b) = n.onb();
            let xi1: f32 = rng.gen();
            let xi2: f32 = rng.gen();

            let wo_ts = Vec3::new(wo.dot(t), wo.dot(b), cos_o);
            let m_ts  = vndf_sample_aniso(wo_ts, alpha_coat, alpha_coat, xi1, xi2);
            let m     = (t * m_ts.x + b * m_ts.y + n * m_ts.z).unit();

            let vo_h  = wo.dot(m).max(0.0);
            let wi    = (2.0 * vo_h * m - wo).unit();
            let cos_i = wi.dot(n);
            if cos_i <= 0.0 { return None; }

            let f_coat = schlick(vo_h, 1.0 / 1.5_f32) * self.clearcoat;
            let g1_wi  = smith_g1(cos_i, alpha_coat);

            let film = if self.film_thickness > 0.0 {
                nacre_color(vo_h, self.film_ior, self.film_thickness)
            } else {
                Color::new(1.0, 1.0, 1.0)
            };

            // VNDF weight: F_coat · film · G1(wi) / p_coat
            Some(ScatterRecord {
                attenuation: film * (f_coat * g1_wi / p_coat),
                ray:         Ray::scatter_from(rec.p, wi, r_in),
                skip_pdf:    true,
            })
        } else {
            // ── Base layer ────────────────────────────────────────────────────
            // All base weights divided by (1 − p_coat) for Russian-roulette correction.
            if rng.gen::<f32>() < p_spec {
                let xi1: f32 = rng.gen();
                let xi2: f32 = rng.gen();

                // Unified VNDF: ax = ay = α at anisotropy=0 is identical to isotropic GGX.
                // Disney convention: ax (tangent) = α/aspect (stretched), ay (bitangent) = α*aspect.
                let aspect = (1.0 - 0.9 * self.anisotropy.clamp(0.0, 1.0)).max(0.001_f32).sqrt();
                let ax = alpha / aspect;
                let ay = alpha * aspect;

                let (t0, b0) = n.onb();
                let (ca, sa) = (self.anisotropy_angle.cos(), self.anisotropy_angle.sin());
                let t = t0 * ca + b0 * sa;
                let b = b0 * ca - t0 * sa;

                let wo_ts = Vec3::new(wo.dot(t), wo.dot(b), cos_o);
                let m_ts  = vndf_sample_aniso(wo_ts, ax, ay, xi1, xi2);
                let m     = (t * m_ts.x + b * m_ts.y + n * m_ts.z).unit();

                let vo_h  = wo.dot(m).max(0.0);
                let wi    = (2.0 * vo_h * m - wo).unit();
                let cos_i = wi.dot(n);
                if cos_i <= 0.0 { return None; }

                let wi_ts = Vec3::new(wi.dot(t), wi.dot(b), cos_i);
                let g1_wi = smith_g1_aniso(wi_ts.z, wi_ts.x, wi_ts.y, ax, ay);
                let f     = schlick_color(vo_h, f0);

                Some(ScatterRecord {
                    attenuation: f * g1_wi / (p_spec * (1.0 - p_coat)),
                    ray:         Ray::scatter_from(rec.p, wi, r_in),
                    skip_pdf:    true,
                })
            } else {
                // ── Diffuse Lambertian + Sheen ────────────────────────────────
                // skip_pdf: false — the integrator samples a cosine-weighted direction
                // and calls scattering_pdf(); rec.normal here is a throwaway placeholder.
                //
                // Sheen is a retroreflective grazing lobe for cloth/velvet (Disney 2012).
                // scattering_pdf() returns cos_i/π so attenuation must equal π × f_total:
                //   attenuation = albedo*(1−m) + π·sheen·C_sheen·F_H
                // where F_H = (1−cos_o)^5 uses the view angle as proxy for the half-angle,
                // correctly peaking at grazing incidence without requiring wi at scatter time.
                let sheen_attn = if self.sheen > 0.0 && metallic < 0.999 {
                    let luma = albedo.luminance();
                    let c_tint = if luma > 1e-6 { albedo / luma } else { Color::new(1.0, 1.0, 1.0) };
                    let sheen_color = Color::new(1.0, 1.0, 1.0) * (1.0 - self.sheen_tint) + c_tint * self.sheen_tint;
                    let f_h = (1.0 - cos_o).powi(5);
                    sheen_color * (self.sheen * (1.0 - metallic) * f_h * PI)
                } else {
                    Color::default()
                };
                Some(ScatterRecord {
                    attenuation: (albedo * (1.0 - metallic) + sheen_attn) / ((1.0 - p_spec) * (1.0 - p_coat)),
                    ray:         Ray::scatter_from(rec.p, rec.normal, r_in),
                    skip_pdf:    false,
                })
            }
        }
    }

    fn emitted(&self, _u: f32, _v: f32, _p: Point3) -> Color {
        self.emission * self.emission_strength
    }

    fn scattering_pdf(&self, _r_in: &Ray, rec: &HitRecord<'_>, scattered: &Ray) -> f32 {
        let cosine = self.tbn_normal(rec).dot(scattered.direction.unit());
        (cosine / PI).max(0.0)
    }

    fn specular_brdf_cos(&self, r_in: &Ray, rec: &HitRecord<'_>, wi: Vec3) -> Color {
        let n     = self.tbn_normal(rec);
        let wo    = (-r_in.direction).unit();
        let cos_o = wo.dot(n);
        let cos_i = wi.dot(n);
        if cos_o <= 0.0 || cos_i <= 0.0 { return Color::default(); }

        let albedo    = self.albedo_at(rec.u, rec.v);
        let roughness = self.roughness_at(rec.u, rec.v);
        let metallic  = self.metallic_at(rec.u, rec.v);
        let h    = (wo + wi).unit();
        let wo_h = wo.dot(h).max(0.0);
        let h_n  = h.dot(n).max(0.0);
        let f0   = Color::new(0.04, 0.04, 0.04) * (1.0 - metallic)
                 + albedo * metallic;

        let mut result = Color::default();

        // Clearcoat lobe: film · D · F_coat · G1(wo) · G1(wi) / (4 · cos_o)
        if self.clearcoat > 1e-3 {
            let alpha_coat = (self.clearcoat_roughness * self.clearcoat_roughness).max(1e-4);
            let d     = ggx_ndf(h_n, alpha_coat);
            let g1_o  = smith_g1(cos_o, alpha_coat);
            let g1_i  = smith_g1(cos_i, alpha_coat);
            let f_c   = schlick(wo_h, 1.0 / 1.5_f32) * self.clearcoat;
            let film  = if self.film_thickness > 0.0 {
                nacre_color(wo_h, self.film_ior, self.film_thickness)
            } else {
                Color::new(1.0, 1.0, 1.0)
            };
            result += film * (d * f_c * g1_o * g1_i / (4.0 * cos_o));
        }

        // Base specular lobe: F · D · G1(wo) · G1(wi) / (4 · cos_o)
        {
            let alpha  = (roughness * roughness).max(1e-4_f32);
            let aspect = (1.0 - 0.9 * self.anisotropy.clamp(0.0, 1.0)).max(0.001_f32).sqrt();
            let ax     = alpha / aspect;
            let ay     = alpha * aspect;
            let (t0, b0) = n.onb();
            let (ca, sa) = (self.anisotropy_angle.cos(), self.anisotropy_angle.sin());
            let t = t0 * ca + b0 * sa;
            let b = b0 * ca - t0 * sa;
            let h_ts  = Vec3::new(h.dot(t),  h.dot(b),  h_n);
            let wo_ts = Vec3::new(wo.dot(t), wo.dot(b), cos_o);
            let wi_ts = Vec3::new(wi.dot(t), wi.dot(b), cos_i);
            let d     = ggx_ndf_aniso(h_ts, ax, ay);
            let g1_o  = smith_g1_aniso(cos_o, wo_ts.x, wo_ts.y, ax, ay);
            let g1_i  = smith_g1_aniso(cos_i, wi_ts.x, wi_ts.y, ax, ay);
            let f     = schlick_color(wo_h, f0);
            result += f * (d * g1_o * g1_i / (4.0 * cos_o));
        }

        result
    }

    fn specular_sampling_pdf(&self, r_in: &Ray, rec: &HitRecord<'_>, wi: Vec3) -> f32 {
        let n     = self.tbn_normal(rec);
        let wo    = (-r_in.direction).unit();
        let cos_o = wo.dot(n);
        let cos_i = wi.dot(n);
        if cos_o <= 0.0 || cos_i <= 0.0 { return 0.0; }

        let albedo   = self.albedo_at(rec.u, rec.v);
        let roughness = self.roughness_at(rec.u, rec.v);
        let metallic  = self.metallic_at(rec.u, rec.v);
        let h   = (wo + wi).unit();
        let h_n = h.dot(n).max(0.0);
        let f0  = Color::new(0.04, 0.04, 0.04) * (1.0 - metallic)
                + albedo * metallic;

        let p_coat = if self.clearcoat > 1e-3 {
            (schlick(cos_o, 1.0 / 1.5_f32) * self.clearcoat).max(0.12 * self.clearcoat)
        } else { 0.0 };
        let f_approx = schlick_color(cos_o, f0);
        let p_spec   = if metallic > 0.999 { 1.0_f32 } else {
            f_approx.luminance().clamp(0.04, 0.9)
        };

        let mut pdf = 0.0;

        // Clearcoat VNDF pdf: p_coat · D · G1(wo) / (4 · cos_o)
        if self.clearcoat > 1e-3 {
            let alpha_coat = (self.clearcoat_roughness * self.clearcoat_roughness).max(1e-4);
            let d    = ggx_ndf(h_n, alpha_coat);
            let g1_o = smith_g1(cos_o, alpha_coat);
            pdf += p_coat * d * g1_o / (4.0 * cos_o);
        }

        // Base specular VNDF pdf: (1−p_coat)·p_spec · D · G1(wo) / (4 · cos_o)
        if (1.0 - p_coat) * p_spec > 1e-6 {
            let alpha  = (roughness * roughness).max(1e-4_f32);
            let aspect = (1.0 - 0.9 * self.anisotropy.clamp(0.0, 1.0)).max(0.001_f32).sqrt();
            let ax     = alpha / aspect;
            let ay     = alpha * aspect;
            let (t0, b0) = n.onb();
            let (ca, sa) = (self.anisotropy_angle.cos(), self.anisotropy_angle.sin());
            let t  = t0 * ca + b0 * sa;
            let b  = b0 * ca - t0 * sa;
            let h_ts  = Vec3::new(h.dot(t),  h.dot(b),  h_n);
            let wo_ts = Vec3::new(wo.dot(t), wo.dot(b), cos_o);
            let d     = ggx_ndf_aniso(h_ts, ax, ay);
            let g1_o  = smith_g1_aniso(cos_o, wo_ts.x, wo_ts.y, ax, ay);
            pdf += (1.0 - p_coat) * p_spec * d * g1_o / (4.0 * cos_o);
        }

        pdf
    }

    fn albedo_hint(&self, u: f32, v: f32, _p: Point3) -> Color { self.albedo_at(u, v) }
    fn can_receive_caustics(&self) -> bool { self.metallic < 0.999 }
}

// ── Pearl sun-direction context ───────────────────────────────────────────────
// The renderer sets this once per frame (before the parallel tile pass) so that
// PearlMaterial::scatter can add a nacre highlight from the sun's incident angle.
// PEARL_SUN_ACTIVE gates all reads: any call path that doesn't set the direction
// first (photon tracing, unit tests) receives None from pearl_sun_dir() and the
// sun highlight is simply omitted — no stale data is used.
// Relaxed ordering is correct: the write happens-before par_iter_mut fires, and
// a one-frame lag on the read side is imperceptible.

static PEARL_SUN_X: AtomicU32 = AtomicU32::new(0);
static PEARL_SUN_Y: AtomicU32 = AtomicU32::new(0x3F80_0000); // f32 1.0
static PEARL_SUN_Z: AtomicU32 = AtomicU32::new(0);
static PEARL_SUN_ACTIVE: AtomicBool = AtomicBool::new(false);

pub fn set_pearl_sun_dir(dir: Vec3) {
    PEARL_SUN_X.store(dir.x.to_bits(), Ordering::Relaxed);
    PEARL_SUN_Y.store(dir.y.to_bits(), Ordering::Relaxed);
    PEARL_SUN_Z.store(dir.z.to_bits(), Ordering::Relaxed);
    // Release: ensures X/Y/Z writes are visible to any thread that observes ACTIVE=true.
    PEARL_SUN_ACTIVE.store(true, Ordering::Release);
}

pub fn clear_pearl_sun_dir() {
    PEARL_SUN_ACTIVE.store(false, Ordering::Relaxed);
}

#[inline]
fn pearl_sun_dir() -> Option<Vec3> {
    // Acquire: pairs with the Release store in set_pearl_sun_dir, guaranteeing X/Y/Z are visible.
    if PEARL_SUN_ACTIVE.load(Ordering::Acquire) {
        Some(Vec3::new(
            f32::from_bits(PEARL_SUN_X.load(Ordering::Relaxed)),
            f32::from_bits(PEARL_SUN_Y.load(Ordering::Relaxed)),
            f32::from_bits(PEARL_SUN_Z.load(Ordering::Relaxed)),
        ))
    } else {
        None
    }
}

// ── Pearl ─────────────────────────────────────────────────────────────────────

// Precomputed LUT for the nacre spectral integral over OPD.
//
// nacre_color's raw output (before the grazing blend) depends only on the
// optical path difference OPD = 2·n·d·cos(θ_t), not on the individual
// (film_ior, film_thickness, cos_theta) parameters.  We table-ise that
// 65-iteration integral once over OPD ∈ [0, 2500 nm] at 256 steps (~10 nm
// per step, far finer than the ~800 nm colour-beat period) and look it up
// with linear interpolation.  Cost drops from ~65 trig ops to 1 lerp.

const NACRE_LUT_SIZE: usize  = 256;
const NACRE_OPD_MAX:  f32    = 2500.0; // nm — covers film thicknesses up to ~850 nm

static NACRE_LUT: OnceLock<Vec<Color>> = OnceLock::new();

fn nacre_lut() -> &'static [Color] {
    NACRE_LUT.get_or_init(|| {
        (0..NACRE_LUT_SIZE).map(|i| {
            let opd = i as f32 / (NACRE_LUT_SIZE - 1) as f32 * NACRE_OPD_MAX;
            let mut color = Color::default();
            let mut lambda = 380.0_f32;
            while lambda <= 700.01 {
                let irid = 0.5 * (1.0 + (2.0 * PI * opd / lambda).cos());
                color += spectral_to_rgb(lambda) * irid;
                lambda += 5.0;
            }
            color / 65.0
        }).collect()
    })
}

/// Thin-film interference colour evaluated via the precomputed OPD LUT.
///
/// OPD = 2·n·d·cos(θ_t) is computed from the parameters, then looked up in
/// `NACRE_LUT` with linear interpolation.  A grazing blend toward (0.5,0.5,0.5)
/// mimics the many-beam suppression of real nacre at oblique angles.
#[inline]
fn nacre_color(cos_theta: f32, film_ior: f32, film_thickness_nm: f32) -> Color {
    let sin_sq = (1.0 - cos_theta * cos_theta).max(0.0);
    let cos_t  = (1.0 - sin_sq / (film_ior * film_ior)).max(0.0).sqrt();
    let opd    = 2.0 * film_ior * film_thickness_nm * cos_t;

    // LUT lookup with linear interpolation.
    let lut = nacre_lut();
    let t   = (opd / NACRE_OPD_MAX).clamp(0.0, 1.0) * (NACRE_LUT_SIZE - 1) as f32;
    let lo  = t as usize;
    let hi  = (lo + 1).min(NACRE_LUT_SIZE - 1);
    let raw = lut[lo] * (1.0 - (t - lo as f32)) + lut[hi] * (t - lo as f32);

    // Grazing blend: suppress saturation toward white luster at oblique angles.
    let blend = cos_theta.powf(0.5);
    raw * blend + Color::new(0.5, 0.5, 0.5) * (1.0 - blend)
}

/// Smooth, aperiodic noise in [−1, 1] — three sine waves at golden-ratio
/// spaced frequencies so the pattern never visibly tiles.
/// Caller scales `p` by `film_scale` to control patch size.
#[inline]
fn oil_noise(p: Point3) -> f32 {
    let a = (p.x * 1.618_f32 + p.z).sin();
    let b = (p.z * 1.618_f32 - p.x).sin();
    let c = (p.x * 0.618_f32 + p.y + p.z).sin();
    (a + b + c) * (1.0 / 3.0)
}

/// Pearl surface material: thin-film nacre iridescence over a Lambertian body.
///
/// Fresnel reflection at the air-nacre interface is tinted by thin-film
/// interference (the "orient") whose colour shifts through the spectrum as the
/// viewing angle changes — rose-pink at perpendicular, cycling through blue and
/// green at oblique angles.  The transmitted fraction scatters diffusely with
/// the pearl's body colour.
pub struct PearlMaterial {
    /// Body colour (cream/white for Akoya, golden for South Sea, black for Tahitian).
    pub base_color:       Color,
    /// Effective nacre IOR.  Average of aragonite (~1.68) and organic (~1.44)
    /// layers; ~1.56 gives a typical Akoya orient cycle.
    pub ior:              f32,
    /// Nacre platelet thickness in **nanometres** (natural pearls: 380–600 nm).
    pub film_thickness:   f32,
    /// How strongly the orient tints the diffuse body colour [0–1].
    pub orient_strength:  f32,
    /// Spatial frequency of the thickness variation (oil-spill pattern).
    pub film_scale:       f32,
    /// GGX roughness of the pearl luster (0 = mirror, 0.05 = soft Akoya sheen).
    pub luster_roughness: f32,
}

impl Material for PearlMaterial {
    fn scatter(&self, r_in: &Ray, rec: &HitRecord<'_>, rng: &mut dyn RngCore) -> Option<ScatterRecord> {
        let n         = rec.normal;
        let wo        = (-r_in.direction).unit();
        let cos_theta = wo.dot(n).clamp(0.0, 1.0);

        let varied = self.film_thickness + oil_noise(rec.p * self.film_scale) * 150.0;
        // Boost the luster sampling probability to a minimum of 10% to reduce the
        // variance caused by near-zero Fresnel at normal incidence (~5% for IOR 1.56).
        // The weight (f_h / f) adjusts inversely, so the mean is exactly preserved.
        let f = schlick(cos_theta, 1.0 / self.ior).max(0.10_f32);

        // Illumination angle: governs the nacre glow on the diffuse path.
        // OPD is set when light enters the aragonite platelet stack, so this
        // component shifts with the sun position, not the camera angle.
        let cos_illumin = match pearl_sun_dir() {
            Some(sun) => n.dot(sun).clamp(0.0, 1.0),
            None      => cos_theta,
        };

        if rng.gen::<f32>() < f {
            // ── GGX luster: view-dependent thin-film iridescence (VNDF) ──────
            let (t, b)    = n.onb();
            let alpha_l   = (self.luster_roughness * self.luster_roughness).max(1e-4);
            let xi1: f32  = rng.gen();
            let xi2: f32  = rng.gen();

            let wo_ts = Vec3::new(wo.dot(t), wo.dot(b), cos_theta);
            let m_ts  = vndf_sample_aniso(wo_ts, alpha_l, alpha_l, xi1, xi2);
            let m     = (t * m_ts.x + b * m_ts.y + n * m_ts.z).unit();

            let vo_h  = wo.dot(m).max(0.0);
            let wi    = (2.0 * vo_h * m - wo).unit();
            let cos_i = wi.dot(n);
            if cos_i <= 0.0 { return None; }

            let f_h    = schlick(vo_h, 1.0 / self.ior);
            let g1_wi  = smith_g1(cos_i, alpha_l);
            let orient = nacre_color(vo_h, self.ior, varied);

            // VNDF weight: orient · F · G1(wi), divided by RR probability f.
            let attenuation = orient * (f_h * g1_wi / f);
            Some(ScatterRecord {
                attenuation,
                ray:      Ray::scatter_from(rec.p, wi, r_in),
                skip_pdf: true,
            })
        } else {
            // ── Diffuse nacre glow: illumination-angle thin film ──────────────
            // Colour is set by how light enters the platelet stack → shifts with
            // sun movement, not camera orbit.  Existing behaviour preserved.
            let orient = nacre_color(cos_illumin, self.ior, varied);
            let s      = self.orient_strength;
            let tinted = self.base_color * (orient * s + Color::new(1.0, 1.0, 1.0) * (1.0 - s));
            Some(ScatterRecord {
                // Divide by (1−f): the diffuse path is selected with probability (1−f),
                // so we must weight up to compensate for Russian roulette.
                attenuation: tinted * (1.0 / (1.0 - f)),
                ray:         Ray::scatter_from(rec.p, n, r_in),
                skip_pdf:    false,
            })
        }
    }

    fn scattering_pdf(&self, _r_in: &Ray, rec: &HitRecord<'_>, scattered: &Ray) -> f32 {
        (rec.normal.dot(scattered.direction.unit()) / PI).max(0.0)
    }

    fn specular_brdf_cos(&self, r_in: &Ray, rec: &HitRecord<'_>, wi: Vec3) -> Color {
        let n     = rec.normal;
        let wo    = (-r_in.direction).unit();
        let cos_o = wo.dot(n).clamp(0.0, 1.0);
        let cos_i = wi.dot(n);
        if cos_i <= 0.0 { return Color::default(); }

        let h     = (wo + wi).unit();
        let wo_h  = wo.dot(h).max(0.0);
        let h_n   = h.dot(n).max(0.0);
        let alpha = (self.luster_roughness * self.luster_roughness).max(1e-4);
        let d     = ggx_ndf(h_n, alpha);
        let g1_o  = smith_g1(cos_o, alpha);
        let g1_i  = smith_g1(cos_i, alpha);
        let f_h   = schlick(wo_h, 1.0 / self.ior);
        let varied = self.film_thickness + oil_noise(rec.p * self.film_scale) * 150.0;
        let orient = nacre_color(wo_h, self.ior, varied);
        orient * (d * f_h * g1_o * g1_i / (4.0 * cos_o))
    }

    fn specular_sampling_pdf(&self, r_in: &Ray, rec: &HitRecord<'_>, wi: Vec3) -> f32 {
        let n     = rec.normal;
        let wo    = (-r_in.direction).unit();
        let cos_o = wo.dot(n).clamp(0.0, 1.0);
        let cos_i = wi.dot(n);
        if cos_i <= 0.0 { return 0.0; }

        let h     = (wo + wi).unit();
        let h_n   = h.dot(n).max(0.0);
        let f     = schlick(cos_o, 1.0 / self.ior).max(0.10_f32); // boosted, matches scatter()
        let alpha = (self.luster_roughness * self.luster_roughness).max(1e-4);
        let d     = ggx_ndf(h_n, alpha);
        let g1_o  = smith_g1(cos_o, alpha);
        f * d * g1_o / (4.0 * cos_o)
    }

    fn albedo_hint(&self, _u: f32, _v: f32, _p: Point3) -> Color { self.base_color }
    fn can_receive_caustics(&self) -> bool { true }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    fn luma(c: Color) -> f32 { c.luminance() }

    // ── GGX NDF ──────────────────────────────────────────────────────────────

    #[test]
    fn ggx_ndf_integrates_to_one() {
        // ∫_hemisphere D(cos_h, α) · cos_h · dω = 1  for any α > 0.
        // Integrate analytically over φ (gives 2π) and numerically over θ ∈ [0, π/2].
        for &alpha in &[0.1f32, 0.3, 0.7, 1.0] {
            let n       = 1000usize;
            let d_theta = std::f32::consts::FRAC_PI_2 / n as f32;
            let mut sum = 0.0f64;
            for i in 0..n {
                let theta = (i as f32 + 0.5) * d_theta;
                let cos_h = theta.cos();
                let sin_h = theta.sin();
                sum += ggx_ndf(cos_h, alpha) as f64
                    * cos_h as f64
                    * sin_h as f64
                    * d_theta as f64
                    * 2.0 * std::f64::consts::PI;
            }
            assert!((sum - 1.0).abs() < 0.01,
                "GGX NDF integral at α={alpha} = {sum:.4} (expected 1.0)");
        }
    }

    // ── Smith G1 ─────────────────────────────────────────────────────────────

    #[test]
    fn smith_g1_is_in_unit_range() {
        for &alpha in &[0.01f32, 0.1, 0.3, 0.5, 0.8, 1.0] {
            for &cos_theta in &[0.01f32, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0] {
                let g = smith_g1(cos_theta, alpha);
                assert!(g >= 0.0 && g <= 1.0,
                    "smith_g1({cos_theta}, {alpha}) = {g:.4} is outside [0, 1]");
            }
        }
    }

    #[test]
    fn smith_g1_smooth_limit_is_one() {
        // At α→0 (perfectly smooth surface): G1 = 2cos / (cos + sqrt(α²+(1−α²)cos²)) → 1.
        for &cos_theta in &[0.1f32, 0.5, 0.9, 1.0] {
            let g = smith_g1(cos_theta, 1e-6);
            assert!((g - 1.0).abs() < 1e-4,
                "smooth G1(cos={cos_theta}, α≈0) = {g:.5} (expected 1.0)");
        }
    }

    // ── White furnace (PbrMaterial) ───────────────────────────────────────────

    // Calls scatter() n times at normal incidence and returns the mean attenuation luminance.
    // For metallic paths (skip_pdf=true), attenuation is already the full Monte Carlo weight.
    fn white_furnace_mean(mat: &PbrMaterial, n: usize, rng: &mut SmallRng) -> f64 {
        let ray = Ray::new(
            Point3::new(0.0, 0.0, -1.0),
            Vec3::new(0.0, 0.0, 1.0),
        );
        let rec = HitRecord {
            p:          Point3::new(0.0, 0.0, 0.0),
            normal:     Vec3::new(0.0, 0.0, -1.0), // outward normal faces toward ray origin
            mat,
            t: 1.0, u: 0.0, v: 0.0,
            front_face: true,
            tangent:    Vec3::default(),
        };
        let mut sum   = 0.0f64;
        let mut count = 0usize;
        for _ in 0..n {
            if let Some(sr) = mat.scatter(&ray, &rec, rng) {
                sum   += luma(sr.attenuation) as f64;
                count += 1;
            }
        }
        if count == 0 { return 0.0; }
        sum / count as f64
    }

    #[test]
    fn white_furnace_metallic_no_energy_gain() {
        // F·G1(wi) ≤ 1 must hold at every roughness — single-scattering GGX
        // loses energy at high roughness (expected) but must never gain it.
        for &roughness in &[0.1f32, 0.5, 0.9] {
            let mat = PbrMaterial {
                albedo:    Color::new(1.0, 1.0, 1.0),
                metallic:  1.0,
                roughness,
                clearcoat: 0.0,
                sheen:     0.0,
                ..PbrMaterial::default()
            };
            let mut rng  = SmallRng::seed_from_u64(42);
            let mean = white_furnace_mean(&mat, 8000, &mut rng);
            assert!(mean <= 1.0 + 1e-6,
                "white metallic furnace (roughness={roughness}) mean={mean:.4} — energy gain");
        }
    }

    #[test]
    fn white_furnace_smooth_metallic_is_nearly_lossless() {
        // At low roughness and normal incidence, G1(wi)≈1, so E[F·G1]≈1.
        // A mirror-like white metal should return almost all incident energy.
        let mat = PbrMaterial {
            albedo:    Color::new(1.0, 1.0, 1.0),
            metallic:  1.0,
            roughness: 0.05,
            clearcoat: 0.0,
            sheen:     0.0,
            ..PbrMaterial::default()
        };
        let mut rng  = SmallRng::seed_from_u64(123);
        let mean = white_furnace_mean(&mat, 8000, &mut rng);
        assert!(mean > 0.95, "smooth white metallic furnace mean={mean:.4} — too lossy");
        assert!(mean <= 1.0 + 1e-6, "smooth white metallic furnace mean={mean:.4} — energy gain");
    }

    // ── Dielectric ────────────────────────────────────────────────────────────

    // HitRecord for a front-face hit at normal incidence (ray going +z, normal −z).
    fn front_hit_normal_incidence(mat: &dyn Material) -> HitRecord<'_> {
        HitRecord {
            p:          Point3::new(0.0, 0.0, 0.0),
            normal:     Vec3::new(0.0, 0.0, -1.0),
            mat,
            t: 1.0, u: 0.0, v: 0.0,
            front_face: true,
            tangent:    Vec3::default(),
        }
    }

    #[test]
    fn dielectric_attenuation_is_always_white() {
        // Glass must be perfectly clear regardless of angle or Fresnel outcome.
        let glass = Dielectric { ir: 1.5 };
        let r   = Ray::new(Point3::new(0.0, 0.0, -5.0), Vec3::new(0.0, 0.0, 1.0));
        let rec = front_hit_normal_incidence(&glass);
        let mut rng = SmallRng::seed_from_u64(0);
        for _ in 0..200 {
            let sr = glass.scatter(&r, &rec, &mut rng).expect("glass always scatters");
            assert!((sr.attenuation.x - 1.0).abs() < 1e-6
                &&  (sr.attenuation.y - 1.0).abs() < 1e-6
                &&  (sr.attenuation.z - 1.0).abs() < 1e-6,
                "attenuation must be white, got {:?}", sr.attenuation);
        }
    }

    #[test]
    fn dielectric_is_always_specular() {
        // skip_pdf=true so the renderer uses the scattered direction directly.
        let glass = Dielectric { ir: 1.5 };
        let r   = Ray::new(Point3::new(0.0, 0.0, -5.0), Vec3::new(0.0, 0.0, 1.0));
        let rec = front_hit_normal_incidence(&glass);
        let mut rng = SmallRng::seed_from_u64(42);
        for _ in 0..100 {
            let sr = glass.scatter(&r, &rec, &mut rng).expect("glass always scatters");
            assert!(sr.skip_pdf, "dielectric scatter must be specular (skip_pdf=true)");
        }
    }

    #[test]
    fn dielectric_tir_always_reflects_at_steep_internal_angle() {
        // From inside glass (n=1.5) at 50° from the surface normal:
        // ratio × sin(50°) ≈ 1.15 > 1 → TIR is guaranteed, no RNG involved.
        // The reflected ray must flip its z-component (going back into the medium).
        let s50 = 50f32.to_radians().sin();
        let c50 = 50f32.to_radians().cos();
        let glass = Dielectric { ir: 1.5 };
        let r = Ray::new(Point3::new(0.0, 0.0, -1.0), Vec3::new(s50, 0.0, c50));
        // back face: outward normal (0,0,1) dotted with direction = c50 > 0 → front_face=false.
        // HitRecord::new would flip to (0,0,−1); we set that directly.
        let rec = HitRecord {
            p: Point3::new(0.0, 0.0, 0.0),
            normal: Vec3::new(0.0, 0.0, -1.0),
            mat: &glass,
            t: 1.0, u: 0.0, v: 0.0,
            front_face: false,
            tangent:    Vec3::default(),
        };
        let mut rng = SmallRng::seed_from_u64(0);
        for _ in 0..50 {
            let sr = glass.scatter(&r, &rec, &mut rng).expect("must scatter");
            assert!(sr.ray.direction.z < 0.0,
                "TIR must reflect back (z < 0), got z={:.4}", sr.ray.direction.z);
        }
    }

    #[test]
    fn dielectric_mostly_transmits_at_normal_incidence() {
        // Schlick r₀ = ((1−n)/(1+n))² ≈ 0.04 for n=1.5.
        // Over 500 samples, > 90% should transmit (z > 0, same direction as input).
        let glass = Dielectric { ir: 1.5 };
        let r   = Ray::new(Point3::new(0.0, 0.0, -5.0), Vec3::new(0.0, 0.0, 1.0));
        let rec = front_hit_normal_incidence(&glass);
        let mut rng = SmallRng::seed_from_u64(7);
        let n = 500usize;
        let transmitted = (0..n)
            .filter(|_| glass.scatter(&r, &rec, &mut rng).expect("must scatter").ray.direction.z > 0.0)
            .count();
        let frac = transmitted as f32 / n as f32;
        assert!(frac > 0.90,
            "≥90% should transmit at normal incidence (Schlick r₀≈4%), got {:.1}%", frac * 100.0);
    }
}
