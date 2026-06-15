use std::f32::consts::PI;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use rand::{Rng, RngCore};
use crate::perlin::Perlin;
use crate::vec3::{Color, Point3, Vec3};
use crate::ray::Ray;
use crate::hittable::{HitRecord, Material, ScatterRecord};
use crate::texture::Texture;
use crate::spectrum::{cauchy_ior, spectral_to_rgb};

pub struct DiffuseLight {
    pub emit: Texture,
}

impl Material for DiffuseLight {
    fn scatter(&self, _r_in: &Ray, _rec: &HitRecord<'_>, _rng: &mut dyn RngCore) -> Option<ScatterRecord> { None }
    fn emitted(&self, u: f32, v: f32, p: Point3) -> Color { self.emit.value(u, v, p) }
    fn albedo_hint(&self, u: f32, v: f32, p: Point3) -> Color { self.emitted(u, v, p) }
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

pub struct Metal {
    pub albedo: Color,
    pub fuzz: f32,
}

impl Material for Metal {
    fn scatter(&self, r_in: &Ray, rec: &HitRecord<'_>, rng: &mut dyn RngCore) -> Option<ScatterRecord> {
        let reflected = r_in.direction.unit().reflect(rec.normal);
        let ray = Ray::scatter_from(rec.p, reflected + self.fuzz * Vec3::random_unit_vector(rng), r_in);
        if ray.direction.dot(rec.normal) > 0.0 {
            Some(ScatterRecord { attenuation: self.albedo, ray, skip_pdf: true })
        } else {
            None
        }
    }
    fn albedo_hint(&self, _u: f32, _v: f32, _p: Point3) -> Color { self.albedo }
}

#[inline]
fn schlick(cosine: f32, ref_idx: f32) -> f32 {
    let r0 = ((1.0 - ref_idx) / (1.0 + ref_idx)).powi(2);
    r0 + (1.0 - r0) * (1.0 - cosine).powi(5)
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
    let direction = if reflected { unit.reflect(rec.normal) } else { unit.refract(rec.normal, ratio) };
    (direction, reflected)
}

// ── GGX / PBR helpers ─────────────────────────────────────────────────────────

/// Schlick Fresnel with a colored F0 (supports metal tints).
fn schlick_color(cos_theta: f32, f0: Color) -> Color {
    let t = (1.0 - cos_theta).max(0.0).powi(5);
    f0 + (Color::new(1.0, 1.0, 1.0) - f0) * t
}

/// Smith G1 term for GGX (height-correlated uncorrelated form).
fn smith_g1(cos_theta: f32, alpha: f32) -> f32 {
    let a2 = alpha * alpha;
    let c2 = cos_theta * cos_theta;
    2.0 * cos_theta / (cos_theta + (a2 + (1.0 - a2) * c2).sqrt())
}

/// Build a tangent + bitangent pair perpendicular to `n`.
fn make_onb(n: Vec3) -> (Vec3, Vec3) {
    let up = if n.x.abs() < 0.999 { Vec3::new(1.0, 0.0, 0.0) } else { Vec3::new(0.0, 1.0, 0.0) };
    let t = n.cross(up).unit();
    let b = n.cross(t);
    (t, b)
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
pub struct SpectralDielectric {
    pub cauchy_b: f32,
    pub cauchy_c: f32,
}

impl Material for SpectralDielectric {
    fn scatter(&self, r_in: &Ray, rec: &HitRecord<'_>, rng: &mut dyn RngCore) -> Option<ScatterRecord> {
        let ior = cauchy_ior(r_in.wavelength, self.cauchy_b, self.cauchy_c);
        let (direction, reflected) = dielectric_boundary(ior, r_in, rec, rng);

        // Apply the CMF weight exactly once per path: on the first refraction.
        // Subsequent refractions use (1,1,1) so the weight doesn't compound.
        // Reflections are always achromatic — Fresnel reflectance is nearly flat
        // across the visible spectrum.
        let weight_this_refraction = !reflected && !r_in.spectral_weighted;
        let attenuation = if weight_this_refraction {
            spectral_to_rgb(r_in.wavelength)
        } else {
            Color::new(1.0, 1.0, 1.0)
        };
        let mut scattered = Ray::scatter_from(rec.p, direction, r_in);
        if weight_this_refraction { scattered.spectral_weighted = true; }
        Some(ScatterRecord { attenuation, ray: scattered, skip_pdf: true })
    }

    fn is_spectral(&self) -> bool { true }
}

/// Glass marble: IOR 1.5 glass exterior with a Perlin-based swirl visible from inside.
///
/// The glass surface behaves like a standard dielectric (Fresnel + TIR).  When a
/// ray travels through the interior and exits (`front_face = false`), the
/// attenuation is sampled from a sine-wave marble pattern modulated by Perlin
/// turbulence, producing the characteristic coloured swirl of a cat's-eye marble.
/// Entry events (`front_face = true`) are unattenuated so the glass shell looks
/// clear from outside.
pub struct MarbleMaterial {
    pub ir:     f32,
    pub color1: Color,   // swirl / ribbon colour
    pub color2: Color,   // clear / base colour (typically near white)
    pub scale:  f32,     // spatial frequency — higher = tighter swirls
    pub perlin: Arc<Perlin>,
}

impl Material for MarbleMaterial {
    fn scatter(&self, r_in: &Ray, rec: &HitRecord<'_>, rng: &mut dyn RngCore) -> Option<ScatterRecord> {
        let (direction, _) = dielectric_boundary(self.ir, r_in, rec, rng);
        // Apply the swirl colour only when the ray is inside the marble (front_face = false).
        // rec.u is the sphere-local azimuthal angle (0..1 = full wrap), so the band
        // pattern always spans the sphere regardless of its world-space position.
        // Perlin turbulence distorts the bands to give organic, marble-like swirls.
        let attenuation = if !rec.front_face {
            let noise = self.perlin.turb(rec.p * self.scale, 7);
            let phi   = rec.u * 2.0 * PI;
            let arg   = phi * 2.0 + noise * 8.0;
            let t     = (0.5 * (1.0 + arg.sin())).clamp(0.0, 1.0);
            self.color1 * t + self.color2 * (1.0 - t)
        } else {
            Color::new(1.0, 1.0, 1.0)
        };
        Some(ScatterRecord { attenuation, ray: Ray::scatter_from(rec.p, direction, r_in), skip_pdf: true })
    }
}

/// Sample a new direction from the Henyey-Greenstein phase function around `wi`.
/// `g` ∈ (-1, 1): 0 = isotropic, >0 = forward-biased, <0 = back-scattered.
fn hg_sample(wi: Vec3, g: f32, rng: &mut dyn RngCore) -> Vec3 {
    let xi1 = rng.gen::<f32>();
    let xi2 = rng.gen::<f32>();
    let cos_theta = if g.abs() < 1e-3 {
        1.0 - 2.0 * xi1
    } else {
        let sq = (1.0 - g * g) / (1.0 - g + 2.0 * g * xi1);
        ((1.0 + g * g - sq * sq) / (2.0 * g)).clamp(-1.0, 1.0)
    };
    let sin_theta = (1.0 - cos_theta * cos_theta).max(0.0).sqrt();
    let phi = 2.0 * PI * xi2;
    let (t, b) = make_onb(wi);
    (t * (sin_theta * phi.cos()) + b * (sin_theta * phi.sin()) + wi * cos_theta).unit()
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
/// - `albedo` — per-scatter tint; channels < 1 bleed energy into surviving
///   wavelengths, naturally creating a coloured glow without explicit `σ_a`.
/// - `density` — σ_t (events per unit length).  For radius-0.15 marbles,
///   `density ≈ 7` gives ≈2 scatters per diameter traversal.
/// - `g` — anisotropy (0 = isotropic; ~0.3 suits glass inclusions).
pub struct SSSMaterial {
    pub albedo:  Color,
    pub ior:     f32,
    pub density: f32,
    pub g:       f32,
}

impl Material for SSSMaterial {
    fn scatter(&self, r_in: &Ray, rec: &HitRecord<'_>, rng: &mut dyn RngCore) -> Option<ScatterRecord> {
        let unit = r_in.direction.unit();

        // Check for a volumetric scatter event before the exit surface.
        if !rec.front_face && self.density > 0.0 {
            let path_length = (rec.p - r_in.origin).length();
            let t_scat      = -(rng.gen::<f32>().max(1e-9).ln()) / self.density;
            if t_scat < path_length {
                let p_scat  = r_in.origin + unit * t_scat;
                let new_dir = hg_sample(unit, self.g, rng);
                return Some(ScatterRecord {
                    attenuation: self.albedo,
                    ray:         Ray::scatter_from(p_scat, new_dir, r_in),
                    skip_pdf:    true,
                });
            }
        }

        // Boundary event (entry or unscattered exit): standard Fresnel.
        let (direction, _) = dielectric_boundary(self.ior, r_in, rec, rng);
        Some(ScatterRecord {
            attenuation: Color::new(1.0, 1.0, 1.0),
            ray:         Ray::scatter_from(rec.p, direction, r_in),
            skip_pdf:    true,
        })
    }

    fn albedo_hint(&self, _u: f32, _v: f32, _p: Point3) -> Color { self.albedo }
}

/// Physically-based material using GGX microfacet specular + Lambertian diffuse.
///
/// The specular lobe uses the GGX NDF (Trowbridge-Reitz) with Smith G2 shadowing
/// and Schlick Fresnel.  Specular vs. diffuse is chosen each scatter event via
/// Russian roulette with probability proportional to the Fresnel reflectance,
/// keeping the estimator unbiased.
///
/// Parameters:
/// - `albedo`: base colour — diffuse tint for dielectrics, F0 for metals.
/// - `roughness`: 0 = perfect mirror, 1 = fully diffuse specular.  The actual
///   α used internally is `roughness²` (perceptual remapping).
/// - `metallic`: 0 = dielectric (F0 ≈ 0.04 gray, tinted diffuse), 1 = conductor
///   (F0 = albedo, no diffuse).
pub struct PbrMaterial {
    pub albedo:    Color,
    pub roughness: f32,
    pub metallic:  f32,
}

impl Material for PbrMaterial {
    fn scatter(&self, r_in: &Ray, rec: &HitRecord<'_>, rng: &mut dyn RngCore) -> Option<ScatterRecord> {
        let n  = rec.normal;
        let wo = (-r_in.direction).unit();
        let cos_o = wo.dot(n);
        if cos_o <= 0.0 { return None; }

        let alpha = (self.roughness * self.roughness).max(1e-4_f32);
        // F0: interpolate between dielectric (0.04) and conductor (albedo)
        let f0 = Color::new(0.04, 0.04, 0.04) * (1.0 - self.metallic)
               + self.albedo * self.metallic;

        // Fresnel at viewing angle: drives specular/diffuse split probability.
        let f_approx = schlick_color(cos_o, f0);
        let p_spec = if self.metallic > 0.999 {
            1.0_f32
        } else {
            (0.2126 * f_approx.x + 0.7152 * f_approx.y + 0.0722 * f_approx.z)
                .clamp(0.04, 0.9)
        };

        if rng.gen::<f32>() < p_spec {
            // ── GGX specular ──────────────────────────────────────────────────
            let (t, b) = make_onb(n);
            let xi1: f32 = rng.gen();
            let xi2: f32 = rng.gen();

            // Sample half-vector from GGX NDF (Walter et al. 2007 inverse CDF)
            let cos_th = ((1.0 - xi2) / (xi2 * (alpha * alpha - 1.0) + 1.0)).max(0.0).sqrt();
            let sin_th = (1.0 - cos_th * cos_th).max(0.0).sqrt();
            let phi    = 2.0 * PI * xi1;
            let h = (t * (sin_th * phi.cos()) + b * (sin_th * phi.sin()) + n * cos_th).unit();

            let vo_h = wo.dot(h).max(0.0);
            let wi   = (2.0 * vo_h * h - wo).unit();   // reflect wo about h
            let cos_i = wi.dot(n);
            if cos_i <= 0.0 { return None; }

            let cos_h = h.dot(n).max(1e-6);
            let g2    = smith_g1(cos_o, alpha) * smith_g1(cos_i, alpha);
            let f     = schlick_color(vo_h, f0);

            // Weight from PDF cancellation: F·G·(vo·h) / (cos_o · cos_h)
            // Divide by p_spec to correct for Russian-roulette sampling.
            let weight = f * (g2 * vo_h / (cos_o.max(1e-6) * cos_h));
            Some(ScatterRecord {
                attenuation: weight / p_spec,
                ray: Ray::scatter_from(rec.p, wi, r_in),
                skip_pdf: true,
            })
        } else {
            // ── Diffuse Lambertian (dielectrics only) ─────────────────────────
            // Divide by (1-p_spec) to correct for Russian-roulette sampling.
            Some(ScatterRecord {
                attenuation: self.albedo * ((1.0 - self.metallic) / (1.0 - p_spec)),
                ray: Ray::scatter_from(rec.p, rec.normal, r_in), // direction overridden by PDF
                skip_pdf: false,
            })
        }
    }

    fn scattering_pdf(&self, _r_in: &Ray, rec: &HitRecord<'_>, scattered: &Ray) -> f32 {
        let cosine = rec.normal.dot(scattered.direction.unit());
        (cosine / PI).max(0.0)
    }

    fn albedo_hint(&self, _u: f32, _v: f32, _p: Point3) -> Color { self.albedo }
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
    PEARL_SUN_ACTIVE.store(true, Ordering::Relaxed);
}

pub fn clear_pearl_sun_dir() {
    PEARL_SUN_ACTIVE.store(false, Ordering::Relaxed);
}

#[inline]
fn pearl_sun_dir() -> Option<Vec3> {
    if PEARL_SUN_ACTIVE.load(Ordering::Relaxed) {
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

/// Spectral thin-film interference colour for nacre, computed via a full
/// spectral integral over the visible range.
///
/// Evaluates OPD = 2 n d cos(θ_t) and integrates the two-beam interference
/// intensity against the CIE 1931 CMFs at every 5 nm step from 380–700 nm.
/// This gives a physically correct, deterministic iridescent colour per bounce
/// without relying on the hero wavelength — avoiding the convergence problem
/// that arises because the interference cosine oscillates ~3 full cycles across
/// the visible range, which would average to near-grey with single-wavelength
/// sampling.
///
/// The integral is normalised so that a non-dispersive surface (constant OPD,
/// irid = 0.5 everywhere) returns (0.5, 0.5, 0.5); blending toward white at
/// grazing angles mimics the many-beam suppression of real nacre.
#[inline]
fn nacre_color(cos_theta: f32, film_ior: f32, film_thickness_nm: f32) -> Color {
    let sin_sq = (1.0 - cos_theta * cos_theta).max(0.0);
    let cos_t  = (1.0 - sin_sq / (film_ior * film_ior)).max(0.0).sqrt();
    let opd    = 2.0 * film_ior * film_thickness_nm * cos_t;

    // Spectral integral: Σ CMF(λ) × irid(λ) over 65 wavelengths, 380–700 nm.
    let mut color = Color::default();
    let mut lambda = 380.0_f32;
    while lambda <= 700.01 {
        let irid = 0.5 * (1.0 + (2.0 * PI * opd / lambda).cos());
        color += spectral_to_rgb(lambda) * irid;
        lambda += 5.0;
    }
    color /= 65.0; // Normalise: mean = (0.5, 0.5, 0.5) for flat interference.

    // Grazing blend: at cos_theta → 0 the many-beam interference in real nacre
    // suppresses saturation; blend toward white luster ((1,1,1) × 0.5).
    let t     = cos_theta.powf(0.5);
    let luster = Color::new(0.5, 0.5, 0.5);
    color * t + luster * (1.0 - t)
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
    pub base_color:      Color,
    /// Effective nacre IOR.  Average of aragonite (~1.68) and organic (~1.44)
    /// layers; ~1.56 gives a typical Akoya orient cycle.
    pub ior:             f32,
    /// Nacre platelet thickness in **nanometres** (natural pearls: 380–600 nm).
    /// Controls which colour appears at normal incidence and how fast it shifts.
    pub film_thickness:  f32,
    /// How strongly the orient tints the diffuse body colour [0–1].
    /// 0 = plain Lambertian cream; 0.3 = visible sheen; 1 = fully replaced.
    pub orient_strength: f32,
    /// Spatial frequency of oil-spill film-thickness variation.
    /// Use ~5.0 for unit-scale objects, ~0.1 for objects ~50 units across.
    pub film_scale:      f32,
}

impl Material for PearlMaterial {
    fn scatter(&self, r_in: &Ray, rec: &HitRecord<'_>, rng: &mut dyn RngCore) -> Option<ScatterRecord> {
        let unit      = r_in.direction.unit();
        let cos_theta = (-unit).dot(rec.normal).clamp(0.0, 1.0);

        let varied = self.film_thickness + oil_noise(rec.p * self.film_scale) * 150.0;
        let f      = schlick(cos_theta, 1.0 / self.ior);

        // Nacre is a diffuse structural-colour material: OPD in the aragonite
        // platelet stack is determined by how light *enters* the film (the
        // illumination angle), not by the specular reflection or view angle.
        // Use the sun–normal angle for both paths so the colour shifts as the
        // sun moves; fall back to the view angle when no sun is available.
        let cos_illumin = match pearl_sun_dir() {
            Some(sun) => rec.normal.dot(sun).clamp(0.0, 1.0),
            None      => cos_theta,
        };

        if rng.gen::<f32>() < f {
            let orient = nacre_color(cos_illumin, self.ior, varied);
            Some(ScatterRecord {
                attenuation: orient,
                ray:         Ray::scatter_from(rec.p, unit.reflect(rec.normal), r_in),
                skip_pdf:    true,
            })
        } else {
            let orient  = nacre_color(cos_illumin, self.ior, varied);
            let s       = self.orient_strength;
            let tinted  = self.base_color * (orient * s + Color::new(1.0, 1.0, 1.0) * (1.0 - s));
            Some(ScatterRecord {
                attenuation: tinted,
                ray:         Ray::scatter_from(rec.p, rec.normal, r_in),
                skip_pdf:    false,
            })
        }
    }

    fn scattering_pdf(&self, _r_in: &Ray, rec: &HitRecord<'_>, scattered: &Ray) -> f32 {
        (rec.normal.dot(scattered.direction.unit()) / PI).max(0.0)
    }

    fn albedo_hint(&self, _u: f32, _v: f32, _p: Point3) -> Color { self.base_color }
    fn can_receive_caustics(&self) -> bool { true }
}
