use std::f32::consts::PI;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use rand::{Rng, RngCore};
use crate::vec3::{Color, Point3, Vec3};
use crate::ray::Ray;
use crate::hittable::{HitRecord, Material, ScatterRecord};
use crate::texture::Texture;
use crate::spectrum::{cauchy_ior, spectral_to_rgb};
use crate::volume::hg_sample;

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

/// Smith G1 term for isotropic GGX.
fn smith_g1(cos_theta: f32, alpha: f32) -> f32 {
    let a2 = alpha * alpha;
    let c2 = cos_theta * cos_theta;
    2.0 * cos_theta / (cos_theta + (a2 + (1.0 - a2) * c2).sqrt())
}

/// Smith G1 for anisotropic GGX.
/// `tx` / `ty` are dot(v, tangent) / dot(v, bitangent); `cos_theta` = dot(v, normal).
fn smith_g1_aniso(cos_theta: f32, tx: f32, ty: f32, ax: f32, ay: f32) -> f32 {
    let denom = cos_theta + (cos_theta*cos_theta + ax*ax*tx*tx + ay*ay*ty*ty).sqrt();
    (2.0 * cos_theta / denom.max(1e-6)).clamp(0.0, 1.0)
}

/// Sample a microfacet normal from the anisotropic GGX VNDF (Heitz 2018).
/// `wo_ts` is the outgoing direction in tangent space (z = dot(wo, n) > 0).
/// Returns the microfacet normal in tangent space.
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
        }
    }
}

impl Material for PbrMaterial {
    fn scatter(&self, r_in: &Ray, rec: &HitRecord<'_>, rng: &mut dyn RngCore) -> Option<ScatterRecord> {
        let n     = rec.normal;
        let wo    = (-r_in.direction).unit();
        let cos_o = wo.dot(n);
        if cos_o <= 0.0 { return None; }

        let alpha = (self.roughness * self.roughness).max(1e-4_f32);
        let f0    = Color::new(0.04, 0.04, 0.04) * (1.0 - self.metallic)
                  + self.albedo * self.metallic;

        // Clearcoat lobe: dielectric IOR 1.5 (F0 = 0.04), scaled by clearcoat weight.
        let p_coat = if self.clearcoat > 1e-3 {
            schlick(cos_o, 1.0 / 1.5_f32) * self.clearcoat
        } else {
            0.0
        };

        // Base specular probability from Fresnel at the view angle.
        let f_approx = schlick_color(cos_o, f0);
        let p_spec = if self.metallic > 0.999 {
            1.0_f32
        } else {
            (0.2126 * f_approx.x + 0.7152 * f_approx.y + 0.0722 * f_approx.z)
                .clamp(0.04, 0.9)
        };

        if rng.gen::<f32>() < p_coat {
            // ── Clearcoat specular (isotropic VNDF) ──────────────────────────
            let alpha_coat = (self.clearcoat_roughness * self.clearcoat_roughness).max(1e-4);
            let (t, b) = make_onb(n);
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
                let aspect = (1.0 - 0.9 * self.anisotropy.clamp(0.0, 1.0)).max(0.001_f32).sqrt();
                let ax = alpha * aspect;
                let ay = alpha / aspect;

                let (t0, b0) = make_onb(n);
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
                // ── Diffuse Lambertian (dielectrics only) ─────────────────────
                Some(ScatterRecord {
                    attenuation: self.albedo * ((1.0 - self.metallic) / ((1.0 - p_spec) * (1.0 - p_coat))),
                    ray:         Ray::scatter_from(rec.p, rec.normal, r_in),
                    skip_pdf:    false,
                })
            }
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
        let f      = schlick(cos_theta, 1.0 / self.ior);

        // Illumination angle: governs the nacre glow on the diffuse path.
        // OPD is set when light enters the aragonite platelet stack, so this
        // component shifts with the sun position, not the camera angle.
        let cos_illumin = match pearl_sun_dir() {
            Some(sun) => n.dot(sun).clamp(0.0, 1.0),
            None      => cos_theta,
        };

        if rng.gen::<f32>() < f {
            // ── GGX luster: view-dependent thin-film iridescence (VNDF) ──────
            let (t, b)    = make_onb(n);
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
                attenuation: tinted,
                ray:         Ray::scatter_from(rec.p, n, r_in),
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
