use std::f32::consts::PI;
use std::sync::Arc;
use rand::{Rng, RngCore};
use crate::perlin::Perlin;
use crate::vec3::{Color, Point3, Vec3};
use crate::ray::Ray;
use crate::hittable::{HitRecord, Material, ScatterRecord};
use crate::texture::Texture;

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
        Some(ScatterRecord { attenuation: albedo, ray: Ray::new_at_time(rec.p, rec.normal, r_in.time), skip_pdf: false })
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
        let ray = Ray::new_at_time(rec.p, reflected + self.fuzz * Vec3::random_unit_vector(rng), r_in.time);
        if ray.direction.dot(rec.normal) > 0.0 {
            Some(ScatterRecord { attenuation: self.albedo, ray, skip_pdf: true })
        } else {
            None
        }
    }
    fn albedo_hint(&self, _u: f32, _v: f32, _p: Point3) -> Color { self.albedo }
}

fn schlick(cosine: f32, ref_idx: f32) -> f32 {
    let r0 = ((1.0 - ref_idx) / (1.0 + ref_idx)).powi(2);
    r0 + (1.0 - r0) * (1.0 - cosine).powi(5)
}

/// Dielectric boundary scatter shared by all glass-like materials.
/// Returns `(exit_direction, reflected)` where `reflected` is true for TIR
/// or Fresnel reflection and false for refraction.
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
        Some(ScatterRecord { attenuation: Color::new(1.0, 1.0, 1.0), ray: Ray::new_at_time(rec.p, direction, r_in.time), skip_pdf: true })
    }
}

/// Dispersive dielectric that refracts each wavelength (R/G/B) at its own IOR.
///
/// At each scatter event a hero wavelength is chosen uniformly at random.
/// Refractions use the wavelength-specific IOR and carry 3× weight on a single
/// channel so the Monte Carlo estimator remains unbiased.  Reflections (TIR or
/// Fresnel) are wavelength-independent and carry full (1,1,1) weight, keeping
/// those paths uncoloured.
///
/// For a round brilliant diamond use  ir_red = 2.407, ir_green = 2.417,
/// ir_blue = 2.426  (Cauchy model fit to measured dispersion data).
pub struct SpectralDielectric {
    pub ir_red:   f32,
    pub ir_green: f32,
    pub ir_blue:  f32,
}

impl Material for SpectralDielectric {
    fn scatter(&self, r_in: &Ray, rec: &HitRecord<'_>, rng: &mut dyn RngCore) -> Option<ScatterRecord> {
        let (ir, channel): (f32, u8) = match rng.gen_range(0u8..3) {
            0 => (self.ir_red,   0),
            1 => (self.ir_green, 1),
            _ => (self.ir_blue,  2),
        };
        let (direction, reflected) = dielectric_boundary(ir, r_in, rec, rng);
        // Reflection is wavelength-independent; refraction isolates the hero channel.
        let attenuation = if reflected {
            Color::new(1.0, 1.0, 1.0)
        } else {
            match channel {
                0 => Color::new(3.0, 0.0, 0.0),
                1 => Color::new(0.0, 3.0, 0.0),
                _ => Color::new(0.0, 0.0, 3.0),
            }
        };
        Some(ScatterRecord { attenuation, ray: Ray::new_at_time(rec.p, direction, r_in.time), skip_pdf: true })
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
        Some(ScatterRecord { attenuation, ray: Ray::new_at_time(rec.p, direction, r_in.time), skip_pdf: true })
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
                    ray:         Ray::new_at_time(p_scat, new_dir, r_in.time),
                    skip_pdf:    true,
                });
            }
        }

        // Boundary event (entry or unscattered exit): standard Fresnel.
        let (direction, _) = dielectric_boundary(self.ior, r_in, rec, rng);
        Some(ScatterRecord {
            attenuation: Color::new(1.0, 1.0, 1.0),
            ray:         Ray::new_at_time(rec.p, direction, r_in.time),
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
                ray: Ray::new_at_time(rec.p, wi, r_in.time),
                skip_pdf: true,
            })
        } else {
            // ── Diffuse Lambertian (dielectrics only) ─────────────────────────
            // Divide by (1-p_spec) to correct for Russian-roulette sampling.
            Some(ScatterRecord {
                attenuation: self.albedo * ((1.0 - self.metallic) / (1.0 - p_spec)),
                ray: Ray::new_at_time(rec.p, rec.normal, r_in.time), // direction overridden by PDF
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

/// Wraps any material and perturbs the surface normal with Perlin turbulence
/// before delegating scatter/PDF, creating a rough/lumpy appearance without
/// changing the underlying geometry.
pub struct BumpMaterial {
    pub inner:    Arc<dyn Material>,
    pub perlin:   Arc<Perlin>,
    /// Spatial frequency of bumps — higher values produce finer detail.
    pub scale:    f32,
    /// Controls how strongly the normal is deflected (0 = smooth, 1 = very rough).
    pub strength: f32,
}

impl BumpMaterial {
    fn perturb<'a>(&self, rec: &HitRecord<'a>) -> HitRecord<'a> {
        let s = self.scale;
        let p = rec.p;
        // Three turbulence samples at permuted coordinates give an independent
        // pseudo-gradient vector for each spatial dimension.
        let t1 = self.perlin.turb(Point3::new(p.x * s, p.y * s, p.z * s), 4) - 0.5;
        let t2 = self.perlin.turb(Point3::new(p.z * s, p.x * s, p.y * s), 4) - 0.5;
        let t3 = self.perlin.turb(Point3::new(p.y * s, p.z * s, p.x * s), 4) - 0.5;
        let normal = (rec.normal + Vec3::new(t1, t2, t3) * self.strength).unit();
        HitRecord { p, normal, mat: rec.mat, t: rec.t, u: rec.u, v: rec.v, front_face: rec.front_face }
    }
}

impl Material for BumpMaterial {
    fn scatter(&self, r_in: &Ray, rec: &HitRecord<'_>, rng: &mut dyn RngCore) -> Option<ScatterRecord> {
        let bumped = self.perturb(rec);
        self.inner.scatter(r_in, &bumped, rng)
    }

    fn scattering_pdf(&self, r_in: &Ray, rec: &HitRecord<'_>, scattered: &Ray) -> f32 {
        let bumped = self.perturb(rec);
        self.inner.scattering_pdf(r_in, &bumped, scattered)
    }

    fn emitted(&self, u: f32, v: f32, p: Point3) -> Color {
        self.inner.emitted(u, v, p)
    }
    fn albedo_hint(&self, u: f32, v: f32, p: Point3) -> Color {
        self.inner.albedo_hint(u, v, p)
    }
    fn can_receive_caustics(&self) -> bool { self.inner.can_receive_caustics() }
}
