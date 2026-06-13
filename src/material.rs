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
}

fn schlick(cosine: f32, ref_idx: f32) -> f32 {
    let r0 = ((1.0 - ref_idx) / (1.0 + ref_idx)).powi(2);
    r0 + (1.0 - r0) * (1.0 - cosine).powi(5)
}

pub struct Dielectric {
    pub ir: f32,
}

impl Material for Dielectric {
    fn scatter(&self, r_in: &Ray, rec: &HitRecord<'_>, rng: &mut dyn RngCore) -> Option<ScatterRecord> {
        let ratio = if rec.front_face { 1.0 / self.ir } else { self.ir };
        let unit = r_in.direction.unit();
        let cos_theta = (-unit).dot(rec.normal).min(1.0);
        let sin_theta = (1.0 - cos_theta * cos_theta).sqrt();
        let direction = if ratio * sin_theta > 1.0 || schlick(cos_theta, ratio) > rng.gen::<f32>() {
            unit.reflect(rec.normal)
        } else {
            unit.refract(rec.normal, ratio)
        };
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

        let ratio     = if rec.front_face { 1.0 / ir } else { ir };
        let unit      = r_in.direction.unit();
        let cos_theta = (-unit).dot(rec.normal).min(1.0);
        let sin_theta = (1.0 - cos_theta * cos_theta).sqrt();

        let reflects = ratio * sin_theta > 1.0 || schlick(cos_theta, ratio) > rng.gen::<f32>();

        if reflects {
            // TIR or Fresnel reflection: direction and attenuation are wavelength-independent.
            Some(ScatterRecord {
                attenuation: Color::new(1.0, 1.0, 1.0),
                ray:         Ray::new_at_time(rec.p, unit.reflect(rec.normal), r_in.time),
                skip_pdf:    true,
            })
        } else {
            // Refraction: direction depends on the hero wavelength; isolate that channel.
            let attenuation = match channel {
                0 => Color::new(3.0, 0.0, 0.0),
                1 => Color::new(0.0, 3.0, 0.0),
                _ => Color::new(0.0, 0.0, 3.0),
            };
            Some(ScatterRecord {
                attenuation,
                ray:      Ray::new_at_time(rec.p, unit.refract(rec.normal, ratio), r_in.time),
                skip_pdf: true,
            })
        }
    }
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
        HitRecord { p: rec.p, normal, mat: rec.mat, t: rec.t, u: rec.u, v: rec.v, front_face: rec.front_face }
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
}
