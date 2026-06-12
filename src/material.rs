use std::f32::consts::PI;
use rand::{Rng, RngCore};
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

pub struct Dielectric {
    pub ir: f32,
}

impl Dielectric {
    fn reflectance(cosine: f32, ref_idx: f32) -> f32 {
        let r0 = ((1.0 - ref_idx) / (1.0 + ref_idx)).powi(2);
        r0 + (1.0 - r0) * (1.0 - cosine).powi(5)
    }
}

impl Material for Dielectric {
    fn scatter(&self, r_in: &Ray, rec: &HitRecord<'_>, rng: &mut dyn RngCore) -> Option<ScatterRecord> {
        let ratio = if rec.front_face { 1.0 / self.ir } else { self.ir };
        let unit = r_in.direction.unit();
        let cos_theta = (-unit).dot(rec.normal).min(1.0);
        let sin_theta = (1.0 - cos_theta * cos_theta).sqrt();
        let direction = if ratio * sin_theta > 1.0 || Self::reflectance(cos_theta, ratio) > rng.gen::<f32>() {
            unit.reflect(rec.normal)
        } else {
            unit.refract(rec.normal, ratio)
        };
        Some(ScatterRecord { attenuation: Color::new(1.0, 1.0, 1.0), ray: Ray::new_at_time(rec.p, direction, r_in.time), skip_pdf: true })
    }
}
