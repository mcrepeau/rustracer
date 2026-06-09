use rand::{Rng, RngCore};
use crate::vec3::{Color, Vec3};
use crate::ray::Ray;
use crate::hittable::{HitRecord, Material};
use crate::texture::Texture;

pub struct DiffuseLight {
    pub emit: Color,
}

impl Material for DiffuseLight {
    fn scatter(&self, _r_in: &Ray, _rec: &HitRecord<'_>, _rng: &mut dyn RngCore) -> Option<(Color, Ray)> { None }
    fn emitted(&self) -> Color { self.emit }
}

pub struct Lambertian {
    pub texture: Texture,
}

impl Material for Lambertian {
    fn scatter(&self, _r_in: &Ray, rec: &HitRecord<'_>, rng: &mut dyn RngCore) -> Option<(Color, Ray)> {
        let mut dir = rec.normal + Vec3::random_unit_vector(rng);
        if dir.near_zero() { dir = rec.normal; }
        let albedo = self.texture.value(rec.u, rec.v, rec.p);
        Some((albedo, Ray::new(rec.p, dir)))
    }
}

pub struct Metal {
    pub albedo: Color,
    pub fuzz: f32,
}

impl Material for Metal {
    fn scatter(&self, r_in: &Ray, rec: &HitRecord<'_>, rng: &mut dyn RngCore) -> Option<(Color, Ray)> {
        let reflected = r_in.direction.unit().reflect(rec.normal);
        let scattered = Ray::new(rec.p, reflected + self.fuzz * Vec3::random_unit_vector(rng));
        if scattered.direction.dot(rec.normal) > 0.0 {
            Some((self.albedo, scattered))
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
    fn scatter(&self, r_in: &Ray, rec: &HitRecord<'_>, rng: &mut dyn RngCore) -> Option<(Color, Ray)> {
        let ratio = if rec.front_face { 1.0 / self.ir } else { self.ir };
        let unit = r_in.direction.unit();
        let cos_theta = (-unit).dot(rec.normal).min(1.0);
        let sin_theta = (1.0 - cos_theta * cos_theta).sqrt();
        let direction = if ratio * sin_theta > 1.0 || Self::reflectance(cos_theta, ratio) > rng.gen::<f32>() {
            unit.reflect(rec.normal)
        } else {
            unit.refract(rec.normal, ratio)
        };
        Some((Color::new(1.0, 1.0, 1.0), Ray::new(rec.p, direction)))
    }
}
