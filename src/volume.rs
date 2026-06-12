use std::sync::Arc;
use rand::Rng;
use crate::aabb::Aabb;
use crate::hittable::{HitRecord, Hittable, Material, ScatterRecord};
use crate::ray::Ray;
use crate::texture::Texture;
use crate::vec3::{Color, Vec3};

// ── Isotropic phase function ──────────────────────────────────────────────────

pub struct Isotropic {
    pub albedo: Texture,
}

impl Material for Isotropic {
    fn scatter(&self, r_in: &Ray, rec: &HitRecord<'_>, rng: &mut dyn rand::RngCore) -> Option<ScatterRecord> {
        let dir    = Vec3::random_unit_vector(rng);
        let albedo = self.albedo.value(rec.u, rec.v, rec.p);
        // albedo: None → skip NEE; isotropic scattering uses indirect illumination only
        Some(ScatterRecord { attenuation: albedo, ray: Ray::new_at_time(rec.p, dir, r_in.time), albedo: None })
    }
}

// ── Constant-density participating medium ─────────────────────────────────────

pub struct ConstantMedium {
    boundary:        Arc<dyn Hittable>,
    neg_inv_density: f32,
    phase_fn:        Arc<dyn Material>,
}

impl ConstantMedium {
    pub fn new(boundary: Arc<dyn Hittable>, density: f32, color: Color) -> Self {
        Self {
            boundary,
            neg_inv_density: -1.0 / density,
            phase_fn: Arc::new(Isotropic { albedo: Texture::Solid(color) }),
        }
    }
}

impl Hittable for ConstantMedium {
    fn hit(&self, r: &Ray, t_min: f32, t_max: f32) -> Option<HitRecord<'_>> {
        // Find entry and exit through the boundary shape
        let mut rec1 = self.boundary.hit(r, f32::NEG_INFINITY, f32::INFINITY)?;
        let mut rec2 = self.boundary.hit(r, rec1.t + 0.0001, f32::INFINITY)?;

        if rec1.t < t_min { rec1.t = t_min; }
        if rec2.t > t_max { rec2.t = t_max; }
        if rec1.t >= rec2.t { return None; }
        if rec1.t < 0.0 { rec1.t = 0.0; }

        let ray_length  = r.direction.length();
        let dist_inside = (rec2.t - rec1.t) * ray_length;

        // Exponential free-path sampling
        let random      = rand::thread_rng().gen::<f32>().max(f32::EPSILON);
        let hit_dist    = self.neg_inv_density * random.ln();
        if hit_dist > dist_inside { return None; }

        let t = rec1.t + hit_dist / ray_length;
        Some(HitRecord {
            p:          r.at(t),
            normal:     Vec3::new(1.0, 0.0, 0.0), // arbitrary for isotropic scattering
            mat:        &*self.phase_fn,
            t,
            u:          0.0,
            v:          0.0,
            front_face: true,
        })
    }

    fn bounding_box(&self) -> Option<Aabb> {
        self.boundary.bounding_box()
    }
}
