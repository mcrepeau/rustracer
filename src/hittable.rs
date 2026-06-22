use std::sync::Arc;
use rand::RngCore;
use crate::aabb::Aabb;
use crate::vec3::{Color, Point3, Vec3};
use crate::ray::Ray;

pub struct ScatterRecord {
    pub attenuation: Color,
    pub ray:         Ray,
    /// true  = specular: use `ray` directly, skip PDF weighting.
    /// false = diffuse:  direction will be overridden by the mixture PDF in ray_color.
    pub skip_pdf:    bool,
}

pub trait Material: Send + Sync {
    fn scatter(&self, r_in: &Ray, rec: &HitRecord<'_>, rng: &mut dyn RngCore) -> Option<ScatterRecord>;
    fn emitted(&self, _u: f32, _v: f32, _p: Point3) -> Color { Color::default() }
    /// f(ωi, ωo) · cos(θi) — used to weight the PDF-sampled contribution.
    /// Only called when skip_pdf = false.
    fn scattering_pdf(&self, _r_in: &Ray, _rec: &HitRecord<'_>, _scattered: &Ray) -> f32 { 0.0 }
    /// Unlit base colour of the surface in [0, 1], used as the OIDN albedo
    /// auxiliary buffer.  Specular and transparent materials return white
    /// (the default); the normal buffer still benefits those pixels.
    #[cfg_attr(not(feature = "denoise"), allow(dead_code))]
    fn albedo_hint(&self, _u: f32, _v: f32, _p: Point3) -> Color { Color::new(1.0, 1.0, 1.0) }
    /// True for diffuse materials that should accumulate caustic photons.
    /// Specular, transmissive, and emissive materials return false (default).
    fn can_receive_caustics(&self) -> bool { false }
    /// True if this material produces spectrally-boosted (3× single-channel)
    /// attenuation that can spike photon power inside a refractive object.
    fn is_spectral(&self) -> bool { false }
}

pub struct HitRecord<'a> {
    pub p: Point3,
    pub normal: Vec3,
    pub mat: &'a dyn Material,
    pub t: f32,
    pub u: f32,
    pub v: f32,
    pub front_face: bool,
}

impl<'a> HitRecord<'a> {
    pub fn new(p: Point3, t: f32, mat: &'a dyn Material, r: &Ray, outward_normal: Vec3) -> Self {
        let front_face = r.direction.dot(outward_normal) < 0.0;
        let normal = if front_face { outward_normal } else { -outward_normal };
        Self { p, normal, mat, t, u: 0.0, v: 0.0, front_face }
    }
}

pub trait Hittable: Send + Sync {
    fn hit(&self, r: &Ray, t_min: f32, t_max: f32) -> Option<HitRecord<'_>>;
    fn bounding_box(&self) -> Option<Aabb>;

    /// Returns true if any object is hit in [t_min, t_max].
    /// Faster than hit() for shadow/occlusion tests — exits at the first intersection.
    fn any_hit(&self, r: &Ray, t_min: f32, t_max: f32) -> bool {
        self.hit(r, t_min, t_max).is_some()
    }

    /// Solid-angle PDF for sampling this hittable from `origin` in direction `dir` at `time`.
    fn pdf_value(&self, _origin: Point3, _dir: Vec3, _time: f32) -> f32 { 0.0 }

    /// Generate a direction toward this hittable from `origin` at `time`.
    fn pdf_generate(&self, _origin: Point3, _rng: &mut dyn RngCore, _time: f32) -> Vec3 {
        Vec3::new(1.0, 0.0, 0.0)
    }
}

pub struct HittableList {
    pub objects: Vec<Arc<dyn Hittable>>,
}

impl HittableList {
    pub fn new() -> Self { Self { objects: Vec::new() } }

    pub fn add(&mut self, obj: impl Hittable + 'static) {
        self.objects.push(Arc::new(obj));
    }

}

impl Hittable for HittableList {
    fn hit(&self, r: &Ray, t_min: f32, t_max: f32) -> Option<HitRecord<'_>> {
        let mut closest = t_max;
        let mut result = None;
        for obj in &self.objects {
            if let Some(rec) = obj.hit(r, t_min, closest) {
                closest = rec.t;
                result = Some(rec);
            }
        }
        result
    }

    fn any_hit(&self, r: &Ray, t_min: f32, t_max: f32) -> bool {
        self.objects.iter().any(|obj| obj.any_hit(r, t_min, t_max))
    }

    fn bounding_box(&self) -> Option<Aabb> {
        if self.objects.is_empty() { return None; }
        let mut result: Option<Aabb> = None;
        for obj in &self.objects {
            let bbox = obj.bounding_box()?;
            result = Some(match result {
                None => bbox,
                Some(prev) => Aabb::surrounding(&prev, &bbox),
            });
        }
        result
    }

    fn pdf_value(&self, origin: Point3, dir: Vec3, time: f32) -> f32 {
        if self.objects.is_empty() { return 0.0; }
        let weight = 1.0 / self.objects.len() as f32;
        self.objects.iter().map(|o| weight * o.pdf_value(origin, dir, time)).sum()
    }

    fn pdf_generate(&self, origin: Point3, rng: &mut dyn RngCore, time: f32) -> Vec3 {
        if self.objects.is_empty() { return Vec3::new(1.0, 0.0, 0.0); }
        let idx = (rng.next_u32() as usize) % self.objects.len();
        self.objects[idx].pdf_generate(origin, rng, time)
    }
}
