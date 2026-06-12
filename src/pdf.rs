use std::f32::consts::PI;
use rand::{Rng, RngCore};
use crate::onb::Onb;
use crate::vec3::{Point3, Vec3};
use crate::hittable::Hittable;

pub trait Pdf {
    fn value(&self, direction: Vec3) -> f32;
    fn generate(&self, rng: &mut dyn RngCore) -> Vec3;
}

fn random_cosine_direction(rng: &mut dyn RngCore) -> Vec3 {
    let r1: f32 = rng.gen();
    let r2: f32 = rng.gen();
    let phi = 2.0 * PI * r1;
    let sqrt_r2 = r2.sqrt();
    Vec3::new(phi.cos() * sqrt_r2, phi.sin() * sqrt_r2, (1.0 - r2).sqrt())
}

// ── Cosine-weighted hemisphere PDF ───────────────────────────────────────────

pub struct CosinePdf {
    uvw: Onb,
}

impl CosinePdf {
    pub fn new(w: Vec3) -> Self {
        Self { uvw: Onb::from_w(w) }
    }
}

impl Pdf for CosinePdf {
    fn value(&self, direction: Vec3) -> f32 {
        let cosine = direction.unit().dot(self.uvw.w);
        (cosine / PI).max(0.0)
    }

    fn generate(&self, rng: &mut dyn RngCore) -> Vec3 {
        self.uvw.local(random_cosine_direction(rng))
    }
}

// ── Hittable (area light) PDF ─────────────────────────────────────────────────

pub struct HittablePdf<'a> {
    objects: &'a dyn Hittable,
    origin:  Point3,
}

impl<'a> HittablePdf<'a> {
    pub fn new(objects: &'a dyn Hittable, origin: Point3) -> Self {
        Self { objects, origin }
    }
}

impl<'a> Pdf for HittablePdf<'a> {
    fn value(&self, direction: Vec3) -> f32 {
        self.objects.pdf_value(self.origin, direction)
    }

    fn generate(&self, rng: &mut dyn RngCore) -> Vec3 {
        self.objects.pdf_generate(self.origin, rng)
    }
}

// ── Mixture PDF ───────────────────────────────────────────────────────────────

pub struct MixturePdf<'a> {
    p0:     &'a dyn Pdf,
    p1:     &'a dyn Pdf,
    weight: f32,  // fraction of samples drawn from p1
}

impl<'a> MixturePdf<'a> {
    pub fn new(p0: &'a dyn Pdf, p1: &'a dyn Pdf, weight: f32) -> Self {
        Self { p0, p1, weight }
    }
}

impl<'a> Pdf for MixturePdf<'a> {
    fn value(&self, direction: Vec3) -> f32 {
        (1.0 - self.weight) * self.p0.value(direction) + self.weight * self.p1.value(direction)
    }

    fn generate(&self, rng: &mut dyn RngCore) -> Vec3 {
        if rng.gen::<f32>() < self.weight {
            self.p1.generate(rng)
        } else {
            self.p0.generate(rng)
        }
    }
}
