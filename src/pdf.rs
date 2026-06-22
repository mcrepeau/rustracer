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
    time:    f32,
}

impl<'a> HittablePdf<'a> {
    pub fn new(objects: &'a dyn Hittable, origin: Point3, time: f32) -> Self {
        Self { objects, origin, time }
    }
}

impl<'a> Pdf for HittablePdf<'a> {
    fn value(&self, direction: Vec3) -> f32 {
        self.objects.pdf_value(self.origin, direction, self.time)
    }

    fn generate(&self, rng: &mut dyn RngCore) -> Vec3 {
        self.objects.pdf_generate(self.origin, rng, self.time)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    #[test]
    fn cosine_pdf_integrates_to_one() {
        // Numerical integration of p(ω) = cos(θ)/π over the upper hemisphere.
        // ∫₀²π ∫₀^{π/2} cos(θ)/π · sin(θ) dθ dφ = 1.
        let pdf = CosinePdf::new(Vec3::new(0.0, 0.0, 1.0));
        let n_theta = 200usize;
        let n_phi   = 200usize;
        let d_theta = std::f32::consts::FRAC_PI_2 / n_theta as f32;
        let d_phi   = 2.0 * PI / n_phi as f32;
        let mut sum = 0.0f64;
        for i in 0..n_theta {
            let theta = (i as f32 + 0.5) * d_theta;
            for j in 0..n_phi {
                let phi = (j as f32 + 0.5) * d_phi;
                let dir = Vec3::new(
                    theta.sin() * phi.cos(),
                    theta.sin() * phi.sin(),
                    theta.cos(),
                );
                sum += (pdf.value(dir) * theta.sin() * d_theta * d_phi) as f64;
            }
        }
        assert!((sum - 1.0).abs() < 0.002, "CosinePdf integral = {sum:.5}");
    }

    #[test]
    fn cosine_pdf_at_normal_equals_inv_pi() {
        // p(ω) at θ=0 (normal direction) = cos(0)/π = 1/π.
        let normal = Vec3::new(0.0, 0.0, 1.0);
        let pdf    = CosinePdf::new(normal);
        let got    = pdf.value(normal);
        let expected = 1.0 / PI;
        assert!((got - expected).abs() < 1e-6, "got {got:.6}, expected {expected:.6}");
    }

    #[test]
    fn cosine_pdf_below_horizon_is_zero() {
        let pdf  = CosinePdf::new(Vec3::new(0.0, 0.0, 1.0));
        let down = Vec3::new(0.5, 0.0, -0.866);
        assert_eq!(pdf.value(down), 0.0, "PDF below horizon must be zero");
    }

    #[test]
    fn cosine_pdf_generate_stays_above_hemisphere() {
        let mut rng    = SmallRng::seed_from_u64(42);
        let normal     = Vec3::new(0.0, 1.0, 0.0);
        let pdf        = CosinePdf::new(normal);
        for _ in 0..2000 {
            let dir = pdf.generate(&mut rng);
            assert!(
                dir.dot(normal) >= -1e-5,
                "generate() produced direction below hemisphere: ({:.3}, {:.3}, {:.3})",
                dir.x, dir.y, dir.z,
            );
        }
    }
}

