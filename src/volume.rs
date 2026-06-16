use std::f32::consts::PI;
use std::sync::Arc;
use rand::Rng;
use crate::aabb::Aabb;
use crate::hittable::{HitRecord, Hittable, Material, ScatterRecord};
use crate::perlin::Perlin;
use crate::ray::Ray;
use crate::texture::Texture;
use crate::vec3::{Color, Point3, Vec3};

// ── Henyey-Greenstein phase function ─────────────────────────────────────────
//
// p(cosθ) = (1 − g²) / (4π (1 + g² − 2g cosθ)^(3/2))
//
// g = 0  : isotropic (flat probability over all directions)
// g > 0  : forward-scattering  (g ≈ 0.85 for water cloud droplets)
// g < 0  : back-scattering
//
// The direction is importance-sampled via the analytical inversion of the CDF,
// so the Monte Carlo weight is just the single-scattering albedo.

pub struct HenyeyGreenstein {
    pub albedo: Texture,
    pub g:      f32,
}

impl Material for HenyeyGreenstein {
    fn scatter(&self, r_in: &Ray, rec: &HitRecord<'_>, rng: &mut dyn rand::RngCore) -> Option<ScatterRecord> {
        let dir    = hg_sample(r_in.direction.unit(), self.g, rng);
        let albedo = self.albedo.value(rec.u, rec.v, rec.p);
        Some(ScatterRecord { attenuation: albedo, ray: Ray::scatter_from(rec.p, dir, r_in), skip_pdf: true })
    }
}

/// Sample a direction from the HG distribution around the incident direction.
/// Uses the exact analytical inversion of the CDF — no rejection needed.
pub(crate) fn hg_sample(v_in: Vec3, g: f32, rng: &mut dyn rand::RngCore) -> Vec3 {
    let xi1 = rng.gen::<f32>();
    let xi2 = rng.gen::<f32>();

    // Sampled cos(θ) relative to the forward direction
    let cos_theta = if g.abs() < 1e-3 {
        1.0 - 2.0 * xi1          // isotropic fallback: uniform in [-1, 1]
    } else {
        let s = (1.0 - g * g) / (1.0 - g + 2.0 * g * xi1);
        (1.0 + g * g - s * s) / (2.0 * g)
    };
    let sin_theta = (1.0 - cos_theta * cos_theta).max(0.0).sqrt();
    let phi       = 2.0 * PI * xi2;

    // Build an ONB with w = incident direction (forward is the +z pole)
    let w      = v_in;     // already unit
    let helper = if w.x.abs() > 0.9 { Vec3::new(0.0, 1.0, 0.0) } else { Vec3::new(1.0, 0.0, 0.0) };
    let u      = w.cross(helper).unit();
    let v      = w.cross(u);

    u * (sin_theta * phi.cos()) + v * (sin_theta * phi.sin()) + w * cos_theta
}

// ── Constant-density participating medium ─────────────────────────────────────

pub struct ConstantMedium {
    boundary:        Arc<dyn Hittable>,
    neg_inv_density: f32,
    phase_fn:        Arc<dyn Material>,
}

impl ConstantMedium {
    /// `g` is the HG asymmetry parameter (0 = isotropic, 0.85 = cloud droplets).
    pub fn new(boundary: Arc<dyn Hittable>, density: f32, color: Color, g: f32) -> Self {
        Self {
            boundary,
            neg_inv_density: -1.0 / density,
            phase_fn: Arc::new(HenyeyGreenstein { albedo: Texture::Solid(color), g }),
        }
    }
}

impl Hittable for ConstantMedium {
    fn hit(&self, r: &Ray, t_min: f32, t_max: f32) -> Option<HitRecord<'_>> {
        let mut rec1 = self.boundary.hit(r, f32::NEG_INFINITY, f32::INFINITY)?;
        let mut rec2 = self.boundary.hit(r, rec1.t + 0.0001, f32::INFINITY)?;

        if rec1.t < t_min { rec1.t = t_min; }
        if rec2.t > t_max { rec2.t = t_max; }
        if rec1.t >= rec2.t { return None; }
        if rec1.t < 0.0 { rec1.t = 0.0; }

        let ray_length  = r.direction.length();
        let dist_inside = (rec2.t - rec1.t) * ray_length;

        let random   = rand::thread_rng().gen::<f32>().max(f32::EPSILON);
        let hit_dist = self.neg_inv_density * random.ln();
        if hit_dist > dist_inside { return None; }

        let t = rec1.t + hit_dist / ray_length;
        Some(HitRecord {
            p:          r.at(t),
            normal:     Vec3::new(1.0, 0.0, 0.0),
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

// ── Noise-driven heterogeneous participating medium ────────────────────────────

pub struct NoiseMedium {
    boundary:      Arc<dyn Hittable>,
    phase_fn:      Arc<dyn Material>,
    noise:         Perlin,
    noise_scale:   f32,
    density_scale: f32,
    threshold:     f32,
}

impl NoiseMedium {
    /// `g` is the HG asymmetry parameter (0 = isotropic, 0.85 = cloud droplets).
    pub fn new(
        boundary:    Arc<dyn Hittable>,
        color:       Color,
        density:     f32,
        noise_scale: f32,
        threshold:   f32,
        g:           f32,
    ) -> Self {
        let mut rng = rand::thread_rng();
        Self {
            boundary,
            phase_fn: Arc::new(HenyeyGreenstein { albedo: Texture::Solid(color), g }),
            noise: Perlin::new(&mut rng),
            noise_scale,
            density_scale: density.max(f32::EPSILON),
            threshold: threshold.clamp(0.0, 0.99),
        }
    }

    fn density_at(&self, p: Point3) -> f32 {
        let n = self.noise.turb(p * self.noise_scale, 7) * 0.5;
        (n - self.threshold).max(0.0) * self.density_scale
    }
}

impl Hittable for NoiseMedium {
    fn hit(&self, r: &Ray, t_min: f32, t_max: f32) -> Option<HitRecord<'_>> {
        let mut rec1 = self.boundary.hit(r, f32::NEG_INFINITY, f32::INFINITY)?;
        let mut rec2 = self.boundary.hit(r, rec1.t + 0.0001, f32::INFINITY)?;

        if rec1.t < t_min { rec1.t = t_min; }
        if rec2.t > t_max { rec2.t = t_max; }
        if rec1.t >= rec2.t { return None; }
        if rec1.t < 0.0    { rec1.t = 0.0; }

        let ray_length = r.direction.length();
        let inv_maj    = 1.0 / (self.density_scale * ray_length);
        let mut rng    = rand::thread_rng();
        let mut t      = rec1.t;

        loop {
            t -= inv_maj * rng.gen::<f32>().max(f32::EPSILON).ln();

            if t >= rec2.t { return None; }

            let local_density = self.density_at(r.at(t));
            if rng.gen::<f32>() < local_density / self.density_scale {
                return Some(HitRecord {
                    p:          r.at(t),
                    normal:     Vec3::new(1.0, 0.0, 0.0),
                    mat:        &*self.phase_fn,
                    t,
                    u: 0.0, v: 0.0,
                    front_face: true,
                });
            }
        }
    }

    fn bounding_box(&self) -> Option<Aabb> {
        self.boundary.bounding_box()
    }
}
