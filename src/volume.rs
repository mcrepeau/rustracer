use std::f32::consts::PI;
use std::sync::Arc;
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::SmallRng;
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
        // At |g| = 1 with xi1 at the extreme, the denominator hits 0 → 0/0 NaN.
        // Clamp it: the limit is s = 0, i.e. cos_theta = ±1 (pure forward/backward).
        let s = (1.0 - g * g) / (1.0 - g + 2.0 * g * xi1).max(1e-12);
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

/// Length of the ray segment inside `boundary`, clamped to [t_min, t_max]
/// (in world units). Returns None if the ray misses the boundary or the
/// clamped segment is empty.
fn inside_length(boundary: &dyn Hittable, r: &Ray, t_min: f32, t_max: f32) -> Option<f32> {
    let rec1 = boundary.hit(r, f32::NEG_INFINITY, f32::INFINITY)?;
    let rec2 = boundary.hit(r, rec1.t + 0.0001, f32::INFINITY)?;
    let t1 = rec1.t.max(0.0).max(t_min);
    let t2 = rec2.t.min(t_max);
    if t1 >= t2 { return None; }
    Some((t2 - t1) * r.direction.length())
}

pub struct ConstantMedium {
    boundary:        Arc<dyn Hittable>,
    density:         f32,
    neg_inv_density: f32,
    phase_fn:        Arc<dyn Material>,
}

impl ConstantMedium {
    /// `g` is the HG asymmetry parameter (0 = isotropic, 0.85 = cloud droplets).
    pub fn new(boundary: Arc<dyn Hittable>, density: f32, color: Color, g: f32) -> Self {
        Self {
            boundary,
            density,
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

        // Seed from ray state so each path sample gets a distinct sequence
        // without requiring rng to be threaded through Hittable::hit().
        // Origin z and direction x are included to avoid correlated decisions
        // between rays that share only the xy origin or differ only in direction.
        let mut vol_rng = SmallRng::seed_from_u64(
            (r.origin.x.to_bits()    as u64).wrapping_mul(6364136223846793005)
                ^ (r.origin.y.to_bits()  as u64).wrapping_mul(0x9E3779B97F4A7C15)
                ^ (r.origin.z.to_bits()  as u64).wrapping_mul(0xBF58476D1CE4E5B9)
                ^ (r.direction.x.to_bits() as u64).wrapping_mul(0x94D049BB133111EB)
                ^ r.wavelength.to_bits() as u64,
        );
        let random   = vol_rng.gen::<f32>().max(f32::EPSILON);
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
            tangent:    Vec3::default(),
            shading:    std::sync::OnceLock::new(),
        })
    }

    fn any_hit(&self, _r: &Ray, _t_min: f32, _t_max: f32) -> bool { false }

    fn transmittance(&self, r: &Ray, t_min: f32, t_max: f32) -> f32 {
        // Homogeneous medium: exact Beer-Lambert over the inside segment.
        match inside_length(&*self.boundary, r, t_min, t_max) {
            Some(l) => (-self.density * l).exp(),
            None    => 1.0,
        }
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
        let mut rng    = SmallRng::seed_from_u64(
            (r.origin.x.to_bits()    as u64).wrapping_mul(6364136223846793005)
                ^ (r.origin.y.to_bits()  as u64).wrapping_mul(0x9E3779B97F4A7C15)
                ^ (r.origin.z.to_bits()  as u64).wrapping_mul(0xBF58476D1CE4E5B9)
                ^ (r.direction.x.to_bits() as u64).wrapping_mul(0x94D049BB133111EB)
                ^ r.wavelength.to_bits() as u64,
        );
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
                    tangent:    Vec3::default(),
                    shading:    std::sync::OnceLock::new(),
                });
            }
        }
    }

    fn any_hit(&self, _r: &Ray, _t_min: f32, _t_max: f32) -> bool { false }

    fn transmittance(&self, r: &Ray, t_min: f32, t_max: f32) -> f32 {
        // Heterogeneous medium: deterministic midpoint quadrature of the optical
        // depth along the inside segment (no RNG — shadow rays stay reproducible).
        let Some(l) = inside_length(&*self.boundary, r, t_min, t_max) else { return 1.0 };
        let rec1 = self.boundary.hit(r, f32::NEG_INFINITY, f32::INFINITY).unwrap();
        let t1 = rec1.t.max(0.0).max(t_min);
        const K: usize = 16;
        let seg_t = l / r.direction.length();
        let mut sum = 0.0f32;
        for i in 0..K {
            let t = t1 + seg_t * (i as f32 + 0.5) / K as f32;
            sum += self.density_at(r.at(t));
        }
        (-sum / K as f32 * l).exp()
    }

    fn bounding_box(&self) -> Option<Aabb> {
        self.boundary.bounding_box()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    struct DummyMat;
    impl Material for DummyMat {
        fn scatter(&self, _: &Ray, _: &HitRecord<'_>, _: &mut dyn rand::RngCore) -> Option<ScatterRecord> { None }
    }

    fn test_sphere(center: Point3, radius: f32) -> Arc<dyn Hittable> {
        Arc::new(crate::sphere::Sphere::new(center, radius, Arc::new(DummyMat)))
    }

    #[test]
    fn constant_medium_transmittance_beer_lambert() {
        // Unit sphere at the origin, density 1.0.
        let med = ConstantMedium::new(test_sphere(Point3::new(0.0, 0.0, 0.0), 1.0), 1.0, Color::new(1.0, 1.0, 1.0), 0.0);

        // Ray through the center: inside chord length 2 → T = exp(-2).
        let r = Ray::new(Point3::new(-5.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0));
        let tr = med.transmittance(&r, 0.001, f32::INFINITY);
        assert!((tr - (-2.0f32).exp()).abs() < 1e-6, "tr = {tr}");

        // Ray that misses the boundary entirely → no attenuation.
        let miss = Ray::new(Point3::new(-5.0, 5.0, 0.0), Vec3::new(1.0, 0.0, 0.0));
        assert_eq!(med.transmittance(&miss, 0.001, f32::INFINITY), 1.0);

        // Origin inside the medium, ray exits: entry clamps to t_min, so the
        // inside segment is [0.001, 1] → length 0.999 → T = exp(-0.999).
        let inside = Ray::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0));
        let tr = med.transmittance(&inside, 0.001, f32::INFINITY);
        assert!((tr - (-0.999f32).exp()).abs() < 1e-6, "tr = {tr}");

        // t_max clipped inside the medium: segment [0.001, 0.5] → length ~0.499.
        let tr = med.transmittance(&inside, 0.001, 0.5);
        assert!((tr - (-(0.5f32 - 0.001)).exp()).abs() < 1e-5, "tr = {tr}");

        // Volumes must not occlude via any_hit (that would zero the shadow ray
        // before transmittance ever gets a say).
        assert!(!med.any_hit(&r, 0.001, f32::INFINITY));
    }

    #[test]
    fn transmittance_composes_across_list() {
        use crate::hittable::HittableList;
        let med = ConstantMedium::new(test_sphere(Point3::new(0.0, 0.0, 0.0), 1.0), 1.0, Color::new(1.0, 1.0, 1.0), 0.0);
        let mut list = HittableList::new();
        list.add(med);
        // Opaque sphere blocking the ray, plus the medium.
        list.add(crate::sphere::Sphere::new(Point3::new(-2.0, 0.0, 0.0), 0.5, Arc::new(DummyMat)));
        let r = Ray::new(Point3::new(-5.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0));
        assert_eq!(list.transmittance(&r, 0.001, f32::INFINITY), 0.0);
        // Ray missing the blocker but crossing the medium at y=0.9:
        // chord = 2*sqrt(1 - 0.81).
        let r2 = Ray::new(Point3::new(-5.0, 0.9, 0.0), Vec3::new(1.0, 0.0, 0.0));
        let l = 2.0 * (1.0f32 - 0.9 * 0.9).sqrt();
        let tr = list.transmittance(&r2, 0.001, f32::INFINITY);
        assert!((tr - (-l).exp()).abs() < 1e-5, "tr = {tr}, l = {l}");
    }

    #[test]
    fn hg_sample_at_extreme_g_is_always_finite() {
        // At |g| = 1 the CDF denominator can hit 0 → 0/0 NaN without the clamp.
        let mut rng = SmallRng::seed_from_u64(1);
        let w = Vec3::new(0.0, 0.0, 1.0);
        for &g in &[1.0f32, -1.0, 0.999999, -0.999999] {
            for _ in 0..20_000 {
                let d = hg_sample(w, g, &mut rng);
                assert!(d.x.is_finite() && d.y.is_finite() && d.z.is_finite(),
                    "hg_sample(g={g}) produced non-finite direction {d:?}");
            }
        }
    }
}
