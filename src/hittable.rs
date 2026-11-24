use std::sync::{Arc, OnceLock};
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
    /// Wavelength-aware emission for hero-wavelength spectral rendering.
    /// `spectral_weighted`: true when the ray's throughput already carries the
    /// CMF weight `spectral_to_rgb(λ)` from a prior spectral bounce.  Spectral
    /// emitters must NOT apply CMF again in that case — return scalar power only.
    /// The default delegates to `emitted()` (flat RGB, no CMF) so non-spectral
    /// emitters (DiffuseLight) work correctly for both spectral and plain paths.
    fn emitted_at(&self, u: f32, v: f32, p: Point3, _lambda: f32, _spectral_weighted: bool) -> Color {
        self.emitted(u, v, p)
    }
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

    /// True if this material emits light -- makes the host shape NEE-sampleable.
    fn is_emissive(&self) -> bool { false }

    /// Evaluate the full specular BRDF × cos(θ_i) for outgoing direction `wi`.
    /// Returns a non-zero Color only for materials that implement specular NEE.
    fn specular_brdf_cos(&self, _r_in: &Ray, _rec: &HitRecord<'_>, _wi: Vec3) -> Color {
        Color::default()
    }

    /// Sampling PDF of scatter() for direction `wi` (specular lobes only, weighted
    /// by Russian-roulette probabilities).  Used for MIS with direct sun sampling.
    fn specular_sampling_pdf(&self, _r_in: &Ray, _rec: &HitRecord<'_>, _wi: Vec3) -> f32 {
        0.0
    }

    /// The shading normal after applying any normal map, in world space.
    /// Defaults to the geometric/interpolated normal; PbrMaterial overrides this
    /// to apply the TBN transform so that PDF sampling and BRDF evaluation are
    /// consistent with the normal-mapped surface.
    fn shading_normal(&self, rec: &HitRecord<'_>) -> Vec3 {
        rec.normal
    }

}

/// Per-hit shading parameters, lazily cached in the HitRecord.
#[derive(Clone, Copy)]
pub struct ShadingParams {
    pub normal:    Vec3,
    pub albedo:    Color,
    pub roughness: f32,
    pub metallic:  f32,
}

pub struct HitRecord<'a> {
    pub p: Point3,
    pub normal: Vec3,
    pub mat: &'a dyn Material,
    pub t: f32,
    pub u: f32,
    pub v: f32,
    pub front_face: bool,
    /// Surface tangent in world space, used for normal-map TBN construction.
    /// Zero for shapes that don't compute tangents (spheres, planes, etc.).
    pub tangent: Vec3,
    /// Lazily-filled per-hit shading cache (filled by PbrMaterial::params).
    pub shading: OnceLock<ShadingParams>,
}

impl<'a> HitRecord<'a> {
    pub fn new(p: Point3, t: f32, mat: &'a dyn Material, r: &Ray, outward_normal: Vec3) -> Self {
        let front_face = r.direction.dot(outward_normal) < 0.0;
        let normal = if front_face { outward_normal } else { -outward_normal };
        Self { p, normal, mat, t, u: 0.0, v: 0.0, front_face, tangent: Vec3::default(), shading: OnceLock::new() }
    }
}

pub trait Hittable: Send + Sync {
    /// Intersect the ray over the CLOSED interval [t_min, t_max]: a hit at
    /// exactly t == t_max is accepted by every shape (rejection is `t > t_max` /
    /// `t > best_t`), and `any_hit` must agree with `hit` on the same interval.
    /// The renderer still shrinks shadow-ray t_max by (1 - 1e-4) so a surface
    /// never shadows itself at the endpoint.
    fn hit(&self, r: &Ray, t_min: f32, t_max: f32) -> Option<HitRecord<'_>>;
    fn bounding_box(&self) -> Option<Aabb>;

    /// Bounds used to size the caustic photon-emission disk.  Proxy geometry that
    /// is effectively infinite (InfinitePlane) returns None so the disk covers only
    /// real geometry; HittableList unions the finite children instead of letting one
    /// proxy poison the whole scene.  Defaults to `bounding_box()`.
    fn photon_bounds(&self) -> Option<Aabb> { self.bounding_box() }

    /// Returns true if any object is hit in [t_min, t_max].
    /// Faster than hit() for shadow/occlusion tests — exits at the first intersection.
    fn any_hit(&self, r: &Ray, t_min: f32, t_max: f32) -> bool {
        self.hit(r, t_min, t_max).is_some()
    }

    /// Transmittance along the ray over [t_min, t_max] — used for shadow rays.
    /// 0 if an opaque surface is hit; otherwise the product of any media
    /// transmittances crossed (Beer-Lambert). Default treats every hit as fully
    /// opaque, i.e. exactly the old any_hit behaviour; media override this.
    fn transmittance(&self, r: &Ray, t_min: f32, t_max: f32) -> f32 {
        if self.any_hit(r, t_min, t_max) { 0.0 } else { 1.0 }
    }

    /// Surface area used as the selection weight in a BVH light-sampling CDF.
    /// 0 = not NEE-sampleable (non-emissive, or infinite/complex area like planes).
    fn light_area(&self) -> f32 { 0.0 }

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

    /// Add an already-shared object (same Arc goes into the world and the light list).
    pub fn add_shared(&mut self, obj: Arc<dyn Hittable>) {
        self.objects.push(obj);
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

    fn transmittance(&self, r: &Ray, t_min: f32, t_max: f32) -> f32 {
        let mut tr = 1.0f32;
        for obj in &self.objects {
            tr *= obj.transmittance(r, t_min, t_max);
            if tr <= 0.0 { return 0.0; }
        }
        tr
    }

    fn light_area(&self) -> f32 {
        self.objects.iter().map(|o| o.light_area()).sum()
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

    fn photon_bounds(&self) -> Option<Aabb> {
        // Union only the finite (non-proxy) children; None children (infinite
        // planes) are skipped rather than failing the whole list.
        let mut result: Option<Aabb> = None;
        for obj in &self.objects {
            if let Some(b) = obj.photon_bounds() {
                result = Some(match result {
                    None => b,
                    Some(prev) => Aabb::surrounding(&prev, &b),
                });
            }
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

/// Exact AABB of a convex polyhedron given as half-spaces `n.dot(x) <= d`
/// (interior on the negative side of each plane).  The extreme points are the
/// vertices: intersections of plane triples that satisfy every other plane.
/// Returns None for a degenerate (empty) plane set.
pub fn aabb_from_planes(planes: &[(Vec3, f32)]) -> Option<Aabb> {
    let eps = 1e-5 * (1.0 + planes.iter().map(|&(_, d)| d.abs()).fold(0.0, f32::max));
    let mut lo = Vec3::new(f32::INFINITY, f32::INFINITY, f32::INFINITY);
    let mut hi = Vec3::new(f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY);
    let mut found = false;
    for i in 0..planes.len() {
        for j in i + 1..planes.len() {
            for k in j + 1..planes.len() {
                let (a, da) = planes[i];
                let (b, db) = planes[j];
                let (c, dc) = planes[k];
                // Cramer: solve [a; b; c] x = [da; db; dc]
                let det = a.dot(b.cross(c));
                if det.abs() < 1e-12 { continue; }
                let p = (b.cross(c) * da + c.cross(a) * db + a.cross(b) * dc) / det;
                if planes.iter().all(|&(n, d)| n.dot(p) <= d + eps) {
                    found = true;
                    lo = Vec3::new(lo.x.min(p.x), lo.y.min(p.y), lo.z.min(p.z));
                    hi = Vec3::new(hi.x.max(p.x), hi.y.max(p.y), hi.z.max(p.z));
                }
            }
        }
    }
    if found { Some(Aabb::new(lo, hi)) } else { None }
}
