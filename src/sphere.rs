use std::sync::Arc;
use std::f32::consts::PI;

use rand::{Rng, RngCore};
use crate::aabb::Aabb;
use crate::onb::Onb;
use crate::vec3::{Point3, Vec3};
use crate::ray::Ray;
use crate::hittable::{HitRecord, Hittable, Material};

fn sphere_uv(p: Vec3) -> (f32, f32) {
    let theta = (-p.y).acos();
    let phi   = (-p.z).atan2(p.x) + PI;
    (phi / (2.0 * PI), theta / PI)
}

#[derive(Clone)]
pub struct Sphere {
    pub center:  Point3,
    pub center1: Option<Point3>,  // Some → moving: lerp from center to center1 over t=0..1
    pub radius:  f32,
    pub mat:     Arc<dyn Material>,
}

impl Sphere {
    pub fn new(center: Point3, radius: f32, mat: Arc<dyn Material>) -> Self {
        Self { center, center1: None, radius, mat }
    }

    pub fn new_moving(c0: Point3, c1: Point3, radius: f32, mat: Arc<dyn Material>) -> Self {
        Self { center: c0, center1: Some(c1), radius, mat }
    }

    fn center_at(&self, time: f32) -> Point3 {
        match self.center1 {
            None     => self.center,
            Some(c1) => self.center + time * (c1 - self.center),
        }
    }
}

impl Hittable for Sphere {
    fn hit(&self, r: &Ray, t_min: f32, t_max: f32) -> Option<HitRecord<'_>> {
        let center = self.center_at(r.time);
        let oc = r.origin - center;
        let a = r.direction.length_squared();
        let half_b = oc.dot(r.direction);
        let c = oc.length_squared() - self.radius * self.radius;
        let disc = half_b * half_b - a * c;
        if disc < 0.0 { return None; }
        let sqrtd = disc.sqrt();
        let mut root = (-half_b - sqrtd) / a;
        if root < t_min || root > t_max {
            root = (-half_b + sqrtd) / a;
            if root < t_min || root > t_max { return None; }
        }
        let p = r.at(root);
        let outward_normal = (p - center) / self.radius;
        let (u, v) = sphere_uv(outward_normal);
        let mut rec = HitRecord::new(p, root, &*self.mat, r, outward_normal);
        rec.u = u;
        rec.v = v;
        Some(rec)
    }

    fn pdf_value(&self, origin: Point3, _dir: Vec3, time: f32) -> f32 {
        let to_center = self.center_at(time) - origin;
        let dist_sq   = to_center.length_squared();
        let r2        = self.radius * self.radius;
        if dist_sq <= r2 { return 0.0; }
        let cos_theta_max = (1.0 - r2 / dist_sq).sqrt();
        1.0 / (2.0 * PI * (1.0 - cos_theta_max))
    }

    fn pdf_generate(&self, origin: Point3, rng: &mut dyn RngCore, time: f32) -> Vec3 {
        let to_center = self.center_at(time) - origin;
        let dist_sq   = to_center.length_squared();
        let r2        = self.radius * self.radius;
        if dist_sq <= r2 { return to_center.unit(); }
        let cos_theta_max = (1.0 - r2 / dist_sq).sqrt();
        let r1: f32 = rng.gen();
        let r2r: f32 = rng.gen();
        let phi       = 2.0 * PI * r1;
        let cos_theta = 1.0 - r2r * (1.0 - cos_theta_max);
        let sin_theta = (1.0 - cos_theta * cos_theta).max(0.0).sqrt();
        let local = Vec3::new(phi.cos() * sin_theta, phi.sin() * sin_theta, cos_theta);
        Onb::from_w(to_center.unit()).local(local)
    }

    fn bounding_box(&self) -> Option<Aabb> {
        let r = Vec3::new(self.radius, self.radius, self.radius);
        let b0 = Aabb::new(self.center - r, self.center + r);
        match self.center1 {
            None     => Some(b0),
            Some(c1) => Some(Aabb::surrounding(&b0, &Aabb::new(c1 - r, c1 + r))),
        }
    }
}
