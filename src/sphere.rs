use std::sync::Arc;
use std::f32::consts::PI;

use crate::aabb::Aabb;
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

    fn bounding_box(&self) -> Option<Aabb> {
        let r = Vec3::new(self.radius, self.radius, self.radius);
        let b0 = Aabb::new(self.center - r, self.center + r);
        match self.center1 {
            None     => Some(b0),
            Some(c1) => Some(Aabb::surrounding(&b0, &Aabb::new(c1 - r, c1 + r))),
        }
    }
}
