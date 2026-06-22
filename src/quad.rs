use std::sync::Arc;
use rand::{Rng, RngCore};
use crate::aabb::Aabb;
use crate::vec3::{Point3, Vec3};
use crate::ray::Ray;
use crate::hittable::{HitRecord, Hittable, HittableList, Material};

#[derive(Clone)]
pub struct Quad {
    q:      Point3,
    u:      Vec3,
    v:      Vec3,
    normal: Vec3,
    d:      f32,
    w:      Vec3,   // n / (n·n), used to compute UV coordinates
    mat:    Arc<dyn Material>,
    area:   f32,
}

impl Quad {
    pub fn new(q: Point3, u: Vec3, v: Vec3, mat: Arc<dyn Material>) -> Self {
        let n      = u.cross(v);
        let area   = n.length();
        let normal = n / area;      // unit normal
        let d      = normal.dot(q);
        let w      = n / n.dot(n);
        Self { q, u, v, normal, d, w, mat, area }
    }
}

impl Hittable for Quad {
    fn hit(&self, r: &Ray, t_min: f32, t_max: f32) -> Option<HitRecord<'_>> {
        let denom = self.normal.dot(r.direction);
        if denom.abs() < 1e-8 { return None; }

        let t = (self.d - self.normal.dot(r.origin)) / denom;
        if t < t_min || t > t_max { return None; }

        let p      = r.at(t);
        let planar = p - self.q;
        let alpha  = self.w.dot(planar.cross(self.v));
        let beta   = self.w.dot(self.u.cross(planar));

        if !(0.0..=1.0).contains(&alpha) || !(0.0..=1.0).contains(&beta) {
            return None;
        }

        let mut rec = HitRecord::new(p, t, &*self.mat, r, self.normal);
        rec.u = alpha;
        rec.v = beta;
        Some(rec)
    }

    fn bounding_box(&self) -> Option<Aabb> {
        let corners = [self.q, self.q + self.u, self.q + self.v, self.q + self.u + self.v];
        let mut min = corners[0];
        let mut max = corners[0];
        for &c in &corners[1..] {
            min = Point3::new(min.x.min(c.x), min.y.min(c.y), min.z.min(c.z));
            max = Point3::new(max.x.max(c.x), max.y.max(c.y), max.z.max(c.z));
        }
        Some(Aabb::new(min, max).pad())
    }

    fn pdf_value(&self, origin: Point3, dir: Vec3, _time: f32) -> f32 {
        let r = Ray::new(origin, dir);
        let Some(rec) = self.hit(&r, 0.001, f32::INFINITY) else { return 0.0; };
        let dist2 = (rec.p - origin).length_squared();
        let cos_theta = dir.unit().dot(rec.normal).abs();
        if cos_theta < 1e-8 { return 0.0; }
        dist2 / (cos_theta * self.area)
    }

    fn pdf_generate(&self, origin: Point3, rng: &mut dyn RngCore, _time: f32) -> Vec3 {
        let r1: f32 = rng.gen();
        let r2: f32 = rng.gen();
        let point = self.q + r1 * self.u + r2 * self.v;
        (point - origin).unit()
    }
}

pub fn make_box(p0: Point3, p1: Point3, mat: Arc<dyn Material>) -> HittableList {
    let mut sides = HittableList::new();
    let min = Point3::new(p0.x.min(p1.x), p0.y.min(p1.y), p0.z.min(p1.z));
    let max = Point3::new(p0.x.max(p1.x), p0.y.max(p1.y), p0.z.max(p1.z));
    let dx = Vec3::new(max.x - min.x, 0.0, 0.0);
    let dy = Vec3::new(0.0, max.y - min.y, 0.0);
    let dz = Vec3::new(0.0, 0.0, max.z - min.z);
    sides.add(Quad::new(Point3::new(min.x, min.y, max.z),  dx,  dy, Arc::clone(&mat)));
    sides.add(Quad::new(Point3::new(max.x, min.y, max.z), -dz,  dy, Arc::clone(&mat)));
    sides.add(Quad::new(Point3::new(max.x, min.y, min.z), -dx,  dy, Arc::clone(&mat)));
    sides.add(Quad::new(Point3::new(min.x, min.y, min.z),  dz,  dy, Arc::clone(&mat)));
    sides.add(Quad::new(Point3::new(min.x, max.y, max.z),  dx, -dz, Arc::clone(&mat)));
    sides.add(Quad::new(Point3::new(min.x, min.y, min.z),  dx,  dz, mat));
    sides
}
