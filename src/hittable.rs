use std::sync::Arc;
use crate::aabb::Aabb;
use crate::vec3::{Color, Point3, Vec3};
use crate::ray::Ray;

pub trait Material: Send + Sync {
    fn scatter(&self, r_in: &Ray, rec: &HitRecord) -> Option<(Color, Ray)>;
    fn emitted(&self) -> Color { Color::default() }
}

pub struct HitRecord {
    pub p: Point3,
    pub normal: Vec3,
    pub mat: Arc<dyn Material>,
    pub t: f64,
    pub u: f64,
    pub v: f64,
    pub front_face: bool,
}

impl HitRecord {
    pub fn new(p: Point3, t: f64, mat: Arc<dyn Material>, r: &Ray, outward_normal: Vec3) -> Self {
        let front_face = r.direction.dot(outward_normal) < 0.0;
        let normal = if front_face { outward_normal } else { -outward_normal };
        Self { p, normal, mat, t, u: 0.0, v: 0.0, front_face }
    }
}

pub trait Hittable: Send + Sync {
    fn hit(&self, r: &Ray, t_min: f64, t_max: f64) -> Option<HitRecord>;
    fn bounding_box(&self) -> Option<Aabb>;
}

pub struct HittableList {
    pub objects: Vec<Arc<dyn Hittable>>,
}

impl HittableList {
    pub fn new() -> Self { Self { objects: Vec::new() } }

    pub fn add(&mut self, obj: impl Hittable + 'static) {
        self.objects.push(Arc::new(obj));
    }

    pub fn add_arc(&mut self, obj: Arc<dyn Hittable>) {
        self.objects.push(obj);
    }
}

impl Hittable for HittableList {
    fn hit(&self, r: &Ray, t_min: f64, t_max: f64) -> Option<HitRecord> {
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

    fn bounding_box(&self) -> Option<Aabb> {
        if self.objects.is_empty() { return None; }
        let mut result: Option<Aabb> = None;
        for obj in &self.objects {
            let Some(bbox) = obj.bounding_box() else { return None; };
            result = Some(match result {
                None => bbox,
                Some(prev) => Aabb::surrounding(&prev, &bbox),
            });
        }
        result
    }
}
