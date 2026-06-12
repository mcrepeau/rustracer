use std::f32::consts::PI;
use rand::Rng;
use crate::hittable::Hittable;
use crate::ray::Ray;
use crate::vec3::{Color, Point3, Vec3};

pub struct Light {
    q:      Point3,
    u:      Vec3,
    v:      Vec3,
    emit:   Color,
    area:   f32,
    normal: Vec3,
}

impl Light {
    pub fn new(q: Point3, u: Vec3, v: Vec3, emit: Color) -> Self {
        let n      = u.cross(v);
        let area   = n.length();
        let normal = n / area;
        Self { q, u, v, emit, area, normal }
    }

    pub fn sample_contribution(
        &self, from: Point3, normal: Vec3, albedo: Color,
        world: &dyn Hittable, rng: &mut impl Rng,
    ) -> Color {
        let p     = self.q + rng.gen::<f32>() * self.u + rng.gen::<f32>() * self.v;
        let delta = p - from;
        let dist2 = delta.length_squared();
        let dist  = dist2.sqrt();
        let dir   = delta / dist;

        let cos_surf  = normal.dot(dir);
        let cos_light = (-dir).dot(self.normal);
        if cos_surf <= 0.0 || cos_light <= 0.0 { return Color::default(); }

        if world.hit(&Ray::new(from, dir), 0.001, dist - 0.001).is_some() {
            return Color::default();
        }

        self.emit * albedo * (cos_surf * cos_light * self.area / (PI * dist2))
    }
}
