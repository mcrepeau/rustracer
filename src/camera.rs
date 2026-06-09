use crate::vec3::{Point3, Vec3};
use crate::ray::Ray;

#[derive(Copy, Clone)]
pub struct Camera {
    origin: Point3,
    lower_left: Point3,
    horizontal: Vec3,
    vertical: Vec3,
    u: Vec3,
    v: Vec3,
    lens_radius: f32,
}

impl Camera {
    pub fn new(
        lookfrom: Point3, lookat: Point3, vup: Vec3,
        vfov_deg: f32, aspect_ratio: f32,
        aperture: f32, focus_dist: f32,
    ) -> Self {
        let h = (vfov_deg.to_radians() / 2.0).tan();
        let viewport_h = 2.0 * h;
        let viewport_w = aspect_ratio * viewport_h;

        let w = (lookfrom - lookat).unit();
        let u = vup.cross(w).unit();
        let v = w.cross(u);

        let horizontal = focus_dist * viewport_w * u;
        let vertical   = focus_dist * viewport_h * v;
        let lower_left = lookfrom - horizontal/2.0 - vertical/2.0 - focus_dist*w;

        Self { origin: lookfrom, lower_left, horizontal, vertical, u, v, lens_radius: aperture/2.0 }
    }

    pub fn get_ray(&self, s: f32, t: f32) -> Ray {
        let rd = self.lens_radius * Vec3::random_in_unit_disk();
        let offset = self.u * rd.x + self.v * rd.y;
        Ray::new(
            self.origin + offset,
            self.lower_left + s*self.horizontal + t*self.vertical - self.origin - offset,
        )
    }
}
