use std::sync::Arc;

use crate::aabb::Aabb;
use crate::hittable::{HitRecord, Hittable, Material};
use crate::ray::Ray;
use crate::vec3::{Point3, Vec3};

/// Infinite flat plane defined by a `point` on the plane and an outward `normal`.
///
/// Optional wave parameters perturb the shading normal with two crossing sine
/// waves, giving the appearance of a gently rippled water surface without any
/// actual geometry displacement.  Set `wave_amplitude = 0` (the default) for a
/// perfectly flat surface.
pub struct InfinitePlane {
    pub point:          Point3,
    pub normal:         Vec3,
    pub wave_amplitude: f32,
    pub wave_scale:     f32,
    pub mat:            Arc<dyn Material>,
    u_axis:             Vec3,
    v_axis:             Vec3,
}

impl InfinitePlane {
    pub fn new(
        point:          Point3,
        normal:         Vec3,
        wave_amplitude: f32,
        wave_scale:     f32,
        mat:            Arc<dyn Material>,
    ) -> Self {
        let n = normal.unit();
        let (u_axis, v_axis) = n.onb();
        Self { point, normal: n, wave_amplitude, wave_scale, mat, u_axis, v_axis }
    }
}

impl Hittable for InfinitePlane {
    fn hit(&self, r: &Ray, t_min: f32, t_max: f32) -> Option<HitRecord<'_>> {
        let denom = self.normal.dot(r.direction);
        if denom.abs() < 1e-8 { return None; }
        let t = self.normal.dot(self.point - r.origin) / denom;
        if t < t_min || t > t_max { return None; }
        let p   = r.at(t);
        let off = p - self.point;

        let shading_normal = if self.wave_amplitude > 1e-6 {
            // Project hit point onto the two plane axes for wave coordinates.
            let u = off.dot(self.u_axis);
            let v = off.dot(self.v_axis);
            let s = self.wave_scale;
            let a = self.wave_amplitude;

            // Two aperiodic crossing sine waves — irrational frequency ratio
            // avoids visible tiling.
            let phi1 = s * u + 0.7  * s * v;
            let phi2 = 1.3 * s * v  - 0.5 * s * u;

            // Gradient of h(u,v) = A*(sin(phi1) + 0.5*sin(phi2))
            let dh_du = a * s * phi1.cos() - 0.25 * a * s * phi2.cos();
            let dh_dv = 0.7 * a * s * phi1.cos() + 0.65 * a * s * phi2.cos();

            // Bump-map: subtract gradient from the flat normal.
            // Derivation: tangent-space Jacobian of the height field gives
            //   n_perturbed = (n − dh/du·u_axis − dh/dv·v_axis).unit()
            (self.normal - dh_du * self.u_axis - dh_dv * self.v_axis).unit()
        } else {
            self.normal
        };

        let mut rec = HitRecord::new(p, t, &*self.mat, r, shading_normal);
        rec.u = off.dot(self.u_axis).rem_euclid(1.0);
        rec.v = off.dot(self.v_axis).rem_euclid(1.0);
        Some(rec)
    }

    fn bounding_box(&self) -> Option<Aabb> {
        // A truly infinite plane can only be bounded along the axis its normal aligns with.
        // For tilted planes all axes are unbounded; use ±1e5 as a finite proxy.
        const B: f32 = 1e5;
        const E: f32 = 1e-3;
        let n = self.normal;
        let p = self.point;
        let range = |ni: f32, pi: f32| -> (f32, f32) {
            if ni.abs() > 0.999 { (pi - E, pi + E) } else { (-B, B) }
        };
        let (x0, x1) = range(n.x, p.x);
        let (y0, y1) = range(n.y, p.y);
        let (z0, z1) = range(n.z, p.z);
        Some(Aabb::new(Point3::new(x0, y0, z0), Point3::new(x1, y1, z1)))
    }
}
