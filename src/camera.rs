use std::f32::consts::PI;

use rand::Rng;
use crate::hittable::Hittable;
use crate::ray::Ray;
use crate::vec3::{Point3, Vec3};

// ── Camera ────────────────────────────────────────────────────────────────────

#[derive(Copy, Clone)]
pub struct Camera {
    origin:          Point3,
    lower_left:      Point3,
    horizontal:      Vec3,
    vertical:        Vec3,
    u:               Vec3,
    v:               Vec3,
    lens_radius:     f32,
    /// 0 = circular (default); ≥ 3 = regular N-gon aperture for polygonal bokeh.
    aperture_blades: u32,
}

impl Camera {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        lookfrom: Point3, lookat: Point3, vup: Vec3,
        vfov_deg: f32, aspect_ratio: f32,
        aperture: f32, focus_dist: f32,
        aperture_blades: u32,
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

        Self { origin: lookfrom, lower_left, horizontal, vertical, u, v,
               lens_radius: aperture/2.0, aperture_blades }
    }

    pub fn get_ray(&self, s: f32, t: f32, rng: &mut (impl Rng + ?Sized)) -> Ray {
        let rd = if self.aperture_blades >= 3 {
            self.lens_radius * sample_polygon(self.aperture_blades, rng)
        } else {
            self.lens_radius * Vec3::random_in_unit_disk(rng)
        };
        let offset = self.u * rd.x + self.v * rd.y;
        Ray::new_at_time(
            self.origin + offset,
            self.lower_left + s*self.horizontal + t*self.vertical - self.origin - offset,
            rng.gen::<f32>(),
        )
    }
}

/// Sample a point uniformly within a regular N-gon inscribed in the unit circle.
/// Uses N-fan triangle decomposition: pick a wedge at random, then uniform-sample
/// the isoceles triangle formed by the origin and the two blade-edge vertices.
/// Rotation is fixed at π/N so a flat edge faces upward (natural lens orientation).
fn sample_polygon(blades: u32, rng: &mut (impl Rng + ?Sized)) -> Vec3 {
    let n   = blades as f32;
    let rot = PI / n;                              // flat-edge-up orientation
    let k   = rng.gen_range(0..blades) as f32;
    let a1  = 2.0 * PI * k       / n + rot;
    let a2  = 2.0 * PI * (k + 1.0) / n + rot;
    let p1  = Vec3::new(a1.cos(), a1.sin(), 0.0);
    let p2  = Vec3::new(a2.cos(), a2.sin(), 0.0);
    // Uniform triangle sample (origin, p1, p2): fold the square into the triangle
    let mut r1: f32 = rng.gen();
    let mut r2: f32 = rng.gen();
    if r1 + r2 > 1.0 { r1 = 1.0 - r1; r2 = 1.0 - r2; }
    p1 * r1 + p2 * r2
}

// ── Scene camera parameters ───────────────────────────────────────────────────

pub struct SceneCameraParams {
    pub pos:             Point3,
    pub lookat:          Point3,
    pub vfov:            f32,
    pub aperture:        f32,
    pub focus_dist:      f32,
    pub move_speed:      f32,
    /// 0 = circular; ≥ 3 = N-blade polygonal aperture.
    pub aperture_blades: u32,
}

// ── Interactive camera state ──────────────────────────────────────────────────

pub struct CameraState {
    pub pos:             Point3,
    pub yaw:             f32,
    pub pitch:           f32,
    pub vfov:            f32,
    pub aperture:        f32,
    pub focus_dist:      f32,
    pub move_speed:      f32,
    pub aperture_blades: u32,
}

impl CameraState {
    pub fn from_params(p: &SceneCameraParams) -> Self {
        let dir = (p.lookat - p.pos).unit();
        Self {
            pos:             p.pos,
            yaw:             dir.x.atan2(-dir.z),
            pitch:           dir.y.asin().clamp(-89f32.to_radians(), 89f32.to_radians()),
            vfov:            p.vfov,
            aperture:        p.aperture,
            focus_dist:      p.focus_dist,
            move_speed:      p.move_speed,
            aperture_blades: p.aperture_blades,
        }
    }

    pub fn forward_horiz(&self) -> Vec3 { Vec3::new( self.yaw.sin(), 0.0, -self.yaw.cos()) }
    pub fn right_horiz(&self)   -> Vec3 { Vec3::new( self.yaw.cos(), 0.0,  self.yaw.sin()) }

    fn fwd(&self) -> Vec3 {
        Vec3::new(
            self.yaw.sin() * self.pitch.cos(),
            self.pitch.sin(),
            -self.yaw.cos() * self.pitch.cos(),
        )
    }

    pub fn to_camera(&self, aspect: f32) -> Camera {
        Camera::new(self.pos, self.pos + self.fwd(), Vec3::new(0.0, 1.0, 0.0),
                    self.vfov, aspect, self.aperture, self.focus_dist,
                    self.aperture_blades)
    }

    pub fn autofocus(&mut self, world: &dyn Hittable) {
        if let Some(rec) = world.hit(&Ray::new(self.pos, self.fwd()), 0.001, f32::INFINITY) {
            self.focus_dist = rec.t;
        }
    }

    /// The point the camera is aimed at (pos + forward × focus_dist).
    /// After autofocus this lands on the nearest scene object along the view axis.
    pub fn look_at(&self) -> Point3 {
        self.pos + self.fwd() * self.focus_dist
    }
}
