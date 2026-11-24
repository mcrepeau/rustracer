use crate::vec3::{Point3, Vec3};

/// Camera state at a single keyframe (or evaluated orbit sample).
#[derive(Clone)]
pub struct Keyframe {
    pub time:       f32,
    pub look_from:  Point3,
    pub look_at:    Point3,
    pub vfov:       f32,
    pub aperture:   f32,
    pub focus_dist: f32,
}

/// True circular orbit: camera travels on a circle in the XZ-plane at fixed Y,
/// always pointing at `look_at`.  Angles are stored in radians internally.
pub struct OrbitData {
    pub center:      Point3,
    pub radius:      f32,
    pub height:      f32,   // world-space Y of the camera
    pub look_at:     Point3,
    pub start_angle: f32,   // radians; 0 = +Z from center
    pub end_angle:   f32,   // radians
    pub vfov:        f32,
    pub aperture:    f32,
    pub focus_dist:  Option<f32>,  // None = compute from distance
}

pub enum AnimationKind {
    Keyframes(Vec<Keyframe>),
    Orbit(OrbitData),
}

pub struct AnimationData {
    pub fps:      f32,
    pub duration: f32,
    pub kind:     AnimationKind,
}

impl AnimationData {
    pub fn new(fps: f32, duration: f32, kind: AnimationKind) -> Option<Self> {
        if fps <= 0.0 || duration <= 0.0 { return None; }
        if let AnimationKind::Keyframes(ref kf) = kind {
            if kf.is_empty() { return None; }
        }
        Some(Self { fps, duration, kind })
    }

    pub fn total_frames(&self) -> u32 {
        (self.fps * self.duration).round() as u32
    }

    pub fn evaluate(&self, t: f32) -> Keyframe {
        match &self.kind {
            AnimationKind::Keyframes(kf) => evaluate_keyframes(kf, t),
            AnimationKind::Orbit(o) => {
                let frac = (t / self.duration).clamp(0.0, 1.0);
                let angle = o.start_angle + (o.end_angle - o.start_angle) * frac;
                let look_from = Point3::new(
                    o.center.x + o.radius * angle.sin(),
                    o.height,
                    o.center.z + o.radius * angle.cos(),
                );
                let focus_dist = o.focus_dist.unwrap_or_else(|| {
                    (look_from - o.look_at).length().max(0.01)
                });
                Keyframe {
                    time: t,
                    look_from,
                    look_at: o.look_at,
                    vfov:     o.vfov,
                    aperture: o.aperture,
                    focus_dist,
                }
            }
        }
    }
}

fn evaluate_keyframes(kf: &[Keyframe], t: f32) -> Keyframe {
    let n = kf.len();
    if n == 1 { return kf[0].clone(); }

    let t = t.clamp(kf[0].time, kf[n - 1].time);

    let i = kf.partition_point(|k| k.time <= t).saturating_sub(1).min(n - 2);

    let t0 = kf[i].time;
    let t1 = kf[i + 1].time;
    let u  = if (t1 - t0).abs() < 1e-8 { 0.0 } else { (t - t0) / (t1 - t0) };

    let p0 = &kf[i.saturating_sub(1)];
    let p1 = &kf[i];
    let p2 = &kf[(i + 1).min(n - 1)];
    let p3 = &kf[(i + 2).min(n - 1)];

    Keyframe {
        time:       t,
        look_from:  cr_vec3(p0.look_from,  p1.look_from,  p2.look_from,  p3.look_from,  u),
        look_at:    cr_vec3(p0.look_at,     p1.look_at,    p2.look_at,    p3.look_at,    u),
        vfov:       lerp(p1.vfov,       p2.vfov,       u),
        aperture:   lerp(p1.aperture,   p2.aperture,   u),
        focus_dist: lerp(p1.focus_dist, p2.focus_dist, u),
    }
}

#[inline] fn lerp(a: f32, b: f32, t: f32) -> f32 { a + (b - a) * t }

#[inline]
fn cr_f32(p0: f32, p1: f32, p2: f32, p3: f32, t: f32) -> f32 {
    let t2 = t * t;
    let t3 = t2 * t;
    0.5 * ((2.0 * p1)
        + (-p0 + p2) * t
        + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
        + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3)
}

#[inline]
fn cr_vec3(p0: Vec3, p1: Vec3, p2: Vec3, p3: Vec3, t: f32) -> Vec3 {
    Vec3::new(
        cr_f32(p0.x, p1.x, p2.x, p3.x, t),
        cr_f32(p0.y, p1.y, p2.y, p3.y, t),
        cr_f32(p0.z, p1.z, p2.z, p3.z, t),
    )
}
