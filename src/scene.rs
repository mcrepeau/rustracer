use std::sync::Arc;
use std::time::Duration;
use crate::aabb::Aabb;
use crate::bvh::BvhTree;
use crate::camera::SceneCameraParams;
use crate::hittable::{Hittable, HittableList, Material};
use crate::renderer::Background;
use crate::sphere::Sphere;
use crate::vec3::{Point3, Vec3};

pub struct Orbit {
    pub parent_idx:  Option<usize>, // None = orbit the fixed origin
    pub radius:      f32,
    pub speed:       f32,           // radians per tick
    pub angle:       f32,           // current angle
    pub inclination: f32,           // tilt of orbital plane from XZ (radians)
}

pub struct DynamicSphere {
    pub center:      Point3,
    pub velocity:    Vec3,
    pub radius:      f32,
    pub mat:         Arc<dyn Material>,
    pub restitution: f32,
    pub is_static:   bool,
    pub orbit:       Option<Orbit>,
}

pub struct SceneData {
    pub world:          Arc<dyn Hittable>,
    pub lights:         HittableList,
    pub background:     Background,
    pub name:           &'static str,
    pub cam_init:       SceneCameraParams,
    pub static_objects: Vec<Arc<dyn Hittable>>,
    pub dynamic:        Vec<DynamicSphere>,
    pub bounds:         Option<Aabb>,
    pub colliders:      Vec<Aabb>,
    pub gravity:        f32,
    pub settled:        bool,
    pub paused:         bool,
    pub max_samples:    u32,
    /// How much wall time must elapse between physics ticks.
    /// Use a larger value for slow-moving scenes (solar system) so the path
    /// tracer can accumulate more samples before each position reset.
    pub physics_dt:     Duration,
}

/// Sphere-vs-AABB collision: push the sphere out along the axis of minimum
/// penetration and reflect its velocity away from the box face.
fn bounce_sphere_off_aabb(center: &mut Point3, velocity: &mut Vec3, radius: f32, restitution: f32, bbox: &Aabb) {
    let lo = Point3::new(bbox.min.x - radius, bbox.min.y - radius, bbox.min.z - radius);
    let hi = Point3::new(bbox.max.x + radius, bbox.max.y + radius, bbox.max.z + radius);
    if center.x < lo.x || center.x > hi.x ||
       center.y < lo.y || center.y > hi.y ||
       center.z < lo.z || center.z > hi.z { return; }

    let dx_lo = center.x - lo.x;  let dx_hi = hi.x - center.x;
    let dy_lo = center.y - lo.y;  let dy_hi = hi.y - center.y;
    let dz_lo = center.z - lo.z;  let dz_hi = hi.z - center.z;

    let min_x = dx_lo.min(dx_hi);
    let min_y = dy_lo.min(dy_hi);
    let min_z = dz_lo.min(dz_hi);

    if min_x <= min_y && min_x <= min_z {
        if dx_lo < dx_hi { center.x = lo.x; velocity.x = -velocity.x.abs() * restitution; }
        else              { center.x = hi.x; velocity.x =  velocity.x.abs() * restitution; }
    } else if min_y <= min_x && min_y <= min_z {
        if dy_lo < dy_hi { center.y = lo.y; velocity.y = -velocity.y.abs() * restitution; }
        else              { center.y = hi.y; velocity.y =  velocity.y.abs() * restitution; }
    } else if dz_lo < dz_hi { center.z = lo.z; velocity.z = -velocity.z.abs() * restitution; }
    else                    { center.z = hi.z; velocity.z =  velocity.z.abs() * restitution; }
}

/// Push the camera point out of an AABB in XZ only (walls span full height).
pub fn resolve_camera_aabb(pos: &mut Point3, radius: f32, bbox: &Aabb) {
    if pos.y - radius >= bbox.max.y || pos.y + radius <= bbox.min.y { return; }
    let x_lo = bbox.min.x - radius;  let x_hi = bbox.max.x + radius;
    let z_lo = bbox.min.z - radius;  let z_hi = bbox.max.z + radius;
    if pos.x <= x_lo || pos.x >= x_hi || pos.z <= z_lo || pos.z >= z_hi { return; }
    let dx_lo = pos.x - x_lo;  let dx_hi = x_hi - pos.x;
    let dz_lo = pos.z - z_lo;  let dz_hi = z_hi - pos.z;
    let min_x = dx_lo.min(dx_hi);
    let min_z = dz_lo.min(dz_hi);
    if min_x <= min_z {
        pos.x = if dx_lo < dx_hi { x_lo } else { x_hi };
    } else {
        pos.z = if dz_lo < dz_hi { z_lo } else { z_hi };
    }
}

impl SceneData {
    pub fn tick(&mut self) -> bool {
        if self.dynamic.is_empty() || self.paused { return false; }
        if self.settled { return false; }

        if self.gravity > 0.0 {
            for ds in &mut self.dynamic {
                if ds.is_static { continue; }
                ds.velocity.y -= self.gravity;
                ds.velocity   *= 0.995;
                ds.center     += ds.velocity;
                if ds.center.y - ds.radius < 0.0 {
                    ds.center.y = ds.radius;
                    if ds.velocity.y < 0.0 {
                        ds.velocity.y = -ds.velocity.y * ds.restitution;
                        ds.velocity.x *= 0.8;
                        ds.velocity.z *= 0.8;
                    }
                    if ds.velocity.y < 0.05 { ds.velocity.y = 0.0; }
                }
            }
        } else {
            // Pass 1: bodies orbiting the fixed origin (planets around the sun)
            for ds in &mut self.dynamic {
                if ds.is_static { continue; }
                match &mut ds.orbit {
                    Some(orbit) if orbit.parent_idx.is_none() => {
                        orbit.angle += orbit.speed;
                        ds.center = Point3::new(
                            orbit.radius * orbit.angle.cos(),
                            orbit.radius * orbit.angle.sin() * orbit.inclination.sin(),
                            orbit.radius * orbit.angle.sin() * orbit.inclination.cos(),
                        );
                    }
                    None => ds.center += ds.velocity,
                    _ => {}
                }
            }
            // Pass 2: bodies orbiting a moving parent (moons) — snapshot parent
            // positions after pass 1 so moons track their planet's new location.
            let centers: Vec<Point3> = self.dynamic.iter().map(|ds| ds.center).collect();
            for ds in &mut self.dynamic {
                if ds.is_static { continue; }
                if let Some(orbit) = &mut ds.orbit {
                    if let Some(pidx) = orbit.parent_idx {
                        orbit.angle += orbit.speed;
                        let p = centers[pidx];
                        ds.center = Point3::new(
                            p.x + orbit.radius * orbit.angle.cos(),
                            p.y + orbit.radius * orbit.angle.sin() * orbit.inclination.sin(),
                            p.z + orbit.radius * orbit.angle.sin() * orbit.inclination.cos(),
                        );
                    }
                }
            }
        }

        if let Some(b) = self.bounds {
            for ds in &mut self.dynamic {
                if ds.is_static { continue; }
                let r = ds.radius;
                if ds.center.x - r < b.min.x { ds.center.x = b.min.x + r; ds.velocity.x =  ds.velocity.x.abs() * ds.restitution; }
                if ds.center.x + r > b.max.x { ds.center.x = b.max.x - r; ds.velocity.x = -ds.velocity.x.abs() * ds.restitution; }
                if ds.center.y - r < b.min.y { ds.center.y = b.min.y + r; ds.velocity.y =  ds.velocity.y.abs() * ds.restitution; }
                if ds.center.y + r > b.max.y { ds.center.y = b.max.y - r; ds.velocity.y = -ds.velocity.y.abs() * ds.restitution; }
                if ds.center.z - r < b.min.z { ds.center.z = b.min.z + r; ds.velocity.z =  ds.velocity.z.abs() * ds.restitution; }
                if ds.center.z + r > b.max.z { ds.center.z = b.max.z - r; ds.velocity.z = -ds.velocity.z.abs() * ds.restitution; }
            }
        }

        for ds in &mut self.dynamic {
            if ds.is_static { continue; }
            let max_step = ds.radius * 2.0;
            let spd = ds.velocity.length();
            if spd > max_step { ds.velocity *= max_step / spd; }
            for bbox in &self.colliders {
                bounce_sphere_off_aabb(&mut ds.center, &mut ds.velocity, ds.radius, ds.restitution, bbox);
            }
        }

        let n = self.dynamic.len();
        for i in 0..n {
            let (left, right) = self.dynamic.split_at_mut(i + 1);
            let a = &mut left[i];
            for b in right.iter_mut() {
                let diff     = a.center - b.center;
                let dist_sq  = diff.length_squared();
                let min_dist = a.radius + b.radius;
                if dist_sq >= min_dist * min_dist { continue; }
                let dist   = dist_sq.sqrt().max(1e-6);
                let normal = diff / dist;
                let rel_v  = (a.velocity - b.velocity).dot(normal);
                if rel_v >= 0.0 { continue; }
                let ma = a.radius * a.radius * a.radius;
                let mb = b.radius * b.radius * b.radius;
                let e  = (a.restitution + b.restitution) * 0.5;
                if a.is_static {
                    b.velocity -= (1.0 + e) * rel_v * normal;
                    b.center    = a.center - normal * min_dist;
                } else if b.is_static {
                    a.velocity -= (1.0 + e) * rel_v * normal;
                    a.center    = b.center + normal * min_dist;
                } else {
                    let j  = -(1.0 + e) * rel_v / (1.0/ma + 1.0/mb);
                    a.velocity += (j / ma) * normal;
                    b.velocity -= (j / mb) * normal;
                    let overlap = min_dist - dist;
                    let ra = mb / (ma + mb);
                    a.center += (overlap * ra) * normal;
                    b.center -= (overlap * (1.0 - ra)) * normal;
                }
            }
        }

        self.rebuild();

        if self.gravity > 0.0 {
            let at_rest = self.dynamic.iter().all(|ds| {
                ds.is_static || ds.velocity.length_squared() < 1e-4
            });
            if at_rest { self.settled = true; }
        }

        true
    }

    pub fn rebuild(&mut self) {
        if self.static_objects.is_empty() && self.dynamic.is_empty() { return; }
        let mut list = HittableList::new();
        for obj in &self.static_objects {
            list.objects.push(Arc::clone(obj));
        }
        for ds in &self.dynamic {
            list.add(Sphere::new(ds.center, ds.radius, Arc::clone(&ds.mat)));
        }
        self.world = Arc::new(BvhTree::from_list(list));
    }
}
