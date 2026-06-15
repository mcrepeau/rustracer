use std::sync::Arc;
use crate::aabb::Aabb;
use crate::bvh::BvhTree;
use crate::camera::SceneCameraParams;
use crate::hittable::{Hittable, HittableList, Material};
use crate::photon::PhotonMap;
use crate::renderer::Background;
use crate::sphere::Sphere;
use crate::vec3::{Color, Point3, Vec3};

pub struct DynamicSphere {
    pub center:      Point3,
    pub velocity:    Vec3,
    pub radius:      f32,
    pub mat:         Arc<dyn Material>,
    pub restitution: f32,
    pub is_static:   bool,
}

// ── Physics ───────────────────────────────────────────────────────────────────

/// All simulation state for a physics-driven scene.
pub struct PhysicsState {
    pub static_objects:   Vec<Arc<dyn Hittable>>,
    pub dynamic:          Vec<DynamicSphere>,
    pub bounds:           Option<Aabb>,
    pub colliders:        Vec<Aabb>,
    /// Convex polyhedra as lists of half-planes (outward normal, offset).
    /// Spheres are kept outside via `bounce_sphere_off_convex`.
    pub convex_colliders: Vec<Vec<(Vec3, f32)>>,
    pub gravity:          f32,
    pub settled:          bool,
    pub paused:           bool,
    /// BVH over `static_objects` and `is_static` dynamic spheres, built once
    /// on the first `build_world` call and reused on every subsequent tick.
    pub(crate) cached_static: Option<Arc<dyn Hittable>>,
}

// ── Scene ─────────────────────────────────────────────────────────────────────

pub struct SceneData {
    pub world:          Arc<dyn Hittable>,
    pub lights:         HittableList,
    pub background:     Background,
    pub name:           &'static str,
    pub cam_init:       SceneCameraParams,
    pub max_samples:    u32,
    /// Enable caustic photon mapping for this scene.
    pub enable_caustics: bool,
    /// Area-light emitter for the photon map when `Background` is not
    /// `Physical`.  Fields: (origin, U-extent, V-extent, emission colour).
    pub caustic_quad:          Option<(Point3, Vec3, Vec3, Color)>,
    /// Photon gather radius in world units.  Must match the scene's spatial
    /// scale: ~0.15 for unit-scale scenes, ~10 for 0–555 coordinate scenes.
    pub caustic_gather_radius: f32,
    /// Caustic photon map, rebuilt after every `rebuild()` when enabled.
    pub photon_map:      Option<Arc<PhotonMap>>,
    pub physics:         PhysicsState,
}

// ── Collision helpers ─────────────────────────────────────────────────────────

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

/// Sphere-vs-convex-polyhedron collision for the physics tick.
///
/// The polyhedron interior is defined by the half-spaces `n · x ≤ d` (outward
/// normal `n`, scalar offset `d`).  For each face we compute the signed
/// separation  sep = n · center − d  (positive = sphere center is on the
/// outside of that face's half-space).  The sphere overlaps when the maximum
/// separation over all faces is less than the sphere radius, meaning the center
/// is inside the Minkowski-expanded polyhedron.  Resolution pushes the center
/// along the face with the largest (least-negative / most-positive) separation,
/// which is the minimum-penetration-depth direction.
fn bounce_sphere_off_convex(
    center: &mut Point3,
    velocity: &mut Vec3,
    radius: f32,
    restitution: f32,
    planes: &[(Vec3, f32)],
) {
    let mut max_sep = f32::NEG_INFINITY;
    let mut best_n  = Vec3::new(0.0, 1.0, 0.0);

    for &(n, d) in planes {
        let sep = n.dot(*center) - d;
        if sep > max_sep {
            max_sep = sep;
            best_n  = n;
        }
    }

    // No overlap when the sphere centre is at least `radius` outside the closest face.
    if max_sep >= radius { return; }

    // Push the centre out along the minimum-penetration face normal.
    *center += best_n * (radius - max_sep);

    // Cancel the velocity component moving into the surface.
    let v_dot_n = velocity.dot(best_n);
    if v_dot_n < 0.0 {
        *velocity -= (1.0 + restitution) * v_dot_n * best_n;
    }
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

// ── PhysicsState impl ─────────────────────────────────────────────────────────

impl PhysicsState {
    /// Advance the simulation by one tick.  Returns `true` when a step
    /// executed and the world geometry may have changed.
    pub fn step(&mut self) -> bool {
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
            for ds in &mut self.dynamic {
                if !ds.is_static { ds.center += ds.velocity; }
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
            for planes in &self.convex_colliders {
                bounce_sphere_off_convex(&mut ds.center, &mut ds.velocity, ds.radius, ds.restitution, planes);
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
                    b.velocity += (1.0 + e) * rel_v * normal;
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

        if self.gravity > 0.0 {
            let at_rest = self.dynamic.iter().all(|ds| {
                ds.is_static || ds.velocity.length_squared() < 1e-4
            });
            if at_rest { self.settled = true; }
        }

        true
    }

    /// Build the world BVH from current state.  Returns `None` when both
    /// `static_objects` and `dynamic` are empty (caller should leave the world
    /// BVH it already has — the scene was built externally and never needs a
    /// physics-driven rebuild).
    fn build_world(&mut self) -> Option<Arc<dyn Hittable>> {
        if self.static_objects.is_empty() && self.dynamic.is_empty() { return None; }

        // Build the static BVH once and reuse it on every subsequent tick.
        // is_static dynamic spheres never move, so they are included here too.
        if self.cached_static.is_none() {
            let mut sl = HittableList::new();
            for obj in &self.static_objects { sl.objects.push(Arc::clone(obj)); }
            for ds in &self.dynamic {
                if ds.is_static { sl.add(Sphere::new(ds.center, ds.radius, Arc::clone(&ds.mat))); }
            }
            if !sl.objects.is_empty() {
                self.cached_static = Some(Arc::new(BvhTree::from_list(sl)));
            }
        }

        // All spheres are static: world = cached_static, no outer wrap needed.
        if self.dynamic.iter().all(|ds| ds.is_static) {
            return self.cached_static.as_ref().map(Arc::clone);
        }

        let mut list = HittableList::new();
        if let Some(s) = &self.cached_static { list.objects.push(Arc::clone(s)); }
        for ds in &self.dynamic {
            if !ds.is_static { list.add(Sphere::new(ds.center, ds.radius, Arc::clone(&ds.mat))); }
        }
        Some(Arc::new(BvhTree::from_list(list)))
    }
}

// ── SceneData impl ────────────────────────────────────────────────────────────

impl SceneData {
    pub fn tick(&mut self) -> bool {
        if !self.physics.step() { return false; }
        if let Some(w) = self.physics.build_world() { self.world = w; }
        true
    }

    pub fn rebuild(&mut self) {
        if let Some(w) = self.physics.build_world() { self.world = w; }
        self.rebuild_caustics();
    }

    /// Rebuild only the photon map, reusing the current world BVH.
    /// Call this after sun-direction changes as well as after explicit rebuilds.
    pub fn rebuild_caustics(&mut self) {
        if !self.enable_caustics { return; }
        let r     = self.caustic_gather_radius;
        let world = Arc::clone(&self.world);
        if let Background::Physical { sun_dir } = self.background {
            // Derive photon power from the actual sky radiance at the sun
            // direction so caustic brightness stays in proportion to the
            // surrounding ground illumination computed by the path tracer.
            let sun_color = self.background.eval(sun_dir) * std::f32::consts::PI;
            self.photon_map = Some(Arc::new(
                PhotonMap::build(world.as_ref(), sun_dir, sun_color, 200_000, r)
            ));
        } else if let Some((origin, u, v, color)) = self.caustic_quad {
            self.photon_map = Some(Arc::new(
                PhotonMap::build_from_quad(world.as_ref(), origin, u, v, color, 200_000, r)
            ));
        }
    }
}
