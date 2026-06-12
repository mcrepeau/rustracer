mod aabb;
mod bvh;
mod vec3;
mod ray;
mod hittable;
mod texture;
mod material;
mod perlin;
mod volume;
mod onb;
mod pdf;
mod sphere;
mod quad;
mod transform;
mod mesh;
mod camera;

use aabb::Aabb;
use vec3::{Color, Point3, Vec3};
use ray::Ray;
use hittable::{Hittable, HittableList};
use texture::Texture;
use material::{Dielectric, DiffuseLight, Lambertian, Metal};
use perlin::Perlin;
use volume::ConstantMedium;
use sphere::Sphere;
use quad::{Quad, make_box};
use transform::{Rotate, Translate};
use mesh::load_obj;
use camera::Camera;
use bvh::BvhTree;
use pdf::{CosinePdf, HittablePdf, MixturePdf, Pdf};

use winit::{
    dpi::PhysicalSize,
    event::{DeviceEvent, ElementState, Event, KeyboardInput, VirtualKeyCode, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::{CursorGrabMode, WindowBuilder},
};
use softbuffer::{Context, Surface};
use std::num::NonZeroU32;
use std::time::{Duration, Instant};
use image::{ImageBuffer, Rgb};
use rayon::prelude::*;
use rand::Rng;
use rand::rngs::SmallRng;
use rand::SeedableRng;
use std::sync::Arc;

const WIDTH: u32         = 1200;
const HEIGHT: u32        = 800;
const MAX_DEPTH: i32     = 50;
const MAX_SAMPLES: u32   = 2000;
const MOUSE_SENS: f32    = 0.002;
const MAX_LUMINANCE: f32      = 10.0;
const TITLE_INTERVAL: Duration  = Duration::from_millis(200);
const PHYSICS_DT:     Duration  = Duration::from_millis(16);  // ~60 Hz fixed physics step
const ADAPTIVE_MIN_SAMPLES: u32 = 32;
const ADAPTIVE_THRESHOLD:   f32 = 0.05; // relative std-dev threshold (coefficient of variation)
const CAM_RADIUS:  f32   = 0.25;

fn ray_color(r: &Ray, world: &dyn Hittable, background: Background, lights: &HittableList, rng: &mut impl Rng) -> Color {
    let mut throughput = Color::new(1.0, 1.0, 1.0);
    let mut color      = Color::default();
    let mut ray        = *r;

    for depth in 0..MAX_DEPTH {
        match world.hit(&ray, 0.001, f32::INFINITY) {
            None => {
                color += throughput * background.eval(ray.direction);
                break;
            }
            Some(rec) => {
                color += throughput * rec.mat.emitted();

                let Some(sr) = rec.mat.scatter(&ray, &rec, rng) else { break; };

                if sr.skip_pdf {
                    // Specular (metal / dielectric / isotropic): exact sampled direction.
                    throughput *= sr.attenuation;
                    ray = sr.ray;
                } else {
                    // Diffuse: importance-sample with mixture PDF (cosine + area light).
                    let scattered_dir;
                    let pdf_val;

                    if lights.objects.is_empty() {
                        let cpdf = CosinePdf::new(rec.normal);
                        scattered_dir = cpdf.generate(rng);
                        pdf_val = cpdf.value(scattered_dir);
                    } else {
                        let cpdf  = CosinePdf::new(rec.normal);
                        let lpdf  = HittablePdf::new(lights, rec.p);
                        let mpdf  = MixturePdf::new(&cpdf, &lpdf);
                        scattered_dir = mpdf.generate(rng);
                        pdf_val       = mpdf.value(scattered_dir);
                    };

                    if pdf_val <= 0.0 { break; }
                    let scattered = Ray::new_at_time(rec.p, scattered_dir, ray.time);
                    let scat_pdf  = rec.mat.scattering_pdf(&ray, &rec, &scattered);
                    if scat_pdf <= 0.0 { break; }  // direction below surface hemisphere

                    throughput *= sr.attenuation * (scat_pdf / pdf_val);
                    ray = scattered;
                }

                // Russian roulette.
                if depth >= 2 {
                    let survive = throughput.x.max(throughput.y).max(throughput.z);
                    if survive <= 0.0 || rng.gen::<f32>() >= survive { break; }
                    throughput /= survive;
                }
            }
        }
    }

    let lum = color.x * 0.2126 + color.y * 0.7152 + color.z * 0.0722;
    if lum > MAX_LUMINANCE { color *= MAX_LUMINANCE / lum; }
    color
}

// ── Camera ───────────────────────────────────────────────────────────────────

struct SceneCameraParams {
    pos:        Point3,
    lookat:     Point3,
    vfov:       f32,
    aperture:   f32,
    focus_dist: f32,
    move_speed: f32,
}

struct CameraState {
    pos:        Point3,
    yaw:        f32,   // radians; 0 = looking toward −Z
    pitch:      f32,   // radians; clamped to ±89°
    vfov:       f32,
    aperture:   f32,
    focus_dist: f32,
    move_speed: f32,
}

impl CameraState {
    fn from_params(p: &SceneCameraParams) -> Self {
        let dir = (p.lookat - p.pos).unit();
        Self {
            pos:        p.pos,
            yaw:        dir.x.atan2(-dir.z),
            pitch:      dir.y.asin().clamp(-89f32.to_radians(), 89f32.to_radians()),
            vfov:       p.vfov,
            aperture:   p.aperture,
            focus_dist: p.focus_dist,
            move_speed: p.move_speed,
        }
    }

    // Horizontal-only forward/right vectors for WASD (no vertical drift when looking up/down).
    fn forward_horiz(&self) -> Vec3 { Vec3::new( self.yaw.sin(), 0.0, -self.yaw.cos()) }
    fn right_horiz(&self)   -> Vec3 { Vec3::new( self.yaw.cos(), 0.0,  self.yaw.sin()) }

    fn fwd(&self) -> Vec3 {
        Vec3::new(
            self.yaw.sin() * self.pitch.cos(),
            self.pitch.sin(),
            -self.yaw.cos() * self.pitch.cos(),
        )
    }

    fn to_camera(&self, aspect: f32) -> Camera {
        Camera::new(self.pos, self.pos + self.fwd(), Vec3::new(0.0, 1.0, 0.0),
                    self.vfov, aspect, self.aperture, self.focus_dist)
    }

    fn autofocus(&mut self, world: &dyn Hittable) {
        if let Some(rec) = world.hit(&Ray::new(self.pos, self.fwd()), 0.001, f32::INFINITY) {
            self.focus_dist = rec.t;
        }
    }
}

// ── Sky / background ─────────────────────────────────────────────────────────

#[derive(Clone, Copy)]
enum Background {
    Solid(Color),
    Physical { sun_dir: Vec3 },
}

impl Background {
    fn eval(self, dir: Vec3) -> Color {
        match self {
            Background::Solid(c) => c,
            Background::Physical { sun_dir } => sky_color(dir, sun_dir),
        }
    }
}


// ── Scenes ───────────────────────────────────────────────────────────────────

struct DynamicSphere {
    center:      Point3,
    velocity:    Vec3,
    radius:      f32,
    mat:         Arc<dyn hittable::Material>,
    restitution: f32,
    is_static:   bool,  // no gravity/movement; acts as infinite-mass wall in collisions
}

struct SceneData {
    world:          Arc<dyn Hittable>,
    lights:         HittableList,
    background:     Background,
    name:           &'static str,
    cam_init:       SceneCameraParams,
    static_objects: Vec<Arc<dyn Hittable>>,
    dynamic:        Vec<DynamicSphere>,
    bounds:         Option<Aabb>,
    colliders:      Vec<Aabb>,
    gravity:        f32,
    settled:        bool,
    paused:         bool,
}

/// Sphere-vs-AABB collision response: if the sphere (center, radius) overlaps
/// the AABB, push it out along the axis of minimum penetration and reflect
/// velocity to point away from the box on that axis.
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
/// Skips if camera is above or below the wall's Y extent.
fn resolve_camera_aabb(pos: &mut Point3, radius: f32, bbox: &Aabb) {
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
    fn tick(&mut self) -> bool {
        if self.dynamic.is_empty() || self.paused { return false; }
        if self.settled { return false; }

        // Gravity, drag, movement, ground bounce — one pass per sphere
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

        // Wall bounds (Cornell box)
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

        // AABB colliders (Cornell box internal boxes)
        for ds in &mut self.dynamic {
            if ds.is_static { continue; }
            // Clamp speed to one diameter per step so the sphere can't tunnel
            // through an AABB wall thinner than its own diameter.
            let max_step = ds.radius * 2.0;
            let spd = ds.velocity.length();
            if spd > max_step { ds.velocity *= max_step / spd; }
            for bbox in &self.colliders {
                bounce_sphere_off_aabb(&mut ds.center, &mut ds.velocity, ds.radius, ds.restitution, bbox);
            }
        }

        // Sphere-sphere collisions; mass ∝ r³ so large spheres barely move when hit
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
                if rel_v >= 0.0 { continue; } // already separating
                let ma = a.radius * a.radius * a.radius;
                let mb = b.radius * b.radius * b.radius;
                let e  = (a.restitution + b.restitution) * 0.5;
                if a.is_static {
                    // a is immovable: only b bounces
                    b.velocity -= (1.0 + e) * rel_v * normal;
                    b.center    = a.center - normal * min_dist;
                } else if b.is_static {
                    // b is immovable: only a bounces
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

        // Settled detection: stop rebuilding once all mobile spheres are at rest
        if self.gravity > 0.0 {
            let at_rest = self.dynamic.iter().all(|ds| {
                ds.is_static || ds.velocity.length_squared() < 1e-4
            });
            if at_rest { self.settled = true; }
        }

        true
    }

    fn rebuild(&mut self) {
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

fn build_random_scene() -> SceneData {
    let mut rng = rand::thread_rng();

    // Ground sphere — never moves, stays in static_objects
    let ground: Arc<dyn Hittable> = Arc::new(Sphere::new(
        Point3::new(0.0, -1000.0, 0.0), 1000.0,
        Arc::new(Lambertian { texture: Texture::Checker {
            scale: 10.0,
            even:  Color::new(0.2, 0.3, 0.1),
            odd:   Color::new(0.9, 0.9, 0.9),
        }}),
    ));
    let static_objects = vec![ground];

    let mut dynamic: Vec<DynamicSphere> = Vec::new();

    // Three feature spheres — is_static: true so they never move but small balls bounce off them
    dynamic.push(DynamicSphere {
        center: Point3::new( 0.0, 1.0, 0.0),
        velocity: Vec3::default(),
        radius: 1.0, mat: Arc::new(Dielectric { ir: 1.5 }),
        restitution: 0.65, is_static: true,
    });
    dynamic.push(DynamicSphere {
        center: Point3::new(-4.0, 1.0, 0.0),
        velocity: Vec3::default(),
        radius: 1.0, mat: Arc::new(Lambertian { texture: Color::new(0.4, 0.2, 0.1).into() }),
        restitution: 0.35, is_static: true,
    });
    dynamic.push(DynamicSphere {
        center: Point3::new( 4.0, 1.0, 0.0),
        velocity: Vec3::default(),
        radius: 1.0, mat: Arc::new(Metal { albedo: Color::new(0.7, 0.6, 0.5), fuzz: 0.0 }),
        restitution: 0.80, is_static: true,
    });

    // Small random balls — fall from random heights, different bounciness by material
    for a in -11i32..11 {
        for b in -11i32..11 {
            let cx = a as f32 + 0.9 * rng.gen::<f32>();
            let cz = b as f32 + 0.9 * rng.gen::<f32>();
            let ground_pos = Point3::new(cx, 0.2, cz);
            // min center-to-center at rest = 1.0 (large) + 0.2 (small) = 1.2
            if (ground_pos - Point3::new( 4.0, 0.2, 0.0)).length() < 1.2 { continue; }
            if (ground_pos - Point3::new( 0.0, 0.2, 0.0)).length() < 1.2 { continue; }
            if (ground_pos - Point3::new(-4.0, 0.2, 0.0)).length() < 1.2 { continue; }
            let choose: f32 = rng.gen();
            let (mat, restitution): (Arc<dyn hittable::Material>, f32) = if choose < 0.80 {
                (Arc::new(Lambertian { texture: (Color::random(&mut rng) * Color::random(&mut rng)).into() }), 0.35)
            } else if choose < 0.95 {
                let fuzz: f32 = rng.gen_range(0.0..0.5);
                (Arc::new(Metal { albedo: Color::random_range(0.5, 1.0, &mut rng), fuzz }), 0.5 + (1.0 - fuzz) * 0.35)
            } else {
                (Arc::new(Dielectric { ir: 1.5 }), 0.65)
            };
            let center = Point3::new(cx, 0.2 + rng.gen_range(3.0..12.0), cz);
            dynamic.push(DynamicSphere {
                center,
                velocity:    Vec3::default(),
                radius:      0.2,
                mat,
                restitution,
                is_static:   false,
            });
        }
    }

    let mut list = HittableList::new();
    for obj in &static_objects { list.objects.push(Arc::clone(obj)); }
    for ds in &dynamic { list.add(Sphere::new(ds.center, ds.radius, Arc::clone(&ds.mat))); }

    SceneData {
        world:          Arc::new(BvhTree::from_list(list)) as Arc<dyn Hittable>,
        lights:         HittableList::new(),
        background:     Background::Physical { sun_dir: Vec3::new(-0.4, 0.9, -0.3).unit() },
        name:           "Random Spheres",
        cam_init:       SceneCameraParams {
            pos: Point3::new(13.0, 2.0, 3.0), lookat: Point3::new(0.0, 0.0, 0.0),
            vfov: 20.0, aperture: 0.1, focus_dist: 10.0, move_speed: 0.3,
        },
        static_objects,
        dynamic,
        bounds:   None,
        colliders:      vec![],
        gravity:        0.03,
        settled:        false,
        paused:         false,
    }
}

/// Creates a matched (world quad, light-sampler quad) pair.  Both share the
/// same geometry so PDF sampling stays in sync with the rendered geometry.
fn emissive_quad(q: Point3, u: Vec3, v: Vec3, emit: Color) -> (Quad, Quad) {
    let mat: Arc<dyn hittable::Material> = Arc::new(DiffuseLight { emit });
    let world   = Quad::new(q, u, v, Arc::clone(&mat));
    let sampler = Quad::new(q, u, v, mat);
    (world, sampler)
}

fn build_cornell_box() -> SceneData {
    let mut list = HittableList::new();
    let red:   Arc<dyn hittable::Material> = Arc::new(Lambertian { texture: Color::new(0.65, 0.05, 0.05).into() });
    let white: Arc<dyn hittable::Material> = Arc::new(Lambertian { texture: Color::new(0.73, 0.73, 0.73).into() });
    let green: Arc<dyn hittable::Material> = Arc::new(Lambertian { texture: Color::new(0.12, 0.45, 0.15).into() });
    let (world_light, sampler_light) = emissive_quad(
        Point3::new(343.0, 554.0, 332.0),
        Vec3::new(-130.0, 0.0, 0.0),
        Vec3::new(0.0, 0.0, -105.0),
        Color::new(15.0, 15.0, 15.0),
    );
    let mut lights = HittableList::new();
    lights.add(sampler_light);

    list.add(Quad::new(Point3::new(555.0, 0.0,   0.0),   Vec3::new(0.0, 555.0,  0.0), Vec3::new(0.0, 0.0,  555.0), green));
    list.add(Quad::new(Point3::new(0.0,   0.0,   0.0),   Vec3::new(0.0, 555.0,  0.0), Vec3::new(0.0, 0.0,  555.0), red));
    list.add(world_light);
    list.add(Quad::new(Point3::new(0.0,   0.0,   0.0),   Vec3::new(555.0, 0.0,  0.0), Vec3::new(0.0, 0.0,  555.0), Arc::clone(&white)));
    list.add(Quad::new(Point3::new(555.0, 555.0, 555.0), Vec3::new(-555.0, 0.0, 0.0), Vec3::new(0.0, 0.0, -555.0), Arc::clone(&white)));
    list.add(Quad::new(Point3::new(0.0,   0.0,   555.0), Vec3::new(555.0, 0.0,  0.0), Vec3::new(0.0, 555.0, 0.0),  Arc::clone(&white)));
    // Build each box at the origin, rotate, then translate into place.
    let tall = Arc::new(make_box(Point3::new(0.0,0.0,0.0), Point3::new(165.0,330.0,165.0), Arc::clone(&white))) as Arc<dyn Hittable>;
    let tall = Arc::new(Rotate::around_y(tall,  15.0)) as Arc<dyn Hittable>;
    let tall = Arc::new(Translate::new(tall, Vec3::new(265.0, 0.0, 295.0))) as Arc<dyn Hittable>;
    let tall_bbox = tall.bounding_box().unwrap();
    list.objects.push(tall);

    let short = Arc::new(make_box(Point3::new(0.0,0.0,0.0), Point3::new(165.0,165.0,165.0), white)) as Arc<dyn Hittable>;
    let short = Arc::new(Rotate::around_y(short, -18.0)) as Arc<dyn Hittable>;
    let short = Arc::new(Translate::new(short, Vec3::new(130.0, 0.0, 65.0))) as Arc<dyn Hittable>;
    let short_bbox = short.bounding_box().unwrap();
    list.objects.push(short);

    // Snapshot static geometry before adding the dynamic sphere.
    let static_objects = list.objects.clone();

    let dynamic = vec![DynamicSphere {
        center:      Point3::new(190.0, 100.0, 190.0),
        velocity:    Vec3::new(3.0, 5.0, 2.0),
        radius:      80.0,
        mat:         Arc::new(Dielectric { ir: 1.5 }),
        restitution: 1.0,
        is_static:   false,
    }];
    let bounds = Aabb::new(
        Point3::new(1.0, 1.0, 1.0),
        Point3::new(554.0, 554.0, 554.0),
    );

    // Add sphere at its initial position for the first BVH build.
    for ds in &dynamic {
        list.add(Sphere::new(ds.center, ds.radius, Arc::clone(&ds.mat)));
    }

    SceneData {
        world:          Arc::new(BvhTree::from_list(list)) as Arc<dyn Hittable>,
        lights,
        background:     Background::Solid(Color::default()),
        name:           "Cornell Box",
        cam_init:       SceneCameraParams {
            pos: Point3::new(278.0, 278.0, -800.0), lookat: Point3::new(278.0, 278.0, 0.0),
            vfov: 40.0, aperture: 0.0, focus_dist: 10.0, move_speed: 8.0,
        },
        static_objects,
        dynamic,
        bounds:    Some(bounds),
        colliders:     vec![tall_bbox, short_bbox],
        gravity:       0.0,
        settled:       false,
        paused:        false,
    }
}

// ── Output ───────────────────────────────────────────────────────────────────

fn build_mesh_scene() -> SceneData {
    const MODEL_PATH: &str = "assets/model.obj";

    let mat: Arc<dyn hittable::Material> =
        Arc::new(Lambertian { texture: Color::new(0.8, 0.8, 0.75).into() });

    // Try loading the mesh; print bounding box so the user knows the scale.
    let (mesh_bvh, cam_init) = match load_obj(MODEL_PATH, 1.0, mat) {
        Err(e) => {
            println!("  Could not load '{MODEL_PATH}': {e}");
            println!("  Drop any OBJ file there and press 3 to view it.");
            // Fallback: shiny sphere so the scene isn't blank.
            let mut list = HittableList::new();
            list.add(Sphere::new(Point3::new(0.0, 1.0, 0.0), 1.0,
                Arc::new(Metal { albedo: Color::new(0.8, 0.85, 0.9), fuzz: 0.02 })));
            (BvhTree::from_list(list), SceneCameraParams {
                pos: Point3::new(0.0, 2.0, 6.0), lookat: Point3::new(0.0, 1.0, 0.0),
                vfov: 40.0, aperture: 0.0, focus_dist: 6.0, move_speed: 0.3,
            })
        }
        Ok(mesh_list) => {
            // Auto-fit camera to the mesh bounding box.
            let bb = mesh_list.bounding_box().unwrap();
            let cx = (bb.min.x + bb.max.x) * 0.5;
            let cy = (bb.min.y + bb.max.y) * 0.5;
            let cz = (bb.min.z + bb.max.z) * 0.5;
            let size = (bb.max.x - bb.min.x).max(bb.max.y - bb.min.y).max(bb.max.z - bb.min.z);
            println!("  center ({cx:.2}, {cy:.2}, {cz:.2}), extent {size:.2}");
            let cam_pos = Point3::new(cx, cy + size * 0.3, cz + size * 1.8);
            (BvhTree::from_list(mesh_list), SceneCameraParams {
                pos: cam_pos, lookat: Point3::new(cx, cy, cz),
                vfov: 40.0, aperture: 0.0, focus_dist: size * 2.0, move_speed: size * 0.02,
            })
        }
    };

    // Ground + overhead light + mesh
    let mut list = HittableList::new();
    list.add(Quad::new(
        Point3::new(-500.0, 0.0, 500.0), Vec3::new(1000.0, 0.0, 0.0), Vec3::new(0.0, 0.0, -1000.0),
        Arc::new(Lambertian { texture: Texture::Checker {
            scale: 1.0, even: Color::new(0.15, 0.15, 0.15), odd: Color::new(0.85, 0.85, 0.85),
        }}),
    ));
    let (overhead_world, overhead_sampler) = emissive_quad(
        Point3::new(-200.0, 500.0, -200.0),
        Vec3::new(400.0, 0.0, 0.0),
        Vec3::new(0.0, 0.0, 400.0),
        Color::new(6.0, 6.0, 6.0),
    );
    let mut lights = HittableList::new();
    lights.add(overhead_sampler);
    list.add(overhead_world);
    list.add(mesh_bvh);

    SceneData {
        world:          Arc::new(BvhTree::from_list(list)) as Arc<dyn Hittable>,
        lights,
        background:     Background::Solid(Color::new(0.05, 0.07, 0.12)),
        name:           "Mesh",
        cam_init,
        static_objects: vec![],
        dynamic:        vec![],
        bounds:         None,
        colliders:      vec![],
        gravity:        0.0,
        settled:        false,
        paused:         false,
    }
}

// ── Next Week final scene ─────────────────────────────────────────────────────

fn build_nextweek_scene() -> SceneData {
    let mut rng = SmallRng::seed_from_u64(42);
    let mut list = HittableList::new();

    // 20×20 ground boxes with a greenish Lambertian material
    let ground_mat: Arc<dyn hittable::Material> =
        Arc::new(Lambertian { texture: Color::new(0.48, 0.83, 0.53).into() });
    let mut ground_boxes = HittableList::new();
    for i in 0..20 {
        for j in 0..20 {
            let w  = 100.0f32;
            let x0 = -1000.0 + i as f32 * w;
            let z0 = -1000.0 + j as f32 * w;
            let y1: f32 = rng.gen_range(1.0..101.0);
            ground_boxes.add(make_box(
                Point3::new(x0, 0.0, z0),
                Point3::new(x0 + w, y1, z0 + w),
                Arc::clone(&ground_mat),
            ));
        }
    }
    list.add(BvhTree::from_list(ground_boxes));

    // Quad light
    let (light_world, light_sampler) = emissive_quad(
        Point3::new(123.0, 554.0, 147.0),
        Vec3::new(300.0,   0.0,   0.0),
        Vec3::new(  0.0,   0.0, 265.0),
        Color::new(7.0, 7.0, 7.0),
    );
    let mut lights = HittableList::new();
    lights.add(light_sampler);
    list.add(light_world);

    // Moving sphere with motion blur (t=0→1, center drifts +30 on X)
    let c0 = Point3::new(400.0, 400.0, 200.0);
    list.add(Sphere::new_moving(
        c0, c0 + Vec3::new(30.0, 0.0, 0.0), 50.0,
        Arc::new(Lambertian { texture: Color::new(0.7, 0.3, 0.1).into() }),
    ));

    // Glass sphere
    list.add(Sphere::new(
        Point3::new(260.0, 150.0, 45.0), 50.0,
        Arc::new(Dielectric { ir: 1.5 }),
    ));

    // Metal sphere
    list.add(Sphere::new(
        Point3::new(0.0, 150.0, 145.0), 50.0,
        Arc::new(Metal { albedo: Color::new(0.8, 0.8, 0.9), fuzz: 1.0 }),
    ));

    // Blue fog sphere: dielectric shell + constant-medium interior
    let fog_boundary: Arc<dyn Hittable> = Arc::new(Sphere::new(
        Point3::new(360.0, 150.0, 145.0), 70.0,
        Arc::new(Dielectric { ir: 1.5 }),
    ));
    list.objects.push(Arc::clone(&fog_boundary));
    list.add(ConstantMedium::new(fog_boundary, 0.2, Color::new(0.2, 0.4, 0.9)));

    // Thin global mist
    let mist_boundary: Arc<dyn Hittable> = Arc::new(Sphere::new(
        Point3::new(0.0, 0.0, 0.0), 5000.0,
        Arc::new(Dielectric { ir: 1.5 }),
    ));
    list.add(ConstantMedium::new(mist_boundary, 0.0001, Color::new(1.0, 1.0, 1.0)));

    // Earth sphere — tries assets/earthmap.jpg, falls back to blue/green checker
    let earth_tex = Texture::load("assets/earthmap.jpg")
        .unwrap_or_else(|_| Texture::Checker {
            scale: 2.0,
            even:  Color::new(0.1, 0.3, 0.7),
            odd:   Color::new(0.2, 0.7, 0.2),
        });
    list.add(Sphere::new(
        Point3::new(400.0, 200.0, 400.0), 100.0,
        Arc::new(Lambertian { texture: earth_tex }),
    ));

    // Marble/noise sphere (Perlin turbulence)
    let perlin = Arc::new(Perlin::new(&mut rng));
    list.add(Sphere::new(
        Point3::new(220.0, 280.0, 300.0), 80.0,
        Arc::new(Lambertian { texture: Texture::Noise { perlin, scale: 0.2 } }),
    ));

    // Cluster of 1000 small white spheres, rotated 15° and translated into place
    let white: Arc<dyn hittable::Material> =
        Arc::new(Lambertian { texture: Color::new(0.73, 0.73, 0.73).into() });
    let mut cluster = HittableList::new();
    for _ in 0..1000 {
        cluster.add(Sphere::new(
            Point3::new(
                rng.gen_range(0.0..165.0),
                rng.gen_range(0.0..165.0),
                rng.gen_range(0.0..165.0),
            ),
            10.0,
            Arc::clone(&white),
        ));
    }
    let cluster: Arc<dyn Hittable> = Arc::new(BvhTree::from_list(cluster));
    let cluster: Arc<dyn Hittable> = Arc::new(Rotate::around_y(cluster, 15.0));
    let cluster: Arc<dyn Hittable> = Arc::new(Translate::new(cluster, Vec3::new(-100.0, 270.0, 395.0)));
    list.objects.push(cluster);

    SceneData {
        world:          Arc::new(BvhTree::from_list(list)) as Arc<dyn Hittable>,
        lights,
        background:     Background::Solid(Color::default()),
        name:           "Next Week",
        cam_init:       SceneCameraParams {
            pos:        Point3::new(478.0, 278.0, -600.0),
            lookat:     Point3::new(278.0, 278.0,    0.0),
            vfov:       40.0,
            aperture:   0.0,
            focus_dist: 10.0,
            move_speed: 8.0,
        },
        static_objects: vec![],
        dynamic:        vec![],
        bounds:         None,
        colliders:      vec![],
        gravity:        0.0,
        settled:        true,
        paused:         false,
    }
}


/// Physically-inspired sky model: blue zenith shading to warm horizon, soft Mie glow.
/// No explicit sun disc — the sun contributes through the bright Mie halo instead,
/// avoiding the need for sphere-light NEE sampling.
fn sky_color(dir: Vec3, sun_dir: Vec3) -> Color {
    let d        = dir.unit();
    let sun      = sun_dir.unit();
    let sun_elev = sun.y.clamp(0.0, 1.0);          // 0 = sunset, 1 = noon
    let cos_a    = d.dot(sun).max(0.0);             // angle toward sun
    let t        = d.y.max(0.0).powf(0.4);          // altitude blend (0 = horizon, 1 = zenith)

    // Zenith: deep blue, dimmer at sunset
    let zenith  = Color::new(0.08, 0.22, 0.75) * (0.4 + 0.6 * sun_elev);
    // Horizon: warm orange-pink at sunset, cool blue-white at noon
    let horizon = Color::new(0.70, 0.55, 0.35) * (1.0 - sun_elev)
                + Color::new(0.65, 0.78, 0.92) *  sun_elev;
    let sky = zenith * t + horizon * (1.0 - t);

    // Mie scattering: broad soft glow in the direction of the sun
    let mie = Color::new(1.0, 0.85, 0.60) * cos_a.powf(8.0) * 0.8 * sun_elev;

    if d.y < 0.0 {
        // Below horizon: fade to near-black ground
        sky * (1.0 + d.y * 5.0).max(0.0)
    } else {
        sky + mie
    }
}

fn aces(x: f32) -> f32 {
    let x = x.max(0.0);
    (x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14)
}

#[inline]
fn tone_map(c: Color, scale: f32) -> [u8; 3] {
    let f = |x: f32| (aces(x * scale).sqrt().clamp(0.0, 0.999) * 256.0) as u8;
    [f(c.x), f(c.y), f(c.z)]
}

fn to_rgb_u32(c: Color, scale: f32) -> u32 {
    let [r, g, b] = tone_map(c, scale);
    (r as u32) << 16 | (g as u32) << 8 | (b as u32)
}

fn save_png(accumulator: &[Color], pixel_samples: &[u32], samples: u32, scene_name: &str, width: u32, height: u32, exposure: f32) {
    if samples == 0 { return; }
    let slug = scene_name.to_lowercase().replace(' ', "_");
    let path = format!("render_{}_{:04}spp.png", slug, samples);
    let img = ImageBuffer::from_fn(width, height, |x, y| {
        let i = (y * width + x) as usize;
        let [r, g, b] = tone_map(accumulator[i], exposure / pixel_samples[i].max(1) as f32);
        Rgb([r, g, b])
    });
    match img.save(&path) {
        Ok(_)  => println!("Saved {path}"),
        Err(e) => eprintln!("Save failed: {e}"),
    }
}

// ── Tile renderer ────────────────────────────────────────────────────────────

/// Render one sample pass into `scratch` in parallel.
/// Pass an empty slice for `conv` to disable adaptive-sampling skipping.
#[allow(clippy::too_many_arguments)]
fn render_tiles(
    scratch:    &mut [Color],
    sample_idx: u32,
    width:      u32,
    height:     u32,
    camera:     &Camera,
    world:      &dyn Hittable,
    background: Background,
    lights:     &HittableList,
    conv:       &[bool],
) {
    let w        = width  as usize;
    let w_denom  = (width  - 1).max(1) as f32;
    let h_denom  = (height - 1).max(1) as f32;
    let adaptive = !conv.is_empty();

    scratch.par_iter_mut().enumerate().for_each(|(i, out)| {
        *out = if adaptive && conv[i] {
            Color::default()
        } else {
            let row = i / w;
            let col = i % w;
            let mut rng = SmallRng::seed_from_u64(
                (i as u64).wrapping_mul(6364136223846793005)
                    ^ (sample_idx as u64).wrapping_mul(0x9E3779B97F4A7C15),
            );
            let ray_y = height - 1 - row as u32;
            let u = (col as f32 + rng.gen::<f32>()) / w_denom;
            let v = (ray_y as f32 + rng.gen::<f32>()) / h_denom;
            ray_color(&camera.get_ray(u, v, &mut rng), world, background, lights, &mut rng)
        };
    });
}

// ── Bench ────────────────────────────────────────────────────────────────────

fn bench_scene(scene: &SceneData, scratch: &mut Vec<Color>, samples: u32) -> Duration {
    let aspect = WIDTH as f32 / HEIGHT as f32;
    let mut cam = CameraState::from_params(&scene.cam_init);
    cam.autofocus(&*scene.world);
    let camera  = cam.to_camera(aspect);
    let world   = scene.world.as_ref();
    let bg      = scene.background;
    scratch.resize((WIDTH * HEIGHT) as usize, Color::default());

    let t0 = Instant::now();
    for s in 0..samples {
        render_tiles(scratch, s, WIDTH, HEIGHT, &camera, world, bg, &scene.lights, &[]);
    }
    t0.elapsed()
}

fn run_bench() {
    const WARMUP:   u32 = 4;
    const SAMPLES:  u32 = 128;

    let threads = rayon::current_num_threads();
    println!("rustracer benchmark — {}×{}  {} threads  {} samples ({} warmup)\n",
        WIDTH, HEIGHT, threads, SAMPLES, WARMUP);

    let header = format!("{:<20}  {:>7}  {:>9}  {:>11}  {:>8}",
        "Scene", "Samples", "Total", "ms/sample", "Mpx/s");
    println!("{}", header);
    println!("{}", "─".repeat(header.len()));

    let builders: &[fn() -> SceneData] = &[
        build_random_scene,
        build_cornell_box,
        build_mesh_scene,
        build_nextweek_scene,
    ];

    let mut scratch: Vec<Color> = Vec::new();

    for build in builders {
        let scene = build();
        print!("  {:<18}  building…\r", scene.name);
        let _ = std::io::Write::flush(&mut std::io::stdout());

        // warmup — brings BVH and scene data into CPU caches
        bench_scene(&scene, &mut scratch, WARMUP);

        let elapsed   = bench_scene(&scene, &mut scratch, SAMPLES);
        let secs      = elapsed.as_secs_f64();
        let ms_per_s  = secs * 1000.0 / SAMPLES as f64;
        let mpx_s     = (WIDTH as f64 * HEIGHT as f64 * SAMPLES as f64) / (secs * 1e6);

        println!("  {:<18}  {:>7}  {:>7.3}s  {:>9.1} ms  {:>7.2}",
            scene.name, SAMPLES, secs, ms_per_s, mpx_s);
    }

    println!("\nDone.");
}

// ── Main loop ────────────────────────────────────────────────────────────────

fn main() {
    if std::env::args().any(|a| a == "--bench") {
        run_bench();
        return;
    }

    println!("Building scenes…");
    println!("Scene 1: Random Spheres");
    let s1 = build_random_scene();
    println!("Scene 2: Cornell Box");
    let s2 = build_cornell_box();
    println!("Scene 3: Mesh");
    let s3 = build_mesh_scene();
    println!("Scene 4: Next Week");
    let s4 = build_nextweek_scene();
    let mut scenes = [s1, s2, s3, s4];
    println!("Ready.  [1/2/3/4] scene  [F] free camera  WASD+mouse  Space/Shift up/down");
    println!("        [P] save  [[] apt  [,.] fov  [-=] exp  [arrows] sun  [C] reset cam");
    println!("        [T] adaptive  [Enter] pause  [R] restart (scene 1)  [Esc] quit");

    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_title("Ray Tracer")
        .with_inner_size(PhysicalSize::new(WIDTH, HEIGHT))
        .build(&event_loop)
        .unwrap();

    let _context = unsafe { Context::new(&window) }.unwrap();
    let mut surface = unsafe { Surface::new(&_context, &window) }.unwrap();
    surface.resize(NonZeroU32::new(WIDTH).unwrap(), NonZeroU32::new(HEIGHT).unwrap()).unwrap();

    let mut win_w       = WIDTH;
    let mut win_h       = HEIGHT;
    let mut scene_idx   = 0usize;
    let mut cam_state   = CameraState::from_params(&scenes[scene_idx].cam_init);
    cam_state.autofocus(scenes[scene_idx].world.as_ref());
    let mut camera      = cam_state.to_camera(win_w as f32 / win_h as f32);
    let mut free_cam      = false;
    let mut adaptive      = false;
    let mut accumulator   = vec![Color::default(); (win_w * win_h) as usize];
    let mut scratch       = vec![Color::default(); (win_w * win_h) as usize];
    let mut pixel_samples = vec![0u32;             (win_w * win_h) as usize];
    let mut welford_mean  = vec![Color::default(); (win_w * win_h) as usize];
    let mut welford_m2    = vec![Color::default(); (win_w * win_h) as usize];
    let mut converged     = vec![false;            (win_w * win_h) as usize];
    let mut samples       = 0u32;
    let mut pressed           = std::collections::HashSet::<VirtualKeyCode>::new();
    let mut exposure          = 1.0f32;
    let mut cam_dirty         = false;
    let mut pending_autofocus = false;
    let mut last_title_update = Instant::now();
    let mut last_frame_time   = Instant::now();
    let mut physics_accum     = Duration::ZERO;

    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Poll;

        macro_rules! reset_accum {
            () => {
                accumulator.fill(Color::default()); pixel_samples.fill(0);
                welford_mean.fill(Color::default()); welford_m2.fill(Color::default());
                converged.fill(false); samples = 0;
            }
        }

        match event {
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::CloseRequested => *control_flow = ControlFlow::Exit,

                WindowEvent::Resized(size) => {
                    let new_w = size.width.max(1);
                    let new_h = size.height.max(1);
                    if new_w != win_w || new_h != win_h {
                        win_w = new_w;
                        win_h = new_h;
                        surface.resize(NonZeroU32::new(win_w).unwrap(), NonZeroU32::new(win_h).unwrap()).unwrap();
                        let n = (win_w * win_h) as usize;
                        accumulator.resize(n, Color::default());   accumulator.fill(Color::default());
                        scratch.resize(n, Color::default());        scratch.fill(Color::default());
                        pixel_samples.resize(n, 0);                pixel_samples.fill(0);
                        welford_mean.resize(n, Color::default());   welford_mean.fill(Color::default());
                        welford_m2.resize(n, Color::default());     welford_m2.fill(Color::default());
                        converged.resize(n, false);                 converged.fill(false);
                        samples = 0;
                        cam_dirty = true;
                    }
                }

                WindowEvent::KeyboardInput {
                    input: KeyboardInput { state, virtual_keycode: Some(key), .. }, ..
                } => {
                    match state {
                        ElementState::Pressed => {
                            pressed.insert(key);
                            match key {
                                VirtualKeyCode::Escape => {
                                    if free_cam {
                                        free_cam = false;
                                        window.set_cursor_grab(CursorGrabMode::None).ok();
                                        window.set_cursor_visible(true);
                                    } else {
                                        *control_flow = ControlFlow::Exit;
                                    }
                                }
                                VirtualKeyCode::F => {
                                    free_cam = !free_cam;
                                    if free_cam {
                                        window.set_cursor_grab(CursorGrabMode::Locked)
                                            .or_else(|_| window.set_cursor_grab(CursorGrabMode::Confined))
                                            .ok();
                                        window.set_cursor_visible(false);
                                        println!("Free camera ON — Esc to release mouse");
                                    } else {
                                        window.set_cursor_grab(CursorGrabMode::None).ok();
                                        window.set_cursor_visible(true);
                                    }
                                }
                                VirtualKeyCode::Key1 | VirtualKeyCode::Key2 | VirtualKeyCode::Key3 | VirtualKeyCode::Key4 => {
                                    let idx = match key { VirtualKeyCode::Key2 => 1, VirtualKeyCode::Key3 => 2, VirtualKeyCode::Key4 => 3, _ => 0 };
                                    scene_idx = idx;
                                    cam_state = CameraState::from_params(&scenes[idx].cam_init);
                                    reset_accum!();
                                    cam_dirty = true;
                                    pending_autofocus = true;
                                }
                                VirtualKeyCode::P => save_png(&accumulator, &pixel_samples, samples, scenes[scene_idx].name, win_w, win_h, exposure),
                                VirtualKeyCode::LBracket => {
                                    cam_state.aperture = (cam_state.aperture - 0.025).max(0.0);
                                    cam_dirty = true;
                                }
                                VirtualKeyCode::RBracket => {
                                    cam_state.aperture += 0.025;
                                    cam_dirty = true;
                                }
                                VirtualKeyCode::Comma => {
                                    cam_state.vfov = (cam_state.vfov - 5.0).max(5.0);
                                    cam_dirty = true;
                                }
                                VirtualKeyCode::Period => {
                                    cam_state.vfov = (cam_state.vfov + 5.0).min(120.0);
                                    cam_dirty = true;
                                }
                                VirtualKeyCode::Minus => {
                                    exposure = (exposure * 0.8).max(0.125);
                                    window.request_redraw();
                                }
                                VirtualKeyCode::Equals => {
                                    exposure = (exposure / 0.8).min(8.0);
                                    window.request_redraw();
                                }
                                VirtualKeyCode::Left | VirtualKeyCode::Right => {
                                    if let Background::Physical { sun_dir } = &mut scenes[scene_idx].background {
                                        let step = if key == VirtualKeyCode::Right { 0.1f32 } else { -0.1f32 };
                                        let (s, c) = step.sin_cos();
                                        *sun_dir = Vec3::new(
                                            sun_dir.x * c - sun_dir.z * s,
                                            sun_dir.y,
                                            sun_dir.x * s + sun_dir.z * c,
                                        ).unit();
                                        accumulator.fill(Color::default()); pixel_samples.fill(0);
                                        welford_mean.fill(Color::default()); welford_m2.fill(Color::default());
                                        converged.fill(false); samples = 0;
                                    }
                                }
                                VirtualKeyCode::Up | VirtualKeyCode::Down => {
                                    if let Background::Physical { sun_dir } = &mut scenes[scene_idx].background {
                                        let step = if key == VirtualKeyCode::Up { 0.1f32 } else { -0.1f32 };
                                        let el = sun_dir.y.asin();
                                        let new_el = (el + step).clamp(-10f32.to_radians(), 85f32.to_radians());
                                        let horiz = Vec3::new(sun_dir.x, 0.0, sun_dir.z).length().max(1e-6);
                                        let scale = new_el.cos() / horiz;
                                        *sun_dir = Vec3::new(sun_dir.x * scale, new_el.sin(), sun_dir.z * scale);
                                        accumulator.fill(Color::default()); pixel_samples.fill(0);
                                        welford_mean.fill(Color::default()); welford_m2.fill(Color::default());
                                        converged.fill(false); samples = 0;
                                    }
                                }
                                VirtualKeyCode::C => {
                                    cam_state = CameraState::from_params(&scenes[scene_idx].cam_init);
                                    cam_dirty = true;
                                    pending_autofocus = true;
                                }
                                VirtualKeyCode::T => {
                                    adaptive = !adaptive;
                                    accumulator.fill(Color::default()); pixel_samples.fill(0);
                                    welford_mean.fill(Color::default()); welford_m2.fill(Color::default());
                                    converged.fill(false); samples = 0;
                                }
                                VirtualKeyCode::Return => {
                                    let pausing = !scenes[scene_idx].paused;
                                    scenes[scene_idx].paused = pausing;
                                    if pausing {
                                        scenes[scene_idx].rebuild();
                                        reset_accum!();
                                    }
                                }
                                VirtualKeyCode::R => {
                                    if scene_idx == 0 {
                                        scenes[0] = build_random_scene();
                                        cam_dirty = true;
                                    }
                                }
                                _ => {}
                            }
                        }
                        ElementState::Released => { pressed.remove(&key); }
                    }
                }
                _ => {}
            }

            Event::DeviceEvent { event: DeviceEvent::MouseMotion { delta: (dx, dy) }, .. } => {
                if free_cam {
                    cam_state.yaw   += dx as f32 * MOUSE_SENS;
                    cam_state.pitch  = (cam_state.pitch - dy as f32 * MOUSE_SENS)
                        .clamp(-89f32.to_radians(), 89f32.to_radians());
                    cam_dirty = true;
                    pending_autofocus = true;
                }
            }

            Event::MainEventsCleared => {
                if free_cam {
                    let spd   = cam_state.move_speed;
                    let fwd_h = cam_state.forward_horiz();
                    let rgt   = cam_state.right_horiz();
                    let mut moved = false;
                    if pressed.contains(&VirtualKeyCode::W)      { cam_state.pos += fwd_h * spd; moved = true; }
                    if pressed.contains(&VirtualKeyCode::S)      { cam_state.pos -= fwd_h * spd; moved = true; }
                    if pressed.contains(&VirtualKeyCode::A)      { cam_state.pos -= rgt   * spd; moved = true; }
                    if pressed.contains(&VirtualKeyCode::D)      { cam_state.pos += rgt   * spd; moved = true; }
                    if pressed.contains(&VirtualKeyCode::Space)  { cam_state.pos.y += spd;       moved = true; }
                    if pressed.contains(&VirtualKeyCode::LShift) { cam_state.pos.y -= spd;       moved = true; }
                    if moved {
                        for bbox in &scenes[scene_idx].colliders {
                            resolve_camera_aabb(&mut cam_state.pos, CAM_RADIUS, bbox);
                        }
                        cam_dirty = true;
                        pending_autofocus = true;
                    }
                }

                if cam_dirty {
                    if pending_autofocus {
                        cam_state.autofocus(scenes[scene_idx].world.as_ref());
                        pending_autofocus = false;
                    }
                    camera = cam_state.to_camera(win_w as f32 / win_h as f32);
                    reset_accum!();
                    cam_dirty = false;
                }

                let now = Instant::now();
                let frame_dt = now.duration_since(last_frame_time);
                // Cap catch-up to 100 ms so a paused/minimised window doesn't
                // run dozens of physics ticks on resume.
                physics_accum += frame_dt.min(Duration::from_millis(100));
                last_frame_time = now;
                let mut physics_ticked = false;
                while physics_accum >= PHYSICS_DT {
                    physics_ticked |= scenes[scene_idx].tick();
                    physics_accum -= PHYSICS_DT;
                }
                if physics_ticked { reset_accum!(); }

                if last_title_update.elapsed() >= TITLE_INTERVAL {
                    let scene    = &scenes[scene_idx];
                    let cam_hint = if free_cam { "FREE CAM [Esc] release" } else { "[F] Free Camera" };
                    let motion_hint = if scene.gravity > 0.0 {
                        if scene.settled     { "  settled — [R] restart" }
                        else if scene.paused { "  PAUSED — [Enter] resume  [R] restart" }
                        else                 { "  [Enter] pause  [R] restart" }
                    } else if !scene.dynamic.is_empty() {
                        if scene.paused { "  PAUSED — [Enter] resume" } else { "" }
                    } else { "" };
                    let adaptive_hint = if adaptive {
                        let pct = converged.iter().filter(|&&c| c).count() * 100
                                / converged.len().max(1);
                        format!("  ADAPTIVE {pct}% conv [T] off")
                    } else {
                        "  [T] adaptive".to_string()
                    };
                    let spp_label = if adaptive && samples > 0 {
                        let min_spp = pixel_samples.iter().copied().min().unwrap_or(0);
                        format!("{min_spp}–{samples} spp")
                    } else {
                        format!("{samples} spp")
                    };
                    let sun_hint = if let Background::Physical { sun_dir } = scene.background {
                        format!("  sun {:.0}° [arrows]", sun_dir.y.asin().to_degrees())
                    } else { String::new() };
                    window.set_title(&format!(
                        "Ray Tracer — {} — {}  |  {}  [P] save  [[] apt {:.2}  [,.] fov {:.0}°  [-=] exp {:.2}x{}{}{}",
                        scene.name, spp_label, cam_hint,
                        cam_state.aperture, cam_state.vfov, exposure,
                        sun_hint, adaptive_hint, motion_hint,
                    ));
                    last_title_update = Instant::now();
                }

                if samples < MAX_SAMPLES {
                    let scene  = &scenes[scene_idx];
                    let bg     = scene.background;
                    let conv   = if adaptive { converged.as_slice() } else { &[] };

                    render_tiles(&mut scratch, samples, win_w, win_h, &camera,
                                 scene.world.as_ref(), bg, &scene.lights, conv);

                    for i in 0..(win_w * win_h) as usize {
                        if adaptive && converged[i] { continue; }
                        let s = scratch[i];
                        accumulator[i] += s;
                        pixel_samples[i] += 1;
                        if adaptive {
                            let n = pixel_samples[i] as f32;
                            let delta = s - welford_mean[i];
                            welford_mean[i] += delta / n;
                            welford_m2[i]   += delta * (s - welford_mean[i]);
                            if pixel_samples[i] >= ADAPTIVE_MIN_SAMPLES {
                                let var      = welford_m2[i] / (n - 1.0);
                                let mean_lum = welford_mean[i].x * 0.2126
                                             + welford_mean[i].y * 0.7152
                                             + welford_mean[i].z * 0.0722;
                                let var_lum  = var.x * 0.2126 + var.y * 0.7152 + var.z * 0.0722;
                                // σ < threshold × (μ + threshold): relative criterion with floor
                                // so genuinely-black pixels converge but dark indirect-lit ones don't lock in early
                                if var_lum.sqrt() < ADAPTIVE_THRESHOLD * (mean_lum + ADAPTIVE_THRESHOLD) {
                                    converged[i] = true;
                                }
                            }
                        }
                    }
                    samples += 1;

                    window.request_redraw();
                } else {
                    *control_flow = ControlFlow::Wait;
                }
            }

            Event::RedrawRequested(_) => {
                let mut buffer = surface.buffer_mut().unwrap();
                for (i, &color) in accumulator.iter().enumerate() {
                    buffer[i] = to_rgb_u32(color, exposure / pixel_samples[i].max(1) as f32);
                }
                buffer.present().unwrap();
            }

            _ => {}
        }
    });
}
