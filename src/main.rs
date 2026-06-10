mod aabb;
mod bvh;
mod vec3;
mod ray;
mod hittable;
mod texture;
mod material;
mod sphere;
mod quad;
mod transform;
mod mesh;
mod camera;
mod light;

use aabb::Aabb;
use vec3::{Color, Point3, Vec3};
use ray::Ray;
use hittable::{Hittable, HittableList};
use texture::Texture;
use material::{Dielectric, DiffuseLight, Lambertian, Metal};
use sphere::Sphere;
use quad::{Quad, make_box};
use transform::{RotateY, Translate};
use mesh::load_obj;
use camera::Camera;
use bvh::BvhTree;
use light::Light;

use winit::{
    dpi::PhysicalSize,
    event::{DeviceEvent, ElementState, Event, KeyboardInput, VirtualKeyCode, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::{CursorGrabMode, WindowBuilder},
};
use softbuffer::{Context, Surface};
use std::num::NonZeroU32;
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
const MAX_LUMINANCE: f32 = 10.0;

fn ray_color(r: &Ray, world: &dyn Hittable, background: Option<Color>, lights: &[Light], rng: &mut impl Rng) -> Color {
    let mut throughput    = Color::new(1.0, 1.0, 1.0);
    let mut color         = Color::default();
    let mut ray           = *r;
    let mut specular_prev = true; // treat camera as specular so first-hit emitters are visible

    for depth in 0..MAX_DEPTH {
        match world.hit(&ray, 0.001, f32::INFINITY) {
            None => {
                let bg = background.unwrap_or_else(|| {
                    let t = 0.5 * (ray.direction.unit().y + 1.0);
                    (1.0 - t) * Color::new(1.0, 1.0, 1.0) + t * Color::new(0.5, 0.7, 1.0)
                });
                color += throughput * bg;
                break;
            }
            Some(rec) => {
                // Count emitted light only from specular/primary paths; diffuse paths get direct
                // light via the explicit NEE shadow ray below to avoid double-counting.
                if specular_prev {
                    color += throughput * rec.mat.emitted();
                }

                let Some(sr) = rec.mat.scatter(&ray, &rec, rng) else { break; };
                let mut attenuation = sr.attenuation;

                if let Some(albedo) = sr.albedo {
                    // Diffuse surface: sample each light explicitly.
                    for light in lights {
                        color += throughput
                            * light.sample_contribution(rec.p, rec.normal, albedo, world, rng);
                    }
                    specular_prev = false;
                } else {
                    specular_prev = true;
                }

                if depth >= 2 {
                    let survive = attenuation.x.max(attenuation.y).max(attenuation.z);
                    if survive <= 0.0 || rng.gen::<f32>() >= survive { break; }
                    attenuation /= survive;
                }
                throughput *= attenuation;
                ray = sr.ray;
            }
        }
    }

    let lum = color.x * 0.2126 + color.y * 0.7152 + color.z * 0.0722;
    if lum > MAX_LUMINANCE { color = color * (MAX_LUMINANCE / lum); }
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
    lights:         Vec<Light>,
    background:     Option<Color>,
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
fn bounce_sphere_off_aabb(center: &mut Point3, velocity: &mut Vec3, radius: f32, bbox: &Aabb) {
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
        if dx_lo < dx_hi { center.x = lo.x; velocity.x = -velocity.x.abs(); }
        else              { center.x = hi.x; velocity.x =  velocity.x.abs(); }
    } else if min_y <= min_x && min_y <= min_z {
        if dy_lo < dy_hi { center.y = lo.y; velocity.y = -velocity.y.abs(); }
        else              { center.y = hi.y; velocity.y =  velocity.y.abs(); }
    } else {
        if dz_lo < dz_hi { center.z = lo.z; velocity.z = -velocity.z.abs(); }
        else              { center.z = hi.z; velocity.z =  velocity.z.abs(); }
    }
}

impl SceneData {
    fn tick(&mut self) -> bool {
        if self.dynamic.is_empty() || self.paused { return false; }
        if self.settled { return false; }

        // Gravity + air drag (skipped for Cornell-style constant-velocity scenes)
        if self.gravity > 0.0 {
            for ds in &mut self.dynamic {
                if ds.is_static { continue; }
                ds.velocity.y -= self.gravity;
                ds.velocity   *= 0.995;
            }
        }

        // Movement
        for ds in &mut self.dynamic {
            if !ds.is_static { ds.center += ds.velocity; }
        }

        // Ground plane bounce (y = 0)
        if self.gravity > 0.0 {
            for ds in &mut self.dynamic {
                if ds.is_static { continue; }
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
            for bbox in &self.colliders {
                bounce_sphere_off_aabb(&mut ds.center, &mut ds.velocity, ds.radius, bbox);
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
                if b.is_static {
                    // Infinite-mass wall: only a bounces
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
                ds.is_static || (ds.velocity.length_squared() < 1e-4 && ds.center.y - ds.radius < 0.05)
            });
            if at_rest { self.settled = true; }
        }

        true
    }

    fn rebuild(&mut self) {
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
        center: Point3::new( 0.0, 1.0, 0.0), velocity: Vec3::default(),
        radius: 1.0, mat: Arc::new(Dielectric { ir: 1.5 }),
        restitution: 0.65, is_static: true,
    });
    dynamic.push(DynamicSphere {
        center: Point3::new(-4.0, 1.0, 0.0), velocity: Vec3::default(),
        radius: 1.0, mat: Arc::new(Lambertian { texture: Color::new(0.4, 0.2, 0.1).into() }),
        restitution: 0.35, is_static: true,
    });
    dynamic.push(DynamicSphere {
        center: Point3::new( 4.0, 1.0, 0.0), velocity: Vec3::default(),
        radius: 1.0, mat: Arc::new(Metal { albedo: Color::new(0.7, 0.6, 0.5), fuzz: 0.0 }),
        restitution: 0.80, is_static: true,
    });

    // Small random balls — fall from random heights, different bounciness by material
    for a in -11i32..11 {
        for b in -11i32..11 {
            let cx = a as f32 + 0.9 * rng.gen::<f32>();
            let cz = b as f32 + 0.9 * rng.gen::<f32>();
            if (Point3::new(cx, 0.2, cz) - Point3::new(4.0, 0.2, 0.0)).length() <= 0.9 { continue; }
            let choose: f32 = rng.gen();
            let (mat, restitution): (Arc<dyn hittable::Material>, f32) = if choose < 0.8 {
                (Arc::new(Lambertian { texture: (Color::random(&mut rng) * Color::random(&mut rng)).into() }), 0.35)
            } else if choose < 0.95 {
                let fuzz: f32 = rng.gen_range(0.0..0.5);
                // smoother metal = more elastic
                (Arc::new(Metal { albedo: Color::random_range(0.5, 1.0, &mut rng), fuzz }), 0.5 + (1.0 - fuzz) * 0.35)
            } else {
                (Arc::new(Dielectric { ir: 1.5 }), 0.65)
            };
            dynamic.push(DynamicSphere {
                center:      Point3::new(cx, 0.2 + rng.gen_range(3.0..12.0), cz),
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
        lights:         vec![],
        background:     None,
        name:           "Random Spheres",
        cam_init:       SceneCameraParams {
            pos: Point3::new(13.0, 2.0, 3.0), lookat: Point3::new(0.0, 0.0, 0.0),
            vfov: 20.0, aperture: 0.1, focus_dist: 10.0, move_speed: 0.3,
        },
        static_objects,
        dynamic,
        bounds:   None,
        colliders: vec![],
        gravity:  0.03,
        settled:  false,
        paused:   false,
    }
}

/// Creates a matched (Quad geometry, Light for NEE) pair from one set of parameters,
/// ensuring geometry and shadow-ray target never drift out of sync.
fn emissive_quad(q: Point3, u: Vec3, v: Vec3, emit: Color) -> (Quad, Light) {
    let mat: Arc<dyn hittable::Material> = Arc::new(DiffuseLight { emit });
    (Quad::new(q, u, v, mat), Light::new(q, u, v, emit))
}

fn build_cornell_box() -> SceneData {
    let mut list = HittableList::new();
    let red:   Arc<dyn hittable::Material> = Arc::new(Lambertian { texture: Color::new(0.65, 0.05, 0.05).into() });
    let white: Arc<dyn hittable::Material> = Arc::new(Lambertian { texture: Color::new(0.73, 0.73, 0.73).into() });
    let green: Arc<dyn hittable::Material> = Arc::new(Lambertian { texture: Color::new(0.12, 0.45, 0.15).into() });
    let (light_quad, nee_light) = emissive_quad(
        Point3::new(343.0, 554.0, 332.0),
        Vec3::new(-130.0, 0.0, 0.0),
        Vec3::new(0.0, 0.0, -105.0),
        Color::new(15.0, 15.0, 15.0),
    );

    list.add(Quad::new(Point3::new(555.0, 0.0,   0.0),   Vec3::new(0.0, 555.0,  0.0), Vec3::new(0.0, 0.0,  555.0), green));
    list.add(Quad::new(Point3::new(0.0,   0.0,   0.0),   Vec3::new(0.0, 555.0,  0.0), Vec3::new(0.0, 0.0,  555.0), red));
    list.add(light_quad);
    list.add(Quad::new(Point3::new(0.0,   0.0,   0.0),   Vec3::new(555.0, 0.0,  0.0), Vec3::new(0.0, 0.0,  555.0), Arc::clone(&white)));
    list.add(Quad::new(Point3::new(555.0, 555.0, 555.0), Vec3::new(-555.0, 0.0, 0.0), Vec3::new(0.0, 0.0, -555.0), Arc::clone(&white)));
    list.add(Quad::new(Point3::new(0.0,   0.0,   555.0), Vec3::new(555.0, 0.0,  0.0), Vec3::new(0.0, 555.0, 0.0),  Arc::clone(&white)));
    // Build each box at the origin, rotate, then translate into place.
    let tall = Arc::new(make_box(Point3::new(0.0,0.0,0.0), Point3::new(165.0,330.0,165.0), Arc::clone(&white))) as Arc<dyn Hittable>;
    let tall = Arc::new(RotateY::new(tall,  15.0)) as Arc<dyn Hittable>;
    let tall = Arc::new(Translate::new(tall, Vec3::new(265.0, 0.0, 295.0))) as Arc<dyn Hittable>;
    let tall_bbox = tall.bounding_box().unwrap();
    list.objects.push(tall);

    let short = Arc::new(make_box(Point3::new(0.0,0.0,0.0), Point3::new(165.0,165.0,165.0), white)) as Arc<dyn Hittable>;
    let short = Arc::new(RotateY::new(short, -18.0)) as Arc<dyn Hittable>;
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
        lights:         vec![nee_light],
        background:     Some(Color::default()),
        name:           "Cornell Box",
        cam_init:       SceneCameraParams {
            pos: Point3::new(278.0, 278.0, -800.0), lookat: Point3::new(278.0, 278.0, 0.0),
            vfov: 40.0, aperture: 0.0, focus_dist: 10.0, move_speed: 8.0,
        },
        static_objects,
        dynamic,
        bounds:    Some(bounds),
        colliders: vec![tall_bbox, short_bbox],
        gravity:   0.0,
        settled:   false,
        paused:    false,
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
    let (overhead_quad, nee_light) = emissive_quad(
        Point3::new(-200.0, 500.0, -200.0),
        Vec3::new(400.0, 0.0, 0.0),
        Vec3::new(0.0, 0.0, 400.0),
        Color::new(6.0, 6.0, 6.0),
    );
    list.add(overhead_quad);
    list.add(mesh_bvh);

    SceneData {
        world:          Arc::new(BvhTree::from_list(list)) as Arc<dyn Hittable>,
        lights:         vec![nee_light],
        background:     Some(Color::new(0.05, 0.07, 0.12)),
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

fn save_png(accumulator: &[Color], samples: u32, scene_name: &str, width: u32, height: u32) {
    if samples == 0 { return; }
    let scale = 1.0 / samples as f32;
    let img = ImageBuffer::from_fn(width, height, |x, y| {
        let [r, g, b] = tone_map(accumulator[(y * width + x) as usize], scale);
        Rgb([r, g, b])
    });
    let slug = scene_name.to_lowercase().replace(' ', "_");
    let path = format!("render_{}_{:04}spp.png", slug, samples);
    match img.save(&path) {
        Ok(_)  => println!("Saved {path}"),
        Err(e) => eprintln!("Save failed: {e}"),
    }
}

// ── Main loop ────────────────────────────────────────────────────────────────

fn main() {
    println!("Building scenes…");
    println!("Scene 1: Random Spheres");
    let s1 = build_random_scene();
    println!("Scene 2: Cornell Box");
    let s2 = build_cornell_box();
    println!("Scene 3: Mesh");
    let s3 = build_mesh_scene();
    let mut scenes = [s1, s2, s3];
    println!("Ready.  [1/2/3] scene  [F] free camera  WASD+mouse  Space/Shift up/down  [P] save  [ ] aperture  [Enter] pause  [R] restart (scene 1)  [Esc] quit");

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
    let mut free_cam    = false;
    let mut accumulator = vec![Color::default(); (win_w * win_h) as usize];
    let mut scratch     = vec![Color::default(); (win_w * win_h) as usize];
    let mut samples     = 0u32;
    let num_threads           = rayon::current_num_threads();
    let mut chunk_size        = ((win_w * win_h) as usize / (num_threads * 4)).max(1);
    let mut pressed           = std::collections::HashSet::<VirtualKeyCode>::new();
    let mut cam_dirty         = false;
    let mut pending_autofocus = false;

    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Poll;

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
                        accumulator.resize((win_w * win_h) as usize, Color::default());
                        accumulator.fill(Color::default());
                        scratch.resize((win_w * win_h) as usize, Color::default());
                        scratch.fill(Color::default());
                        chunk_size = ((win_w * win_h) as usize / (num_threads * 4)).max(1);
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
                                VirtualKeyCode::Key1 | VirtualKeyCode::Key2 | VirtualKeyCode::Key3 => {
                                    let idx = match key { VirtualKeyCode::Key2 => 1, VirtualKeyCode::Key3 => 2, _ => 0 };
                                    scene_idx = idx;
                                    cam_state = CameraState::from_params(&scenes[idx].cam_init);
                                    accumulator.fill(Color::default());
                                    samples = 0;
                                    cam_dirty = true;
                                    pending_autofocus = true;
                                }
                                VirtualKeyCode::P => save_png(&accumulator, samples, scenes[scene_idx].name, win_w, win_h),
                                VirtualKeyCode::LBracket => {
                                    cam_state.aperture = (cam_state.aperture - 0.025).max(0.0);
                                    cam_dirty = true;
                                }
                                VirtualKeyCode::RBracket => {
                                    cam_state.aperture += 0.025;
                                    cam_dirty = true;
                                }
                                VirtualKeyCode::Return => {
                                    scenes[scene_idx].paused = !scenes[scene_idx].paused;
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
                    let spd = cam_state.move_speed;
                    let fwd = cam_state.forward_horiz();
                    let rgt = cam_state.right_horiz();
                    let mut moved = false;
                    if pressed.contains(&VirtualKeyCode::W)      { cam_state.pos += fwd * spd; moved = true; }
                    if pressed.contains(&VirtualKeyCode::S)      { cam_state.pos -= fwd * spd; moved = true; }
                    if pressed.contains(&VirtualKeyCode::A)      { cam_state.pos -= rgt * spd; moved = true; }
                    if pressed.contains(&VirtualKeyCode::D)      { cam_state.pos += rgt * spd; moved = true; }
                    if pressed.contains(&VirtualKeyCode::Space)  { cam_state.pos.y += spd;     moved = true; }
                    if pressed.contains(&VirtualKeyCode::LShift) { cam_state.pos.y -= spd;     moved = true; }
                    if moved { cam_dirty = true; pending_autofocus = true; }
                }

                if cam_dirty {
                    camera = cam_state.to_camera(win_w as f32 / win_h as f32);
                    accumulator.fill(Color::default());
                    samples = 0;
                    cam_dirty = false;
                } else if pending_autofocus {
                    cam_state.autofocus(scenes[scene_idx].world.as_ref());
                    camera = cam_state.to_camera(win_w as f32 / win_h as f32);
                    accumulator.fill(Color::default());
                    samples = 0;
                    pending_autofocus = false;
                }

                if scenes[scene_idx].tick() {
                    accumulator.fill(Color::default());
                    samples = 0;
                }

                {
                    let scene      = &scenes[scene_idx];
                    let cam_hint   = if free_cam { "FREE CAM [Esc] release" } else { "[F] Free Camera" };
                    let motion_hint = if scene.gravity > 0.0 {
                        if scene.settled     { "  settled — [R] restart" }
                        else if scene.paused { "  PAUSED — [Enter] resume  [R] restart" }
                        else                 { "  [Enter] pause  [R] restart" }
                    } else if !scene.dynamic.is_empty() {
                        if scene.paused { "  PAUSED — [Enter] resume" } else { "" }
                    } else { "" };
                    window.set_title(&format!(
                        "Ray Tracer — {} — {} spp  |  [1/2/3] scene  {}  [P] Save  [ ] aperture {:.2}{}",
                        scene.name, samples, cam_hint, cam_state.aperture, motion_hint,
                    ));
                }

                if samples < MAX_SAMPLES {
                    let scene     = &scenes[scene_idx];
                    let world_ref = &scene.world;
                    let bg        = scene.background;
                    let lights    = scene.lights.as_slice();

                    scratch.par_chunks_mut(chunk_size).enumerate().for_each(|(ci, chunk)| {
                        let mut rng = SmallRng::seed_from_u64(
                            (ci as u64).wrapping_mul(6364136223846793005)
                                ^ (samples as u64).wrapping_mul(0x9E3779B97F4A7C15)
                        );
                        for (li, out) in chunk.iter_mut().enumerate() {
                            let i     = ci * chunk_size + li;
                            let px    = (i % win_w as usize) as u32;
                            let py    = (i / win_w as usize) as u32;
                            let ray_y = win_h - 1 - py;
                            let u = (px as f32 + rng.gen::<f32>()) / (win_w - 1) as f32;
                            let v = (ray_y as f32 + rng.gen::<f32>()) / (win_h - 1) as f32;
                            *out = ray_color(&camera.get_ray(u, v, &mut rng), world_ref.as_ref(), bg, lights, &mut rng);
                        }
                    });

                    for (acc, s) in accumulator.iter_mut().zip(scratch.iter()) { *acc += *s; }
                    samples += 1;

                    window.request_redraw();
                } else {
                    *control_flow = ControlFlow::Wait;
                }
            }

            Event::RedrawRequested(_) => {
                let mut buffer = surface.buffer_mut().unwrap();
                let scale = 1.0 / samples.max(1) as f32;
                for (i, &color) in accumulator.iter().enumerate() {
                    buffer[i] = to_rgb_u32(color, scale);
                }
                buffer.present().unwrap();
            }

            _ => {}
        }
    });
}
