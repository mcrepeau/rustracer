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
use bvh::BvhNode;
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

                let Some((mut attenuation, scattered)) = rec.mat.scatter(&ray, &rec, rng) else {
                    break;
                };

                if let Some(albedo) = rec.mat.albedo_at(rec.u, rec.v, rec.p) {
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
                    attenuation = attenuation / survive;
                }
                throughput = throughput * attenuation;
                ray = scattered;
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

    fn to_camera(&self, aspect: f32) -> Camera {
        let fwd = Vec3::new(
            self.yaw.sin() * self.pitch.cos(),
            self.pitch.sin(),
            -self.yaw.cos() * self.pitch.cos(),
        );
        Camera::new(self.pos, self.pos + fwd, Vec3::new(0.0, 1.0, 0.0),
                    self.vfov, aspect, self.aperture, self.focus_dist)
    }
}

// ── Scenes ───────────────────────────────────────────────────────────────────

struct SceneData {
    world:      Arc<dyn Hittable>,
    lights:     Vec<Light>,
    background: Option<Color>,
    name:       &'static str,
    cam_init:   SceneCameraParams,
}

fn build_random_scene() -> SceneData {
    let mut list = HittableList::new();
    let mut rng  = rand::thread_rng();

    list.add(Sphere::new(Point3::new(0.0, -1000.0, 0.0), 1000.0,
        Arc::new(Lambertian { texture: Texture::Checker {
            scale: 10.0,
            even:  Color::new(0.2, 0.3, 0.1),
            odd:   Color::new(0.9, 0.9, 0.9),
        }})));

    for a in -11..11 {
        for b in -11..11 {
            let center = Point3::new(a as f32 + 0.9*rng.gen::<f32>(), 0.2, b as f32 + 0.9*rng.gen::<f32>());
            if (center - Point3::new(4.0, 0.2, 0.0)).length() <= 0.9 { continue; }
            let choose: f32 = rng.gen();
            let mat: Arc<dyn hittable::Material> = if choose < 0.8 {
                Arc::new(Lambertian { texture: (Color::random(&mut rng) * Color::random(&mut rng)).into() })
            } else if choose < 0.95 {
                Arc::new(Metal { albedo: Color::random_range(0.5, 1.0, &mut rng), fuzz: rng.gen_range(0.0..0.5) })
            } else {
                Arc::new(Dielectric { ir: 1.5 })
            };
            list.add(Sphere::new(center, 0.2, mat));
        }
    }
    list.add(Sphere::new(Point3::new( 0.0, 1.0, 0.0), 1.0, Arc::new(Dielectric { ir: 1.5 })));
    list.add(Sphere::new(Point3::new(-4.0, 1.0, 0.0), 1.0, Arc::new(Lambertian { texture: Color::new(0.4, 0.2, 0.1).into() })));
    list.add(Sphere::new(Point3::new( 4.0, 1.0, 0.0), 1.0, Arc::new(Metal { albedo: Color::new(0.7, 0.6, 0.5), fuzz: 0.0 })));

    SceneData {
        world:      Arc::new(BvhNode::from_list(list)) as Arc<dyn Hittable>,
        lights:     vec![],
        background: None,
        name:       "Random Spheres",
        cam_init:   SceneCameraParams {
            pos: Point3::new(13.0, 2.0, 3.0), lookat: Point3::new(0.0, 0.0, 0.0),
            vfov: 20.0, aperture: 0.0, focus_dist: 10.0, move_speed: 0.3,
        },
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
    list.add_arc(Arc::new(Translate::new(tall, Vec3::new(265.0, 0.0, 295.0))));

    let short = Arc::new(make_box(Point3::new(0.0,0.0,0.0), Point3::new(165.0,165.0,165.0), white)) as Arc<dyn Hittable>;
    let short = Arc::new(RotateY::new(short, -18.0)) as Arc<dyn Hittable>;
    list.add_arc(Arc::new(Translate::new(short, Vec3::new(130.0, 0.0, 65.0))));

    SceneData {
        world:      Arc::new(BvhNode::from_list(list)) as Arc<dyn Hittable>,
        lights:     vec![nee_light],
        background: Some(Color::default()),
        name:       "Cornell Box",
        cam_init:   SceneCameraParams {
            pos: Point3::new(278.0, 278.0, -800.0), lookat: Point3::new(278.0, 278.0, 0.0),
            vfov: 40.0, aperture: 0.0, focus_dist: 10.0, move_speed: 8.0,
        },
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
            (BvhNode::from_list(list), SceneCameraParams {
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
            (BvhNode::from_list(mesh_list), SceneCameraParams {
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
    list.add_arc(Arc::new(mesh_bvh));

    SceneData {
        world:      Arc::new(BvhNode::from_list(list)) as Arc<dyn Hittable>,
        lights:     vec![nee_light],
        background: Some(Color::new(0.05, 0.07, 0.12)),
        name:       "Mesh",
        cam_init,
    }
}

fn aces(x: f32) -> f32 {
    let x = x.max(0.0);
    (x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14)
}

fn to_rgb_u32(c: Color, scale: f32) -> u32 {
    let r = (aces(c.x * scale).sqrt().clamp(0.0, 0.999) * 256.0) as u32;
    let g = (aces(c.y * scale).sqrt().clamp(0.0, 0.999) * 256.0) as u32;
    let b = (aces(c.z * scale).sqrt().clamp(0.0, 0.999) * 256.0) as u32;
    r << 16 | g << 8 | b
}

fn save_png(accumulator: &[Color], samples: u32, scene_name: &str) {
    if samples == 0 { return; }
    let scale = 1.0 / samples as f32;
    let img = ImageBuffer::from_fn(WIDTH, HEIGHT, |x, y| {
        let c = accumulator[(y * WIDTH + x) as usize];
        let r = aces(c.x * scale).sqrt().clamp(0.0, 0.999);
        let g = aces(c.y * scale).sqrt().clamp(0.0, 0.999);
        let b = aces(c.z * scale).sqrt().clamp(0.0, 0.999);
        Rgb([(256.0 * r) as u8, (256.0 * g) as u8, (256.0 * b) as u8])
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
    let scenes = [s1, s2, s3];
    println!("Ready.  [1/2/3] scene  [F] free camera  WASD+mouse  Space/Shift up/down  [P] save  [Esc] quit");

    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_title("Ray Tracer")
        .with_inner_size(PhysicalSize::new(WIDTH, HEIGHT))
        .build(&event_loop)
        .unwrap();

    let _context = unsafe { Context::new(&window) }.unwrap();
    let mut surface = unsafe { Surface::new(&_context, &window) }.unwrap();
    surface.resize(NonZeroU32::new(WIDTH).unwrap(), NonZeroU32::new(HEIGHT).unwrap()).unwrap();

    let mut scene_idx   = 0usize;
    let mut cam_state   = CameraState::from_params(&scenes[scene_idx].cam_init);
    let mut camera      = cam_state.to_camera(WIDTH as f32 / HEIGHT as f32);
    let mut free_cam    = false;
    let mut accumulator = vec![Color::default(); (WIDTH * HEIGHT) as usize];
    let mut scratch     = vec![Color::default(); (WIDTH * HEIGHT) as usize];
    let mut samples     = 0u32;
    let mut pressed     = std::collections::HashSet::<VirtualKeyCode>::new();
    let mut cam_dirty   = false;

    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Poll;

        match event {
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::CloseRequested => *control_flow = ControlFlow::Exit,

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
                                VirtualKeyCode::Key1 => {
                                    scene_idx = 0;
                                    cam_state = CameraState::from_params(&scenes[0].cam_init);
                                    accumulator.fill(Color::default());
                                    samples = 0;
                                    cam_dirty = true;
                                }
                                VirtualKeyCode::Key2 => {
                                    scene_idx = 1;
                                    cam_state = CameraState::from_params(&scenes[1].cam_init);
                                    accumulator.fill(Color::default());
                                    samples = 0;
                                    cam_dirty = true;
                                }
                                VirtualKeyCode::Key3 => {
                                    scene_idx = 2;
                                    cam_state = CameraState::from_params(&scenes[2].cam_init);
                                    accumulator.fill(Color::default());
                                    samples = 0;
                                    cam_dirty = true;
                                }
                                VirtualKeyCode::P => save_png(&accumulator, samples, scenes[scene_idx].name),
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
                }
            }

            Event::MainEventsCleared => {
                if free_cam {
                    let spd = cam_state.move_speed;
                    let fwd = cam_state.forward_horiz();
                    let rgt = cam_state.right_horiz();
                    if pressed.contains(&VirtualKeyCode::W)      { cam_state.pos += fwd * spd; cam_dirty = true; }
                    if pressed.contains(&VirtualKeyCode::S)      { cam_state.pos -= fwd * spd; cam_dirty = true; }
                    if pressed.contains(&VirtualKeyCode::A)      { cam_state.pos -= rgt * spd; cam_dirty = true; }
                    if pressed.contains(&VirtualKeyCode::D)      { cam_state.pos += rgt * spd; cam_dirty = true; }
                    if pressed.contains(&VirtualKeyCode::Space)  { cam_state.pos.y += spd;     cam_dirty = true; }
                    if pressed.contains(&VirtualKeyCode::LShift) { cam_state.pos.y -= spd;     cam_dirty = true; }
                }

                if cam_dirty {
                    camera = cam_state.to_camera(WIDTH as f32 / HEIGHT as f32);
                    accumulator.fill(Color::default());
                    samples = 0;
                    cam_dirty = false;
                }

                if samples < MAX_SAMPLES {
                    let scene     = &scenes[scene_idx];
                    let world_ref = &scene.world;
                    let bg        = scene.background;
                    let lights    = scene.lights.as_slice();

                    scratch.par_chunks_mut(64).enumerate().for_each(|(ci, chunk)| {
                        let mut rng = SmallRng::seed_from_u64(
                            (ci as u64).wrapping_mul(6364136223846793005)
                                ^ (samples as u64).wrapping_mul(2654435761)
                        );
                        for (li, out) in chunk.iter_mut().enumerate() {
                            let i     = ci * 64 + li;
                            let px    = (i % WIDTH as usize) as u32;
                            let py    = (i / WIDTH as usize) as u32;
                            let ray_y = HEIGHT - 1 - py;
                            let u = (px as f32 + rng.gen::<f32>()) / (WIDTH  - 1) as f32;
                            let v = (ray_y as f32 + rng.gen::<f32>()) / (HEIGHT - 1) as f32;
                            *out = ray_color(&camera.get_ray(u, v, &mut rng), world_ref.as_ref(), bg, lights, &mut rng);
                        }
                    });

                    for (acc, s) in accumulator.iter_mut().zip(scratch.iter()) { *acc += *s; }
                    samples += 1;

                    let cam_hint = if free_cam { "FREE CAM [Esc] release" } else { "[F] Free Camera" };
                    window.set_title(&format!(
                        "Ray Tracer — {} — {} spp  |  [1] Random [2] Cornell [3] Mesh  {}  [P] Save",
                        scene.name, samples, cam_hint,
                    ));

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
