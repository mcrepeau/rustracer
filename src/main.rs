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
mod cylinder;
mod cone;
mod disk;
mod plane;
mod transform;
mod camera;
mod renderer;
mod scene;
mod scenes;
mod scene_file;
mod output;
mod diamond;
mod photon;
#[cfg(feature = "denoise")]
mod denoise;

use vec3::Color;
use camera::CameraState;
#[cfg(not(feature = "denoise"))]
use renderer::{Background, render_tiles};
#[cfg(feature = "denoise")]
use renderer::{Background, render_tiles, render_aux_pass};
use scene::{SceneData, resolve_camera_aabb};
use scenes::{build_random_scene, build_cornell_box, build_nextweek_scene};
use output::{to_rgb_u32, save_png};

use winit::{
    dpi::PhysicalSize,
    event::{DeviceEvent, ElementState, Event, KeyboardInput, VirtualKeyCode, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::{CursorGrabMode, WindowBuilder},
};
use softbuffer::{Context, Surface};
use std::num::NonZeroU32;
#[cfg(feature = "denoise")]
use std::sync::{Arc, Mutex};
#[cfg(feature = "denoise")]
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

const WIDTH:  u32 = 1200;
const HEIGHT: u32 = 800;
const MOUSE_SENS:  f32 = 0.002;
const TITLE_INTERVAL: Duration = Duration::from_millis(200);
const CAM_RADIUS: f32 = 0.25;

// ── Bench ─────────────────────────────────────────────────────────────────────

fn bench_scene(scene: &SceneData, scratch: &mut Vec<Color>, samples: u32) -> Duration {
    let aspect = WIDTH as f32 / HEIGHT as f32;
    let mut cam = CameraState::from_params(&scene.cam_init);
    cam.autofocus(&*scene.world);
    let camera = cam.to_camera(aspect);
    let world  = scene.world.as_ref();
    let bg     = scene.background;
    scratch.resize((WIDTH * HEIGHT) as usize, Color::default());

    let strata = (samples as f32).sqrt() as u32;
    let t0 = Instant::now();
    for s in 0..samples {
        render_tiles(scratch, s, strata, WIDTH, HEIGHT, &camera, world, bg, &scene.lights, 1.0, scene.photon_map.as_deref());
    }
    t0.elapsed()
}

fn run_bench() {
    const WARMUP:  u32 = 4;
    const SAMPLES: u32 = 128;

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
        build_nextweek_scene,
    ];

    let mut scratch: Vec<Color> = Vec::new();
    for build in builders {
        let scene = build();
        print!("  {:<18}  building…\r", scene.name);
        let _ = std::io::Write::flush(&mut std::io::stdout());

        bench_scene(&scene, &mut scratch, WARMUP);

        let elapsed  = bench_scene(&scene, &mut scratch, SAMPLES);
        let secs     = elapsed.as_secs_f64();
        let ms_per_s = secs * 1000.0 / SAMPLES as f64;
        let mpx_s    = (WIDTH as f64 * HEIGHT as f64 * SAMPLES as f64) / (secs * 1e6);

        println!("  {:<18}  {:>7}  {:>7.3}s  {:>9.1} ms  {:>7.2}",
            scene.name, SAMPLES, secs, ms_per_s, mpx_s);
    }
    println!("\nDone.");
}

// ── Main loop ─────────────────────────────────────────────────────────────────

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
    println!("Scene 3: Next Week");
    let s3 = build_nextweek_scene();
    let mut scenes: Vec<SceneData> = vec![s1, s2, s3];

    // Scene 4: hot-reloadable file scene
    match scene_file::load("scene.toml") {
        Ok(s)  => { println!("Scene 4: {} (scene.toml)", s.name); scenes.push(s); }
        Err(e) => println!("scene.toml not loaded — {e}"),
    }

    println!("Ready.  [1-4] scene  [F] free camera  WASD+mouse  Space/Shift up/down");
    println!("        [P] save  [[] apt  [,.] fov  [-=] exp  [arrows] sun  [C] reset cam");
    println!("        [Enter] pause  [R] restart/reload  [Esc] quit");

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
    let mut camera        = cam_state.to_camera(win_w as f32 / win_h as f32);
    let mut free_cam      = false;
    #[cfg(feature = "denoise")]
    let mut oidn_on       = false;
    #[cfg(feature = "denoise")]
    let mut denoise_blend = 0.8f32;  // 1.0 = full OIDN, 0.0 = raw
    #[cfg(feature = "denoise")]
    let denoised: Arc<Mutex<Vec<Color>>> = Arc::new(Mutex::new(Vec::new()));
    #[cfg(feature = "denoise")]
    let mut aux_albedo: Vec<f32> = Vec::new();
    #[cfg(feature = "denoise")]
    let mut aux_normal: Vec<f32> = Vec::new();
    #[cfg(feature = "denoise")]
    let denoise_running: Arc<AtomicBool> = Arc::new(AtomicBool::new(false));
    #[cfg(feature = "denoise")]
    let denoise_epoch:   Arc<AtomicU64>  = Arc::new(AtomicU64::new(0));
    let mut accumulator   = vec![Color::default(); (win_w * win_h) as usize];
    let mut scratch       = vec![Color::default(); (win_w * win_h) as usize];
    let mut samples       = 0u32;
    let mut strata = (scenes[scene_idx].max_samples as f32).sqrt() as u32;
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
                accumulator.fill(Color::default());
                samples = 0;
                #[cfg(feature = "denoise")]
                {
                    denoised.lock().unwrap().clear();
                    denoise_epoch.fetch_add(1, Ordering::Relaxed);
                    aux_albedo.clear();
                    aux_normal.clear();
                }
            }
        }
        macro_rules! switch_scene {
            ($idx:expr) => {{
                let i = $idx;
                scene_idx        = i;
                cam_state        = CameraState::from_params(&scenes[i].cam_init);
                strata           = (scenes[i].max_samples as f32).sqrt() as u32;
                physics_accum    = Duration::ZERO;
                reset_accum!();
                cam_dirty        = true;
                pending_autofocus = true;
            }};
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
                        accumulator.clear(); accumulator.resize(n, Color::default());
                        scratch.clear();     scratch.resize(n, Color::default());
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
                                    switch_scene!(idx);
                                }
                                VirtualKeyCode::Key4 => {
                                    if scenes.len() >= 4 {
                                        switch_scene!(3);
                                    } else {
                                        println!("scene.toml not loaded — edit the file and press [R] to reload it");
                                    }
                                }
                                VirtualKeyCode::P => {
                                    #[cfg(feature = "denoise")]
                                    {
                                        let denoised_guard = denoised.lock().unwrap();
                                        if oidn_on && denoised_guard.len() == (win_w * win_h) as usize {
                                            let sc = 1.0 / samples.max(1) as f32;
                                            let blended: Vec<Color> = accumulator.iter().zip(denoised_guard.iter())
                                                .map(|(acc, den)| *den * denoise_blend + *acc * sc * (1.0 - denoise_blend))
                                                .collect();
                                            save_png(&blended, 1, scenes[scene_idx].name, win_w, win_h, exposure);
                                        } else {
                                            save_png(&accumulator, samples, scenes[scene_idx].name, win_w, win_h, exposure);
                                        }
                                    }
                                    #[cfg(not(feature = "denoise"))]
                                    save_png(&accumulator, samples, scenes[scene_idx].name, win_w, win_h, exposure);
                                }
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
                                        *sun_dir = vec3::Vec3::new(
                                            sun_dir.x * c - sun_dir.z * s,
                                            sun_dir.y,
                                            sun_dir.x * s + sun_dir.z * c,
                                        ).unit();
                                        reset_accum!();
                                        scenes[scene_idx].rebuild_caustics();
                                    }
                                }
                                VirtualKeyCode::Up | VirtualKeyCode::Down => {
                                    if let Background::Physical { sun_dir } = &mut scenes[scene_idx].background {
                                        let step = if key == VirtualKeyCode::Up { 0.1f32 } else { -0.1f32 };
                                        let el = sun_dir.y.asin();
                                        let new_el = (el + step).clamp(-10f32.to_radians(), 85f32.to_radians());
                                        let horiz = vec3::Vec3::new(sun_dir.x, 0.0, sun_dir.z).length().max(1e-6);
                                        let scale = new_el.cos() / horiz;
                                        *sun_dir = vec3::Vec3::new(sun_dir.x * scale, new_el.sin(), sun_dir.z * scale);
                                        reset_accum!();
                                        scenes[scene_idx].rebuild_caustics();
                                    }
                                }
                                VirtualKeyCode::C => {
                                    cam_state = CameraState::from_params(&scenes[scene_idx].cam_init);
                                    cam_dirty = true;
                                    pending_autofocus = true;
                                }
                                #[cfg(feature = "denoise")]
                                VirtualKeyCode::N => {
                                    oidn_on = !oidn_on;
                                    if oidn_on && samples > 0 && !denoise_running.load(Ordering::Relaxed) {
                                        let w = win_w; let h = win_h;
                                        let sc = 1.0 / samples.max(1) as f32;
                                        let input: Vec<f32> = accumulator.iter()
                                            .flat_map(|c| [c.x * sc, c.y * sc, c.z * sc])
                                            .collect();
                                        let alb     = aux_albedo.clone();
                                        let nrm     = aux_normal.clone();
                                        let dst     = Arc::clone(&denoised);
                                        let running = Arc::clone(&denoise_running);
                                        let epoch   = denoise_epoch.load(Ordering::Relaxed);
                                        let ep_ref  = Arc::clone(&denoise_epoch);
                                        running.store(true, Ordering::Relaxed);
                                        std::thread::spawn(move || {
                                            if let Some(result) = denoise::denoise_rgb(w, h, input, alb, nrm) {
                                                if ep_ref.load(Ordering::Relaxed) == epoch {
                                                    *dst.lock().unwrap() = result;
                                                }
                                            }
                                            running.store(false, Ordering::Relaxed);
                                        });
                                    }
                                    window.request_redraw();
                                }
                                #[cfg(feature = "denoise")]
                                VirtualKeyCode::J => {
                                    denoise_blend = (denoise_blend - 0.1).max(0.0);
                                    window.request_redraw();
                                }
                                #[cfg(feature = "denoise")]
                                VirtualKeyCode::K => {
                                    denoise_blend = (denoise_blend + 0.1).min(1.0);
                                    window.request_redraw();
                                }
                                VirtualKeyCode::Return => {
                                    let pausing = !scenes[scene_idx].paused;
                                    scenes[scene_idx].paused = pausing;
                                    if pausing {
                                        scenes[scene_idx].rebuild();
                                        reset_accum!();
                                    }
                                }
                                VirtualKeyCode::R if scene_idx == 0 => {
                                    scenes[0] = build_random_scene();
                                    strata = (scenes[0].max_samples as f32).sqrt() as u32;
                                    physics_accum = Duration::ZERO;
                                    reset_accum!();
                                    cam_dirty = true;
                                    pending_autofocus = true;
                                }
                                VirtualKeyCode::R if scene_idx == 3 => {
                                    match scene_file::load("scene.toml") {
                                        Ok(s) => {
                                            println!("Reloaded: {}", s.name);
                                            scenes[3] = s;
                                            cam_state = CameraState::from_params(&scenes[3].cam_init);
                                            strata = (scenes[3].max_samples as f32).sqrt() as u32;
                                            reset_accum!();
                                            cam_dirty = true;
                                            pending_autofocus = true;
                                        }
                                        Err(e) => println!("Reload failed — {e}"),
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

            Event::DeviceEvent { event: DeviceEvent::MouseMotion { delta: (dx, dy) }, .. }
                if free_cam =>
            {
                cam_state.yaw   += dx as f32 * MOUSE_SENS;
                cam_state.pitch  = (cam_state.pitch - dy as f32 * MOUSE_SENS)
                    .clamp(-89f32.to_radians(), 89f32.to_radians());
                cam_dirty = true;
                pending_autofocus = true;
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
                physics_accum += frame_dt.min(Duration::from_millis(100));
                last_frame_time = now;
                let mut physics_ticked = false;
                while physics_accum >= Duration::from_millis(16) {
                    physics_ticked |= scenes[scene_idx].tick();
                    physics_accum -= Duration::from_millis(16);
                }
                if physics_ticked {
                    reset_accum!();
                }

                if last_title_update.elapsed() >= TITLE_INTERVAL {
                    let scene    = &scenes[scene_idx];
                    let cam_hint = if free_cam { "FREE CAM [Esc] release" } else { "[F] Free Camera" };
                    let motion_hint = if scene.gravity > 0.0 {
                        if scene.settled     { "  settled — [R] restart" }
                        else if scene.paused { "  PAUSED — [Enter] resume  [R] restart" }
                        else                 { "  [Enter] pause  [R] restart" }
                    } else if !scene.dynamic.is_empty() {
                        if scene.paused { "  PAUSED — [Enter] resume" } else { "  [Enter] pause" }
                    } else { "" };
                    #[cfg(feature = "denoise")]
                    let oidn_hint = if oidn_on {
                        let state = if denoise_running.load(Ordering::Relaxed) { "running…" } else { "on" };
                        format!("  OIDN {state} blend:{:.0}% [JK] [N] off", denoise_blend * 100.0)
                    } else {
                        "  [N] denoise".to_string()
                    };
                    #[cfg(not(feature = "denoise"))]
                    let oidn_hint = "";
                    let spp_label = format!("{samples} spp");
                    let sun_hint = if let Background::Physical { sun_dir } = scene.background {
                        format!("  sun {:.0}° [arrows]", sun_dir.y.asin().to_degrees())
                    } else { String::new() };
                    window.set_title(&format!(
                        "Ray Tracer — {} — {}  |  {}  [P] save  [[] apt {:.2}  [,.] fov {:.0}°  [-=] exp {:.2}x{}{}{}",
                        scene.name, spp_label, cam_hint,
                        cam_state.aperture, cam_state.vfov, exposure,
                        sun_hint, oidn_hint, motion_hint,
                    ));
                    last_title_update = Instant::now();
                }

                if samples < scenes[scene_idx].max_samples {
                    let scene = &scenes[scene_idx];
                    let bg    = scene.background;

                    let bg_scale = 1.0;
                    render_tiles(&mut scratch, samples, strata, win_w, win_h, &camera,
                                 scene.world.as_ref(), bg, &scene.lights, bg_scale,
                                 scene.photon_map.as_deref());

                    for i in 0..(win_w * win_h) as usize {
                        accumulator[i] += scratch[i];
                    }
                    samples += 1;

                    // Build the aux buffers once per render sequence (first sample),
                    // so they are ready before the first OIDN invocation at sample 32.
                    // Skipped for scenes where first-hit geometry doesn't correlate
                    // with the final pixel colour (indirect lighting, volumes).
                    #[cfg(feature = "denoise")]
                    if samples == 1 && matches!(scene.background, Background::Physical { .. }) {
                        let (alb, nrm) = render_aux_pass(win_w, win_h, &camera,
                                                         scene.world.as_ref(), bg);
                        aux_albedo = alb;
                        aux_normal = nrm;
                    }

                    #[cfg(feature = "denoise")]
                    if oidn_on && samples % 32 == 0 && !denoise_running.load(Ordering::Relaxed) {
                        let w = win_w; let h = win_h;
                        let sc = 1.0 / samples.max(1) as f32;
                        let input: Vec<f32> = accumulator.iter()
                            .flat_map(|c| [c.x * sc, c.y * sc, c.z * sc])
                            .collect();
                        let alb     = aux_albedo.clone();
                        let nrm     = aux_normal.clone();
                        let dst     = Arc::clone(&denoised);
                        let running = Arc::clone(&denoise_running);
                        let epoch   = denoise_epoch.load(Ordering::Relaxed);
                        let ep_ref  = Arc::clone(&denoise_epoch);
                        running.store(true, Ordering::Relaxed);
                        std::thread::spawn(move || {
                            if let Some(result) = denoise::denoise_rgb(w, h, input, alb, nrm) {
                                if ep_ref.load(Ordering::Relaxed) == epoch {
                                    *dst.lock().unwrap() = result;
                                }
                            }
                            running.store(false, Ordering::Relaxed);
                        });
                    }

                    window.request_redraw();
                } else {
                    *control_flow = ControlFlow::Wait;
                }
            }

            Event::RedrawRequested(_) => {
                let mut buffer = surface.buffer_mut().unwrap();
                #[cfg(feature = "denoise")]
                {
                    let denoised_guard = denoised.lock().unwrap();
                    let use_denoised = oidn_on && denoised_guard.len() == (win_w * win_h) as usize;
                    let sc = 1.0 / samples.max(1) as f32;
                    for i in 0..(win_w * win_h) as usize {
                        let raw = accumulator[i] * sc;
                        let color = if use_denoised {
                            denoised_guard[i] * denoise_blend + raw * (1.0 - denoise_blend)
                        } else {
                            raw
                        };
                        buffer[i] = to_rgb_u32(color, exposure);
                    }
                }
                #[cfg(not(feature = "denoise"))]
                {
                    let sc = 1.0 / samples.max(1) as f32;
                    for i in 0..(win_w * win_h) as usize {
                        buffer[i] = to_rgb_u32(accumulator[i] * sc, exposure);
                    }
                }
                buffer.present().unwrap();
            }

            _ => {}
        }
    });
}
