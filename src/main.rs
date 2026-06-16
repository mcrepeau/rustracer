mod aabb;
mod bvh;
mod vec3;
mod ray;
mod spectrum;
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

use rayon::prelude::*;
use vec3::Color;
use camera::CameraState;
#[cfg(not(feature = "denoise"))]
use renderer::{Background, render_tiles};
#[cfg(feature = "denoise")]
use renderer::{Background, render_tiles, render_aux_pass};
use scene::SceneData;
use scenes::{build_random_scene, build_cornell_box, build_nextweek_scene};
use output::{to_rgb_u32, save_png, ToneMapper};

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
        render_tiles(scratch, None, s, strata, WIDTH, HEIGHT, &camera, world, bg, &scene.lights, 1.0, scene.photon_map.as_deref());
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

// ── Headless render ───────────────────────────────────────────────────────────

struct RenderArgs {
    scene:      String,
    samples:    Option<u32>,
    width:      u32,
    height:     u32,
    exposure:   f32,
    output:     Option<String>,
    tonemapper: ToneMapper,
}

impl Default for RenderArgs {
    fn default() -> Self {
        Self {
            scene:      "random".to_string(),
            samples:    None,
            width:      WIDTH,
            height:     HEIGHT,
            exposure:   1.0,
            output:     None,
            tonemapper: ToneMapper::AgX,
        }
    }
}

fn print_controls() {
    println!("Controls:");
    println!("  [1-4]        Switch scene          [R]      Restart / reload scene");
    println!("  [F]          Toggle free camera    [C]      Reset camera");
    println!("  WASD         Move (free cam)       [Space]  Move up");
    println!("  Mouse        Look (free cam)       [Shift]  Move down");
    println!("  [,] [.]      Decrease / increase FOV");
    println!("  [-] [=]      Decrease / increase exposure");
    println!("  [[]  []]     Decrease / increase aperture");
    println!("  [Arrows]     Rotate sun");
    println!("  [P]          Save PNG");
    println!("  [Enter]      Pause rendering");
    println!("  [T]          Toggle tonemapper (AgX / ACES)");
    println!("  [V]          Toggle adaptive sampling");
    #[cfg(feature = "denoise")]
    println!("  [N]          Toggle OIDN denoiser");
    #[cfg(feature = "denoise")]
    println!("  [J] [K]      Decrease / increase denoise blend");
    println!("  [?]          Reprint this list");
    println!("  [Esc]        Release mouse / quit");
    println!();
}

fn print_help() {
    println!("rustracer — path tracer\n");
    println!("USAGE:");
    println!("  rustracer                      Interactive viewer");
    println!("  rustracer --render [options]   Headless render to PNG");
    println!("  rustracer --bench              Performance benchmark");
    println!("  rustracer --help               Show this message\n");
    println!("RENDER OPTIONS:");
    println!("  --scene <name>       random|1  cornell|2  nextweek|3  <path.toml>");
    println!("                       Default: random");
    println!("  --samples <n>        Samples per pixel (default: scene maximum)");
    println!("  --width  <n>         Output width  in pixels (default: {WIDTH})");
    println!("  --height <n>         Output height in pixels (default: {HEIGHT})");
    println!("  --exposure <f>       Exposure multiplier     (default: 1.0)");
    println!("  --output <path>      Output PNG path (default: auto-generated)");
    println!("  --tonemapper <name>  agx or aces             (default: agx)");
}

fn parse_render_args() -> Result<RenderArgs, String> {
    let mut r  = RenderArgs::default();
    let mut it = std::env::args().skip(2); // skip binary + "--render"
    while let Some(flag) = it.next() {
        macro_rules! val {
            () => { it.next().ok_or_else(|| format!("'{flag}' requires a value"))? }
        }
        match flag.as_str() {
            "--scene"      => r.scene    = val!(),
            "--samples"    => r.samples  = Some(val!().parse::<u32>().map_err(|e| format!("--samples: {e}"))?),
            "--width"      => r.width    = val!().parse::<u32>().map_err(|e| format!("--width: {e}"))?,
            "--height"     => r.height   = val!().parse::<u32>().map_err(|e| format!("--height: {e}"))?,
            "--exposure"   => r.exposure = val!().parse::<f32>().map_err(|e| format!("--exposure: {e}"))?,
            "--output"     => r.output   = Some(val!()),
            "--tonemapper" => r.tonemapper = match val!().to_lowercase().as_str() {
                "agx"  => ToneMapper::AgX,
                "aces" => ToneMapper::Aces,
                s      => return Err(format!("Unknown tonemapper '{s}': use agx or aces")),
            },
            other => return Err(format!("Unknown flag '{other}'. Run with --help for usage.")),
        }
    }
    Ok(r)
}

fn run_render(args: RenderArgs) {
    let scene = {
        let sl = args.scene.to_lowercase();
        match sl.as_str() {
            "random"  | "1" => build_random_scene(),
            "cornell" | "2" => build_cornell_box(),
            "nextweek"| "3" => build_nextweek_scene(),
            _ => match scene_file::load(&args.scene) {
                Ok(s)  => s,
                Err(e) => { eprintln!("Failed to load '{}': {e}", args.scene); std::process::exit(1); }
            },
        }
    };

    let samples = args.samples.unwrap_or(scene.max_samples);
    let strata  = (samples as f32).sqrt() as u32;
    let n_px    = (args.width * args.height) as usize;

    let mut cam = CameraState::from_params(&scene.cam_init);
    cam.autofocus(scene.world.as_ref());
    let camera = cam.to_camera(args.width as f32 / args.height as f32);

    let mut accumulator = vec![Color::default(); n_px];
    let mut scratch     = vec![Color::default(); n_px];

    let tm_name = if args.tonemapper == ToneMapper::AgX { "AgX" } else { "ACES" };
    println!("Scene:      {}", scene.name);
    println!("Resolution: {}×{}", args.width, args.height);
    println!("Samples:    {samples}  (strata {strata}×{strata})");
    println!("Tonemapper: {tm_name}  exposure {:.2}", args.exposure);
    println!();

    let t0 = Instant::now();
    for s in 0..samples {
        render_tiles(&mut scratch, None, s, strata,
                     args.width, args.height, &camera,
                     scene.world.as_ref(), scene.background,
                     &scene.lights, 1.0, scene.photon_map.as_deref());

        accumulator.par_iter_mut().zip(scratch.par_iter()).for_each(|(a, &c)| *a += c);

        if s % 5 == 4 || s == samples - 1 {
            let done    = s + 1;
            let elapsed = t0.elapsed().as_secs_f64();
            let eta     = if done < samples { elapsed / done as f64 * (samples - done) as f64 } else { 0.0 };
            const BAR: usize = 32;
            let filled = (done as usize * BAR / samples as usize).min(BAR);
            let bar    = format!("{}{}", "█".repeat(filled), "░".repeat(BAR - filled));
            print!("\r  {done:>5}/{samples}  [{bar}]  {elapsed:5.0}s elapsed  ETA {eta:4.0}s  ");
            let _ = std::io::Write::flush(&mut std::io::stdout());
        }
    }

    let elapsed = t0.elapsed();
    println!("\n\nRendered in {:.2}s  ({:.1} ms/spp)\n",
             elapsed.as_secs_f64(), elapsed.as_millis() as f64 / samples as f64);

    let out = args.output.as_deref();
    save_png(&accumulator, samples, None, scene.name,
             args.width, args.height, args.exposure, args.tonemapper, None, out);
}

// ── Shared helpers ────────────────────────────────────────────────────────────

fn write_tonemap(buf: &mut [u32], accumulator: &[Color], sc: f32, exposure: f32, tm: ToneMapper) {
    buf.par_iter_mut()
        .zip(accumulator.par_iter())
        .for_each(|(dst, &acc)| *dst = to_rgb_u32(acc * sc, exposure, tm));
}

/// Adaptive-sampling display: each pixel uses its own sample count.
fn write_tonemap_adaptive(buf: &mut [u32], accumulator: &[Color], pixel_samples: &[u32], exposure: f32, tm: ToneMapper) {
    buf.par_iter_mut()
        .zip(accumulator.par_iter())
        .zip(pixel_samples.par_iter())
        .for_each(|((dst, &acc), &n)| {
            *dst = to_rgb_u32(acc, exposure / n.max(1) as f32, tm);
        });
}

/// Minimum samples before a pixel can be declared converged.
const MIN_ADAPTIVE_SAMPLES: u32 = 16;
/// Maximum relative standard error (σ/μ) to consider a pixel converged.
const ADAPTIVE_THRESHOLD:   f32 = 0.05;

#[cfg(feature = "denoise")]
fn spawn_denoiser(
    win_w: u32,
    win_h: u32,
    accumulator:     &[Color],
    samples:         u32,
    pixel_samples:   Option<&[u32]>,
    aux_albedo:      &[f32],
    aux_normal:      &[f32],
    denoised:        &Arc<Mutex<Vec<Color>>>,
    denoise_running: &Arc<AtomicBool>,
    denoise_epoch:   &Arc<AtomicU64>,
) {
    let input = accumulator.iter().enumerate()
        .flat_map(|(i, c)| {
            let n = pixel_samples.map_or(samples, |ps| ps[i]).max(1) as f32;
            [c.x / n, c.y / n, c.z / n]
        })
        .collect::<Vec<f32>>();
    let alb     = aux_albedo.to_vec();
    let nrm     = aux_normal.to_vec();
    let dst     = Arc::clone(denoised);
    let running = Arc::clone(denoise_running);
    let epoch   = denoise_epoch.load(Ordering::Relaxed);
    let ep_ref  = Arc::clone(denoise_epoch);
    running.store(true, Ordering::Relaxed);
    std::thread::spawn(move || {
        if let Some(result) = denoise::denoise_rgb(win_w, win_h, input, alb, nrm) {
            if ep_ref.load(Ordering::Relaxed) == epoch {
                *dst.lock().unwrap() = result;
            }
        }
        running.store(false, Ordering::Release);
    });
}

// ── Main loop ─────────────────────────────────────────────────────────────────

fn main() {
    match std::env::args().nth(1).as_deref() {
        Some("--bench")  => { run_bench(); return; }
        Some("--help") | Some("-h") => { print_help(); return; }
        Some("--render") => {
            match parse_render_args() {
                Ok(args) => run_render(args),
                Err(e)   => { eprintln!("Error: {e}"); std::process::exit(1); }
            }
            return;
        }
        _ => {}
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

    print_controls();

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
    // Adaptive sampling state
    let mut tonemapper     = ToneMapper::AgX;
    let mut adaptive_on   = false;
    let n_px              = (win_w * win_h) as usize;
    let mut pixel_samples = vec![0u32;  n_px];  // per-pixel sample count
    let mut var_m2_lum    = vec![0.0f32; n_px]; // Welford M2 for luminance
    let mut adap_conv     = vec![false;  n_px]; // convergence mask
    let mut n_converged   = 0usize;
    let mut pressed           = std::collections::HashSet::<VirtualKeyCode>::new();
    let mut exposure          = 1.0f32;
    let mut cam_dirty         = false;
    let mut pending_autofocus = false;
    let mut last_title_update = Instant::now();

    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Poll;

        macro_rules! reset_accum {
            () => {
                accumulator.fill(Color::default());
                pixel_samples.fill(0);
                var_m2_lum.fill(0.0);
                adap_conv.fill(false);
                n_converged = 0;
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
                        accumulator.clear();  accumulator.resize(n, Color::default());
                        scratch.clear();      scratch.resize(n, Color::default());
                        pixel_samples.clear(); pixel_samples.resize(n, 0);
                        var_m2_lum.clear();    var_m2_lum.resize(n, 0.0);
                        adap_conv.clear();     adap_conv.resize(n, false);
                        n_converged = 0;
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
                                            save_png(&blended, 1, None, scenes[scene_idx].name, win_w, win_h, exposure, tonemapper, Some(samples), None);
                                        } else {
                                            let ps = if adaptive_on { Some(pixel_samples.as_slice()) } else { None };
                                            save_png(&accumulator, samples, ps, scenes[scene_idx].name, win_w, win_h, exposure, tonemapper, None, None);
                                        }
                                    }
                                    #[cfg(not(feature = "denoise"))]
                                    {
                                        let ps = if adaptive_on { Some(pixel_samples.as_slice()) } else { None };
                                        save_png(&accumulator, samples, ps, scenes[scene_idx].name, win_w, win_h, exposure, tonemapper, None, None);
                                    }
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
                                    if let Background::Physical { sun_dir, .. } = &mut scenes[scene_idx].background {
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
                                    if let Background::Physical { sun_dir, .. } = &mut scenes[scene_idx].background {
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
                                    if oidn_on && samples > 0 && !denoise_running.load(Ordering::Acquire) {
                                        let ps = if adaptive_on { Some(pixel_samples.as_slice()) } else { None };
                                        spawn_denoiser(win_w, win_h, &accumulator, samples, ps,
                                            &aux_albedo, &aux_normal,
                                            &denoised, &denoise_running, &denoise_epoch);
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
                                VirtualKeyCode::R if scene_idx == 0 => {
                                    scenes[0] = build_random_scene();
                                    strata = (scenes[0].max_samples as f32).sqrt() as u32;
                                    reset_accum!();
                                    cam_dirty = true;
                                    pending_autofocus = true;
                                }
                                VirtualKeyCode::T => {
                                    tonemapper = if tonemapper == ToneMapper::AgX { ToneMapper::Aces } else { ToneMapper::AgX };
                                    window.request_redraw();
                                }
                                VirtualKeyCode::Slash => { print_controls(); }
                                VirtualKeyCode::V => {
                                    adaptive_on = !adaptive_on;
                                    reset_accum!();
                                    window.request_redraw();
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

                if last_title_update.elapsed() >= TITLE_INTERVAL {
                    let scene    = &scenes[scene_idx];
                    let cam_state_str = if free_cam { "FREE CAM" } else { "" };
                    #[cfg(feature = "denoise")]
                    let oidn_str = if oidn_on {
                        let state = if denoise_running.load(Ordering::Relaxed) { "running…" } else { "on" };
                        format!("  OIDN {state} {:.0}%", denoise_blend * 100.0)
                    } else { String::new() };
                    #[cfg(not(feature = "denoise"))]
                    let oidn_str = "";
                    let sun_str = if let Background::Physical { sun_dir, .. } = scene.background {
                        format!("  sun {:.0}°", sun_dir.y.asin().to_degrees())
                    } else { String::new() };
                    let adaptive_str = if adaptive_on {
                        let pct = n_converged * 100 / adap_conv.len().max(1);
                        format!("  adaptive {pct}% conv")
                    } else { String::new() };
                    let tm_str = if tonemapper == ToneMapper::AgX { "AgX" } else { "ACES" };
                    let cam_str = if cam_state_str.is_empty() {
                        format!("apt {:.2}  fov {:.0}°", cam_state.aperture, cam_state.vfov)
                    } else {
                        cam_state_str.to_string()
                    };
                    window.set_title(&format!(
                        "rustracer — {} — {samples} spp  |  {cam_str}  exp {:.2}  {tm_str}{}{}{}",
                        scene.name, exposure, sun_str, oidn_str, adaptive_str,
                    ));
                    last_title_update = Instant::now();
                }

                let all_conv = adaptive_on && n_converged == adap_conv.len();
                if samples < scenes[scene_idx].max_samples && !all_conv {
                    let scene = &scenes[scene_idx];
                    let bg    = scene.background;

                    let bg_scale = 1.0;
                    let conv_mask = if adaptive_on { Some(adap_conv.as_slice()) } else { None };
                    render_tiles(&mut scratch, conv_mask, samples, strata, win_w, win_h, &camera,
                                 scene.world.as_ref(), bg, &scene.lights, bg_scale,
                                 scene.photon_map.as_deref());

                    // Accumulate samples and update per-pixel Welford statistics.
                    // Converged pixels are skipped (scratch[i] == black, flag checked).
                    let lum = |c: Color| c.x * 0.2126 + c.y * 0.7152 + c.z * 0.0722;
                    accumulator.par_iter_mut()
                        .zip(scratch.par_iter())
                        .zip(var_m2_lum.par_iter_mut())
                        .zip(pixel_samples.par_iter_mut())
                        .zip(adap_conv.par_iter())
                        .for_each(|((((a, &s), m2), n), &conv)| {
                            if adaptive_on && conv { return; }
                            let s_lum   = lum(s);
                            let old_n   = *n;
                            let old_mean = if old_n > 0 { lum(*a) / old_n as f32 } else { 0.0 };
                            *a += s;
                            *n += 1;
                            let new_mean = lum(*a) / *n as f32;
                            *m2 += (s_lum - old_mean) * (s_lum - new_mean);
                        });
                    samples += 1;

                    // Mark pixels as converged when relative std error < threshold.
                    if adaptive_on && samples >= MIN_ADAPTIVE_SAMPLES {
                        adap_conv.par_iter_mut()
                            .zip(pixel_samples.par_iter())
                            .zip(var_m2_lum.par_iter())
                            .zip(accumulator.par_iter())
                            .for_each(|(((conv, &n), &m2), &acc)| {
                                if *conv || n < MIN_ADAPTIVE_SAMPLES { return; }
                                let mean_lum = lum(acc) / n as f32;
                                if mean_lum < 1e-4 { *conv = true; return; }
                                let variance = m2 / (n - 1).max(1) as f32;
                                let std_err  = (variance / n as f32).sqrt();
                                if std_err / mean_lum < ADAPTIVE_THRESHOLD { *conv = true; }
                            });
                        n_converged = adap_conv.iter().filter(|&&c| c).count();
                    }

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
                    if oidn_on && samples % 32 == 0 && !denoise_running.load(Ordering::Acquire) {
                        let ps = if adaptive_on { Some(pixel_samples.as_slice()) } else { None };
                        spawn_denoiser(win_w, win_h, &accumulator, samples, ps,
                            &aux_albedo, &aux_normal,
                            &denoised, &denoise_running, &denoise_epoch);
                    }

                    window.request_redraw();
                } else {
                    *control_flow = ControlFlow::Wait;
                }
            }

            Event::RedrawRequested(_) => {
                let mut buffer = surface.buffer_mut().unwrap();
                let sc = 1.0 / samples.max(1) as f32;
                let buf: &mut [u32] = &mut buffer;
                #[cfg(feature = "denoise")]
                {
                    let denoised_guard = denoised.lock().unwrap();
                    let use_denoised = oidn_on && denoised_guard.len() == (win_w * win_h) as usize;
                    if use_denoised {
                        buf.par_iter_mut()
                            .zip(accumulator.par_iter())
                            .zip(denoised_guard.par_iter())
                            .zip(pixel_samples.par_iter())
                            .for_each(|(((dst, &acc), &den), &n)| {
                                let n_sc = if adaptive_on { n.max(1) as f32 } else { samples.max(1) as f32 };
                                let raw = acc / n_sc;
                                *dst = to_rgb_u32(den * denoise_blend + raw * (1.0 - denoise_blend), exposure, tonemapper);
                            });
                    } else if adaptive_on {
                        write_tonemap_adaptive(buf, &accumulator, &pixel_samples, exposure, tonemapper);
                    } else {
                        write_tonemap(buf, &accumulator, sc, exposure, tonemapper);
                    }
                }
                #[cfg(not(feature = "denoise"))]
                if adaptive_on {
                    write_tonemap_adaptive(buf, &accumulator, &pixel_samples, exposure, tonemapper);
                } else {
                    write_tonemap(buf, &accumulator, sc, exposure, tonemapper);
                }
                buffer.present().unwrap();
            }

            _ => {}
        }
    });
}
