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
mod triangle;
mod mesh;
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
use scenes::{build_random_scene, build_cornell_box, build_nextweek_scene, build_benchmark_scene};
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

fn compute_strata(samples: u32) -> u32 { (samples as f32).sqrt() as u32 }

// ── Bench ─────────────────────────────────────────────────────────────────────

fn bench_scene(scene: &SceneData, scratch: &mut Vec<Color>, samples: u32) -> Duration {
    let aspect = WIDTH as f32 / HEIGHT as f32;
    let mut cam = CameraState::from_params(&scene.cam_init);
    cam.autofocus(&*scene.world);
    let camera = cam.to_camera(aspect);
    let world  = scene.world.as_ref();
    scratch.resize((WIDTH * HEIGHT) as usize, Color::default());

    let strata = compute_strata(samples);
    let t0 = Instant::now();
    for s in 0..samples {
        render_tiles(scratch, None, s, strata, WIDTH, HEIGHT, &camera, world, &scene.background, &scene.lights, 1.0, scene.photon_map.as_deref());
    }
    t0.elapsed()
}

fn run_bench() {
    const WARMUP:  u32 = 4;
    const SAMPLES: u32 = 128;

    let threads = rayon::current_num_threads();
    println!("rustracer benchmark — {}×{}  {} threads\n", WIDTH, HEIGHT, threads);

    let mut scratch: Vec<Color> = Vec::new();
    let mut scene = build_benchmark_scene();

    print!("Building scene… "); let _ = std::io::Write::flush(&mut std::io::stdout());
    scene.rebuild_caustics();
    println!("done.");

    print!("Warming up…     "); let _ = std::io::Write::flush(&mut std::io::stdout());
    bench_scene(&scene, &mut scratch, WARMUP);
    println!("done.\n");

    let elapsed  = bench_scene(&scene, &mut scratch, SAMPLES);
    let secs     = elapsed.as_secs_f64();
    let ms_per_s = secs * 1000.0 / SAMPLES as f64;
    let mpx_s    = (WIDTH as f64 * HEIGHT as f64 * SAMPLES as f64) / (secs * 1e6);

    println!("Score: {:.2} Mpx/s  ({} samples, {:.1} ms/sample, {:.1}s total)",
        mpx_s, SAMPLES, ms_per_s, secs);
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
    adaptive:         bool,
    convergence_pct:  f32,
    no_photon_map:    bool,
    #[cfg(feature = "denoise")]
    denoise:    bool,
}

impl Default for RenderArgs {
    fn default() -> Self {
        Self {
            scene:           "random".to_string(),
            samples:         None,
            width:           WIDTH,
            height:          HEIGHT,
            exposure:        1.0,
            output:          None,
            tonemapper:      ToneMapper::AgX,
            adaptive:        false,
            convergence_pct: 99.0,
            no_photon_map:   false,
            #[cfg(feature = "denoise")]
            denoise:    false,
        }
    }
}

fn print_controls() {
    println!("Controls:");
    println!("  [0]          Benchmark scene        [1-4]    Switch scene");
    println!("  [R]          Restart / reload scene");
    println!("  [F]          Toggle free camera    [C]      Reset camera");
    println!("  WASD         Move (free cam)       [Space]  Move up");
    println!("  Mouse        Look (free cam)       [Shift]  Move down");
    println!("  [,] [.]      Decrease / increase FOV");
    println!("  [-] [=]      Decrease / increase exposure");
    println!("  [I]  [O]     Decrease / increase aperture");
    println!("  [Arrows]     Rotate sun");
    println!("  [L]          Print camera position (look_from / look_at for scene.toml)");
    println!("  [P]          Save PNG");
    println!("  [Enter]      Pause rendering");
    println!("  [T]          Toggle tonemapper (AgX / ACES)");
    println!("  [V]          Toggle adaptive sampling");
    println!("  [M]          Toggle photon map (caustics on/off)");
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
    println!("  --scene <name>       benchmark|0  random|1  cornell|2  nextweek|3  <path.toml>");
    println!("                       Default: random");
    println!("  --samples <n>        Samples per pixel (default: scene maximum; unused with --adaptive)");
    println!("  --width  <n>         Output width  in pixels (default: {WIDTH})");
    println!("  --height <n>         Output height in pixels (default: {HEIGHT})");
    println!("  --exposure <f>       Exposure multiplier     (default: 1.0)");
    println!("  --output <path>      Output PNG path (default: auto-generated)");
    println!("  --tonemapper <name>  agx or aces             (default: agx)");
    println!("  --adaptive           Adaptive sampling: run until convergence target is reached");
    println!("  --convergence <pct>  Convergence target in %% (default: 99.0; requires --adaptive)");
    println!("  --no-photon-map      Disable caustic photon map (faster, no caustics)");
    #[cfg(feature = "denoise")]
    println!("  --denoise            Run OIDN after render and save denoised PNG");
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
            "--adaptive"      => r.adaptive      = true,
            "--convergence"   => r.convergence_pct = val!().parse::<f32>()
                .map_err(|e| format!("--convergence: {e}"))?.clamp(0.0, 100.0),
            "--no-photon-map" => r.no_photon_map = true,
            "--denoise"    => {
                #[cfg(feature = "denoise")]
                { r.denoise = true; }
                #[cfg(not(feature = "denoise"))]
                return Err("--denoise requires the 'denoise' feature (rebuild with: cargo run --features denoise)".to_string());
            }
            other => return Err(format!("Unknown flag '{other}'. Run with --help for usage.")),
        }
    }
    Ok(r)
}

fn run_render(args: RenderArgs) {
    let mut scene = {
        let sl = args.scene.to_lowercase();
        match sl.as_str() {
            "benchmark"| "0" => build_benchmark_scene(),
            "random"   | "1" => build_random_scene(),
            "cornell"  | "2" => build_cornell_box(),
            "nextweek" | "3" => build_nextweek_scene(),
            _ => match scene_file::load(&args.scene) {
                Ok(s)  => s,
                Err(e) => { eprintln!("Failed to load '{}': {e}", args.scene); std::process::exit(1); }
            },
        }
    };
    if !args.no_photon_map { scene.rebuild_caustics(); }

    let samples_max = args.samples.unwrap_or(scene.max_samples);
    let strata      = compute_strata(samples_max);
    let n_px        = (args.width * args.height) as usize;

    let mut cam = CameraState::from_params(&scene.cam_init);
    cam.autofocus(scene.world.as_ref());
    let camera = cam.to_camera(args.width as f32 / args.height as f32);

    let mut accumulator = vec![Color::default(); n_px];
    let mut scratch     = vec![Color::default(); n_px];

    let tm_name = if args.tonemapper == ToneMapper::AgX { "AgX" } else { "ACES" };
    println!("Scene:      {}", scene.name);
    println!("Resolution: {}×{}", args.width, args.height);
    if args.adaptive {
        println!("Samples:    adaptive  (target: {:.1}% convergence)", args.convergence_pct);
    } else {
        println!("Samples:    {samples_max}  (strata {strata}×{strata})");
    }
    println!("Tonemapper: {tm_name}  exposure {:.2}", args.exposure);
    if args.no_photon_map { println!("Photon map: off"); }
    #[cfg(feature = "denoise")]
    if args.denoise { println!("Denoiser:   OIDN"); }
    println!();

    let t0 = Instant::now();

    let adaptive        = args.adaptive;
    let convergence_pct = args.convergence_pct;
    let mut pixel_samples = if adaptive { vec![0u32;   n_px] } else { Vec::new() };
    let mut var_m2_lum    = if adaptive { vec![0.0f32; n_px] } else { Vec::new() };
    let mut adap_conv     = if adaptive { vec![false;  n_px] } else { Vec::new() };
    let mut n_converged   = 0usize;

    let mut s = 0u32;
    loop {
        if !adaptive && s >= samples_max { break; }

        render_tiles(&mut scratch,
                     if adaptive { Some(adap_conv.as_slice()) } else { None },
                     s, strata, args.width, args.height, &camera,
                     scene.world.as_ref(), &scene.background,
                     &scene.lights, 1.0, scene.photon_map.as_deref());

        accumulate_sample(&mut accumulator, &scratch, s,
                          &mut pixel_samples, &mut var_m2_lum, &adap_conv);
        s += 1;

        if adaptive && s >= MIN_ADAPTIVE_SAMPLES {
            n_converged = mark_converged(&mut adap_conv, &pixel_samples, &var_m2_lum, &accumulator);
        }

        let target_reached = n_converged as f32 >= n_px as f32 * convergence_pct / 100.0;
        let print_now = if adaptive {
            s.is_multiple_of(5) || target_reached
        } else {
            s % 5 == 4 || s == samples_max - 1
        };
        if print_now {
            let elapsed = t0.elapsed().as_secs_f64();
            if adaptive {
                let pct = n_converged as f32 * 100.0 / n_px.max(1) as f32;
                print!("\r  {s:>5} spp  {pct:>5.1}% converged  {elapsed:5.0}s  ");
            } else {
                let done   = s;
                let eta    = if done < samples_max { elapsed / done as f64 * (samples_max - done) as f64 } else { 0.0 };
                const BAR: usize = 32;
                let filled = (done as usize * BAR / samples_max as usize).min(BAR);
                let bar    = format!("{}{}", "█".repeat(filled), "░".repeat(BAR - filled));
                print!("\r  {done:>5}/{samples_max}  [{bar}]  {elapsed:5.0}s elapsed  ETA {eta:4.0}s  ");
            }
            let _ = std::io::Write::flush(&mut std::io::stdout());
        }
        if adaptive && target_reached { break; }
    }
    let actual_spp        = s;
    let pixel_samples_opt = if adaptive { Some(pixel_samples) } else { None };

    let elapsed = t0.elapsed();
    println!("\n\nRendered in {:.2}s  ({:.1} ms/spp)\n",
             elapsed.as_secs_f64(), elapsed.as_millis() as f64 / actual_spp.max(1) as f64);

    // ── OIDN denoising ────────────────────────────────────────────────────────
    #[cfg(feature = "denoise")]
    if args.denoise {
        let color: Vec<f32> = accumulator.iter().enumerate()
            .flat_map(|(i, c)| {
                let n = pixel_samples_opt.as_ref().map_or(actual_spp, |ps| ps[i]).max(1) as f32;
                [c.x / n, c.y / n, c.z / n]
            })
            .collect();
        print!("Denoising with OIDN…  ");
        let _ = std::io::Write::flush(&mut std::io::stdout());
        let (alb, nrm) = render_aux_pass(args.width, args.height, &camera,
                                         scene.world.as_ref(), &scene.background);
        let spp_label = actual_spp;
        match denoise::denoise_rgb(args.width, args.height, color, alb, nrm) {
            Some(denoised) => {
                let denoised_path = args.output.clone().unwrap_or_else(|| {
                    let slug = scene.name.to_lowercase().replace(' ', "_");
                    format!("render_{}_{:04}spp_denoised.png", slug, spp_label)
                });
                // denoised buffer is already per-pixel-normalized; pass samples=1 so save_png
                // applies exposure directly without a second division
                save_png(&denoised, 1, None, scene.name,
                         args.width, args.height, args.exposure, args.tonemapper,
                         Some(spp_label), Some(&denoised_path));
            }
            None => eprintln!("OIDN denoising failed."),
        }
        return;
    }

    // ── Raw PNG save ──────────────────────────────────────────────────────────
    let ps    = pixel_samples_opt.as_deref();
    let label: Option<u32> = None;
    save_png(&accumulator, actual_spp, ps, scene.name,
             args.width, args.height, args.exposure, args.tonemapper,
             label, args.output.as_deref());
}

// ── Shared helpers ────────────────────────────────────────────────────────────

fn luminance(c: Color) -> f32 { c.x * 0.2126 + c.y * 0.7152 + c.z * 0.0722 }

/// Accumulates one rendered sample into the accumulator with firefly clamping.
/// When `pixel_samples` is non-empty, per-pixel adaptive bookkeeping is updated;
/// otherwise the global `cur_sample` count is used for the clamp threshold.
fn accumulate_sample(
    accumulator:   &mut [Color],
    scratch:       &[Color],
    cur_sample:    u32,
    pixel_samples: &mut [u32],
    var_m2_lum:    &mut [f32],
    adap_conv:     &[bool],
) {
    if pixel_samples.is_empty() {
        accumulator.par_iter_mut()
            .zip(scratch.par_iter())
            .for_each(|(a, &sc)| {
                let old_mean = if cur_sample > 0 { luminance(*a) / cur_sample as f32 } else { 0.0 };
                let sc = if cur_sample >= FIREFLY_MIN_SAMPLES && old_mean > 1e-6 {
                    let ratio = luminance(sc) / old_mean;
                    if ratio > FIREFLY_CLAMP { sc * (FIREFLY_CLAMP / ratio) } else { sc }
                } else { sc };
                *a += sc;
            });
    } else {
        accumulator.par_iter_mut()
            .zip(scratch.par_iter())
            .zip(var_m2_lum.par_iter_mut())
            .zip(pixel_samples.par_iter_mut())
            .zip(adap_conv.par_iter())
            .for_each(|((((a, &sc), m2), n), &conv)| {
                if conv { return; }
                let old_n    = *n;
                let old_mean = if old_n > 0 { luminance(*a) / old_n as f32 } else { 0.0 };
                let sc = if old_n >= FIREFLY_MIN_SAMPLES && old_mean > 1e-6 {
                    let ratio = luminance(sc) / old_mean;
                    if ratio > FIREFLY_CLAMP { sc * (FIREFLY_CLAMP / ratio) } else { sc }
                } else { sc };
                *a  += sc;
                *n  += 1;
                let sc_lum   = luminance(sc);
                let new_mean = luminance(*a) / *n as f32;
                *m2 += (sc_lum - old_mean) * (sc_lum - new_mean);
            });
    }
}

/// Updates per-pixel convergence flags and returns the count of converged pixels.
fn mark_converged(
    adap_conv:     &mut [bool],
    pixel_samples: &[u32],
    var_m2_lum:    &[f32],
    accumulator:   &[Color],
) -> usize {
    adap_conv.par_iter_mut()
        .zip(pixel_samples.par_iter())
        .zip(var_m2_lum.par_iter())
        .zip(accumulator.par_iter())
        .for_each(|(((conv, &n), &m2), &acc)| {
            if *conv || n < MIN_ADAPTIVE_SAMPLES { return; }
            let mean_lum = luminance(acc) / n as f32;
            if mean_lum < 1e-4 { *conv = true; return; }
            let variance = m2 / (n - 1).max(1) as f32;
            let std_err  = (variance / n as f32).sqrt();
            if std_err / mean_lum < ADAPTIVE_THRESHOLD { *conv = true; }
        });
    adap_conv.iter().filter(|&&c| c).count()
}

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
/// Per-sample firefly clamp: a new sample whose luminance exceeds this multiple
/// of the running pixel mean is scaled down to this ratio × mean.  Adapts to
/// local brightness so genuinely bright pixels (all samples bright) are not
/// biased, while rare spikes in otherwise dark pixels are suppressed.
/// Applied only after FIREFLY_MIN_SAMPLES are accumulated.
const FIREFLY_CLAMP:        f32 = 8.0;
const FIREFLY_MIN_SAMPLES:  u32 = 4;

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
    rayon::ThreadPoolBuilder::new()
        .stack_size(2 * 1024 * 1024)
        .build_global()
        .unwrap();
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
    println!("Scene 0: Benchmark (32 768-triangle mesh + spectral glass + caustics)");
    let s_bench = build_benchmark_scene();
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

    scenes.push(s_bench);
    let bench_idx = scenes.len() - 1;

    // Build the photon map only for the initially active scene (scene 0).
    // All other scenes build their maps lazily when first switched to.
    scenes[0].rebuild_caustics();

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
    let mut strata = compute_strata(scenes[scene_idx].max_samples);
    // Adaptive sampling state
    let mut tonemapper     = ToneMapper::AgX;
    let mut adaptive_on    = false;
    let mut photon_map_on  = true;
    let mut pixel_samples: Vec<u32>   = Vec::new(); // allocated only when adaptive_on
    let mut var_m2_lum:   Vec<f32>    = Vec::new();
    let mut adap_conv:    Vec<bool>   = Vec::new();
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
                strata           = compute_strata(scenes[i].max_samples);
                reset_accum!();
                cam_dirty        = true;
                pending_autofocus = true;
                // Build photon map on first visit; subsequent visits reuse the cached map.
                // Sun-direction changes rebuild it separately via rebuild_caustics() below.
                if scenes[i].photon_map.is_none() {
                    scenes[i].rebuild_caustics();
                }
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
                        if adaptive_on {
                            pixel_samples.clear(); pixel_samples.resize(n, 0);
                            var_m2_lum.clear();    var_m2_lum.resize(n, 0.0);
                            adap_conv.clear();     adap_conv.resize(n, false);
                            n_converged = 0;
                        }
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
                                VirtualKeyCode::Key0 => { switch_scene!(bench_idx); }
                                VirtualKeyCode::Key1 | VirtualKeyCode::Key2 | VirtualKeyCode::Key3 => {
                                    let idx = match key { VirtualKeyCode::Key2 => 1, VirtualKeyCode::Key3 => 2, _ => 0 };
                                    switch_scene!(idx);
                                }
                                VirtualKeyCode::Key4 => {
                                    if scenes.len() >= 5 {
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
                                VirtualKeyCode::L => {
                                    let from = cam_state.pos;
                                    let at   = cam_state.look_at();
                                    println!("# paste into [camera] in scene.toml");
                                    println!("look_from = [{:.4}, {:.4}, {:.4}]", from.x, from.y, from.z);
                                    println!("look_at   = [{:.4}, {:.4}, {:.4}]", at.x,   at.y,   at.z);
                                    println!("vfov      = {:.1}", cam_state.vfov);
                                    println!("aperture  = {:.3}", cam_state.aperture);
                                }
                                VirtualKeyCode::I => {
                                    cam_state.aperture = (cam_state.aperture - 0.025).max(0.0);
                                    cam_dirty = true;
                                }
                                VirtualKeyCode::O => {
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
                                    if oidn_on {
                                        if samples > 0 && !denoise_running.load(Ordering::Acquire) {
                                            let ps = if adaptive_on { Some(pixel_samples.as_slice()) } else { None };
                                            spawn_denoiser(win_w, win_h, &accumulator, samples, ps,
                                                &aux_albedo, &aux_normal,
                                                &denoised, &denoise_running, &denoise_epoch);
                                        }
                                    } else {
                                        // Free denoised output and aux buffers when OIDN is turned off.
                                        *denoised.lock().unwrap() = Vec::new();
                                        aux_albedo = Vec::new();
                                        aux_normal = Vec::new();
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
                                    scenes[0].rebuild_caustics();
                                    strata = compute_strata(scenes[0].max_samples);
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
                                    if adaptive_on {
                                        let n = (win_w * win_h) as usize;
                                        pixel_samples = vec![0u32;   n];
                                        var_m2_lum    = vec![0.0f32; n];
                                        adap_conv     = vec![false;  n];
                                    } else {
                                        pixel_samples = Vec::new();
                                        var_m2_lum    = Vec::new();
                                        adap_conv     = Vec::new();
                                        n_converged   = 0;
                                    }
                                    reset_accum!();
                                    window.request_redraw();
                                }
                                VirtualKeyCode::M => {
                                    photon_map_on = !photon_map_on;
                                    reset_accum!();
                                    window.request_redraw();
                                }
                                VirtualKeyCode::R if scene_idx == 3 => {
                                    match scene_file::load("scene.toml") {
                                        Ok(s) => {
                                            println!("Reloaded: {}", s.name);
                                            scenes[3] = s;
                                            scenes[3].rebuild_caustics();
                                            cam_state = CameraState::from_params(&scenes[3].cam_init);
                                            strata = compute_strata(scenes[3].max_samples);
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
                    let sun_str = if let Background::Physical { sun_dir, .. } = &scene.background {
                        format!("  sun {:.0}°", sun_dir.y.asin().to_degrees())
                    } else { String::new() };
                    let adaptive_str = if adaptive_on {
                        let pct = n_converged * 100 / adap_conv.len().max(1);
                        format!("  adaptive {pct}% conv")
                    } else { String::new() };
                    let pm_str = if !photon_map_on && scene.photon_map.is_some() {
                        "  [photon map OFF]"
                    } else { "" };
                    let tm_str = if tonemapper == ToneMapper::AgX { "AgX" } else { "ACES" };
                    let cam_str = if cam_state_str.is_empty() {
                        format!("apt {:.2}  fov {:.0}°", cam_state.aperture, cam_state.vfov)
                    } else {
                        cam_state_str.to_string()
                    };
                    window.set_title(&format!(
                        "rustracer — {} — {samples} spp  |  {cam_str}  exp {:.2}  {tm_str}{}{}{}{}",
                        scene.name, exposure, sun_str, oidn_str, adaptive_str, pm_str,
                    ));
                    last_title_update = Instant::now();
                }

                let all_conv = adaptive_on && n_converged == adap_conv.len();
                if samples < scenes[scene_idx].max_samples && !all_conv {
                    let scene = &scenes[scene_idx];
                    let bg_scale = 1.0;
                    let conv_mask = if adaptive_on { Some(adap_conv.as_slice()) } else { None };
                    let pm = if photon_map_on { scene.photon_map.as_deref() } else { None };
                    render_tiles(&mut scratch, conv_mask, samples, strata, win_w, win_h, &camera,
                                 scene.world.as_ref(), &scene.background, &scene.lights, bg_scale,
                                 pm);

                    accumulate_sample(&mut accumulator, &scratch, samples,
                                      &mut pixel_samples, &mut var_m2_lum, &adap_conv);
                    samples += 1;

                    if adaptive_on && samples >= MIN_ADAPTIVE_SAMPLES {
                        n_converged = mark_converged(&mut adap_conv, &pixel_samples, &var_m2_lum, &accumulator);
                    }

                    // Build the aux buffers once per render sequence (first sample),
                    // so they are ready before the first OIDN invocation at sample 32.
                    #[cfg(feature = "denoise")]
                    if oidn_on && samples == 1 {
                        let (alb, nrm) = render_aux_pass(win_w, win_h, &camera,
                                                         scene.world.as_ref(), &scene.background);
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
                        if adaptive_on {
                            buf.par_iter_mut()
                                .zip(accumulator.par_iter())
                                .zip(denoised_guard.par_iter())
                                .zip(pixel_samples.par_iter())
                                .for_each(|(((dst, &acc), &den), &n)| {
                                    let raw = acc / n.max(1) as f32;
                                    *dst = to_rgb_u32(den * denoise_blend + raw * (1.0 - denoise_blend), exposure, tonemapper);
                                });
                        } else {
                            let sc = 1.0 / samples.max(1) as f32;
                            buf.par_iter_mut()
                                .zip(accumulator.par_iter())
                                .zip(denoised_guard.par_iter())
                                .for_each(|((dst, &acc), &den)| {
                                    let raw = acc * sc;
                                    *dst = to_rgb_u32(den * denoise_blend + raw * (1.0 - denoise_blend), exposure, tonemapper);
                                });
                        }
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
