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
mod scene_file;
mod output;
mod animation;
mod diamond;
mod step_cut;
mod photon;
#[cfg(feature = "denoise")]
mod denoise;

use rayon::prelude::*;
use vec3::Color;
use camera::{Camera, CameraState};
use vec3::Vec3;
#[cfg(not(feature = "denoise"))]
use renderer::{Background, render_tiles};
#[cfg(feature = "denoise")]
use renderer::{Background, render_tiles, render_aux_pass};
use scene::SceneData;
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
        render_tiles(scratch, None, s, strata, WIDTH, HEIGHT, &camera, world, &scene.background, &scene.lights, 1.0, scene.photon_map.as_deref(), None);
    }
    t0.elapsed()
}

fn run_bench() {
    const WARMUP:  u32 = 4;
    const SAMPLES: u32 = 128;

    let threads = rayon::current_num_threads();
    println!("rustracer benchmark — {}×{}  {} threads\n", WIDTH, HEIGHT, threads);

    let mut scratch: Vec<Color> = Vec::new();
    // The benchmark scene is a plain TOML file (bench.toml by default) so the
    // reference stays visible/editable; pass another file to bench that instead.
    let path = std::env::args().nth(2).filter(|p| p.ends_with(".toml"))
        .unwrap_or_else(|| "scenes/bench.toml".to_string());
    println!("Scene: {path}\n");
    let mut scene = match scene_file::load(&path) {
        Ok(s)  => s,
        Err(e) => { eprintln!("Failed to load '{path}': {e}"); std::process::exit(1); }
    };

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
    min_samples:      u32,
    no_photon_map:    bool,
    frames:          Option<u32>,
    #[cfg(feature = "denoise")]
    denoise:    Option<f32>,
}

impl Default for RenderArgs {
    fn default() -> Self {
        Self {
            scene:           "scenes/scene.toml".to_string(),
            samples:         None,
            width:           WIDTH,
            height:          HEIGHT,
            exposure:        1.0,
            output:          None,
            tonemapper:      ToneMapper::AgX,
            adaptive:        false,
            convergence_pct: 99.0,
            min_samples:     128,
            no_photon_map:   false,
            frames:          None,
            #[cfg(feature = "denoise")]
            denoise:    None,
        }
    }
}

fn print_controls() {
    println!("Controls:");
    println!("  [R]          Reload scene file");
    println!("  [F]          Toggle free camera    [C]      Reset camera");
    println!("  WASD         Move (free cam)       [Space]  Move up");
    println!("  Mouse        Look (free cam)       [Shift]  Move down");
    println!("  [,] [.]      Decrease / increase FOV");
    println!("  [-] [=]      Decrease / increase exposure");
    println!("  [I]  [O]     Decrease / increase aperture");
    println!("  [Arrows]     Rotate sun");
    println!("  [L]          Print camera position (look_from / look_at for scenes/scene.toml)");
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
    println!("  rustracer                      Interactive viewer (default scene: scenes/scene.toml)");
    println!("  rustracer --scene <path.toml>  Interactive viewer for that scene file ([R] reloads)");
    println!("  rustracer --render [options]   Headless render to PNG");
    println!("  rustracer --bench [scene.toml] Performance benchmark (default: scenes/bench.toml)");
    println!("  rustracer --help               Show this message\n");
    println!("RENDER OPTIONS:");
    println!("  --scene <path.toml>  Scene file to render (default: scenes/scene.toml)");
    println!("  --samples <n>        Samples per pixel (default: scene maximum; with --adaptive, the hard SPP cap)");
    println!("  --preset <name>      Measured recipes: preview = 16 spp + OIDN 0.6, final = 64 spp + OIDN 1.0,");
    println!("                       reference = adaptive, cap 4096 spp.  Explicit flags override the preset.");
    println!("  --width  <n>         Output width  in pixels (default: {WIDTH})");
    println!("  --height <n>         Output height in pixels (default: {HEIGHT})");
    println!("  --exposure <f>       Exposure multiplier     (default: 1.0)");
    println!("  --output <path>      Output PNG path (default: auto-generated)");
    println!("  --tonemapper <name>  agx or aces             (default: agx)");
    println!("  --adaptive           Adaptive sampling: run until convergence target is reached");
    println!("  --convergence <pct>  Convergence target in %% (default: 99.0; requires --adaptive)");
    println!("  --min-samples <n>    Minimum SPP before any pixel may be marked converged (default: 128; requires --adaptive)");
    println!("  --no-photon-map      Disable caustic photon map (faster, no caustics)");
    println!("  --frames <n>         Render only the first N frames of a scene animation; 1 = single still (default: all)");
    #[cfg(feature = "denoise")]
    println!("  --denoise [strength] Run OIDN after render; optional strength in [0.0, 1.0] (default: 1.0)");
}

#[cfg(feature = "denoise")]
fn set_denoise(r: &mut RenderArgs, strength: f32) { r.denoise = Some(strength); }
#[cfg(not(feature = "denoise"))]
fn set_denoise(_r: &mut RenderArgs, _strength: f32) {
    eprintln!("note: preset wants OIDN denoising but this binary lacks the 'denoise' feature; rendering raw (rebuild: cargo build --release --features denoise)");
}

/// Measured render recipes (gem scene, 640x400, reference = adaptive 1024 spp):
/// 32 spp + OIDN reached 33 dB PSNR in ~0.4 s, while the raw adaptive reference
/// path took 10.3 s.  Low fixed spp + OIDN is the compute-per-quality winner
/// for stills; video wants a little more spp per frame (OIDN runs per frame).
fn apply_preset(r: &mut RenderArgs, name: &str) -> Result<(), String> {
    match name {
        "preview"   => { r.samples = Some(16);  set_denoise(r, 0.6); }
        "final"     => { r.samples = Some(64);  set_denoise(r, 1.0); }
        "reference" => {
            r.adaptive        = true;
            r.min_samples     = 128;
            r.convergence_pct = 99.0;
            r.samples         = Some(4096);
        }
        other => return Err(format!("Unknown preset '{other}': use preview, final or reference")),
    }
    Ok(())
}

fn parse_render_args() -> Result<RenderArgs, String> {
    let mut r  = RenderArgs::default();
    let argv: Vec<String> = std::env::args().skip(2).collect(); // skip binary + "--render"
    // Apply --preset first so any explicit flag later in the command line
    // overrides the preset value, regardless of flag order.
    if let Some(pos) = argv.iter().position(|a| a == "--preset") {
        let name = argv.get(pos + 1).ok_or("--preset requires a value (preview|final|reference)")?;
        apply_preset(&mut r, name)?;
    }
    let mut it = argv.into_iter().peekable();
    while let Some(flag) = it.next() {
        macro_rules! val {
            () => { it.next().ok_or_else(|| format!("'{flag}' requires a value"))? }
        }
        match flag.as_str() {
            "--preset"    => { val!(); } // applied by the pre-scan above
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
            "--min-samples"   => r.min_samples = val!().parse::<u32>()
                .map_err(|e| format!("--min-samples: {e}"))?,
            "--no-photon-map" => r.no_photon_map = true,
            "--frames"    => r.frames = Some(val!().parse::<u32>().map_err(|e| format!("--frames: {e}"))?),
            "--denoise"    => {
                #[cfg(feature = "denoise")]
                {
                    let strength = match it.peek() {
                        Some(s) if s.parse::<f32>().is_ok() => {
                            it.next().unwrap().parse::<f32>().unwrap().clamp(0.0, 1.0)
                        }
                        _ => 1.0,
                    };
                    r.denoise = Some(strength);
                }
                #[cfg(not(feature = "denoise"))]
                return Err("--denoise requires the 'denoise' feature (rebuild with: cargo run --features denoise)".to_string());
            }
            other => return Err(format!("Unknown flag '{other}'. Run with --help for usage.")),
        }
    }
    Ok(r)
}

fn run_render(args: RenderArgs) {
    let mut scene = match scene_file::load(&args.scene) {
        Ok(s)  => s,
        Err(e) => {
            eprintln!("Failed to load '{}': {e}", args.scene);
            eprintln!("Scenes are .toml files now - e.g. --scene scenes/scene.toml");
            std::process::exit(1);
        }
    };
    if !args.no_photon_map { scene.rebuild_caustics(); }

    let samples_max = args.samples.unwrap_or(scene.max_samples);
    let strata      = compute_strata(samples_max);
    let n_px        = (args.width * args.height) as usize;
    let aspect      = args.width as f32 / args.height as f32;
    let slug        = scene.name.to_lowercase().replace(' ', "_");

    // Take animation out of the scene (if present) so we own it independently.
    let animation   = scene.animation.take();
    let total_frames = animation.as_ref().map_or(1u32, |a| a.total_frames().max(1));
    // --frames N caps the frame count; --frames 1 renders a plain single still
    // (--output is then a file path, not a directory).
    let total_frames = args.frames.map_or(total_frames, |n| n.min(total_frames).max(1));
    let is_anim      = animation.is_some() && total_frames > 1;

    // ── One-time header ───────────────────────────────────────────────────────
    let tm_name = if args.tonemapper == ToneMapper::AgX { "AgX" } else { "ACES" };
    println!("Scene:      {}", scene.name);
    println!("Resolution: {}×{}", args.width, args.height);
    if is_anim {
        let a = animation.as_ref().unwrap();
        println!("Animation:  {} frames  ·  {:.0} fps  ·  {:.1}s",
                 total_frames, a.fps, a.duration);
        let out_dir = args.output.as_deref().unwrap_or(".");
        println!("Output dir: {out_dir}");
    }
    if args.adaptive {
        println!("Samples:    adaptive  (target: {:.1}% convergence)", args.convergence_pct);
    } else {
        println!("Samples:    {samples_max}  (strata {strata}×{strata})");
    }
    println!("Tonemapper: {tm_name}  exposure {:.2}", args.exposure);
    if args.no_photon_map { println!("Photon map: off"); }
    #[cfg(feature = "denoise")]
    if let Some(blend) = args.denoise {
        if blend < 1.0 { println!("Denoiser:   OIDN ({:.0}% blend)", blend * 100.0); }
        else           { println!("Denoiser:   OIDN"); }
    }
    println!();

    // ── Per-frame loop ────────────────────────────────────────────────────────
    let mut accumulator = vec![Color::default(); n_px];
    let mut scratch     = vec![Color::default(); n_px];

    let adaptive        = args.adaptive;
    let convergence_pct = args.convergence_pct;
    let min_samples     = args.min_samples;

    // Animation frames are written into the output dir - make sure it exists
    // (save_png does not create directories).
    if is_anim {
        let out_dir = args.output.as_deref().unwrap_or(".");
        let _ = std::fs::create_dir_all(out_dir);
    }

    for frame_idx in 0..total_frames {
        // ── Camera for this frame ─────────────────────────────────────────────
        let camera: Camera = if let Some(anim) = &animation {
            let t  = frame_idx as f32 / anim.fps;
            let kf = anim.evaluate(t);
            Camera::new(kf.look_from, kf.look_at, Vec3::new(0.0, 1.0, 0.0),
                        kf.vfov, aspect, kf.aperture, kf.focus_dist,
                        scene.cam_init.aperture_blades)
        } else {
            let mut cam = CameraState::from_params(&scene.cam_init);
            cam.autofocus(scene.world.as_ref());
            cam.to_camera(aspect)
        };

        if is_anim {
            println!("[Frame {:>4} / {}]", frame_idx + 1, total_frames);
        }

        // ── Reset per-frame state ─────────────────────────────────────────────
        accumulator.iter_mut().for_each(|c| *c = Color::default());
        scratch.iter_mut().for_each(|c| *c = Color::default());
        let mut pixel_samples = vec![0u32; n_px];
        let mut var_m2        = vec![Color::default(); n_px];
        let mut adap_conv     = vec![false; n_px];
        let mut n_converged   = 0usize;

        // ── Render loop ───────────────────────────────────────────────────────
        // Animation: each frame motion-blurs over its own slice of the timeline
        // [frame/fps, (frame+1)/fps), not over the whole [0, 1) range.
        let frame_time = animation.as_ref().map(|a| {
            let t0 = frame_idx as f32 / a.fps;
            (t0, t0 + 1.0 / a.fps)
        });
        let t0 = Instant::now();
        let mut s = 0u32;
        loop {
            // Hard SPP cap in both modes. In adaptive mode this is --samples (or
            // the scene default) even when the convergence target is not reached,
            // so a few never-converging pixels cannot hang the render forever.
            if s >= samples_max { break; }

            render_tiles(&mut scratch,
                         if adaptive { Some(adap_conv.as_slice()) } else { None },
                         s, strata, args.width, args.height, &camera,
                         scene.world.as_ref(), &scene.background,
                         &scene.lights, 1.0, scene.photon_map.as_deref(), frame_time);

            accumulate_sample(&mut accumulator, &scratch,
                              &mut pixel_samples, &mut var_m2, &adap_conv);
            s += 1;

            if adaptive && s >= min_samples {
                n_converged = mark_converged(&mut adap_conv, &pixel_samples, &var_m2, &accumulator);
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
        if adaptive && (n_converged as f32) < (n_px as f32) * convergence_pct / 100.0 {
            let pct = n_converged as f32 * 100.0 / n_px.max(1) as f32;
            println!("\n  Warning: {:.1}% convergence target not reached — stopped at the {samples_max}-spp cap ({pct:.1}% converged).",
                     convergence_pct);
        }
        let actual_spp        = s;
        let pixel_samples_opt = Some(pixel_samples);

        let elapsed = t0.elapsed();
        println!("\n\n  Rendered in {:.2}s  ({:.1} ms/spp)\n",
                 elapsed.as_secs_f64(), elapsed.as_millis() as f64 / actual_spp.max(1) as f64);

        // ── Save (OIDN or raw) ────────────────────────────────────────────────
        #[allow(unused_mut)]
        let mut saved = false;

        #[cfg(feature = "denoise")]
        if let Some(blend) = args.denoise {
            let raw: Vec<Color> = accumulator.iter().enumerate()
                .map(|(i, c)| {
                    let n = pixel_samples_opt.as_ref().map_or(actual_spp, |ps| ps[i]).max(1) as f32;
                    Color::new(c.x / n, c.y / n, c.z / n)
                })
                .collect();
            let color: Vec<f32> = raw.iter().flat_map(|c| [c.x, c.y, c.z]).collect();
            print!("  Denoising with OIDN…  ");
            let _ = std::io::Write::flush(&mut std::io::stdout());
            let (alb, nrm) = render_aux_pass(args.width, args.height, &camera,
                                             scene.world.as_ref(), &scene.background);
            match denoise::denoise_rgb(args.width, args.height, color, alb, nrm) {
                Some(denoised) => {
                    let output: Vec<Color> = if blend < 1.0 {
                        denoised.iter().zip(raw.iter())
                            .map(|(&den, &r)| den * blend + r * (1.0 - blend))
                            .collect()
                    } else {
                        denoised
                    };
                    let path = if is_anim {
                        let out_dir = args.output.as_deref().unwrap_or(".");
                        format!("{out_dir}/{slug}_{frame_idx:04}.png")
                    } else {
                        args.output.clone().unwrap_or_else(|| {
                            format!("render_{}_{:04}spp.png", slug, actual_spp)
                        })
                    };
                    // output is already per-pixel-normalized; samples=1 so save_png
                    // applies exposure directly without a second division
                    save_png(&output, 1, None, scene.name,
                             args.width, args.height, args.exposure, args.tonemapper,
                             Some(actual_spp), Some(&path));
                }
                None => eprintln!("OIDN denoising failed."),
            }
            saved = true;
        }

        if !saved {
            let ps = pixel_samples_opt.as_deref();
            let out_path: Option<String> = if is_anim {
                let out_dir = args.output.as_deref().unwrap_or(".");
                Some(format!("{out_dir}/{slug}_{frame_idx:04}.png"))
            } else {
                args.output.clone()
            };
            save_png(&accumulator, actual_spp, ps, scene.name,
                     args.width, args.height, args.exposure, args.tonemapper,
                     None, out_path.as_deref());
        }
    }

    if is_anim {
        let out_dir = args.output.as_deref().unwrap_or(".");
        println!("\nAll frames saved to {out_dir}/");
        println!("Assemble with ffmpeg:");
        let fps = animation.as_ref().map_or(24.0, |a| a.fps);
        println!("  ffmpeg -r {fps:.0} -i \"{out_dir}/{slug}_%04d.png\" -c:v libx264 -pix_fmt yuv420p output.mp4");
    }
}

// ── Shared helpers ────────────────────────────────────────────────────────────

fn luminance(c: Color) -> f32 { c.luminance() }

/// Accumulates one rendered sample into the accumulator with firefly clamping.
/// Per-pixel sample counts are always tracked; `adap_conv` is all-false when
/// adaptive sampling is off, so converged-pixel skipping is a no-op there.
fn accumulate_sample(
    accumulator:   &mut [Color],
    scratch:       &[Color],
    pixel_samples: &mut [u32],
    var_m2:        &mut [Color],
    adap_conv:     &[bool],
) {
    accumulator.par_iter_mut()
        .zip(scratch.par_iter())
        .zip(pixel_samples.par_iter_mut())
        .zip(var_m2.par_iter_mut())
        .zip(adap_conv.par_iter())
        .for_each(|((((a, &sc), n), m2), &conv)| {
            if conv { return; }
            // Drop non-finite samples entirely (no accumulate, no count): an inf/NaN
            // sample would otherwise poison the running mean and, via the firefly
            // clamp (inf * 0 = NaN), the accumulator forever. Because `n` counts only
            // finite samples, the clamp divisor stays exact even when drops occur.
            if !sc.x.is_finite() || !sc.y.is_finite() || !sc.z.is_finite() { return; }
            let old_n    = *n;
            let old_mean = if old_n > 0 { *a / old_n as f32 } else { Color::default() };
            // Firefly clamp keys on luminance (a chromatic spike is still a spike).
            // The threshold is max(FIREFLY_CLAMP * old_lum, FIREFLY_ABS_FLOOR): without
            // the absolute floor, a pixel whose running mean is still ~0 (e.g. a dark
            // pixel whose first samples are all zero) would clamp its first legitimate
            // bright sample to ~0 and render black despite a healthy true mean.
            let old_lum  = luminance(old_mean);
            let sc = if old_n >= FIREFLY_MIN_SAMPLES {
                let l      = luminance(sc);
                let thresh = (FIREFLY_CLAMP * old_lum).max(FIREFLY_ABS_FLOOR);
                if l > thresh { sc * (thresh / l) } else { sc }
            } else { sc };
            *a  += sc;
            *n  += 1;
            // Per-channel Welford update: variance is tracked for R, G, B
            // independently so chromatic noise is visible to mark_converged.
            let new_mean = *a / *n as f32;
            *m2 += (sc - old_mean) * (sc - new_mean);
        });
}

/// Updates per-pixel convergence flags and returns the count of converged pixels.
fn mark_converged(
    adap_conv:     &mut [bool],
    pixel_samples: &[u32],
    var_m2:    &[Color],
    accumulator:   &[Color],
) -> usize {
    adap_conv.par_iter_mut()
        .zip(pixel_samples.par_iter())
        .zip(var_m2.par_iter())
        .zip(accumulator.par_iter())
        .for_each(|(((conv, &n), &m2), &acc)| {
            if *conv { return; }
            // n == 0 means every sample was dropped as non-finite — leave it
            // unconverged rather than freezing the pixel as black.
            if n == 0 { return; }
            // Per-channel relative standard error: a pixel is converged only when
            // EVERY non-dark channel is below the threshold. Channels with
            // mean < 1e-4 are exempt, so a pixel whose luminance is dark but that
            // has one bright channel (e.g. a saturated caustic) is still tracked.
            let mean = acc / n.max(1) as f32;
            let mut all_ok = true;
            for c in 0..3 {
                let m = mean[c];
                if m < 1e-4 { continue; }
                let variance = m2[c] / (n - 1).max(1) as f32;
                let std_err  = (variance / n as f32).sqrt();
                if std_err / m >= ADAPTIVE_THRESHOLD { all_ok = false; break; }
            }
            if all_ok { *conv = true; }
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

/// Maximum relative standard error (σ/μ) to consider a pixel converged.
const ADAPTIVE_THRESHOLD:   f32 = 0.05;
/// Per-sample firefly clamp: a new sample whose luminance exceeds this multiple
/// of the running pixel mean is scaled down to this ratio × mean.
///
/// **Bias:** this intentionally introduces a downward bias on high-variance
/// pixels.  Once a pixel has accumulated many samples its running mean tracks
/// its true brightness, so the clamp rarely fires on consistently bright
/// pixels (e.g. direct emitter hits).  The bias is most visible *early* in
/// the accumulation: a legitimately bright sample (caustic specular path,
/// first hit on an area light) can be clamped against a low running mean
/// before enough samples have been gathered to raise it.  This is an
/// acceptable tradeoff for interactive rendering — use OIDN denoising
/// (`--denoise`) for bias-free firefly suppression in final renders.
///
/// Photon-map caustics are unaffected: the photon irradiance estimate is
/// constant across samples, so it contributes equally to every sample and
/// never appears as a spike relative to the running mean.
const FIREFLY_CLAMP:        f32 = 8.0;
const FIREFLY_MIN_SAMPLES:  u32 = 4;
/// Absolute floor (linear luminance) for the firefly-clamp threshold — see
/// accumulate_sample. Samples below this are never clamped.
const FIREFLY_ABS_FLOOR:    f32 = 5.0;

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

    // Top-level `--scene <path.toml>` (interactive mode only): the hot-reloadable
    // file scene (slot 4, [R] to reload) loads this file instead of scene.toml.
    let mut file_scene = String::from("scenes/scene.toml");
    {
        let args: Vec<String> = std::env::args().skip(1).collect();
        if let Some(pos) = args.iter().position(|a| a == "--scene") {
            if let Some(p) = args.get(pos + 1) {
                if !p.ends_with(".toml") || p == "--render" {
                    // not a file path — leave default
                } else {
                    file_scene = p.clone();
                }
            }
        }
    }

    println!("Loading scene: {file_scene}");
    let mut scene: SceneData = match scene_file::load(&file_scene) {
        Ok(s)  => s,
        Err(e) => { eprintln!("Failed to load '{file_scene}': {e}"); std::process::exit(1); }
    };
    scene.rebuild_caustics();

    print_controls();

    let file_scene_evt = file_scene.clone();
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
    let mut cam_state   = CameraState::from_params(&scene.cam_init);
    cam_state.autofocus(scene.world.as_ref());
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
    let mut strata = compute_strata(scene.max_samples);
    // Adaptive sampling state
    let mut tonemapper          = ToneMapper::AgX;
    let mut adaptive_on         = false;
    let     min_adaptive_samples: u32 = 128;
    let mut photon_map_on  = true;
    let mut pixel_samples: Vec<u32>   = vec![0; (WIDTH * HEIGHT) as usize];
    let mut var_m2:   Vec<Color>    = vec![Color::default(); (WIDTH * HEIGHT) as usize];
    let mut adap_conv:    Vec<bool>   = vec![false; (WIDTH * HEIGHT) as usize];
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
                var_m2.fill(Color::default());
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
                        var_m2.clear();    var_m2.resize(n, Color::default());
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
                                VirtualKeyCode::P => {
                                    #[cfg(feature = "denoise")]
                                    {
                                        let denoised_guard = denoised.lock().unwrap();
                                        if oidn_on && denoised_guard.len() == (win_w * win_h) as usize {
                                            let sc = 1.0 / samples.max(1) as f32;
                                            let blended: Vec<Color> = accumulator.iter().zip(denoised_guard.iter())
                                                .map(|(acc, den)| *den * denoise_blend + *acc * sc * (1.0 - denoise_blend))
                                                .collect();
                                            save_png(&blended, 1, None, scene.name, win_w, win_h, exposure, tonemapper, Some(samples), None);
                                        } else {
                                            let ps = Some(pixel_samples.as_slice());
                                            save_png(&accumulator, samples, ps, scene.name, win_w, win_h, exposure, tonemapper, None, None);
                                        }
                                    }
                                    #[cfg(not(feature = "denoise"))]
                                    {
                                        let ps = Some(pixel_samples.as_slice());
                                        save_png(&accumulator, samples, ps, scene.name, win_w, win_h, exposure, tonemapper, None, None);
                                    }
                                }
                                VirtualKeyCode::L => {
                                    let from = cam_state.pos;
                                    let at   = cam_state.look_at();
                                    println!("# paste into [camera] in scenes/scene.toml");
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
                                    if let Background::Physical { sun_dir, .. } = &mut scene.background {
                                        let step = if key == VirtualKeyCode::Right { 0.1f32 } else { -0.1f32 };
                                        let (s, c) = step.sin_cos();
                                        *sun_dir = vec3::Vec3::new(
                                            sun_dir.x * c - sun_dir.z * s,
                                            sun_dir.y,
                                            sun_dir.x * s + sun_dir.z * c,
                                        ).unit();
                                        reset_accum!();
                                        scene.rebuild_caustics();
                                    }
                                }
                                VirtualKeyCode::Up | VirtualKeyCode::Down => {
                                    if let Background::Physical { sun_dir, .. } = &mut scene.background {
                                        let step = if key == VirtualKeyCode::Up { 0.1f32 } else { -0.1f32 };
                                        let el = sun_dir.y.asin();
                                        let new_el = (el + step).clamp(-10f32.to_radians(), 85f32.to_radians());
                                        let horiz = vec3::Vec3::new(sun_dir.x, 0.0, sun_dir.z).length().max(1e-6);
                                        let scale = new_el.cos() / horiz;
                                        *sun_dir = vec3::Vec3::new(sun_dir.x * scale, new_el.sin(), sun_dir.z * scale);
                                        reset_accum!();
                                        scene.rebuild_caustics();
                                    }
                                }
                                VirtualKeyCode::C => {
                                    cam_state = CameraState::from_params(&scene.cam_init);
                                    cam_dirty = true;
                                    pending_autofocus = true;
                                }
                                #[cfg(feature = "denoise")]
                                VirtualKeyCode::N => {
                                    oidn_on = !oidn_on;
                                    if oidn_on {
                                        if samples > 0 && !denoise_running.load(Ordering::Acquire) {
                                            let ps = Some(pixel_samples.as_slice());
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
                                VirtualKeyCode::T => {
                                    tonemapper = if tonemapper == ToneMapper::AgX { ToneMapper::Aces } else { ToneMapper::AgX };
                                    window.request_redraw();
                                }
                                VirtualKeyCode::Slash => { print_controls(); }
                                VirtualKeyCode::V => {
                                    adaptive_on = !adaptive_on;
                                    n_converged = 0;
                                    reset_accum!();
                                    window.request_redraw();
                                }
                                VirtualKeyCode::M => {
                                    photon_map_on = !photon_map_on;
                                    reset_accum!();
                                    window.request_redraw();
                                }
                                VirtualKeyCode::R => {
                                    match scene_file::load(&file_scene_evt) {
                                        Ok(s) => {
                                            println!("Reloaded: {}", s.name);
                                            scene = s;
                                            scene.rebuild_caustics();
                                            cam_state = CameraState::from_params(&scene.cam_init);
                                            strata = compute_strata(scene.max_samples);
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
                        cam_state.autofocus(scene.world.as_ref());
                        pending_autofocus = false;
                    }
                    camera = cam_state.to_camera(win_w as f32 / win_h as f32);
                    reset_accum!();
                    cam_dirty = false;
                }

                if last_title_update.elapsed() >= TITLE_INTERVAL {
                    let scene    = &scene;
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
                if samples < scene.max_samples && !all_conv {
                    let bg_scale = 1.0;
                    let conv_mask = if adaptive_on { Some(adap_conv.as_slice()) } else { None };
                    let pm = if photon_map_on { scene.photon_map.as_deref() } else { None };
                    render_tiles(&mut scratch, conv_mask, samples, strata, win_w, win_h, &camera,
                                 scene.world.as_ref(), &scene.background, &scene.lights, bg_scale,
                                 pm, None);

                    accumulate_sample(&mut accumulator, &scratch,
                                      &mut pixel_samples, &mut var_m2, &adap_conv);
                    samples += 1;

                    if adaptive_on && samples >= min_adaptive_samples {
                        n_converged = mark_converged(&mut adap_conv, &pixel_samples, &var_m2, &accumulator);
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
                        let ps = Some(pixel_samples.as_slice());
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
