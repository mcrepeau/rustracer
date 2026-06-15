use std::sync::Arc;

use serde::Deserialize;

use crate::bvh::BvhTree;
use crate::camera::SceneCameraParams;
use crate::hittable::{Hittable, HittableList, Material};
use crate::material::{
    Dielectric, DiffuseLight, Lambertian, Metal, PbrMaterial, PearlMaterial, SpectralDielectric,
};
use crate::cone::Cone;
use crate::cylinder::Cylinder;
use crate::disk::Disk;
use crate::plane::InfinitePlane;
use crate::quad::{make_box, Quad};
use crate::renderer::Background;
use crate::scene::SceneData;
use crate::sphere::Sphere;
use crate::texture::Texture;
use crate::vec3::{Color, Point3, Vec3};

// ── Serde types ───────────────────────────────────────────────────────────────

#[derive(Deserialize)]
pub struct SceneFile {
    #[serde(default = "default_name")]
    pub name:        String,
    pub camera:      CameraConfig,
    pub background:  BackgroundConfig,
    #[serde(default)]
    pub objects:     Vec<ObjectConfig>,
    #[serde(default)]
    pub lights:      Vec<LightConfig>,
    pub caustics:    Option<CausticsConfig>,
    #[serde(default = "default_max_samples")]
    pub max_samples: u32,
}

fn default_name()        -> String { "Custom Scene".to_string() }
fn default_max_samples() -> u32    { 2000 }

#[derive(Deserialize)]
pub struct CameraConfig {
    pub look_from:  [f32; 3],
    pub look_at:    [f32; 3],
    #[serde(default = "default_vfov")]
    pub vfov:       f32,
    #[serde(default)]
    pub aperture:   f32,
    /// If omitted, computed as the distance from look_from to look_at.
    pub focus_dist: Option<f32>,
    #[serde(default = "default_move_speed")]
    pub move_speed: f32,
}

fn default_vfov()       -> f32 { 40.0 }
fn default_move_speed() -> f32 { 0.5  }

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum BackgroundConfig {
    /// Physically-inspired sky with a sun.
    Physical {
        /// Degrees clockwise from +Z axis when viewed from above.
        #[serde(default)]
        sun_azimuth:   f32,
        /// Degrees above the horizon (0 = horizon, 90 = zenith).
        #[serde(default = "default_sun_elevation")]
        sun_elevation: f32,
    },
    /// Uniform colour background.
    Solid { color: [f32; 3] },
}

fn default_sun_elevation() -> f32 { 30.0 }

#[derive(Deserialize)]
pub struct ObjectConfig {
    pub shape:    ShapeConfig,
    pub material: MaterialConfig,
}

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ShapeConfig {
    Sphere   { center: [f32; 3], radius: f32 },
    Quad     { corner: [f32; 3], u: [f32; 3], v: [f32; 3] },
    Box      { p_min:  [f32; 3], p_max: [f32; 3] },
    Cylinder      { center: [f32; 3], radius: f32, height: f32 },
    Cone          { center: [f32; 3], radius: f32, height: f32 },
    Disk          { center: [f32; 3], normal: [f32; 3], radius: f32 },
    InfinitePlane {
        point:  [f32; 3],
        normal: [f32; 3],
        #[serde(default)]
        wave_amplitude: f32,
        #[serde(default = "default_wave_scale")]
        wave_scale: f32,
    },
}

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum MaterialConfig {
    Lambertian { color: [f32; 3] },
    Metal {
        color: [f32; 3],
        #[serde(default)]
        fuzz:  f32,
    },
    Dielectric { ior: f32 },
    /// Dispersive glass — use for diamonds and other high-IOR materials.
    SpectralDielectric {
        ior: f32,
        /// Half-spread of IOR across R/G/B (0 = no dispersion).
        /// Diamond ≈ 0.04, glass ≈ 0.01.
        #[serde(default)]
        dispersion: f32,
    },
    DiffuseLight { color: [f32; 3] },
    Pbr {
        albedo:    [f32; 3],
        #[serde(default)]
        metallic:  f32,
        #[serde(default = "default_roughness")]
        roughness: f32,
    },
    Pearl {
        #[serde(default = "default_pearl_color")]
        base_color:      [f32; 3],
        #[serde(default = "default_pearl_ior")]
        ior:             f32,
        #[serde(default = "default_film_thickness")]
        film_thickness:  f32,
        #[serde(default = "default_orient_strength")]
        orient_strength: f32,
        #[serde(default = "default_film_scale")]
        film_scale:      f32,
    },
}

fn default_roughness()       -> f32       { 0.5 }
fn default_pearl_color()     -> [f32; 3]  { [0.98, 0.93, 0.88] }
fn default_pearl_ior()       -> f32       { 1.56 }
fn default_film_thickness()  -> f32       { 450.0 }
fn default_orient_strength() -> f32       { 0.30 }
fn default_film_scale()      -> f32       { 3.0 }

/// A quad that emits light — added to both the world and the NEE light list.
#[derive(Deserialize)]
pub struct LightConfig {
    pub corner: [f32; 3],
    pub u:      [f32; 3],
    pub v:      [f32; 3],
    pub color:  [f32; 3],
}

#[derive(Deserialize)]
pub struct CausticsConfig {
    #[serde(default = "default_true")]
    pub enabled:       bool,
    #[serde(default = "default_gather_radius")]
    pub gather_radius: f32,
}

fn default_true()          -> bool { true  }
fn default_gather_radius() -> f32  { 0.15  }
fn default_wave_scale()    -> f32  { 1.0   }

// ── Public entry point ────────────────────────────────────────────────────────

/// Parse `path` (TOML) and return a fully built `SceneData`, or a human-readable
/// error string suitable for printing to the console.
pub fn load(path: &str) -> Result<SceneData, String> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| format!("cannot read {path}: {e}"))?;
    let file: SceneFile = toml::from_str(&text)
        .map_err(|e| format!("parse error in {path}:\n{e}"))?;
    build(file)
}

// ── Scene builder ─────────────────────────────────────────────────────────────

fn build(file: SceneFile) -> Result<SceneData, String> {
    // ── Camera ────────────────────────────────────────────────────────────────
    let from = p3(file.camera.look_from);
    let at   = p3(file.camera.look_at);
    let focus_dist = file.camera.focus_dist
        .unwrap_or_else(|| (from - at).length().max(0.01));
    let cam_init = SceneCameraParams {
        pos:        from,
        lookat:     at,
        vfov:       file.camera.vfov,
        aperture:   file.camera.aperture,
        focus_dist,
        move_speed: file.camera.move_speed,
    };

    // ── Background ────────────────────────────────────────────────────────────
    let background = match file.background {
        BackgroundConfig::Physical { sun_azimuth, sun_elevation } => {
            let el = sun_elevation.to_radians();
            let az = sun_azimuth.to_radians();
            Background::Physical {
                sun_dir: Vec3::new(
                    el.cos() * az.sin(),
                    el.sin(),
                    el.cos() * az.cos(),
                ).unit(),
            }
        }
        BackgroundConfig::Solid { color } => Background::Solid(col(color)),
    };

    // ── Static objects ────────────────────────────────────────────────────────
    let mut static_objects: Vec<Arc<dyn Hittable>> = Vec::new();

    for (i, obj) in file.objects.into_iter().enumerate() {
        let mat = build_material(obj.material)
            .map_err(|e| format!("objects[{i}].material: {e}"))?;
        let hittable: Arc<dyn Hittable> = match obj.shape {
            ShapeConfig::Sphere { center, radius } =>
                Arc::new(Sphere::new(p3(center), radius, mat)),
            ShapeConfig::Quad { corner, u, v } =>
                Arc::new(Quad::new(p3(corner), v3(u), v3(v), mat)),
            ShapeConfig::Box { p_min, p_max } =>
                Arc::new(BvhTree::from_list(make_box(p3(p_min), p3(p_max), mat))),
            ShapeConfig::Cylinder { center, radius, height } =>
                Arc::new(Cylinder { center: p3(center), radius, height, mat }),
            ShapeConfig::Cone { center, radius, height } =>
                Arc::new(Cone { center: p3(center), radius, height, mat }),
            ShapeConfig::Disk { center, normal, radius } =>
                Arc::new(Disk::new(p3(center), v3(normal), radius, mat)),
            ShapeConfig::InfinitePlane { point, normal, wave_amplitude, wave_scale } =>
                Arc::new(InfinitePlane::new(p3(point), v3(normal), wave_amplitude, wave_scale, mat)),
        };
        static_objects.push(hittable);
    }

    // ── Lights ────────────────────────────────────────────────────────────────
    // Each entry creates a DiffuseLight quad in both the world and the NEE list.
    let mut lights = HittableList::new();

    for (i, lc) in file.lights.iter().enumerate() {
        let _ = i;
        let mat: Arc<dyn Material> =
            Arc::new(DiffuseLight { emit: Texture::from(col(lc.color)) });
        let corner = p3(lc.corner);
        let u      = v3(lc.u);
        let v      = v3(lc.v);
        static_objects.push(Arc::new(Quad::new(corner, u, v, Arc::clone(&mat))));
        lights.add(Quad::new(corner, u, v, mat));
    }

    // ── Caustics ──────────────────────────────────────────────────────────────
    let (enable_caustics, caustic_gather_radius) = match file.caustics {
        Some(c) => (c.enabled, c.gather_radius),
        None    => (false, 0.15),
    };

    // ── Assemble SceneData ────────────────────────────────────────────────────
    // Leak the name string so it lives for 'static — the leak is at most a few
    // bytes per reload, negligible over the life of the process.
    let name: &'static str = Box::leak(file.name.into_boxed_str());

    let mut scene = SceneData {
        world:         Arc::new(HittableList::new()),
        lights,
        background,
        name,
        cam_init,
        static_objects,
        dynamic:       Vec::new(),
        bounds:        None,
        colliders:     Vec::new(),
        convex_colliders: Vec::new(),
        gravity:       0.0,
        settled:       false,
        paused:        false,
        max_samples:   file.max_samples,
        enable_caustics,
        caustic_quad:          None,
        caustic_gather_radius,
        photon_map:    None,
        cached_static: None,
    };
    scene.rebuild();
    Ok(scene)
}

// ── Material builder ──────────────────────────────────────────────────────────

fn build_material(cfg: MaterialConfig) -> Result<Arc<dyn Material>, String> {
    Ok(match cfg {
        MaterialConfig::Lambertian { color } =>
            Arc::new(Lambertian { texture: Texture::from(col(color)) }),

        MaterialConfig::Metal { color, fuzz } =>
            Arc::new(Metal { albedo: col(color), fuzz }),

        MaterialConfig::Dielectric { ior } =>
            Arc::new(Dielectric { ir: ior }),

        MaterialConfig::SpectralDielectric { ior, dispersion } => {
            let h = dispersion / 2.0;
            Arc::new(SpectralDielectric {
                ir_red:   ior - h,
                ir_green: ior,
                ir_blue:  ior + h,
            })
        }

        MaterialConfig::DiffuseLight { color } =>
            Arc::new(DiffuseLight { emit: Texture::from(col(color)) }),

        MaterialConfig::Pbr { albedo, metallic, roughness } =>
            Arc::new(PbrMaterial { albedo: col(albedo), metallic, roughness }),

        MaterialConfig::Pearl { base_color, ior, film_thickness, orient_strength, film_scale } =>
            Arc::new(PearlMaterial { base_color: col(base_color), ior, film_thickness, orient_strength, film_scale }),
    })
}

// ── Tiny array-to-vec3/point3/color helpers ───────────────────────────────────

#[inline] fn v3([x, y, z]: [f32; 3]) -> Vec3   { Vec3::new(x, y, z) }
#[inline] fn p3([x, y, z]: [f32; 3]) -> Point3 { Point3::new(x, y, z) }
#[inline] fn col([r, g, b]: [f32; 3]) -> Color  { Color::new(r, g, b) }
