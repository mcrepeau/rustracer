use std::sync::Arc;

use serde::Deserialize;

use crate::bvh::BvhTree;
use crate::camera::SceneCameraParams;
use crate::hittable::{Hittable, HittableList, Material};
use crate::material::{
    Dielectric, DiffuseLight, Lambertian, PbrMaterial, PearlMaterial, SpectralDielectric,
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
use crate::volume::{ConstantMedium, NoiseMedium};

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
    /// Aperture blade count for polygonal bokeh (0 = circular, 5/6/8 = typical lenses).
    #[serde(default)]
    pub aperture_blades: u32,
}

fn default_vfov()       -> f32 { 40.0 }
fn default_move_speed() -> f32 { 0.5  }

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum BackgroundConfig {
    /// Preetham atmospheric sky with a physical sun.
    Physical {
        /// Degrees clockwise from +Z axis when viewed from above.
        #[serde(default)]
        sun_azimuth:   f32,
        /// Degrees above the horizon (0 = horizon, 90 = zenith).
        #[serde(default = "default_sun_elevation")]
        sun_elevation: f32,
        /// Atmospheric turbidity T (1 = ideal clear, 3 = clear, 5 = light haze, 10 = heavy haze).
        #[serde(default = "default_turbidity")]
        turbidity:     f32,
    },
    /// Uniform colour background.
    Solid { color: [f32; 3] },
}

fn default_sun_elevation() -> f32 { 30.0 }
fn default_turbidity()     -> f32 {  3.0 }

#[derive(Deserialize)]
pub struct ObjectConfig {
    pub shape:    ShapeConfig,
    /// Required for surface shapes; omit (or leave out entirely) for volume shapes.
    #[serde(default)]
    pub material: Option<MaterialConfig>,
}

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ShapeConfig {
    Sphere        { center: [f32; 3], radius: f32 },
    Quad          { corner: [f32; 3], u: [f32; 3], v: [f32; 3] },
    Box           { p_min:  [f32; 3], p_max: [f32; 3] },
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
    // ── Volume shapes — `material` field is ignored; color+density are inline ──
    /// Uniform-density participating medium enclosed in a sphere.
    /// `g`: Henyey-Greenstein asymmetry (0 = isotropic, 0.85 = cloud droplets).
    ConstantVolumeSphere {
        center: [f32; 3], radius: f32, density: f32, color: [f32; 3],
        #[serde(default)] g: f32,
    },
    /// Uniform-density participating medium enclosed in an axis-aligned box.
    ConstantVolumeBox {
        p_min: [f32; 3], p_max: [f32; 3], density: f32, color: [f32; 3],
        #[serde(default)] g: f32,
    },
    /// Perlin-noise–driven heterogeneous medium enclosed in a sphere.
    /// `noise_scale` controls feature size (larger = smaller features).
    /// `threshold` [0, 1) clips the noise: 0 = full volume, 0.5 = patchy clouds.
    /// `g`: Henyey-Greenstein asymmetry (0 = isotropic, 0.85 = cloud droplets).
    NoiseVolumeSphere {
        center:  [f32; 3],
        radius:  f32,
        density: f32,
        color:   [f32; 3],
        #[serde(default = "default_noise_scale")]
        noise_scale: f32,
        #[serde(default)]
        threshold:   f32,
        #[serde(default)]
        g:           f32,
    },
    /// Perlin-noise–driven heterogeneous medium enclosed in an axis-aligned box.
    NoiseVolumeBox {
        p_min:   [f32; 3],
        p_max:   [f32; 3],
        density: f32,
        color:   [f32; 3],
        #[serde(default = "default_noise_scale")]
        noise_scale: f32,
        #[serde(default)]
        threshold:   f32,
        #[serde(default)]
        g:           f32,
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
        #[serde(default)]
        anisotropy: f32,
        #[serde(default)]
        anisotropy_angle: f32,
        #[serde(default)]
        clearcoat: f32,
        #[serde(default = "default_clearcoat_roughness")]
        clearcoat_roughness: f32,
        #[serde(default)]
        film_thickness: f32,
        #[serde(default = "default_film_ior")]
        film_ior: f32,
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
        #[serde(default = "default_luster_roughness")]
        luster_roughness: f32,
    },
}

fn default_roughness()          -> f32 { 0.5 }
fn default_clearcoat_roughness() -> f32 { 0.03 }
fn default_film_ior()           -> f32 { 1.5 }
fn default_pearl_color()        -> [f32; 3] { [0.98, 0.93, 0.88] }
fn default_pearl_ior()          -> f32 { 1.56 }
fn default_film_thickness()     -> f32 { 450.0 }
fn default_orient_strength()    -> f32 { 0.30 }
fn default_film_scale()         -> f32 { 3.0 }
fn default_luster_roughness()   -> f32 { 0.05 }

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
fn default_noise_scale()   -> f32  { 1.0   }

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
        pos:             from,
        lookat:          at,
        vfov:            file.camera.vfov,
        aperture:        file.camera.aperture,
        focus_dist,
        move_speed:      file.camera.move_speed,
        aperture_blades: file.camera.aperture_blades,
    };

    // ── Background ────────────────────────────────────────────────────────────
    let background = match file.background {
        BackgroundConfig::Physical { sun_azimuth, sun_elevation, turbidity } => {
            let el = sun_elevation.to_radians();
            let az = sun_azimuth.to_radians();
            Background::Physical {
                sun_dir: Vec3::new(
                    el.cos() * az.sin(),
                    el.sin(),
                    el.cos() * az.cos(),
                ).unit(),
                turbidity,
            }
        }
        BackgroundConfig::Solid { color } => Background::Solid(col(color)),
    };

    // ── Static objects ────────────────────────────────────────────────────────
    let mut static_objects: Vec<Arc<dyn Hittable>> = Vec::new();

    // A throwaway Lambertian is used as the boundary-shape material for volume
    // types; it is never actually sampled — the HG phase function inside the
    // medium is what scatters photons.
    let dummy_mat = || -> Arc<dyn Material> {
        Arc::new(Lambertian { texture: Texture::from(Color::new(0.5, 0.5, 0.5)) })
    };

    for (i, obj) in file.objects.into_iter().enumerate() {
        let hittable: Arc<dyn Hittable> = match obj.shape {
            // ── Volume shapes (material field is ignored) ──────────────────
            ShapeConfig::ConstantVolumeSphere { center, radius, density, color, g } => {
                let boundary = Arc::new(Sphere::new(p3(center), radius, dummy_mat()));
                Arc::new(ConstantMedium::new(boundary, density, col(color), g))
            }
            ShapeConfig::ConstantVolumeBox { p_min, p_max, density, color, g } => {
                let boundary = Arc::new(BvhTree::from_list(make_box(p3(p_min), p3(p_max), dummy_mat())));
                Arc::new(ConstantMedium::new(boundary, density, col(color), g))
            }
            ShapeConfig::NoiseVolumeSphere { center, radius, density, color, noise_scale, threshold, g } => {
                let boundary = Arc::new(Sphere::new(p3(center), radius, dummy_mat()));
                Arc::new(NoiseMedium::new(boundary, col(color), density, noise_scale, threshold, g))
            }
            ShapeConfig::NoiseVolumeBox { p_min, p_max, density, color, noise_scale, threshold, g } => {
                let boundary = Arc::new(BvhTree::from_list(make_box(p3(p_min), p3(p_max), dummy_mat())));
                Arc::new(NoiseMedium::new(boundary, col(color), density, noise_scale, threshold, g))
            }
            // ── Surface shapes (material field required) ───────────────────
            shape => {
                let mat = build_material(
                    obj.material.ok_or_else(|| format!("objects[{i}]: material is required for this shape type"))?
                ).map_err(|e| format!("objects[{i}].material: {e}"))?;
                match shape {
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
                    _ => unreachable!(), // volume variants handled above
                }
            }
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

    let mut world_list = HittableList::new();
    for obj in static_objects { world_list.objects.push(obj); }
    let world: Arc<dyn Hittable> = Arc::new(BvhTree::from_list(world_list));

    let mut scene = SceneData {
        world,
        lights,
        background,
        name,
        cam_init,
        max_samples:   file.max_samples,
        enable_caustics,
        caustic_quad:          None,
        caustic_gather_radius,
        photon_map:    None,
    };
    scene.rebuild_caustics();
    Ok(scene)
}

// ── Material builder ──────────────────────────────────────────────────────────

fn build_material(cfg: MaterialConfig) -> Result<Arc<dyn Material>, String> {
    Ok(match cfg {
        MaterialConfig::Lambertian { color } =>
            Arc::new(Lambertian { texture: Texture::from(col(color)) }),

        MaterialConfig::Metal { color, fuzz } =>
            Arc::new(PbrMaterial { albedo: col(color), metallic: 1.0, roughness: fuzz, ..Default::default() }),

        MaterialConfig::Dielectric { ior } =>
            Arc::new(Dielectric { ir: ior }),

        MaterialConfig::SpectralDielectric { ior, dispersion } => {
            // Convert (ior at ~546 nm, spread over visible) to Cauchy n(λ)=B+C/λ²
            // C determined from spread: Δn = C × (1/λ_b² − 1/λ_r²), λ in μm.
            // λ_b = 0.435 μm, λ_r = 0.700 μm → denominator ≈ 3.250 μm⁻².
            let cauchy_c = dispersion / 3.250;
            // B so that n(550 nm) ≈ ior: B = ior − C / 0.550².
            let cauchy_b = ior - cauchy_c / (0.550 * 0.550);
            Arc::new(SpectralDielectric { cauchy_b, cauchy_c })
        }

        MaterialConfig::DiffuseLight { color } =>
            Arc::new(DiffuseLight { emit: Texture::from(col(color)) }),

        MaterialConfig::Pbr { albedo, metallic, roughness, anisotropy, anisotropy_angle,
                              clearcoat, clearcoat_roughness, film_thickness, film_ior } =>
            Arc::new(PbrMaterial { albedo: col(albedo), metallic, roughness,
                                   anisotropy, anisotropy_angle,
                                   clearcoat, clearcoat_roughness, film_thickness, film_ior }),

        MaterialConfig::Pearl { base_color, ior, film_thickness, orient_strength, film_scale, luster_roughness } =>
            Arc::new(PearlMaterial { base_color: col(base_color), ior, film_thickness, orient_strength, film_scale, luster_roughness }),
    })
}

// ── Tiny array-to-vec3/point3/color helpers ───────────────────────────────────

#[inline] fn v3([x, y, z]: [f32; 3]) -> Vec3   { Vec3::new(x, y, z) }
#[inline] fn p3([x, y, z]: [f32; 3]) -> Point3 { Point3::new(x, y, z) }
#[inline] fn col([r, g, b]: [f32; 3]) -> Color  { Color::new(r, g, b) }
