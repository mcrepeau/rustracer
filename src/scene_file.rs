use std::sync::Arc;

use serde::Deserialize;

use crate::animation::{AnimationData, AnimationKind, Keyframe, OrbitData};
use crate::bvh::BvhTree;
use crate::camera::SceneCameraParams;
use crate::hittable::{Hittable, HittableList, Material};
use crate::material::{
    BlackbodyLight, Dielectric, DiffuseLight, Lambertian, PbrMaterial, PearlMaterial, SpectralDielectric,
    SpectralMetal, SpectralMetalVariant, SSSMaterial,
};
use crate::cone::Cone;
use crate::cylinder::Cylinder;
use crate::diamond::{Diamond, DiamondParams};
use crate::step_cut::{StepCut, StepCutParams, StepOutline};
use crate::disk::Disk;
use crate::plane::InfinitePlane;
use crate::quad::{make_box, Quad};
use crate::renderer::{Background, EnvMapData};
use crate::scene::SceneData;
use crate::sphere::Sphere;
use crate::texture::Texture;
use crate::transform::{Rotate, UniformScale, Translate};
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
    /// Gemstones described by the key fields of a grading report (IGI/GIA).
    #[serde(default)]
    pub gems:        Vec<GemConfig>,
    pub caustics:    Option<CausticsConfig>,
    #[serde(default = "default_max_samples")]
    pub max_samples: u32,
    pub animation:   Option<AnimationConfig>,
}

fn default_name()        -> String { "Custom Scene".to_string() }
fn default_max_samples() -> u32    { 2000 }

#[derive(Deserialize)]
pub struct AnimationConfig {
    pub fps:      f32,
    pub duration: f32,
    /// Keyframe-based path — Catmull-Rom spline through each control point.
    #[serde(default)]
    pub keyframes: Vec<KeyframeConfig>,
    /// True circular orbit — camera travels on a perfect circle in the XZ-plane.
    /// Takes precedence over `keyframes` when both are present.
    pub orbit: Option<OrbitConfig>,
}

#[derive(Deserialize)]
pub struct OrbitConfig {
    /// World-space center of the orbit circle (only XZ used; Y is set by `height`).
    pub center:      [f32; 3],
    /// Orbit radius in scene units.
    pub radius:      f32,
    /// World-space Y position of the camera throughout the orbit.
    pub height:      f32,
    /// Fixed look-at point (can differ from `center`).
    pub look_at:     [f32; 3],
    /// Starting angle in degrees; 0° = +Z from center (clockwise when viewed from above).
    #[serde(default)]
    pub start_angle: f32,
    /// Ending angle in degrees; 360° = full orbit back to start.
    #[serde(default = "default_end_angle")]
    pub end_angle:   f32,
    /// Overrides the scene camera vfov for the orbit.
    pub vfov:        Option<f32>,
    /// Overrides the scene camera aperture for the orbit.
    pub aperture:    Option<f32>,
    /// Fixed focus distance; defaults to camera→look_at distance if omitted.
    pub focus_dist:  Option<f32>,
}

fn default_end_angle() -> f32 { 360.0 }

#[derive(Deserialize)]
pub struct KeyframeConfig {
    pub time:       f32,
    pub look_from:  [f32; 3],
    pub look_at:    [f32; 3],
    /// Overrides the scene camera vfov for this keyframe.
    pub vfov:       Option<f32>,
    /// Overrides the scene camera aperture for this keyframe.
    pub aperture:   Option<f32>,
    /// Overrides the scene camera focus distance; defaults to look_from→look_at distance.
    pub focus_dist: Option<f32>,
}

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
        /// Atmospheric turbidity T (1.7 = min valid, 2 = very clear, 3 = clear, 5 = light haze, 10 = heavy haze).
        #[serde(default = "default_turbidity")]
        turbidity:     f32,
    },
    /// Uniform colour background.
    Solid { color: [f32; 3] },
    /// Equirectangular HDR environment map (EXR file).
    EnvMap { path: String },
}

fn default_sun_elevation() -> f32 { 30.0 }
fn default_turbidity()     -> f32 {  3.0 }

#[derive(Deserialize)]
pub struct ObjectConfig {
    pub shape:    ShapeConfig,
    /// Required for surface shapes; omit (or leave out entirely) for volume shapes.
    #[serde(default)]
    pub material: Option<MaterialConfig>,
    /// Uniform scale applied before rotation and translation (e.g. 0.01 converts cm → m).
    #[serde(default)]
    pub scale:     Option<f32>,
    /// Rotation around the world Y axis in degrees, applied after scale.
    #[serde(default)]
    pub rotate_y:  Option<f32>,
    /// World-space translation applied last (after scale and rotation).
    #[serde(default)]
    pub translate: Option<[f32; 3]>,
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
    /// Round brilliant diamond as an analytic convex polyhedron.
    /// Proportions default to the Tolkowsky ideal cut when omitted.
    Diamond {
        center:   [f32; 3],
        radius:   f32,
        #[serde(default = "default_table_pct")]
        table_pct:      f32,
        #[serde(default = "default_crown_angle")]
        crown_angle:    f32,
        #[serde(default = "default_pavilion_angle")]
        pavilion_angle: f32,
        /// Girdle stretch along X (length/width); 1.0 = round.  radius = semi-minor axis.
        #[serde(default = "default_oval_ratio")]
        oval_ratio:     f32,
        /// Total girdle segments (even). 16 = classic round brilliant; ovals
        /// want 64+ so the stretched outline reads as a true ellipse.
        #[serde(default = "default_girdle_facets")]
        girdle_facets:  u32,
    },
    /// Step-family cut (emerald/princess/cushion/trillion/marquise/...) as an
    /// analytic convex polyhedron: polygon girdle outline, stepped crown and
    /// pavilion, flat table/culet.
    StepCut {
        center:   [f32; 3],
        /// Full length along X and width along Z (scene units).
        length:   f32,
        width:    f32,
        #[serde(default = "default_table_pct")]
        table_pct:      f32,
        #[serde(default = "default_crown_angle")]
        crown_angle:    f32,
        #[serde(default = "default_pavilion_angle")]
        pavilion_angle: f32,
        /// Girdle outline: "rect" (default), "rounded_rect" (cushion),
        /// "triangle" (trillion), "hexagon", "marquise", "pear", or
        /// "polygon" (requires `points`).
        #[serde(default)]
        outline:        Option<String>,
        /// Explicit outline vertices for outline = "polygon" (XZ, relative
        /// to `center`).
        #[serde(default)]
        points:         Option<Vec<[f32; 2]>>,
        /// Corner chamfer as % of the smaller semi-axis (rect outlines).
        #[serde(default = "default_corner_pct")]
        corner_pct:     f32,
        /// Arc segments per corner (rounded_rect).
        #[serde(default = "default_corner_segs")]
        corner_segs:    u32,
        /// Outline segments for marquise/pear ovals.
        #[serde(default = "default_oval_segs")]
        oval_segs:      u32,
        /// Round one end and point the other (marquise/pear).
        #[serde(default)]
        pear:           bool,
        #[serde(default = "default_steps")]
        crown_steps:    u32,
        #[serde(default = "default_steps")]
        pavilion_steps: u32,
        /// Culet flat as % of the width (0 = pointed culet).
        #[serde(default = "default_culet_pct")]
        culet_pct:      f32,
        /// Angle spread between girdle and table rows (0 = single slope).
        #[serde(default = "default_step_spread")]
        step_spread:    f32,
        /// Diagonal chevron rows per outline vertex (0 = none).
        #[serde(default)]
        chevrons:       u32,
    },
    InfinitePlane {
        point:  [f32; 3],
        normal: [f32; 3],
        #[serde(default)]
        wave_amplitude: f32,
        #[serde(default = "default_wave_scale")]
        wave_scale: f32,
    },
    /// OBJ mesh file — path relative to the working directory.
    /// Material is applied to all triangles.
    Mesh { path: String },
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
    /// Lambertian with a 3-D checker texture.  `scale` controls tile size in world units.
    Checker {
        even:  [f32; 3],
        odd:   [f32; 3],
        #[serde(default = "default_checker_scale")]
        scale: f32,
    },
    /// Lambertian with an sRGB image texture (e.g. a world map), path relative to the cwd.
    Image { path: String },
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
        /// Beer-Lambert absorption coefficient per unit length (RGB, scene units).
        /// [0,0,0] = clear glass. Example: [0.05, 0.01, 0.0] gives amber tint.
        #[serde(default)]
        absorption: [f32; 3],
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
        #[serde(default)]
        sheen: f32,
        #[serde(default = "default_sheen_tint")]
        sheen_tint: f32,
        #[serde(default)]
        emission: [f32; 3],
        #[serde(default)]
        emission_strength: f32,
    },
    /// PBR material driven by image texture maps.
    /// `albedo_path` is required; the others are optional (fall back to scalar defaults).
    /// Normal maps are not yet supported.
    TexturedPbr {
        albedo_path:    String,
        #[serde(default)]
        roughness_path: Option<String>,
        #[serde(default)]
        metallic_path:  Option<String>,
        #[serde(default)]
        ao_path:        Option<String>,
        #[serde(default)]
        normal_path:    Option<String>,
        #[serde(default = "default_roughness")]
        roughness:      f32,
        #[serde(default)]
        metallic:       f32,
    },
    /// Conductor Fresnel from J&C 1972 tabulated IOR data.
    SpectralMetal {
        /// "gold", "copper", or "silver"
        variant:   String,
        #[serde(default)]
        roughness: f32,
    },
    /// Subsurface scattering (jade, marble, skin): albedo tint, absorption sigma_a,
    /// boundary ior, density (scattering-coefficient scale) and HG asymmetry g.
    Sss {
        albedo:  [f32; 3],
        sigma_a: [f32; 3],
        ior:     f32,
        density: f32,
        g:       f32,
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
fn default_sheen_tint()         -> f32 { 0.5 }

/// A quad that emits light — added to both the world and the NEE light list.
#[derive(Deserialize)]
pub struct LightConfig {
    pub corner: [f32; 3],
    pub u:      [f32; 3],
    pub v:      [f32; 3],
    pub color:  [f32; 3],
    /// If set, the emitter is a blackbody at this temperature (K) with `intensity`
    /// (Planck spectrum, spectrally weighted); `color` is then ignored.
    #[serde(default)]
    pub temp_k:    Option<f32>,
    #[serde(default)]
    pub intensity: Option<f32>,
}

#[derive(Deserialize)]
pub struct CausticsConfig {
    #[serde(default = "default_true")]
    pub enabled:       bool,
    #[serde(default = "default_gather_radius")]
    pub gather_radius: f32,
    /// Photons to trace.  Increase for small or far-away objects where most
    /// photons miss the target (e.g. 5_000_000 for a small diamond).
    #[serde(default = "default_num_photons")]
    pub num_photons:   u32,
}

fn default_checker_scale() -> f32  { 1.0   }
fn default_num_photons()   -> u32  { 500_000 }
fn default_true()          -> bool { true  }
fn default_gather_radius() -> f32  { 0.15  }
fn default_wave_scale()    -> f32  { 1.0   }

fn default_gem_scale_mm()  -> f32  { 0.001 }  // mm -> metres
fn default_color_grade()   -> String { "D".to_string() }

/// A gemstone ingested from a grading report (IGI/GIA).  The size comes from
/// `measurements_mm` (L x W x depth, mm) when present, otherwise from `carat`
/// via the round-brilliant approximation ct ~= 0.00355 * d_mm^3.  The cut is
/// table_pct / crown_angle / pavilion_angle, exactly as reported.  The colour
/// grade maps to a faint yellow-brown body tint (heuristic Beer-Lambert).
/// `material` selects a gem preset (diamond/sapphire/ruby/emerald) and `cut`
/// selects "brilliant" (default) or "step" (rectangular emerald cut).
#[derive(Deserialize)]
pub struct GemConfig {
    /// Free-text provenance, e.g. the report number.
    #[serde(default)]
    pub report:          String,
    #[serde(default)]
    pub carat:           Option<f32>,
    /// [length, width, depth] in mm — overrides carat for the girdle diameter.
    #[serde(default)]
    pub measurements_mm: Option<[f32; 3]>,
    #[serde(default = "default_table_pct")]
    pub table_pct:       f32,
    #[serde(default = "default_crown_angle")]
    pub crown_angle:     f32,
    #[serde(default = "default_pavilion_angle")]
    pub pavilion_angle:  f32,
    /// IGI-style proportions diagram: crown height as % of the minor axis.
    /// When present, overrides crown_angle (converted via the facet geometry).
    #[serde(default)]
    pub crown_pct:       Option<f32>,
    /// IGI-style proportions diagram: pavilion depth as % of the minor axis.
    /// When present, overrides pavilion_angle.
    #[serde(default)]
    pub pavilion_pct:    Option<f32>,
    /// Reported total depth % — cross-checked against the angles (warning only).
    #[serde(default)]
    pub depth_pct:       Option<f32>,
    #[serde(default = "default_color_grade")]
    pub color_grade:     String,
    /// Gem material preset: "diamond" (default), "sapphire", "ruby" or "emerald".
    /// Sets IOR/dispersion and a body-colour absorption (Beer-Lambert); the
    /// color_grade tint and explicit cauchy_b/cauchy_c/absorption add on top.
    #[serde(default)]
    pub material:        Option<String>,
    /// Cut style: "brilliant" (default; round/oval), "step" (rectangular
    /// emerald cut), "princess" (square brilliant + chevrons), "asscher",
    /// "radiant", "baguette", "cushion", "trillion", "marquise" or "pear".
    #[serde(default)]
    pub cut:             Option<String>,
    #[serde(default)]
    pub cauchy_b:        Option<f32>,
    #[serde(default)]
    pub cauchy_c:        Option<f32>,
    #[serde(default)]
    pub absorption:      Option<[f32; 3]>,
    /// Diagonal chevron rows per corner for step/princess cuts
    /// (default 0 for "step", 3 for "princess").
    #[serde(default)]
    pub chevrons:        Option<u32>,
    /// Total girdle segments (even).  None = 16 for round, 64 for oval
    /// (ratio > 1.05) so the girdle outline is a smooth ellipse.
    #[serde(default)]
    pub girdle_facets:   Option<u32>,
    /// Girdle centre in scene units.
    pub center:          [f32; 3],
    /// mm -> scene units applied to the size.
    #[serde(default = "default_gem_scale_mm")]
    pub scale_mm:        f32,
    /// Same SRT placement transforms as objects.
    #[serde(default)]
    pub scale:           Option<f32>,
    #[serde(default)]
    pub rotate_y:        Option<f32>,
    #[serde(default)]
    pub translate:       Option<[f32; 3]>,
}
fn default_noise_scale()   -> f32  { 1.0   }
fn default_table_pct()     -> f32  { 53.0  }
fn default_crown_angle()   -> f32  { 34.5  }
fn default_pavilion_angle()-> f32  { 40.75 }
fn default_oval_ratio()    -> f32  { 1.0   }
fn default_girdle_facets()  -> u32  { 16    }
fn default_corner_pct()     -> f32  { 12.0  }
fn default_corner_segs()    -> u32  { 4     }
fn default_oval_segs()      -> u32  { 24    }
fn default_steps()          -> u32  { 3     }
fn default_culet_pct()      -> f32  { 8.0   }
fn default_step_spread()    -> f32  { 0.25  }

// ── Public entry point ────────────────────────────────────────────────────────

/// Parse `path` (TOML) and return a fully built `SceneData`, or a human-readable
/// error string suitable for printing to the console.
pub fn load(path: &str) -> Result<SceneData, String> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| format!("cannot read {path}: {e}"))?;
    let file: SceneFile = toml::from_str(&text)
        .map_err(|e| format!("parse error in {path}:\n{e}"))?;
    // Asset paths in the file resolve relative to the scene file's own
    // directory first (so scenes/ can carry its own assets/), then cwd.
    let base = std::path::Path::new(path)
        .parent()
        .unwrap_or_else(|| std::path::Path::new("."))
        .to_path_buf();
    build(file, &base)
}

/// Resolve an asset path from a scene file: a relative path that exists next
/// to the scene file wins; otherwise the path is used as-is (cwd-relative).
fn resolve_asset(base: &std::path::Path, p: &str) -> String {
    if std::path::Path::new(p).is_relative() {
        let cand = base.join(p);
        if cand.exists() { return cand.to_string_lossy().into_owned(); }
    }
    p.to_string()
}

// ── Scene builder ─────────────────────────────────────────────────────────────

/// Material presets for `[[gems]]`: `(cauchy_b, cauchy_c, absorption)`.
/// Cauchy B/C in µm² (see [`crate::spectrum::cauchy_ior`]); the absorption is a
/// Beer-Lambert sigma per scene unit, added on top of the color_grade tint.
fn gem_preset(name: &str) -> Result<(f32, f32, Color), String> {
    match name {
        // Diamond: nD = 2.417, nF-nC ~= 0.044 (textbook dispersion — the old
        // C = 0.00585 gave only 0.011, so refracted light barely separated: no fire).
        "diamond"  => Ok((2.351, 0.0230, Color::new(0.0, 0.0, 0.0))),
        // Corundum (sapphire / ruby): nD 1.768, dispersion ~ 0.018.  Blue
        // sapphire absorbs red+yellow (Cr3+/Fe-Ti charge transfer); ruby
        // absorbs green+violet (Cr3+).
        "sapphire" => Ok((1.741, 0.0094, Color::new(650.0, 320.0, 25.0))),
        "ruby"     => Ok((1.741, 0.0094, Color::new(25.0, 550.0, 450.0))),
        // Beryl (emerald): nD 1.577, dispersion ~ 0.014; Cr3+ absorbs red+violet.
        "emerald"  => Ok((1.556, 0.0073, Color::new(480.0, 25.0, 380.0))),
        other => Err(format!(
            "unknown gem material '{other}' (expected diamond, sapphire, ruby or emerald)")),
    }
}

/// Map a `step_cut` shape's `outline` string to a [`StepOutline`].
fn step_outline_name(name: &str, corner_pct: f32, corner_segs: u32, oval_segs: u32, pear: bool,
                     points: Option<Vec<[f32; 2]>>) -> Result<StepOutline, String>
{
    Ok(match name {
        "rect"         => StepOutline::Rect { corner_pct },
        "rounded_rect" => StepOutline::RoundedRect { corner_pct, segs: corner_segs },
        "triangle"     => StepOutline::Triangle,
        "hexagon"      => StepOutline::Hexagon,
        "marquise"     => StepOutline::PointedOval { segs: oval_segs, pear: false },
        "pear"         => StepOutline::PointedOval { segs: oval_segs, pear },
        "polygon"      => StepOutline::Polygon(points.ok_or("outline = \"polygon\" requires `points`")?),
        other => return Err(format!(
            "unknown step_cut outline '{other}' (expected rect, rounded_rect, triangle, hexagon, marquise, pear, polygon)")),
    })
}

/// Named cut presets for `[[gems]] cut = "..."` (other than "brilliant"):
/// (outline, culet_pct, crown_steps, pavilion_steps, step_spread, default chevrons).
fn step_cut_preset(cut: &str, g: &GemConfig, crown_angle: f32, pavilion_angle: f32)
    -> Result<StepCutParams, String>
{
    use StepOutline::*;
    let ch = g.chevrons;
    let (outline, culet_pct, crown_steps, pavilion_steps, step_spread, chev) = match cut {
        "step"     => (Rect { corner_pct: 12.0 }, 8.0, 3, 3, 0.25, ch.unwrap_or(0)),
        "princess" => (Rect { corner_pct: 0.0 },  0.0, 4, 4, 0.15, ch.unwrap_or(3)),
        "asscher"  => (Rect { corner_pct: 22.0 }, 8.0, 3, 3, 0.20, ch.unwrap_or(0)),
        "radiant"  => (Rect { corner_pct: 12.0 }, 0.0, 4, 4, 0.15, ch.unwrap_or(2)),
        "baguette" => (Rect { corner_pct: 4.0 },  8.0, 2, 2, 0.10, ch.unwrap_or(0)),
        "cushion"  => (RoundedRect { corner_pct: 30.0, segs: 4 }, 8.0, 3, 3, 0.15, ch.unwrap_or(0)),
        "trillion" => (Triangle,                  0.0, 4, 4, 0.15, ch.unwrap_or(2)),
        "marquise" => (PointedOval { segs: 24, pear: false }, 0.0, 4, 4, 0.15, ch.unwrap_or(2)),
        "pear"     => (PointedOval { segs: 24, pear: true },  0.0, 4, 4, 0.15, ch.unwrap_or(2)),
        other => return Err(format!(
            "unknown gem cut '{other}' (expected brilliant, step, princess, asscher, radiant, baguette, cushion, trillion, marquise or pear)")),
    };
    Ok(StepCutParams {
        table_pct: g.table_pct, crown_angle, pavilion_angle,
        outline, culet_pct, crown_steps, pavilion_steps, step_spread, chevrons: chev,
    })
}

/// Build an analytic gem from a grading report's key fields: a round/oval
/// brilliant `Diamond` (cut = "brilliant", default) or a step-family `StepCut`
/// (cut = "step"/"princess"/"asscher"/"radiant"/"baguette"/"cushion"/
/// "trillion"/"marquise"/"pear"), with a preset or overridden
/// `SpectralDielectric`.
fn build_gem(g: &GemConfig) -> Result<Arc<dyn Hittable>, String> {
    // Size: explicit measurements win, else carat weight.  With measurements,
    // the minor axis W sets the radius and L/W the oval ratio (round if ~equal).
    let (w_mm, oval_ratio) = if let Some(m) = g.measurements_mm {
        if m[0] <= 0.0 || m[1] <= 0.0 { return Err("measurements_mm must be positive".into()); }
        let (l, w) = if m[0] >= m[1] { (m[0], m[1]) } else { (m[1], m[0]) };
        let ratio = if (l - w) / w > 0.02 { l / w } else { 1.0 };
        (w, ratio)
    } else if let Some(ct) = g.carat {
        if ct <= 0.0 { return Err("carat must be positive".into()); }
        ((ct / 0.00355).cbrt(), 1.0)   // round brilliant: ct ~= 0.00355 * d_mm^3
    } else {
        return Err("provide either carat or measurements_mm".into());
    };
    if g.scale_mm <= 0.0 { return Err("scale_mm must be positive".into()); }
    let radius = 0.5 * w_mm * g.scale_mm;

    if !(g.table_pct > 0.0 && g.table_pct < 100.0) {
        return Err(format!("gem '{}': table_pct must be in (0, 100), got {}", g.report, g.table_pct));
    }

    // IGI-style crown/pavilion percentages are fractions of the MINOR axis;
    // convert to facet angles via the (round-space) facet geometry.
    let table_r = (g.table_pct / 100.0) * radius;
    let crown_angle = match g.crown_pct {
        Some(p) => (p / 100.0 * w_mm * g.scale_mm / (radius - table_r)).atan().to_degrees(),
        None    => g.crown_angle,
    };
    let pavilion_angle = match g.pavilion_pct {
        Some(p) => (p / 100.0 * w_mm * g.scale_mm / radius).atan().to_degrees(),
        None    => g.pavilion_angle,
    };

    let params = DiamondParams {
        table_pct:      g.table_pct,
        crown_angle,
        pavilion_angle,
        oval_ratio,
        girdle_facets:  g.girdle_facets.unwrap_or(if oval_ratio > 1.05 { 64 } else { 16 }),
    };

    // Cross-check the reported total depth against the facet heights (warning
    // only; the reported depth includes the girdle, which we do not model).
    if let Some(depth) = g.depth_pct {
        let crown_h    = (radius - table_r) * crown_angle.to_radians().tan();
        let pavilion_h = radius * pavilion_angle.to_radians().tan();
        let calc = (crown_h + pavilion_h) / (2.0 * radius) * 100.0;
        if (calc - depth).abs() > 6.0 {
            eprintln!("warning: gem '{}': reported depth {depth}% but proportions give {calc:.1}% (girdle excluded)",
                      g.report);
        }
    }

    // Colour grade D..Z -> faint yellow-brown body tint (heuristic Beer-Lambert,
    // per scene unit; a 6.5 mm stone at scale_mm=0.001 spans ~0.0065 units).
    let sigma = match g.color_grade.trim().to_uppercase().chars().next() {
        Some('D') | Some('E') | Some('F') => 0.0,
        Some('G') => 50.0,
        Some('H') => 80.0,
        Some('I') => 120.0,
        Some('J') => 170.0,
        Some('K') => 230.0,
        Some('L') | Some('M') => 300.0,
        _         => 400.0,
    };
    // Material preset (IOR / dispersion / body colour) + D..Z grade tint, which
    // adds in the Beer-Lambert exponent, + optional explicit overrides.
    let grade_tint = Color::new(1.0, 0.75, 0.35) * sigma;
    let (mut cauchy_b, mut cauchy_c, preset_abs) = gem_preset(g.material.as_deref().unwrap_or("diamond"))?;
    if let Some(v) = g.cauchy_b { cauchy_b = v; }
    if let Some(v) = g.cauchy_c { cauchy_c = v; }
    let absorption = preset_abs + grade_tint + g.absorption.map(col).unwrap_or_default();
    let mat: Arc<dyn Material> = Arc::new(SpectralDielectric { cauchy_b, cauchy_c, absorption });

    // Cut style: brilliant (round/oval) or a step-family preset.
    let cut = g.cut.as_deref().unwrap_or("brilliant");
    let mut hittable: Arc<dyn Hittable> = match cut {
        "brilliant" =>
            Arc::new(Diamond::new(p3(g.center), radius, params, mat)),
        other => {
            let sp = step_cut_preset(other, g, crown_angle, pavilion_angle)?;
            let cut_obj = StepCut::new(p3(g.center), radius * oval_ratio, radius, sp, mat)
                .map_err(|e| format!("gem '{}': {e}", g.report))?;
            Arc::new(cut_obj)
        }
    };
    if let Some(s) = g.scale     { hittable = Arc::new(UniformScale::new(hittable, s)); }
    if let Some(a) = g.rotate_y  { hittable = Arc::new(Rotate::around_y(hittable, a)); }
    if let Some(t) = g.translate { hittable = Arc::new(Translate::new(hittable, v3(t))); }
    Ok(hittable)
}

fn build(file: SceneFile, base: &std::path::Path) -> Result<SceneData, String> {
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
        BackgroundConfig::EnvMap { path } => {
            let path = resolve_asset(base, &path);
            let mut reader = image::ImageReader::open(&path)
                .map_err(|e| format!("cannot load environment map '{path}': {e}"))?
                .with_guessed_format()
                .map_err(|e| format!("cannot load environment map '{path}': {e}"))?;
            reader.limits(image::Limits::no_limits());
            let img = reader
                .decode()
                .map_err(|e| format!("cannot load environment map '{path}': {e}"))?
                .into_rgb32f();
            Background::EnvMap(Arc::new(EnvMapData::new(img)))
        }
    };

    // ── Static objects ────────────────────────────────────────────────────────
    let mut static_objects: Vec<Arc<dyn Hittable>> = Vec::new();
    // Created up front so emissive objects found below can join the NEE light list.
    let mut lights = HittableList::new();

    // A throwaway Lambertian is used as the boundary-shape material for volume
    // types; it is never actually sampled — the HG phase function inside the
    // medium is what scatters photons.
    let dummy_mat = || -> Arc<dyn Material> {
        Arc::new(Lambertian { texture: Texture::from(Color::new(0.5, 0.5, 0.5)) })
    };

    for (i, obj) in file.objects.into_iter().enumerate() {
        let (scale, rotate_y, translate) = (obj.scale, obj.rotate_y, obj.translate);
        let mut hittable: Arc<dyn Hittable> = match obj.shape {
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
                    obj.material.ok_or_else(|| format!("objects[{i}]: material is required for this shape type"))?,
                    base,
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
                    ShapeConfig::Diamond { center, radius, table_pct, crown_angle, pavilion_angle, oval_ratio, girdle_facets } =>
                        Arc::new(Diamond::new(
                            p3(center), radius,
                            DiamondParams { table_pct, crown_angle, pavilion_angle, oval_ratio, girdle_facets },
                            mat,
                        )),
                    ShapeConfig::StepCut { center, length, width, table_pct, crown_angle, pavilion_angle, outline, points, corner_pct, corner_segs, oval_segs, pear, crown_steps, pavilion_steps, culet_pct, step_spread, chevrons } => {
                        let ol = step_outline_name(outline.as_deref().unwrap_or("rect"), corner_pct, corner_segs, oval_segs, pear, points)
                            .map_err(|e| format!("objects[{i}].shape: {e}"))?;
                        let sc = StepCut::new(
                            p3(center), 0.5 * length, 0.5 * width,
                            StepCutParams { table_pct, crown_angle, pavilion_angle, outline: ol, crown_steps, pavilion_steps, culet_pct, step_spread, chevrons },
                            mat,
                        ).map_err(|e| format!("objects[{i}].shape: {e}"))?;
                        Arc::new(sc)
                    }
                    ShapeConfig::InfinitePlane { point, normal, wave_amplitude, wave_scale } =>
                        Arc::new(InfinitePlane::new(p3(point), v3(normal), wave_amplitude, wave_scale, mat)),
                    ShapeConfig::Mesh { path } =>
                        crate::mesh::load_obj(&resolve_asset(base, &path), mat)
                            .map_err(|e| format!("objects[{i}]: {e}"))?,
                    _ => unreachable!(), // volume variants handled above
                }
            }
        };
        // Apply SRT transforms: scale → rotate → translate.
        if let Some(s) = scale    { hittable = Arc::new(UniformScale::new(hittable, s)); }
        if let Some(a) = rotate_y { hittable = Arc::new(Rotate::around_y(hittable, a)); }
        if let Some(t) = translate { hittable = Arc::new(Translate::new(hittable, v3(t))); }
        // Emissive, NEE-sampleable objects (e.g. an OBJ mesh with a light material,
        // an emissive sphere) join the light list so NEE targets them explicitly;
        // the same Arc serves both world and lights.
        if hittable.light_area() > 0.0 {
            lights.add_shared(Arc::clone(&hittable));
        }
        static_objects.push(hittable);
    }

    // ── Gems from grading reports ─────────────────────────────────────────────
    for (i, g) in file.gems.iter().enumerate() {
        let hittable = build_gem(g).map_err(|e| format!("gems[{i}]: {e}"))?;
        static_objects.push(hittable);
    }

    // ── Lights ────────────────────────────────────────────────────────────────
    // Each entry creates a DiffuseLight quad in both the world and the NEE list.
    for (i, lc) in file.lights.iter().enumerate() {
        let _ = i;
        let mat: Arc<dyn Material> = match lc.temp_k {
            Some(t) => Arc::new(BlackbodyLight::new(t, lc.intensity.unwrap_or(1.0))),
            None    => Arc::new(DiffuseLight { emit: Texture::from(col(lc.color)) }),
        };
        let corner = p3(lc.corner);
        let u      = v3(lc.u);
        let v      = v3(lc.v);
        static_objects.push(Arc::new(Quad::new(corner, u, v, Arc::clone(&mat))));
        lights.add(Quad::new(corner, u, v, mat));
    }

    // ── Caustics ──────────────────────────────────────────────────────────────
    let (enable_caustics, caustic_gather_radius, caustic_num_photons) = match file.caustics {
        Some(c) => (c.enabled, c.gather_radius, c.num_photons),
        None    => (false, 0.15, 500_000),
    };

    // ── Assemble SceneData ────────────────────────────────────────────────────
    // Leak the name string so it lives for 'static — the leak is at most a few
    // bytes per reload, negligible over the life of the process.
    let name: &'static str = Box::leak(file.name.into_boxed_str());

    let mut world_list = HittableList::new();
    for obj in static_objects { world_list.objects.push(obj); }
    let world: Arc<dyn Hittable> = Arc::new(BvhTree::from_list(world_list));

    // ── Animation ─────────────────────────────────────────────────────────────
    let animation = file.animation.and_then(|ac| {
        let kind = if let Some(oc) = ac.orbit {
            let look_at_pt = p3(oc.look_at);
            AnimationKind::Orbit(OrbitData {
                center:      p3(oc.center),
                radius:      oc.radius,
                height:      oc.height,
                look_at:     look_at_pt,
                start_angle: oc.start_angle.to_radians(),
                end_angle:   oc.end_angle.to_radians(),
                vfov:        oc.vfov      .unwrap_or(cam_init.vfov),
                aperture:    oc.aperture  .unwrap_or(cam_init.aperture),
                focus_dist:  oc.focus_dist,
            })
        } else {
            let keyframes = ac.keyframes.into_iter().map(|kc| {
                let from = p3(kc.look_from);
                let at   = p3(kc.look_at);
                Keyframe {
                    time:       kc.time,
                    look_from:  from,
                    look_at:    at,
                    vfov:       kc.vfov      .unwrap_or(cam_init.vfov),
                    aperture:   kc.aperture  .unwrap_or(cam_init.aperture),
                    focus_dist: kc.focus_dist.unwrap_or_else(|| (from - at).length().max(0.01)),
                }
            }).collect();
            AnimationKind::Keyframes(keyframes)
        };
        AnimationData::new(ac.fps, ac.duration, kind)
    });

    let scene = SceneData {
        world,
        lights,
        background,
        name,
        cam_init,
        max_samples:   file.max_samples,
        enable_caustics,
        caustic_quad:          file.lights.first().map(|lc| (p3(lc.corner), v3(lc.u), v3(lc.v), col(lc.color))),
        caustic_gather_radius,
        caustic_num_photons,
        photon_map:    None,
        animation,
    };
    Ok(scene)
}

// ── Material builder ──────────────────────────────────────────────────────────

fn load_img(path: &str) -> Result<Arc<image::RgbImage>, String> {
    image::open(path)
        .map_err(|e| format!("cannot load texture '{path}': {e}"))
        .map(|img| Arc::new(img.into_rgb8()))
}

fn build_material(cfg: MaterialConfig, base: &std::path::Path) -> Result<Arc<dyn Material>, String> {
    Ok(match cfg {
        MaterialConfig::Lambertian { color } =>
            Arc::new(Lambertian { texture: Texture::from(col(color)) }),

        MaterialConfig::Checker { even, odd, scale } =>
            Arc::new(Lambertian { texture: Texture::Checker { scale, even: col(even), odd: col(odd) } }),

        MaterialConfig::Image { path } =>
            Arc::new(Lambertian { texture: Texture::load(&resolve_asset(base, &path)).map_err(|e| format!("texture '{path}': {e}"))? }),

        MaterialConfig::Metal { color, fuzz } =>
            Arc::new(PbrMaterial { albedo: col(color), metallic: 1.0, roughness: fuzz, ..Default::default() }),

        MaterialConfig::Dielectric { ior } =>
            Arc::new(Dielectric { ir: ior }),

        MaterialConfig::SpectralDielectric { ior, dispersion, absorption } => {
            // Convert (ior at d-line, spread over visible) to Cauchy n(λ)=B+C/λ²
            // C determined from spread: Δn = C × (1/λ_b² − 1/λ_r²), λ in μm.
            // λ_b = 0.435 μm, λ_r = 0.700 μm → denominator ≈ 3.250 μm⁻².
            let cauchy_c = dispersion / 3.250;
            // B so that n(d-line, 587.56 nm) = ior: B = ior − C / 0.58756².
            // The d-line (He 587.56 nm) is the standard reference for optical glass IOR.
            let cauchy_b = ior - cauchy_c / (0.58756 * 0.58756);
            Arc::new(SpectralDielectric {
                cauchy_b,
                cauchy_c,
                absorption: Color::new(absorption[0], absorption[1], absorption[2]),
            })
        }

        MaterialConfig::DiffuseLight { color } =>
            Arc::new(DiffuseLight { emit: Texture::from(col(color)) }),

        MaterialConfig::Pbr { albedo, metallic, roughness, anisotropy, anisotropy_angle,
                              clearcoat, clearcoat_roughness, film_thickness, film_ior,
                              sheen, sheen_tint, emission, emission_strength } =>
            Arc::new(PbrMaterial { albedo: col(albedo), metallic, roughness,
                                   anisotropy, anisotropy_angle,
                                   clearcoat, clearcoat_roughness, film_thickness, film_ior,
                                   sheen, sheen_tint,
                                   emission: col(emission), emission_strength,
                                   ..PbrMaterial::default() }),

        MaterialConfig::TexturedPbr { albedo_path, roughness_path, metallic_path, ao_path, normal_path, roughness, metallic } =>
            Arc::new(PbrMaterial {
                albedo_tex:    Some(load_img(&resolve_asset(base, &albedo_path))?),
                roughness_tex: roughness_path.as_deref().map(|p| load_img(&resolve_asset(base, p))).transpose()?,
                metallic_tex:  metallic_path.as_deref().map(|p| load_img(&resolve_asset(base, p))).transpose()?,
                ao_tex:        ao_path.as_deref().map(|p| load_img(&resolve_asset(base, p))).transpose()?,
                normal_tex:    normal_path.as_deref().map(|p| load_img(&resolve_asset(base, p))).transpose()?,
                roughness,
                metallic,
                ..PbrMaterial::default()
            }),

        MaterialConfig::SpectralMetal { variant, roughness } => {
            let v = match variant.to_lowercase().as_str() {
                "copper" => SpectralMetalVariant::Copper,
                "silver" => SpectralMetalVariant::Silver,
                _        => SpectralMetalVariant::Gold,
            };
            Arc::new(SpectralMetal::new(v, roughness))
        }

        MaterialConfig::Pearl { base_color, ior, film_thickness, orient_strength, film_scale, luster_roughness } =>
            Arc::new(PearlMaterial { base_color: col(base_color), ior, film_thickness, orient_strength, film_scale, luster_roughness }),

        MaterialConfig::Sss { albedo, sigma_a, ior, density, g } =>
            Arc::new(SSSMaterial { albedo: col(albedo), sigma_a: col(sigma_a), ior, density, g }),
    })
}

// ── Tiny array-to-vec3/point3/color helpers ───────────────────────────────────

#[inline] fn v3([x, y, z]: [f32; 3]) -> Vec3   { Vec3::new(x, y, z) }
#[inline] fn p3([x, y, z]: [f32; 3]) -> Point3 { Point3::new(x, y, z) }
#[inline] fn col([r, g, b]: [f32; 3]) -> Color  { Color::new(r, g, b) }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sample_scene_file_loads() {
        // scenes/sample.toml is the documented parameter reference (README points
        // to it) — it must always parse and build.
        let scene = load("scenes/sample.toml").expect("scenes/sample.toml must load");
        assert_eq!(scene.name, "Sample scene");
        assert!(!scene.lights.objects.is_empty(), "sample scene should have one light");
        assert!(scene.animation.is_none(), "sample scene ships with animation commented out");
    }

    fn parse_gem(gem_block: &str) -> GemConfig {
        let text = format!(
            "name = \"t\"
[camera]
look_from = [0,0,-1]
look_at = [0,0,0]
[background]
type = \"solid\"
color = [0,0,0]
{gem_block}"
        );
        let file: SceneFile = toml::from_str(&text).unwrap();
        assert_eq!(file.gems.len(), 1);
        file.gems.into_iter().next().unwrap()
    }

    #[test]
    fn gem_carat_maps_to_girdle_diameter() {
        let g = parse_gem("[[gems]]
carat = 1.0
center = [0,0,0]
");
        let h = build_gem(&g).unwrap();
        let bb = h.bounding_box().unwrap();
        let width = bb.max.x - bb.min.x;
        // 1 ct -> d = (1/0.00355)^(1/3) ~= 6.55 mm; scale_mm default 0.001 -> 0.00655 units.
        // The bbox uses the 16-gon circumradius r/cos(pi/16), so width = d*scale/cos(pi/16).
        let expected = (1.0f32 / 0.00355).cbrt() * 0.001 / (std::f32::consts::PI / 16.0).cos();
        assert!((width - expected).abs() < 1e-6, "width {width} vs expected {expected}");
    }

    #[test]
    fn gem_measurements_override_carat() {
        let g = parse_gem("[[gems]]
carat = 99.0
measurements_mm = [6.50, 6.40, 4.01]
center = [0,0,0]
");
        let h = build_gem(&g).unwrap();
        let bb = h.bounding_box().unwrap();
        let width = bb.max.x - bb.min.x;
        // minor axis W = 6.40 sets the radius; L/W = 1.016 < 2% -> round
        let expected = 6.40 * 0.001 / (std::f32::consts::PI / 16.0).cos();
        assert!((width - expected).abs() < 1e-6, "width {width} vs expected {expected}");
    }

    #[test]
    fn gem_requires_a_size_source() {
        let g = parse_gem("[[gems]]
center = [0,0,0]
");
        assert!(build_gem(&g).is_err());
    }

    #[test]
    fn gem_oval_measurements_set_ratio_and_radius() {
        // IGI LG625469749-style: 8.99 x 6.42 x 3.97 mm oval
        let g = parse_gem("[[gems]]
measurements_mm = [8.99, 6.42, 3.97]
table_pct = 58.0
center = [0,0,0]
");
        let h = build_gem(&g).unwrap();
        let bb = h.bounding_box().unwrap();
        let wx = bb.max.x - bb.min.x;
        let wz = bb.max.z - bb.min.z;
        let circ = (std::f32::consts::PI / 64.0).cos();   // ovals auto-use 64 girdle facets
        // semi-minor r = 6.42/2 mm -> z width = 6.42*0.001/circ; x stretched by 8.99/6.42
        assert!((wz - 6.42 * 0.001 / circ).abs() < 1e-6, "wz {wz}");
        assert!((wx / wz - 8.99 / 6.42).abs() < 1e-4, "ratio {} vs {}", wx / wz, 8.99 / 6.42);
    }

    #[test]
    fn gem_igi_percentages_convert_to_angles() {
        // 16% crown / 41% pavilion of the 6.42 mm minor axis, table 58%:
        // crown_h = 1.0272 mm over run (r - table_r) = 1.3482 mm -> 37.3 deg
        // pavilion_h = 2.6322 mm over r = 3.21 mm -> 39.35 deg
        let g = parse_gem("[[gems]]
measurements_mm = [8.99, 6.42, 3.97]
table_pct = 58.0
crown_pct = 16.0
pavilion_pct = 41.0
center = [0,0,0]
");
        let r_mm = 6.42 / 2.0f32;
        let table_r = 0.58 * r_mm;
        let exp_crown = (0.16 * 6.42 / (r_mm - table_r)).atan().to_degrees();
        let exp_pav   = (0.41 * 6.42 / r_mm).atan().to_degrees();
        // build_gem doesn't expose params; rebuild the same math here and check
        // the constructed diamond's bbox height matches crown_h + pavilion_h.
        let h = build_gem(&g).unwrap();
        let bb = h.bounding_box().unwrap();
        let h_mm = (bb.max.y - bb.min.y) / 0.001;
        let exp_h = 0.16 * 6.42 + 0.41 * 6.42;
        assert!((h_mm - exp_h).abs() < 0.02, "height {h_mm} mm vs expected {exp_h} mm");
        assert!((exp_crown - 37.3).abs() < 0.2 && (exp_pav - 39.35).abs() < 0.2,
                "angles {exp_crown} / {exp_pav}");
    }

    #[test]
    fn gem_defaults_to_tolkowsky_cut() {
        let g = parse_gem("[[gems]]
carat = 0.5
center = [0,0,0]
");
        assert!((g.table_pct - 53.0).abs() < 1e-6);
        assert!((g.crown_angle - 34.5).abs() < 1e-6);
        assert!((g.pavilion_angle - 40.75).abs() < 1e-6);
    }

    #[test]
    fn gem_material_presets_have_distinct_ior_and_absorption() {
        let d = gem_preset("diamond").unwrap();
        let s = gem_preset("sapphire").unwrap();
        let r = gem_preset("ruby").unwrap();
        let e = gem_preset("emerald").unwrap();
        assert!((d.0 - 2.351).abs() < 1e-6 && d.2.length_squared() == 0.0);
        // Diamond dispersion must be physical: nF - nC ~= 0.044, not 0.011.
        let nf = crate::spectrum::cauchy_ior(486.1, d.0, d.1);
        let nc = crate::spectrum::cauchy_ior(656.3, d.0, d.1);
        assert!((nf - nc - 0.044).abs() < 0.002, "diamond nF-nC = {:.4}", nf - nc);
        assert!((s.0 - 1.741).abs() < 1e-6 && (e.0 - 1.556).abs() < 1e-6);
        assert!(s.2.x > 100.0 && s.2.z < 100.0, "sapphire: absorb red, transmit blue");
        assert!(r.2.y > 100.0 && r.2.x < 100.0, "ruby: absorb green, transmit red");
        assert!(e.2.y < 100.0 && e.2.x > 100.0 && e.2.z > 100.0, "emerald: transmit green only");
        assert!(gem_preset("topaz").is_err());
    }

    #[test]
    fn gem_step_cut_uses_measurements_as_rectangle() {
        let g = parse_gem("[[gems]]
measurements_mm = [9.0, 6.0, 3.9]
cut = \"step\"
material = \"emerald\"
center = [0,0,0]
");
        let h = build_gem(&g).unwrap();
        let bb = h.bounding_box().unwrap();
        assert!((bb.max.x - bb.min.x - 0.009).abs() < 1e-6, "length 9 mm -> 0.009 units");
        assert!((bb.max.z - bb.min.z - 0.006).abs() < 1e-6, "width 6 mm -> 0.006 units");
    }

    #[test]
    fn gem_princess_cut_is_square() {
        let g = parse_gem("[[gems]]
measurements_mm = [7.0, 7.0, 4.2]
cut = \"princess\"
material = \"sapphire\"
center = [0,0,0]
");
        let h = build_gem(&g).unwrap();
        let bb = h.bounding_box().unwrap();
        assert!((bb.max.x - bb.min.x - 0.007).abs() < 1e-6, "square side 7 mm");
        assert!((bb.max.z - bb.min.z - 0.007).abs() < 1e-6, "square side 7 mm");
    }

    #[test]
    fn gem_princess_with_chevrons_builds() {
        let g = parse_gem("[[gems]]
measurements_mm = [7.0, 7.0, 4.2]
cut = \"princess\"
chevrons = 2
center = [0,0,0]
");
        let h = build_gem(&g).unwrap();
        let bb = h.bounding_box().unwrap();
        assert!((bb.max.x - bb.min.x - 0.007).abs() < 1e-6);
    }

    #[test]
    fn gem_trillion_and_marquise_build() {
        let g = parse_gem("[[gems]]
measurements_mm = [7.0, 7.0, 4.2]
cut = \"trillion\"
center = [0,0,0]
");
        let h = build_gem(&g).unwrap();
        let bb = h.bounding_box().unwrap();
        // triangle vertices (0, b), (+-0.866 a, -0.5 b), a = b = 3.5 mm * 0.001
        let a = 0.0035f32;
        assert!((bb.max.x - bb.min.x - 2.0 * 0.866 * a).abs() < 1e-5, "w {}", bb.max.x - bb.min.x);
        assert!((bb.max.z - bb.min.z - 1.5 * a).abs() < 1e-5, "d {}", bb.max.z - bb.min.z);

        let g = parse_gem("[[gems]]
measurements_mm = [10.0, 4.0, 3.0]
cut = \"marquise\"
center = [0,0,0]
");
        let h = build_gem(&g).unwrap();
        let bb = h.bounding_box().unwrap();
        assert!((bb.max.x - bb.min.x - 0.010).abs() < 1e-6, "length 10 mm");
        assert!((bb.max.z - bb.min.z - 0.004).abs() < 1e-6, "width 4 mm");
    }

    #[test]
    fn gem_unknown_cut_or_material_errors() {
        let g = parse_gem("[[gems]]
carat = 1.0
cut = \"heart\"
center = [0,0,0]
");
        assert!(build_gem(&g).err().map(|e| e.contains("cut")).unwrap_or(false));
        let g = parse_gem("[[gems]]
carat = 1.0
material = \"topaz\"
center = [0,0,0]
");
        assert!(build_gem(&g).err().map(|e| e.contains("material")).unwrap_or(false));
    }
}
