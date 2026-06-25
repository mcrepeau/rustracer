use std::sync::Arc;
use crate::animation::AnimationData;
use crate::camera::SceneCameraParams;
use crate::hittable::{Hittable, HittableList};
use crate::photon::PhotonMap;
use crate::renderer::Background;
use crate::vec3::{Color, Point3, Vec3};

pub struct SceneData {
    pub world:          Arc<dyn Hittable>,
    pub lights:         HittableList,
    pub background:     Background,
    pub name:           &'static str,
    pub cam_init:       SceneCameraParams,
    pub max_samples:    u32,
    /// Enable caustic photon mapping for this scene.
    pub enable_caustics: bool,
    /// Area-light emitter for the photon map when `Background` is not
    /// `Physical`.  Fields: (origin, U-extent, V-extent, emission colour).
    pub caustic_quad:          Option<(Point3, Vec3, Vec3, Color)>,
    /// Photon gather radius in world units.  Must match the scene's spatial
    /// scale: ~0.15 for unit-scale scenes, ~10 for 0–555 coordinate scenes.
    pub caustic_gather_radius: f32,
    /// Number of photons to trace when building the caustic map.
    pub caustic_num_photons: u32,
    /// Caustic photon map, rebuilt after every sun-direction change when enabled.
    pub photon_map:      Option<Arc<PhotonMap>>,
    /// Camera animation; `None` for static scenes.
    pub animation:       Option<AnimationData>,
}

impl SceneData {
    /// Rebuild only the photon map, reusing the current world BVH.
    /// Call this after sun-direction changes.
    pub fn rebuild_caustics(&mut self) {
        if !self.enable_caustics { return; }
        let r   = self.caustic_gather_radius;
        let n   = self.caustic_num_photons;
        let world = Arc::clone(&self.world);
        if let Background::Physical { sun_dir, .. } = &self.background {
            let sun_dir = *sun_dir;
            let perp = if sun_dir.x.abs() < 0.9 {
                Vec3::new(0.0, -sun_dir.z, sun_dir.y).unit()
            } else {
                Vec3::new(-sun_dir.z, 0.0, sun_dir.x).unit()
            };
            let power_dir = (sun_dir + perp * 0.04).unit();
            let sun_color = self.background.eval(power_dir) * std::f32::consts::PI;
            self.photon_map = Some(Arc::new(
                PhotonMap::build(world.as_ref(), sun_dir, sun_color, n, r)
            ));
        } else if let Some((origin, u, v, color)) = self.caustic_quad {
            self.photon_map = Some(Arc::new(
                PhotonMap::build_from_quad(world.as_ref(), origin, u, v, color, n, r)
            ));
        }
    }
}
