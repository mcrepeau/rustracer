use crate::vec3::{Point3, Vec3};

#[derive(Clone, Copy)]
pub struct Ray {
    pub origin:            Point3,
    pub direction:         Vec3,
    pub inv_dir:           Vec3,
    pub time:              f32,
    /// Hero wavelength in nanometres [380, 700].  Set once per camera ray and
    /// preserved through all scatters via `Ray::scatter_from`.  Used by
    /// SpectralDielectric to select the wavelength-dependent IOR via Cauchy's
    /// equation; non-spectral materials ignore it.
    pub wavelength:        f32,
    /// Set to true after the first dispersive refraction in a path.  Subsequent
    /// SpectralDielectric refractions use (1,1,1) attenuation so the CMF weight
    /// is applied exactly once per path, preventing compounding brightness.
    pub spectral_weighted: bool,
}

impl Ray {
    pub fn new(origin: Point3, direction: Vec3) -> Self {
        Self::new_at_time(origin, direction, 0.0)
    }

    pub fn new_at_time(origin: Point3, direction: Vec3, time: f32) -> Self {
        let inv_dir = Vec3::new(1.0 / direction.x, 1.0 / direction.y, 1.0 / direction.z);
        Self { origin, direction, inv_dir, time, wavelength: 550.0, spectral_weighted: false }
    }

    /// Create a scattered ray that inherits `time`, `wavelength`, and
    /// `spectral_weighted` from the parent, keeping all spectral state
    /// consistent along the full path.
    #[inline]
    pub fn scatter_from(origin: Point3, direction: Vec3, source: &Ray) -> Self {
        let inv_dir = Vec3::new(1.0 / direction.x, 1.0 / direction.y, 1.0 / direction.z);
        Self {
            origin, direction, inv_dir,
            time:              source.time,
            wavelength:        source.wavelength,
            spectral_weighted: source.spectral_weighted,
        }
    }

    pub fn at(self, t: f32) -> Point3 { self.origin + t * self.direction }
}
