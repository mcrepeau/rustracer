use crate::vec3::Color;

/// Denoise a linear-HDR RGB framebuffer using OIDN.
///
/// `color` is a flat f32 slice of `width × height × 3` values (already
/// averaged, not summed).  Pass non-empty `albedo` and `normal` buffers
/// (same layout) to enable the auxiliary-pass mode: OIDN uses the unlit
/// surface colour and world-space normals to guide denoising across
/// geometry boundaries, producing sharper results especially at low spp.
/// Pass empty slices to fall back to colour-only denoising.
///
/// Returns `None` if OIDN reports an error.
pub fn denoise_rgb(
    width:  u32,
    height: u32,
    color:  Vec<f32>,
    albedo: Vec<f32>,
    normal: Vec<f32>,
) -> Option<Vec<Color>> {
    let device = oidn::Device::new();
    let mut output = vec![0.0f32; color.len()];
    let mut filter = oidn::RayTracing::new(&device);
    filter
        .hdr(true)
        .image_dimensions(width as usize, height as usize);
    if !albedo.is_empty() && !normal.is_empty() {
        filter.albedo_normal(&albedo, &normal);
    }
    filter.filter(&color, &mut output).ok()?;
    Some(
        output
            .chunks_exact(3)
            .map(|c| Color::new(c[0], c[1], c[2]))
            .collect(),
    )
}
