use crate::vec3::Color;

/// Denoise a linear-HDR RGB framebuffer using OIDN.
/// `input` is a flat f32 slice of width×height×3 values (averaged, not summed).
/// Returns None if OIDN reports an error.
pub fn denoise_rgb(width: u32, height: u32, input: Vec<f32>) -> Option<Vec<Color>> {
    let device = oidn::Device::new();
    let mut output = vec![0.0f32; input.len()];
    let mut filter = oidn::RayTracing::new(&device);
    filter
        .hdr(true)
        .image_dimensions(width as usize, height as usize);
    filter.filter(&input, &mut output).ok()?;
    Some(
        output
            .chunks_exact(3)
            .map(|c| Color::new(c[0], c[1], c[2]))
            .collect(),
    )
}
