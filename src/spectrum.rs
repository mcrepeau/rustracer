use std::sync::OnceLock;
use crate::vec3::Color;

// ── CIE 1931 2° standard observer, 5 nm steps, 380–700 nm (65 entries) ────────
// Source: CIE standard tabulation.

const CIE_N:     usize = 65;
const CIE_START: f32   = 380.0;
const CIE_STEP:  f32   = 5.0;

#[rustfmt::skip]
#[allow(clippy::excessive_precision)]
static CIE_X: [f32; CIE_N] = [
    0.001368, 0.002236, 0.004243, 0.007650, 0.014310,
    0.023190, 0.043510, 0.077630, 0.134380, 0.214770,
    0.283900, 0.328500, 0.348280, 0.348060, 0.336200,
    0.318700, 0.290800, 0.251100, 0.195360, 0.142100,
    0.095640, 0.057950, 0.032010, 0.014700, 0.004900,
    0.002400, 0.009300, 0.029100, 0.063270, 0.109600,
    0.165500, 0.225750, 0.290400, 0.359700, 0.433450,
    0.512050, 0.594500, 0.678400, 0.762100, 0.842500,
    0.916300, 0.978600, 1.026300, 1.056700, 1.062200,
    1.045600, 1.002600, 0.938400, 0.854450, 0.751400,
    0.642400, 0.541900, 0.447900, 0.360800, 0.283500,
    0.218700, 0.164900, 0.121200, 0.087400, 0.063600,
    0.046770, 0.032900, 0.022700, 0.015840, 0.011359,
];

#[rustfmt::skip]
static CIE_Y: [f32; CIE_N] = [
    0.000039, 0.000064, 0.000120, 0.000217, 0.000396,
    0.000640, 0.001210, 0.002180, 0.004000, 0.007300,
    0.011600, 0.016840, 0.023000, 0.029800, 0.038000,
    0.048000, 0.060000, 0.073900, 0.090980, 0.112600,
    0.139020, 0.169300, 0.208020, 0.258600, 0.323000,
    0.407300, 0.503000, 0.608200, 0.710000, 0.793200,
    0.862000, 0.914850, 0.954000, 0.980300, 0.994950,
    1.000000, 0.995000, 0.978600, 0.952000, 0.915400,
    0.870000, 0.816300, 0.757000, 0.694900, 0.631000,
    0.566800, 0.503000, 0.441200, 0.381000, 0.321000,
    0.265000, 0.217000, 0.175000, 0.138200, 0.107000,
    0.081600, 0.061000, 0.044580, 0.032000, 0.023200,
    0.017000, 0.011920, 0.008210, 0.005723, 0.004102,
];

#[rustfmt::skip]
#[allow(clippy::excessive_precision)]
static CIE_Z: [f32; CIE_N] = [
    0.006450, 0.010550, 0.020050, 0.036210, 0.067850,
    0.110200, 0.207400, 0.371300, 0.645600, 1.039050,
    1.385600, 1.622960, 1.747060, 1.782600, 1.772110,
    1.744100, 1.669200, 1.528100, 1.287640, 1.041900,
    0.812950, 0.616200, 0.465180, 0.353300, 0.272000,
    0.212300, 0.158200, 0.111700, 0.078250, 0.057250,
    0.042160, 0.029840, 0.020300, 0.013400, 0.008750,
    0.005750, 0.003900, 0.002750, 0.002100, 0.001800,
    0.001650, 0.001400, 0.001100, 0.001000, 0.000800,
    0.000600, 0.000340, 0.000240, 0.000190, 0.000100,
    0.000050, 0.000030, 0.000020, 0.000010, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
];

// ── XYZ → linear sRGB (D65 white point) ───────────────────────────────────────

#[inline]
fn xyz_to_srgb_clamped(x: f32, y: f32, z: f32) -> [f32; 3] {
    [
        ( 3.2406 * x - 1.5372 * y - 0.4986 * z).max(0.0),
        (-0.9689 * x + 1.8758 * y + 0.0415 * z).max(0.0),
        ( 0.0557 * x - 0.2040 * y + 1.0570 * z).max(0.0),
    ]
}

// ── Per-channel normalisation (computed once) ──────────────────────────────────
// Ensures E_λ[spectral_to_rgb(λ)] = (1,1,1) for uniform λ ∈ [380, 700 nm].

static NORMS: OnceLock<[f32; 3]> = OnceLock::new();

fn norms() -> [f32; 3] {
    *NORMS.get_or_init(|| {
        let mut sum = [0.0f32; 3];
        for i in 0..CIE_N {
            let rgb = xyz_to_srgb_clamped(CIE_X[i], CIE_Y[i], CIE_Z[i]);
            sum[0] += rgb[0];
            sum[1] += rgb[1];
            sum[2] += rgb[2];
        }
        [
            CIE_N as f32 / sum[0].max(1e-6),
            CIE_N as f32 / sum[1].max(1e-6),
            CIE_N as f32 / sum[2].max(1e-6),
        ]
    })
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Map a wavelength λ (nm, clamped to [380, 700]) to a normalised linear-sRGB
/// colour such that E_λ[spectral_to_rgb(λ)] = (1,1,1) for uniform λ.
///
/// This is the hero-wavelength weight: multiply a scalar spectral radiance by
/// this colour to accumulate it into the RGB image buffer.
pub fn spectral_to_rgb(lambda: f32) -> Color {
    let t    = ((lambda - CIE_START) / CIE_STEP).clamp(0.0, (CIE_N - 1) as f32);
    let i    = (t as usize).min(CIE_N - 2);
    let frac = t - i as f32;
    let x = CIE_X[i] + frac * (CIE_X[i + 1] - CIE_X[i]);
    let y = CIE_Y[i] + frac * (CIE_Y[i + 1] - CIE_Y[i]);
    let z = CIE_Z[i] + frac * (CIE_Z[i + 1] - CIE_Z[i]);
    let [r, g, b] = xyz_to_srgb_clamped(x, y, z);
    let n = norms();
    Color::new(r * n[0], g * n[1], b * n[2])
}

/// Planck spectral radiance, un-normalised.
///
/// Returns a relative power value proportional to B(λ, T).  To get a
/// normalised weight (mean = 1 over [380, 700 nm]) divide by the mean
/// of this function over that range — `BlackbodyLight::new` does this
/// once at construction time and stores the reciprocal as a `norm` field.
///
/// `lambda_nm` in nanometres, `temp_k` in Kelvin.
#[inline]
pub fn planck_raw(lambda_nm: f32, temp_k: f32) -> f32 {
    const HC_OVER_K: f32 = 14_388_000.0; // hc/k in nm·K
    let x = HC_OVER_K / (lambda_nm * temp_k);
    1.0 / (lambda_nm.powi(5) * (x.exp() - 1.0).max(1e-30))
}

/// Cauchy dispersion equation: n(λ) = B + C/λ² (λ in **micrometres**).
///
/// Pass `lambda_nm` in nanometres; the function converts internally.
/// Common values (B, C in μm²):
/// - Crown glass:  B ≈ 1.507, C ≈ 0.00375
/// - Dense flint:  B ≈ 1.612, C ≈ 0.00950
/// - Diamond:      B ≈ 2.395, C ≈ 0.00585
#[inline]
pub fn cauchy_ior(lambda_nm: f32, b: f32, c: f32) -> f32 {
    let lam_um = lambda_nm * 1e-3;
    b + c / (lam_um * lam_um)
}
