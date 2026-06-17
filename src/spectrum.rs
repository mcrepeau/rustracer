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

// ── Spectral metal IOR tables (Johnson & Christy 1972) ────────────────────────
// 13 samples at 25 nm intervals from 380 nm to 680 nm.

const METAL_N:     usize = 13;
const METAL_START: f32   = 380.0;
const METAL_STEP:  f32   = 25.0;

// Gold (Au) — characteristic interband edge ≈ 480–520 nm produces the warm yellow.
#[rustfmt::skip] static GOLD_N:   [f32; METAL_N] = [1.69, 1.65, 1.54, 1.33, 0.97, 0.54, 0.30, 0.23, 0.20, 0.19, 0.18, 0.16, 0.16];
#[rustfmt::skip] static GOLD_K:   [f32; METAL_N] = [1.91, 1.94, 1.95, 1.92, 1.80, 1.93, 2.41, 3.00, 3.56, 4.07, 4.56, 4.99, 5.41];
// Copper (Cu) — similar edge, shifted toward the red; characteristic orange tint.
#[rustfmt::skip] static COPPER_N: [f32; METAL_N] = [1.23, 1.21, 1.16, 1.10, 1.04, 0.88, 0.60, 0.38, 0.26, 0.22, 0.21, 0.20, 0.21];
#[rustfmt::skip] static COPPER_K: [f32; METAL_N] = [2.00, 2.26, 2.49, 2.59, 2.64, 2.69, 2.84, 3.21, 3.70, 4.17, 4.64, 5.07, 5.48];
// Silver (Ag) — nearly flat and very high reflectance across the visible; near-white.
#[rustfmt::skip] static SILVER_N: [f32; METAL_N] = [0.18, 0.14, 0.12, 0.13, 0.13, 0.13, 0.14, 0.14, 0.15, 0.16, 0.16, 0.17, 0.17];
#[rustfmt::skip] static SILVER_K: [f32; METAL_N] = [1.57, 2.01, 2.50, 3.02, 3.55, 4.05, 4.54, 5.02, 5.50, 5.98, 6.42, 6.84, 7.22];

#[inline]
fn interp_metal(ns: &[f32; METAL_N], ks: &[f32; METAL_N], lambda_nm: f32) -> (f32, f32) {
    let t = ((lambda_nm - METAL_START) / METAL_STEP).clamp(0.0, (METAL_N - 1) as f32);
    let i = (t as usize).min(METAL_N - 2);
    let f = t - i as f32;
    (ns[i] + f * (ns[i + 1] - ns[i]), ks[i] + f * (ks[i + 1] - ks[i]))
}

/// Complex IOR `(n, k)` for gold at `lambda_nm` (nanometres), from J&C 1972.
pub fn gold_ior(lambda_nm:   f32) -> (f32, f32) { interp_metal(&GOLD_N,   &GOLD_K,   lambda_nm) }
/// Complex IOR `(n, k)` for copper at `lambda_nm` (nanometres), from J&C 1972.
pub fn copper_ior(lambda_nm: f32) -> (f32, f32) { interp_metal(&COPPER_N, &COPPER_K, lambda_nm) }
/// Complex IOR `(n, k)` for silver at `lambda_nm` (nanometres), from J&C 1972.
pub fn silver_ior(lambda_nm: f32) -> (f32, f32) { interp_metal(&SILVER_N, &SILVER_K, lambda_nm) }

/// Exact Fresnel reflectance for a conductor (unpolarized, vacuum incidence).
///
/// `cos_theta_i` — cosine of the angle of incidence (1 = normal, 0 = grazing).
/// `n`, `k`       — real and imaginary parts of the complex IOR at the relevant λ.
#[inline]
pub fn fresnel_conductor(cos_theta_i: f32, n: f32, k: f32) -> f32 {
    let cos2 = cos_theta_i * cos_theta_i;
    let sin2 = (1.0 - cos2).max(0.0);
    let n2   = n * n;
    let k2   = k * k;
    let t0       = n2 - k2 - sin2;
    let a2plusb2 = (t0 * t0 + 4.0 * n2 * k2).sqrt();
    let t1       = a2plusb2 + cos2;
    let a        = (0.5 * (a2plusb2 + t0)).max(0.0).sqrt();
    let t2       = 2.0 * cos_theta_i * a;
    let rs       = (t1 - t2) / (t1 + t2).max(1e-12);
    let t3       = cos2 * a2plusb2 + sin2 * sin2;
    let t4       = t2 * sin2;
    let rp       = rs * (t3 - t4) / (t3 + t4).max(1e-12);
    (0.5 * (rp + rs)).clamp(0.0, 1.0)
}

// ── Cauchy dispersion ─────────────────────────────────────────────────────────

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalization_mean_is_white() {
        // The normalization in norms() is defined over the 65 CIE table entries
        // (380–700 nm, 5 nm steps). Averaging spectral_to_rgb at exactly those
        // wavelengths must yield (1, 1, 1) — that is the invariant the renderer relies on.
        let mut sum = [0.0f64; 3];
        for i in 0..CIE_N {
            let lambda = CIE_START + i as f32 * CIE_STEP;
            let c = spectral_to_rgb(lambda);
            sum[0] += c.x as f64;
            sum[1] += c.y as f64;
            sum[2] += c.z as f64;
        }
        let mean = [sum[0] / CIE_N as f64, sum[1] / CIE_N as f64, sum[2] / CIE_N as f64];
        assert!((mean[0] - 1.0).abs() < 1e-4, "R mean = {}", mean[0]);
        assert!((mean[1] - 1.0).abs() < 1e-4, "G mean = {}", mean[1]);
        assert!((mean[2] - 1.0).abs() < 1e-4, "B mean = {}", mean[2]);
    }

    #[test]
    fn wavelength_colors_match_visible_spectrum() {
        let blue  = spectral_to_rgb(450.0);
        let green = spectral_to_rgb(550.0);
        let red   = spectral_to_rgb(650.0);

        assert!(blue.z  > blue.x  && blue.z  > blue.y,  "450 nm should be blue-dominant");
        assert!(green.y > green.x && green.y > green.z, "550 nm should be green-dominant");
        assert!(red.x   > red.y   && red.x   > red.z,   "650 nm should be red-dominant");
    }

    #[test]
    fn out_of_range_wavelengths_clamp_to_boundary() {
        let at_380  = spectral_to_rgb(380.0);
        let below   = spectral_to_rgb(300.0);
        let at_700  = spectral_to_rgb(700.0);
        let above   = spectral_to_rgb(800.0);

        assert!((at_380.x - below.x).abs() < 1e-5);
        assert!((at_380.y - below.y).abs() < 1e-5);
        assert!((at_380.z - below.z).abs() < 1e-5);
        assert!((at_700.x - above.x).abs() < 1e-5);
        assert!((at_700.y - above.y).abs() < 1e-5);
        assert!((at_700.z - above.z).abs() < 1e-5);
    }

    #[test]
    fn planck_hotter_blackbody_is_bluer() {
        // At 6500 K (daylight) the blue/red ratio is higher than at 3000 K (warm tungsten).
        // Wien's displacement law: λ_peak ∝ 1/T.
        let ratio_hot  = planck_raw(450.0, 6500.0) / planck_raw(650.0, 6500.0);
        let ratio_warm = planck_raw(450.0, 3000.0) / planck_raw(650.0, 3000.0);
        assert!(ratio_hot > ratio_warm, "6500K blue/red ratio should exceed 3000K");
    }

    #[test]
    fn planck_monotone_in_temperature() {
        // Hotter blackbody emits more radiance at every wavelength.
        let lam = 550.0;
        assert!(planck_raw(lam, 6000.0)  > planck_raw(lam, 3000.0));
        assert!(planck_raw(lam, 10000.0) > planck_raw(lam, 6000.0));
    }

    #[test]
    fn cauchy_crown_glass_sodium_d_line() {
        // Crown glass at the sodium D line (589.3 nm): expected IOR ≈ 1.52.
        // Reference: standard optics tables.
        let n = cauchy_ior(589.3, 1.507, 0.00375);
        assert!((n - 1.52).abs() < 0.005, "crown glass IOR at 589 nm = {n:.4}");
    }

    #[test]
    fn cauchy_normal_dispersion() {
        // IOR must decrease monotonically from blue to red for positive C.
        let (b, c) = (1.507, 0.00375);
        let n_blue  = cauchy_ior(450.0, b, c);
        let n_green = cauchy_ior(550.0, b, c);
        let n_red   = cauchy_ior(650.0, b, c);
        assert!(n_blue > n_green, "IOR should decrease toward red (blue > green)");
        assert!(n_green > n_red,  "IOR should decrease toward red (green > red)");
    }

    #[test]
    fn fresnel_conductor_grazing_approaches_one() {
        // At near-grazing incidence, reflectance → 1 for any conductor.
        let r = fresnel_conductor(0.001, 0.18, 3.0);
        assert!(r > 0.99, "grazing reflectance should be > 0.99, got {r:.4}");
    }

    #[test]
    fn fresnel_conductor_normal_incidence_matches_formula() {
        // At normal incidence (cos θ = 1), the conductor Fresnel formula simplifies to:
        //   R = ((n-1)² + k²) / ((n+1)² + k²)
        let (n, k) = (0.23_f32, 3.00_f32);
        let expected = ((n - 1.0).powi(2) + k * k) / ((n + 1.0).powi(2) + k * k);
        let got = fresnel_conductor(1.0, n, k);
        assert!((got - expected).abs() < 1e-4, "got {got:.5}, expected {expected:.5}");
    }

    #[test]
    fn fresnel_conductor_result_in_unit_range() {
        for &cos_theta in &[0.0f32, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0] {
            let r = fresnel_conductor(cos_theta, 1.5, 2.0);
            assert!(r >= 0.0 && r <= 1.0,
                "Fresnel({cos_theta}) = {r:.4} is outside [0, 1]");
        }
    }

    #[test]
    fn gold_reflects_much_more_red_than_blue() {
        // Gold's characteristic warm colour comes from its interband absorption edge
        // around 480–520 nm: low reflectance in blue, high in red/yellow.
        // Values derived from the J&C 1972 tabulation in GOLD_N / GOLD_K.
        let (n_blue, k_blue) = gold_ior(440.0); // deep blue
        let (n_red,  k_red)  = gold_ior(660.0); // red
        let r_blue = fresnel_conductor(1.0, n_blue, k_blue);
        let r_red  = fresnel_conductor(1.0, n_red,  k_red);
        assert!(r_blue < 0.5, "gold blue reflectance should be < 0.5, got {r_blue:.3}");
        assert!(r_red  > 0.9, "gold red reflectance should be > 0.9, got {r_red:.3}");
        assert!(r_red > 2.0 * r_blue, "gold red/blue ratio should exceed 2×");
    }

    #[test]
    fn silver_is_spectrally_flat_and_bright() {
        // Silver appears neutral (white/grey) because it reflects all visible
        // wavelengths at high and nearly equal reflectance.
        let lambdas = [450.0f32, 550.0, 650.0];
        let reflectances: Vec<f32> = lambdas.iter().map(|&lam| {
            let (n, k) = silver_ior(lam);
            fresnel_conductor(1.0, n, k)
        }).collect();

        let r_min = reflectances.iter().cloned().fold(f32::INFINITY,     f32::min);
        let r_max = reflectances.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        assert!(r_min > 0.9, "silver reflectance should be > 0.9 at all visible wavelengths, min = {r_min:.3}");
        assert!(r_max - r_min < 0.06, "silver spectral spread should be < 0.06, got {:.3}", r_max - r_min);
    }
}
