use std::f32::consts::PI;
use rand::Rng;
use rand::rngs::SmallRng;
use rand::SeedableRng;
use rayon::prelude::*;
use crate::hittable::Hittable;
use crate::ray::Ray;
use crate::vec3::{Color, Point3, Vec3};

struct KdPhoton {
    pos:   [f32; 3],
    power: [f32; 3],
    nor:   [f32; 3],
    /// Split axis used when this node was inserted (x=0, y=1, z=2).
    /// Only meaningful for interior nodes; leaves keep the default 0.
    axis:  u8,
}

const DISK_R: f32 = 20.0;
pub const PHOTON_MAX_DEPTH: i32 = 12;

/// Balanced-kd-tree caustic photon map with surface-normal hemisphere filtering.
///
/// Photons are emitted from a light source and stored only when they reach a
/// diffuse surface after at least one specular/transmissive bounce (caustic
/// path — plain glass included, not just spectral materials).
/// `irradiance()` returns the Epanechnikov-filtered estimate
/// pre-divided by π, so the caller multiplies by albedo to get radiance.
/// Photons whose stored surface normal opposes the shading normal are
/// rejected, preventing caustic "leakage" through opaque surfaces.
pub struct PhotonMap {
    nodes:     Vec<KdPhoton>,
    gather_r2: f32,
    /// 2 / (π² R²) — Epanechnikov kernel normaliser pre-divided by π.
    norm:      f32,
}

impl PhotonMap {
    /// Trace `num_photons` from a directional disk facing `sun_dir` and build
    /// the map.  `sun_color` should be `background.eval(sun_dir) * PI`.
    pub fn build(
        world:         &dyn Hittable,
        sun_dir:       Vec3,
        sun_color:     Color,
        num_photons:   u32,
        gather_radius: f32,
    ) -> Self {
        let sun_down    = (-sun_dir).unit();
        let up          = if sun_down.x.abs() < 0.999 { Vec3::new(1.0, 0.0, 0.0) } else { Vec3::new(0.0, 1.0, 0.0) };
        let t           = sun_down.cross(up).unit();
        let b           = sun_down.cross(t);
        // Size the emission disk to the *real* geometry: photon_bounds() excludes
        // proxy shapes (InfinitePlane reports ±1e5), so in a plane-floor scene the
        // disk hugs the actual objects instead of wasting ~every photon on empty
        // floor (hit fraction ~(object_r / disk_r)^2).  The disk is centred on the
        // bounds, pushed out along the sun direction, with the bounding-sphere radius,
        // which provably covers the geometry's projection onto the disk plane.
        // ±SCENE_CLIP remains as a sanity bound for huge real scenes; the radius is
        // floored so disk_area stays > 0 for degenerate (point-like) bounds.
        const SCENE_CLIP: f32 = 1.0e3;
        let (disk_center, disk_r) = match world.photon_bounds() {
            Some(bb) => {
                let clip = |v: f32| v.clamp(-SCENE_CLIP, SCENE_CLIP);
                let min = Vec3::new(clip(bb.min.x), clip(bb.min.y), clip(bb.min.z));
                let max = Vec3::new(clip(bb.max.x), clip(bb.max.y), clip(bb.max.z));
                let c = (min + max) * 0.5;
                let r = ((max - min).length() * 0.5).max(1.0e-3);
                (c + sun_dir.unit() * r, r)
            }
            None => (sun_dir.unit() * 30.0, DISK_R), // only proxy geometry: old fallback
        };
        let disk_area   = PI * disk_r * disk_r;
        let power       = sun_color * (disk_area / (num_photons as f32 * PI));
        // Scale the storage luminance clamp with the per-photon mean power so it
        // stays proportional to the sun at any disk size (15.0 was tuned for R=20).
        let lum_clamp   = 15.0 * disk_area / (PI * DISK_R * DISK_R);

        let mut photons: Vec<KdPhoton> = (0..num_photons)
            .into_par_iter()
            .filter_map(|i| {
                let mut rng = SmallRng::seed_from_u64(
                    (i as u64).wrapping_mul(6_364_136_223_846_793_005) ^ 0x9E3779B97F4A7C15,
                );
                let r   = disk_r * rng.gen::<f32>().sqrt();
                let phi = 2.0 * PI * rng.gen::<f32>();
                let origin = disk_center + t * (r * phi.cos()) + b * (r * phi.sin());
                trace_photon(world, origin, sun_down, lum_clamp, power, &mut rng)
            })
            .collect();

        photons.shrink_to_fit();
        kd_build(&mut photons);
        Self::from_nodes(photons, gather_radius)
    }

    /// Trace `num_photons` emitted as a Lambertian source from a quad and build.
    pub fn build_from_quad(
        world:         &dyn Hittable,
        light_origin:  Point3,
        light_u:       Vec3,
        light_v:       Vec3,
        light_color:   Color,
        num_photons:   u32,
        gather_radius: f32,
    ) -> Self {
        let cross     = light_u.cross(light_v);
        let quad_area = cross.length();
        let normal    = cross / quad_area; // unit normal, emission side
        let (t_axis, b_axis) = normal.onb();
        let power     = light_color * (quad_area * PI / num_photons as f32);

        let mut photons: Vec<KdPhoton> = (0..num_photons)
            .into_par_iter()
            .filter_map(|i| {
                let mut rng = SmallRng::seed_from_u64(
                    (i as u64).wrapping_mul(6_364_136_223_846_793_005) ^ 0x9E3779B97F4A7C15,
                );
                let s = rng.gen::<f32>();
                let t = rng.gen::<f32>();
                let origin    = light_origin + light_u * s + light_v * t;
                let cos_theta = rng.gen::<f32>().sqrt();
                let sin_theta = (1.0 - cos_theta * cos_theta).sqrt();
                let phi       = 2.0 * PI * rng.gen::<f32>();
                // Cosine-weighted hemisphere sample in the quad's own frame.
                let dir = sin_theta * phi.cos() * t_axis
                        + sin_theta * phi.sin() * b_axis
                        + cos_theta * normal;
                trace_photon(world, origin, dir, 15.0, power, &mut rng)
            })
            .collect();

        photons.shrink_to_fit();
        kd_build(&mut photons);
        Self::from_nodes(photons, gather_radius)
    }

    fn from_nodes(nodes: Vec<KdPhoton>, gather_radius: f32) -> Self {
        Self {
            nodes,
            gather_r2: gather_radius * gather_radius,
            norm:      2.0 / (PI * PI * gather_radius * gather_radius),
        }
    }

    /// Epanechnikov-filtered irradiance at `pos` on a surface with outward
    /// `normal`, pre-divided by π.  Photons on the opposite hemisphere
    /// (stored normal · shading normal ≤ 0) are silently skipped.
    pub fn irradiance(&self, pos: Point3, normal: Vec3) -> Color {
        if self.nodes.is_empty() {
            return Color::default();
        }
        let p = [pos.x, pos.y, pos.z];
        let n = [normal.x, normal.y, normal.z];
        let mut acc = Color::default();
        kd_gather(&self.nodes, &p, self.gather_r2, &n, &mut acc);
        acc * self.norm
    }
}

// ── kd-tree build ─────────────────────────────────────────────────────────────
// Balanced implicit kd-tree: the root of nodes[lo..=hi] is always at the
// median index mid = (lo + hi) / 2 after partitioning by that node's axis.
// Using subslices keeps the bookkeeping index-free.

fn kd_build(nodes: &mut [KdPhoton]) {
    let n = nodes.len();
    if n == 0 { return; }

    // Choose the axis with the widest spread across this subset.
    let mut min = [f32::INFINITY; 3];
    let mut max = [f32::NEG_INFINITY; 3];
    for nd in nodes.iter() {
        for k in 0..3 {
            if nd.pos[k] < min[k] { min[k] = nd.pos[k]; }
            if nd.pos[k] > max[k] { max[k] = nd.pos[k]; }
        }
    }
    let axis = (0..3)
        .max_by(|&a, &b| (max[a] - min[a]).partial_cmp(&(max[b] - min[b])).unwrap())
        .unwrap();

    // Partial-sort so the median lands at nodes[mid].
    let mid = n / 2;
    nodes.select_nth_unstable_by(mid, |a, b| {
        a.pos[axis].partial_cmp(&b.pos[axis]).unwrap_or(std::cmp::Ordering::Equal)
    });
    nodes[mid].axis = axis as u8;

    kd_build(&mut nodes[..mid]);
    kd_build(&mut nodes[mid + 1..]);
}

// ── kd-tree gather ────────────────────────────────────────────────────────────

fn kd_gather(nodes: &[KdPhoton], pos: &[f32; 3], r2: f32, normal: &[f32; 3], acc: &mut Color) {
    let n = nodes.len();
    if n == 0 { return; }

    let mid  = n / 2;
    let node = &nodes[mid];

    let dx = node.pos[0] - pos[0];
    let dy = node.pos[1] - pos[1];
    let dz = node.pos[2] - pos[2];
    let d2 = dx * dx + dy * dy + dz * dz;

    if d2 < r2 {
        // Hemisphere check: skip photons stored on the opposite side of the surface.
        let dot = node.nor[0] * normal[0] + node.nor[1] * normal[1] + node.nor[2] * normal[2];
        if dot > 0.0 {
            *acc += Color::new(node.power[0], node.power[1], node.power[2]) * (1.0 - d2 / r2);
        }
    }

    // Prune subtrees using the split plane distance.
    let axis = node.axis as usize;
    let sq   = pos[axis] - node.pos[axis]; // positive → query is right-of-split

    if sq <= 0.0 {
        // Query is on the left (low) side: always search left, only right if sphere extends past split.
        kd_gather(&nodes[..mid], pos, r2, normal, acc);
        if sq * sq < r2 { kd_gather(&nodes[mid + 1..], pos, r2, normal, acc); }
    } else {
        // Query is on the right (high) side.
        kd_gather(&nodes[mid + 1..], pos, r2, normal, acc);
        if sq * sq < r2 { kd_gather(&nodes[..mid], pos, r2, normal, acc); }
    }
}

// ── photon tracing ────────────────────────────────────────────────────────────

fn trace_photon(
    world:     &dyn Hittable,
    origin:    Point3,
    dir:       Vec3,
    lum_clamp: f32,
    power:  Color,
    rng:    &mut SmallRng,
) -> Option<KdPhoton> {
    let mut pos          = origin;
    let mut dir          = dir;
    let mut power        = power;
    let mut spec_depth   = 0u32;
    let wavelength       = rng.gen_range(380.0_f32..700.0);

    for _ in 0..PHOTON_MAX_DEPTH {
        let mut ray = Ray::new_at_time(pos, dir, 0.0);
        ray.wavelength = wavelength;
        let rec = world.hit(&ray, 0.001, f32::INFINITY)?;
        let sr  = rec.mat.scatter(&ray, &rec, rng)?;

        if sr.skip_pdf {
            // Any specular/transmissive bounce counts toward the caustic path —
            // plain (non-spectral) glass included.  Spectral CMF spikes are handled
            // by the luminance clamp at storage below.
            power     *= sr.attenuation;
            pos        = sr.ray.origin;
            dir        = sr.ray.direction;
            spec_depth += 1;
        } else {
            if spec_depth > 0 && rec.mat.can_receive_caustics() {
                // Clamp spectral spikes once at storage, not per bounce.
                let lum = power.luminance();
                if lum > lum_clamp { power *= lum_clamp / lum; }
                let nor = rec.normal;
                return Some(KdPhoton {
                    pos:   [rec.p.x, rec.p.y, rec.p.z],
                    power: [power.x, power.y, power.z],
                    nor:   [nor.x,   nor.y,   nor.z],
                    axis:  0,
                });
            }
            return None;
        }
    }
    None
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::bvh::BvhTree;
    use crate::hittable::{HittableList, Material};
    use crate::material::{Dielectric, Lambertian};
    use crate::plane::InfinitePlane;
    use crate::sphere::Sphere;
    use crate::texture::Texture;
    use crate::transform::{Translate, UniformScale};
    use std::sync::Arc;

    fn diffuse() -> Arc<dyn Material> {
        Arc::new(Lambertian { texture: Texture::from(Color::new(0.7, 0.7, 0.7)) })
    }

    #[test]
    fn photon_bounds_skip_proxy_and_forward_through_wrappers() {
        let plane = InfinitePlane::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0), 0.0, 0.0, diffuse());
        assert!(plane.photon_bounds().is_none());
        assert!(plane.bounding_box().is_some()); // full bbox keeps the ±1e5 proxy

        // List: union of finite children only.
        let mut list = HittableList::new();
        list.add(plane);
        list.add(Sphere::new(Point3::new(3.0, 1.0, 0.0), 0.5, diffuse()));
        let bb = list.photon_bounds().unwrap();
        assert!((bb.min.x - 2.5).abs() < 1e-4 && (bb.max.x - 3.5).abs() < 1e-4);

        // Lone plane -> None (caller falls back to the DISK_R disk).
        let mut lone = HittableList::new();
        lone.add(InfinitePlane::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0), 0.0, 0.0, diffuse()));
        assert!(lone.photon_bounds().is_none());

        // Wrappers forward the finite bounds.
        let t = Translate::new(Arc::new(Sphere::new(Point3::new(0.0, 1.0, 0.0), 0.5, diffuse())), Vec3::new(10.0, 0.0, 0.0));
        let bb = t.photon_bounds().unwrap();
        assert!((bb.min.x - 9.5).abs() < 1e-4 && (bb.max.x - 10.5).abs() < 1e-4);

        let s = UniformScale::new(Arc::new(Sphere::new(Point3::new(0.0, 1.0, 0.0), 0.5, diffuse())), 2.0);
        let bb = s.photon_bounds().unwrap();
        assert!((bb.max.x - 1.0).abs() < 1e-4 && (bb.min.y - 1.0).abs() < 1e-4); // centre y=1, r=0.5, x2
    }

    #[test]
    fn gem_scene_photon_map_is_populated_and_finite() {
        // Scene-level regression test: the tracked bench scene (checker floor +
        // two gem cuts, caustics on) must produce a populated, finite caustic map —
        // with the old plane-inclusive disk bounds it was essentially empty.
        let mut scene = crate::scene_file::load("scenes/bench.toml").unwrap();
        scene.rebuild_caustics();
        let pm = scene.photon_map.as_ref().expect("caustics enabled in bench.toml");
        // bench.toml: dusk sun through a window, small stones -> thousands of stored photons
        // (the regression this guards against was an essentially EMPTY map, ~0 nodes).
        assert!(pm.nodes.len() > 1_000, "caustic map suspiciously empty: {}", pm.nodes.len());
        for nd in &pm.nodes {
            assert!(nd.power[0].is_finite() && nd.power[1].is_finite() && nd.power[2].is_finite());
            assert!(nd.pos[0].is_finite() && nd.pos[1].is_finite() && nd.pos[2].is_finite());
        }
        // Floor point right under the bench diamond (girdle centre [0, 1.30, 2.5]):
        // positive, finite irradiance.
        let irr = pm.irradiance(Point3::new(0.0, 0.0, 2.5), Vec3::new(0.0, 1.0, 0.0));
        assert!(irr.x.is_finite() && irr.x > 0.0, "expected caustic light under the gem, got {irr:?}");
    }

    #[test]
    fn photon_disk_hugs_real_geometry_in_plane_scenes() {
        // Infinite floor + small glass sphere above it: the emission disk must be
        // sized from the sphere only, so ~all photons hit the sphere and, after one
        // specular bounce, land on the floor as stored caustic photons.  With the
        // old plane-inclusive bounds the disk radius was ~50 and the hit fraction
        // ~(0.87/50)^2 ~ 3e-4, i.e. an essentially empty caustic map.
        let mut list = HittableList::new();
        list.add(InfinitePlane::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0), 0.0, 0.0, diffuse()));
        list.add(Sphere::new(Point3::new(0.0, 1.0, 0.0), 0.5, Arc::new(Dielectric { ir: 1.5 })));
        let world = BvhTree::from_list(list);

        let bb = world.photon_bounds().expect("finite geometry present");
        assert!(bb.max.x - bb.min.x < 2.0, "photon disk bounds must hug the sphere");
        assert!(world.bounding_box().unwrap().max.x > 1.0e4, "full bbox still has the plane proxy");

        let map = PhotonMap::build(&world, Vec3::new(0.0, 1.0, 0.0), Color::new(10.0, 10.0, 10.0), 20_000, 0.5);
        assert!(map.nodes.len() > 2_000,
                "expected most photons to reach the floor via the glass sphere, got {}", map.nodes.len());
    }
}
