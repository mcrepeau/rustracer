use std::sync::Arc;
use crate::bvh::BvhTree;
use crate::camera::SceneCameraParams;
use crate::hittable::{Hittable, HittableList, Material};
use crate::material::{BlackbodyLight, Lambertian, PbrMaterial, SpectralDielectric, SSSMaterial};
use crate::quad::Quad;
use crate::renderer::Background;
use crate::scene::SceneData;
use crate::sphere::Sphere;
use crate::texture::Texture;
use crate::triangle::Triangle;
use crate::vec3::{Color, Point3, Vec3};

/// Creates a matched (world quad, light-sampler quad) pair with an arbitrary material.
fn emissive_quad_mat(q: Point3, u: Vec3, v: Vec3, mat: Arc<dyn Material>) -> (Quad, Quad) {
    let world   = Quad::new(q, u, v, Arc::clone(&mat));
    let sampler = Quad::new(q, u, v, mat);
    (world, sampler)
}

/// Procedural UV sphere as a triangle mesh wrapped in a BVH.
/// `lat` × `lon` quads → 2 × lat × lon triangles with smooth per-vertex normals and tangents.
fn make_uv_sphere_mesh(
    center: Point3, radius: f32, lat: usize, lon: usize,
    mat: Arc<dyn Material>,
) -> Arc<dyn Hittable> {
    use std::f32::consts::PI;
    let stride = lon + 1;
    let n_verts = (lat + 1) * stride;
    let mut verts    = Vec::with_capacity(n_verts);
    let mut normals  = Vec::with_capacity(n_verts);
    let mut tangents = Vec::with_capacity(n_verts);

    for i in 0..=lat {
        let phi     = PI * i as f32 / lat as f32;
        let sin_phi = phi.sin();
        let cos_phi = phi.cos();
        for j in 0..=lon {
            let theta     = 2.0 * PI * j as f32 / lon as f32;
            let sin_theta = theta.sin();
            let cos_theta = theta.cos();
            let n = Vec3::new(sin_phi * cos_theta, cos_phi, sin_phi * sin_theta);
            verts.push(center + n * radius);
            normals.push(n);
            // ∂pos/∂θ normalised — well-defined everywhere including poles
            tangents.push(Vec3::new(-sin_theta, 0.0, cos_theta));
        }
    }

    let mut list = HittableList::new();
    for i in 0..lat {
        for j in 0..lon {
            let i00 = i * stride + j;
            let i10 = (i + 1) * stride + j;
            let i11 = (i + 1) * stride + (j + 1);
            let i01 = i * stride + (j + 1);
            let uv00 = (j as f32 / lon as f32,       i as f32 / lat as f32);
            let uv10 = (j as f32 / lon as f32,       (i + 1) as f32 / lat as f32);
            let uv11 = ((j + 1) as f32 / lon as f32, (i + 1) as f32 / lat as f32);
            let uv01 = ((j + 1) as f32 / lon as f32, i as f32 / lat as f32);

            // Upper triangle
            if (verts[i10] - verts[i00]).cross(verts[i11] - verts[i00]).length_squared() > 1e-16 {
                list.add(Triangle::new(
                    verts[i00], verts[i10], verts[i11],
                    normals[i00], normals[i10], normals[i11],
                    tangents[i00], tangents[i10], tangents[i11],
                    uv00, uv10, uv11, Arc::clone(&mat),
                ));
            }
            // Lower triangle
            if (verts[i11] - verts[i00]).cross(verts[i01] - verts[i00]).length_squared() > 1e-16 {
                list.add(Triangle::new(
                    verts[i00], verts[i11], verts[i01],
                    normals[i00], normals[i11], normals[i01],
                    tangents[i00], tangents[i11], tangents[i01],
                    uv00, uv11, uv01, Arc::clone(&mat),
                ));
            }
        }
    }
    Arc::new(BvhTree::from_list(list))
}

/// Benchmark scene designed to stress all major rendering code paths simultaneously:
///
/// - BVH traversal: 128×128 UV sphere mesh (32 768 triangles)
/// - Spectral material: SpectralDielectric glass sphere with strong dispersion
/// - Photon map / caustics: glass sphere under Preetham sun
/// - Area NEE: overhead BlackbodyLight quad
/// - Sun NEE: Physical background with Preetham sky
/// - Parallel scaling: fixed 256 spp, no adaptive exit
pub fn build_benchmark_scene() -> SceneData {
    let mut list   = HittableList::new();
    let mut lights = HittableList::new();

    // Room materials — saturated so they read clearly under warm dusk sky light
    let floor_mat: Arc<dyn Material> = Arc::new(Lambertian { texture: Texture::Checker {
        scale: 0.8,
        even:  Color::new(0.12, 0.12, 0.12),
        odd:   Color::new(0.88, 0.88, 0.88),
    }});
    let white: Arc<dyn Material>  = Arc::new(Lambertian { texture: Color::new(0.80, 0.80, 0.80).into() });
    let wall_l: Arc<dyn Material> = Arc::new(Lambertian { texture: Color::new(0.65, 0.10, 0.05).into() });
    let wall_r: Arc<dyn Material> = Arc::new(Lambertian { texture: Color::new(0.05, 0.15, 0.65).into() });

    // Room bounds: x ∈ [−5, 5], y ∈ [0, 5], z ∈ [−2, 9]

    // Floor
    list.add(Quad::new(Point3::new(-5.0, 0.0, -2.0), Vec3::new(10.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 11.0), floor_mat));
    // Left wall (warm red)
    list.add(Quad::new(Point3::new(-5.0, 0.0, -2.0), Vec3::new(0.0, 0.0, 11.0), Vec3::new(0.0, 5.0, 0.0), wall_l));
    // Right wall (cool blue)
    list.add(Quad::new(Point3::new( 5.0, 0.0, -2.0), Vec3::new(0.0, 0.0, 11.0), Vec3::new(0.0, 5.0, 0.0), wall_r));

    // Back wall: four panels around a window opening at x ∈ [−2, 2], y ∈ [1.2, 4.2].
    // Dusk sun shines through this opening and hits the glass sphere head-on.
    list.add(Quad::new(Point3::new(-5.0, 0.0,  9.0), Vec3::new(10.0, 0.0, 0.0), Vec3::new(0.0, 1.2, 0.0), Arc::clone(&white))); // sill
    list.add(Quad::new(Point3::new(-5.0, 4.2,  9.0), Vec3::new(10.0, 0.0, 0.0), Vec3::new(0.0, 0.8, 0.0), Arc::clone(&white))); // lintel
    list.add(Quad::new(Point3::new(-5.0, 1.2,  9.0), Vec3::new( 3.0, 0.0, 0.0), Vec3::new(0.0, 3.0, 0.0), Arc::clone(&white))); // left jamb
    list.add(Quad::new(Point3::new( 2.0, 1.2,  9.0), Vec3::new( 3.0, 0.0, 0.0), Vec3::new(0.0, 3.0, 0.0), Arc::clone(&white))); // right jamb

    // Ceiling: white panels surrounding the BlackbodyLight panel (x ∈ [−1.5, 1.5], z ∈ [−0.5, 2.5])
    list.add(Quad::new(Point3::new(-5.0, 5.0, -2.0), Vec3::new(10.0, 0.0, 0.0), Vec3::new(0.0, 0.0,  1.5), Arc::clone(&white))); // front
    list.add(Quad::new(Point3::new(-5.0, 5.0,  2.5), Vec3::new(10.0, 0.0, 0.0), Vec3::new(0.0, 0.0,  6.5), Arc::clone(&white))); // back
    list.add(Quad::new(Point3::new(-5.0, 5.0, -0.5), Vec3::new( 3.5, 0.0, 0.0), Vec3::new(0.0, 0.0,  3.0), Arc::clone(&white))); // left
    list.add(Quad::new(Point3::new( 1.5, 5.0, -0.5), Vec3::new( 3.5, 0.0, 0.0), Vec3::new(0.0, 0.0,  3.0), Arc::clone(&white))); // right

    // BlackbodyLight ceiling panel — area NEE + spectral emission weighting
    let (world_light, sampler_light) = emissive_quad_mat(
        Point3::new(-1.5, 5.0, -0.5),
        Vec3::new(3.0, 0.0, 0.0),
        Vec3::new(0.0, 0.0, 3.0),
        Arc::new(BlackbodyLight::new(6500.0, 8.0)),
    );
    lights.add(sampler_light);
    list.add(world_light);

    // Flint glass sphere — centered on the window axis so dusk photons travel through
    // and hit it directly.  Bottom touches the floor; caustic falls toward the camera.
    list.add(Sphere::new(
        Point3::new(0.0, 1.5, 2.5), 1.5,
        Arc::new(SpectralDielectric { cauchy_b: 1.612, cauchy_c: 0.00950, ..Default::default() }),
    ));

    // White marble sphere — SSS with warm absorption tint, exercises volumetric scattering
    // Symmetric to the gold sphere on the left side of the room.
    // Density scaled to radius 1.2: density = 2 scatters/diameter / (2 × 1.2) ≈ 0.85.
    list.add(Sphere::new(
        Point3::new(-3.0, 1.2, 1.0), 1.2,
        Arc::new(SSSMaterial {
            albedo:  Color::new(0.90, 0.87, 0.82),
            sigma_a: Color::new(0.10, 0.08, 0.05),
            ior:     1.5,
            density: 0.85,
            g:       0.30,
        }),
    ));

    // Brushed gold PBR sphere — 128×128 UV mesh (32 768 triangles), stresses BVH + GGX
    let gold: Arc<dyn Material> = Arc::new(PbrMaterial {
        albedo:    Color::new(1.0, 0.78, 0.34),
        roughness: 0.25,
        metallic:  1.0,
        ..Default::default()
    });
    list.objects.push(make_uv_sphere_mesh(
        Point3::new(3.0, 1.2, 1.0), 1.2, 128, 128, gold,
    ));

    SceneData {
        world:      Arc::new(BvhTree::from_list(list)) as Arc<dyn Hittable>,
        lights,
        // Dusk sun at ~10° elevation, directly behind the scene, shining through the window.
        // High turbidity gives warm orange-red sky and soft shadows.
        background: Background::Physical {
            sun_dir:   Vec3::new(0.0, 0.18, 1.0).unit(),
            turbidity: 4.5,
        },
        name:       "Benchmark",
        cam_init:   SceneCameraParams {
            pos:             Point3::new(0.0, 3.4, -9.0),
            lookat:          Point3::new(0.0, 2.247, 1.1993),
            vfov:            36.0,
            aperture:        0.0,
            focus_dist:      10.0,
            move_speed:      0.3,
            aperture_blades: 0,
        },
        max_samples:           256,
        enable_caustics:       true,
        caustic_quad:          None,
        caustic_gather_radius: 0.5,
        caustic_num_photons:   500_000,
        photon_map:            None,
        animation:             None,
    }
}
