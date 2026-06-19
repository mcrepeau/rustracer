use std::sync::Arc;
use crate::bvh::BvhTree;
use crate::camera::SceneCameraParams;
use crate::hittable::{Hittable, HittableList, Material};
use crate::material::{BlackbodyLight, Dielectric, DiffuseLight, Lambertian, PbrMaterial, PearlMaterial, SpectralDielectric, SpectralMetal, SpectralMetalVariant, SSSMaterial};
use crate::quad::{Quad, make_box};
use crate::renderer::Background;
use crate::scene::SceneData;
use crate::sphere::Sphere;
use crate::texture::Texture;
use crate::diamond::Diamond;
use crate::transform::{Rotate, Translate};
use crate::vec3::{Color, Point3, Vec3};
use crate::volume::{ConstantMedium, NoiseMedium};
use rand::Rng;
use rand::rngs::SmallRng;
use rand::SeedableRng;

/// Creates a matched (world quad, light-sampler quad) pair with an arbitrary material.
fn emissive_quad_mat(q: Point3, u: Vec3, v: Vec3, mat: Arc<dyn Material>) -> (Quad, Quad) {
    let world   = Quad::new(q, u, v, Arc::clone(&mat));
    let sampler = Quad::new(q, u, v, mat);
    (world, sampler)
}

/// Creates a matched (world sphere, light-sampler sphere) pair with an arbitrary material.
fn emissive_sphere_mat(center: Point3, radius: f32, mat: Arc<dyn Material>) -> (Sphere, Sphere) {
    let world   = Sphere::new(center, radius, Arc::clone(&mat));
    let sampler = Sphere::new(center, radius, mat);
    (world, sampler)
}

pub fn build_random_scene() -> SceneData {
    use std::f32::consts::PI;
    let mut rng = rand::thread_rng();

    let ground: Arc<dyn Hittable> = Arc::new(Sphere::new(
        Point3::new(0.0, -1000.0, 0.0), 1000.0,
        Arc::new(Lambertian { texture: Texture::Checker {
            scale: 10.0,
            even:  Color::new(0.2, 0.3, 0.1),
            odd:   Color::new(0.9, 0.9, 0.9),
        }}),
    ));

    let mut list = HittableList::new();
    list.objects.push(ground);

    // Centre: emissive PBR sphere — a glowing hot ember that casts warm light on the ring.
    // A sampler copy goes into `lights` so NEE explicitly targets it every bounce.
    let (ember_world, ember_sampler) = emissive_sphere_mat(
        Point3::new(0.0, 1.0, 0.0), 1.0,
        Arc::new(PbrMaterial {
            albedo:            Color::new(0.12, 0.04, 0.01),
            roughness:         0.8,
            metallic:          0.6,
            emission:          Color::new(1.0, 0.30, 0.02),
            emission_strength: 6.0,
            ..Default::default()
        }),
    );
    list.add(ember_world);
    let mut lights = HittableList::new();
    lights.add(ember_sampler);

    // ── Hero spheres in a circle ──────────────────────────────────────────────────
    // 7 positions evenly at radius HERO_R.  Position 3 hosts a NoiseMedium (requires
    // two scene objects: a glass boundary + the medium), handled before the material loop.
    const HERO_R:   f32 = 4.0;
    const N_HEROES: u32 = 7;

    // Hero 3: noise-driven cloud volume trapped inside a glass sphere.
    {
        let angle = 3.0_f32 * 2.0 * PI / N_HEROES as f32;
        let pos   = Point3::new(HERO_R * angle.cos(), 1.0, HERO_R * angle.sin());
        let cloud_boundary: Arc<dyn Hittable> = Arc::new(Sphere::new(
            pos, 1.0, Arc::new(Dielectric { ir: 1.5 }),
        ));
        list.objects.push(Arc::clone(&cloud_boundary));
        list.add(NoiseMedium::new(
            cloud_boundary,
            Color::new(0.65, 0.80, 1.0),  // pale blue-white — cloud / nebula
            12.0,  // density
            1.8,   // noise_scale: controls cloud texture size within the sphere
            0.0,   // threshold: 0 = fully filled, higher = patchier
            0.7,   // g: forward-scattering (water-droplet clouds)
        ));
    }

    // Heroes at slots 0,1,2,4,5,6 — material-only spheres sharing the hero loop.
    let hero_slots: Vec<(u32, Arc<dyn Material>)> = vec![
        (0, Arc::new(SpectralDielectric { cauchy_b: 1.507, cauchy_c: 0.00375, ..Default::default() })),
        (1, Arc::new(PbrMaterial { albedo: Color::new(0.4, 0.2, 0.1), roughness: 0.85, ..Default::default() })),
        (2, Arc::new(SpectralMetal::new(SpectralMetalVariant::Gold, 0.04))),
        (4, Arc::new(SSSMaterial {
            albedo:  Color::new(0.55, 0.97, 0.55),
            sigma_a: Color::new(3.0, 0.2, 3.0),
            ior: 1.5, density: 5.0, g: 0.3,
        })),
        (5, Arc::new(PearlMaterial {
            base_color: Color::new(0.98, 0.93, 0.88), ior: 1.56,
            film_thickness: 450.0, orient_strength: 0.30,
            film_scale: 5.0, luster_roughness: 0.05,
        })),
        (6, Arc::new(PbrMaterial {
            albedo: Color::new(0.02, 0.01, 0.03), roughness: 0.6,
            clearcoat: 0.9, clearcoat_roughness: 0.03,
            film_thickness: 480.0, film_ior: 1.45,
            ..Default::default()
        })),
    ];
    for (k, mat) in hero_slots {
        let angle = k as f32 * 2.0 * PI / N_HEROES as f32;
        list.add(Sphere::new(
            Point3::new(HERO_R * angle.cos(), 1.0, HERO_R * angle.sin()),
            1.0, mat,
        ));
    }

    // ── Random small spheres outside the hero circle ──────────────────────────────
    // Exclusion radius: HERO_R + hero sphere radius (1) + random sphere radius (0.2) + clearance.
    const EXCL_R: f32 = HERO_R + 1.5;
    let sss_colors: &[Color] = &[
        Color::new(0.55, 0.97, 0.55),  // jade green
        Color::new(0.38, 0.52, 0.97),  // cobalt blue
        Color::new(0.97, 0.68, 0.14),  // amber
        Color::new(0.97, 0.28, 0.18),  // ruby red
        Color::new(0.52, 0.18, 0.97),  // violet
        Color::new(0.97, 0.82, 0.28),  // gold
        Color::new(0.18, 0.97, 0.82),  // teal
        Color::new(0.97, 0.33, 0.68),  // rose
    ];
    let sss_sigma_a: &[Color] = &[
        Color::new(3.0, 0.2, 3.0),  // jade:   absorb R+B, pass G
        Color::new(3.0, 1.5, 0.2),  // cobalt: absorb R, moderate G, pass B
        Color::new(0.2, 1.5, 4.0),  // amber:  absorb B strongly, moderate G, pass R
        Color::new(0.2, 4.0, 4.0),  // ruby:   absorb G+B, pass R
        Color::new(1.0, 4.0, 0.2),  // violet: absorb G, moderate R, pass B
        Color::new(0.2, 0.5, 3.5),  // gold:   absorb B, pass R+G
        Color::new(4.0, 0.2, 0.2),  // teal:   absorb R, pass G+B
        Color::new(0.2, 3.5, 1.5),  // rose:   absorb G, moderate B, pass R
    ];
    for a in -15i32..15 {
        for b in -15i32..15 {
            let cx = a as f32 + 0.9 * rng.gen::<f32>();
            let cz = b as f32 + 0.9 * rng.gen::<f32>();
            if (cx * cx + cz * cz).sqrt() < EXCL_R { continue; }
            let choose: f32 = rng.gen();
            let mat: Arc<dyn Material> = if choose < 0.60 {
                let idx     = rng.gen_range(0..sss_colors.len());
                let density = rng.gen_range(3.0_f32..9.0);
                Arc::new(SSSMaterial { albedo: sss_colors[idx], sigma_a: sss_sigma_a[idx], ior: 1.5, density, g: 0.30 })
            } else if choose < 0.75 {
                let albedo    = Color::random(&mut rng) * Color::random(&mut rng);
                let roughness = rng.gen_range(0.5_f32..1.0);
                Arc::new(PbrMaterial { albedo, roughness, ..Default::default() })
            } else if choose < 0.92 {
                let albedo    = Color::random_range(0.5, 1.0, &mut rng);
                let roughness = rng.gen_range(0.0_f32..0.5);
                Arc::new(PbrMaterial { albedo, roughness, metallic: 1.0, ..Default::default() })
            } else {
                Arc::new(Dielectric { ir: 1.5 })
            };
            list.add(Sphere::new(Point3::new(cx, 0.2, cz), 0.2, mat));
        }
    }

    let mut scene = SceneData {
        world:               Arc::new(BvhTree::from_list(list)) as Arc<dyn Hittable>,
        lights,
        background:          Background::Physical { sun_dir: Vec3::new(-0.4, 0.9, -0.3).unit(), turbidity: 3.0 },
        name:                "Random Spheres",
        cam_init:            SceneCameraParams {
            pos: Point3::new(3.72, 7.1, 18.36), lookat: Point3::new(0.20, 1.09, 0.98),
            vfov: 20.0, aperture: 0.1, focus_dist: 10.0, move_speed: 0.3,
            aperture_blades: 6,
        },
        max_samples:         2000,
        enable_caustics:     true,
        caustic_quad:        None,
        caustic_gather_radius: 0.5,
        photon_map:          None,
    };
    scene.rebuild_caustics();
    scene
}

pub fn build_cornell_box() -> SceneData {
    let mut list = HittableList::new();
    let red:   Arc<dyn Material> = Arc::new(Lambertian { texture: Color::new(0.65, 0.05, 0.05).into() });
    let white: Arc<dyn Material> = Arc::new(Lambertian { texture: Color::new(0.73, 0.73, 0.73).into() });
    let green: Arc<dyn Material> = Arc::new(Lambertian { texture: Color::new(0.12, 0.45, 0.15).into() });

    let (world_light, sampler_light) = emissive_quad_mat(
        Point3::new(343.0, 554.0, 332.0),
        Vec3::new(-130.0, 0.0, 0.0),
        Vec3::new(0.0, 0.0, -105.0),
        // 6500 K (D65 daylight) — close to neutral white; spectral path tracer
        // uses the Planck weight at each hero wavelength so the glass sphere
        // produces a physically correct rainbow under this illuminant.
        Arc::new(BlackbodyLight::new(6500.0, 15.0)),
    );
    let mut lights = HittableList::new();
    lights.add(sampler_light);

    list.add(Quad::new(Point3::new(555.0, 0.0,   0.0),   Vec3::new(0.0, 555.0,  0.0), Vec3::new(0.0, 0.0,  555.0), green));
    list.add(Quad::new(Point3::new(0.0,   0.0,   0.0),   Vec3::new(0.0, 555.0,  0.0), Vec3::new(0.0, 0.0,  555.0), red));
    list.add(world_light);
    list.add(Quad::new(Point3::new(0.0,   0.0,   0.0),   Vec3::new(555.0, 0.0,  0.0), Vec3::new(0.0, 0.0,  555.0), Arc::clone(&white)));
    list.add(Quad::new(Point3::new(555.0, 555.0, 555.0), Vec3::new(-555.0, 0.0, 0.0), Vec3::new(0.0, 0.0, -555.0), Arc::clone(&white)));
    list.add(Quad::new(Point3::new(0.0,   0.0,   555.0), Vec3::new(555.0, 0.0,  0.0), Vec3::new(0.0, 555.0, 0.0),  Arc::clone(&white)));

    let tall = Arc::new(make_box(Point3::new(0.0,0.0,0.0), Point3::new(165.0,330.0,165.0), Arc::clone(&white))) as Arc<dyn Hittable>;
    let tall = Arc::new(Rotate::around_y(tall,  15.0)) as Arc<dyn Hittable>;
    let tall = Arc::new(Translate::new(tall, Vec3::new(265.0, 0.0, 295.0))) as Arc<dyn Hittable>;
    list.objects.push(tall);

    let short = Arc::new(make_box(Point3::new(0.0,0.0,0.0), Point3::new(165.0,165.0,165.0), white)) as Arc<dyn Hittable>;
    let short = Arc::new(Rotate::around_y(short, -18.0)) as Arc<dyn Hittable>;
    let short = Arc::new(Translate::new(short, Vec3::new(130.0, 0.0, 65.0))) as Arc<dyn Hittable>;
    list.objects.push(short);

    // Dense flint glass (B=1.612, C=0.00950 μm²): IOR ranges 1.620–1.678
    // across the visible spectrum — high dispersion like an optical prism,
    // producing visible rainbow fringes in the caustic and refraction.
    list.add(Sphere::new(Point3::new(190.0, 245.0, 190.0), 80.0,
        Arc::new(SpectralDielectric { cauchy_b: 1.612, cauchy_c: 0.00950, ..Default::default() })));

    let mut scene = SceneData {
        world:      Arc::new(BvhTree::from_list(list)) as Arc<dyn Hittable>,
        lights,
        background: Background::Solid(Color::default()),
        name:       "Cornell Box",
        cam_init:   SceneCameraParams {
            pos: Point3::new(278.0, 278.0, -800.0), lookat: Point3::new(278.0, 278.0, 0.0),
            vfov: 40.0, aperture: 0.0, focus_dist: 10.0, move_speed: 8.0,
            aperture_blades: 0,
        },
        max_samples:    2000,
        enable_caustics:       true,
        caustic_quad:          Some((
            Point3::new(343.0, 554.0, 332.0),
            Vec3::new(-130.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, -105.0),
            Color::new(15.0, 15.0, 15.0),
        )),
        caustic_gather_radius: 20.0,
        photon_map:            None,
    };
    scene.rebuild_caustics();
    scene
}


pub fn build_nextweek_scene() -> SceneData {
    let mut rng = SmallRng::seed_from_u64(42);
    let mut list = HittableList::new();

    // Ground terrain: 20×20 grid of random-height boxes.
    let ground_mat: Arc<dyn Material> =
        Arc::new(Lambertian { texture: Color::new(0.48, 0.83, 0.53).into() });
    let mut ground_boxes = HittableList::new();
    for i in 0..20 {
        for j in 0..20 {
            let w  = 100.0f32;
            let x0 = -1000.0 + i as f32 * w;
            let z0 = -1000.0 + j as f32 * w;
            let y1: f32 = rng.gen_range(1.0..101.0);
            ground_boxes.add(make_box(
                Point3::new(x0, 0.0, z0),
                Point3::new(x0 + w, y1, z0 + w),
                Arc::clone(&ground_mat),
            ));
        }
    }
    list.add(BvhTree::from_list(ground_boxes));

    // Warm-tungsten area light (3 200 K blackbody).  The physically-based SPD
    // gives dispersive materials (diamond, glass) a warm-biased rainbow fire
    // compared to the neutral-white DiffuseLight.
    let (light_world, light_sampler) = emissive_quad_mat(
        Point3::new(123.0, 554.0, 147.0),
        Vec3::new(300.0,   0.0,   0.0),
        Vec3::new(  0.0,   0.0, 265.0),
        Arc::new(BlackbodyLight::new(3200.0, 7.0)),
    );
    let mut lights = HittableList::new();
    lights.add(light_sampler);
    list.add(light_world);

    // ── Row 2: PBR material showcase (z = 225) ───────────────────────────────────
    // Velvet: sheen=1 brightens grazing incidence; sheen_tint=0.3 keeps the
    // rim mostly white against the deep violet base.
    list.add(Sphere::new(Point3::new( 80.0, 150.0, 350.0), 50.0,
        Arc::new(PbrMaterial {
            albedo:     Color::new(0.08, 0.04, 0.18),
            roughness:  0.9,
            sheen:      1.0,
            sheen_tint: 0.3,
            ..Default::default()
        })));

    // Brushed aluminium: anisotropic GGX streaks the highlight tangentially.
    list.add(Sphere::new(Point3::new(300.0, 150.0, 220.0), 50.0,
        Arc::new(PbrMaterial {
            albedo:           Color::new(0.78, 0.78, 0.80),
            roughness:        0.3,
            metallic:         1.0,
            anisotropy:       0.8,
            anisotropy_angle: 0.0,
            ..Default::default()
        })));

    // Motion-blurred orange Lambertian sphere (demonstrates motion blur).
    let c0 = Point3::new(400.0, 400.0, 200.0);
    list.add(Sphere::new_moving(
        c0, c0 + Vec3::new(30.0, 0.0, 0.0), 50.0,
        Arc::new(Lambertian { texture: Color::new(0.7, 0.3, 0.1).into() }),
    ));

    // Thin global mist (homogeneous constant-density medium).
    let mist_boundary: Arc<dyn Hittable> = Arc::new(Sphere::new(
        Point3::new(0.0, 0.0, 0.0), 5000.0,
        Arc::new(Dielectric { ir: 1.5 }),
    ));
    list.add(ConstantMedium::new(mist_boundary, 0.0003, Color::new(1.0, 1.0, 1.0), 0.0));

    // Spectral diamond: Tolkowsky-cut polyhedron with Cauchy dispersive IOR.
    // The 3200 K light produces warm-tinted rainbow fire inside the gem.
    let diamond_r = 100.0_f32;
    list.objects.push(Arc::new(Diamond::new(
        Point3::new(220.0, 325.0, 300.0),
        diamond_r,
        Arc::new(SpectralDielectric { cauchy_b: 2.395, cauchy_c: 0.00585, ..Default::default() }),
    )) as Arc<dyn Hittable>);

    // Earth texture (textured Lambertian).
    let earth_tex = Texture::load("assets/earthmap.jpg")
        .unwrap_or_else(|_| Texture::Checker {
            scale: 2.0,
            even:  Color::new(0.1, 0.3, 0.7),
            odd:   Color::new(0.2, 0.7, 0.2),
        });
    list.add(Sphere::new(
        Point3::new(400.0, 220.0, 400.0), 100.0,
        Arc::new(Lambertian { texture: earth_tex }),
    ));

    // Cloud of 1 000 white diffuse micro-spheres (BVH stress test).
    let white: Arc<dyn Material> =
        Arc::new(Lambertian { texture: Color::new(0.73, 0.73, 0.73).into() });
    let mut cluster = HittableList::new();
    for _ in 0..1000 {
        cluster.add(Sphere::new(
            Point3::new(
                rng.gen_range(0.0..165.0),
                rng.gen_range(0.0..165.0),
                rng.gen_range(0.0..165.0),
            ),
            10.0,
            Arc::clone(&white),
        ));
    }
    let cluster: Arc<dyn Hittable> = Arc::new(BvhTree::from_list(cluster));
    let cluster: Arc<dyn Hittable> = Arc::new(Rotate::around_y(cluster, 15.0));
    let cluster: Arc<dyn Hittable> = Arc::new(Translate::new(cluster, Vec3::new(-100.0, 300.0, 395.0)));
    list.objects.push(cluster);

    // Caustic quad geometry mirrors the area light for photon-map emission.
    let caustic_quad = (
        Point3::new(123.0, 554.0, 147.0),
        Vec3::new(300.0,   0.0,   0.0),
        Vec3::new(  0.0,   0.0, 265.0),
        Color::new(7.0, 7.0, 7.0),
    );

    let mut scene = SceneData {
        world:      Arc::new(BvhTree::from_list(list)) as Arc<dyn Hittable>,
        lights,
        background: Background::Solid(Color::default()),
        name:       "Next Week",
        cam_init:   SceneCameraParams {
            pos:             Point3::new(452.0, 366.0, -425.0),
            lookat:          Point3::new(266.0, 315.0,  225.0),
            vfov:            45.0,
            aperture:        0.0,
            focus_dist:      10.0,
            move_speed:      8.0,
            aperture_blades: 0,
        },
        max_samples:    2000,
        enable_caustics:       true,
        caustic_quad:          Some(caustic_quad),
        caustic_gather_radius: 25.0,
        photon_map:            None,
    };
    scene.rebuild_caustics();
    scene
}

