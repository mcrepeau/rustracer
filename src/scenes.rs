use std::sync::Arc;
use crate::bvh::BvhTree;
use crate::camera::SceneCameraParams;
use crate::hittable::{Hittable, HittableList, Material};
use crate::material::{Dielectric, DiffuseLight, Lambertian, PbrMaterial, PearlMaterial, SpectralDielectric, SSSMaterial};
use crate::perlin::Perlin;
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

/// Creates a matched (world quad, light-sampler quad) pair with identical geometry.
fn emissive_quad(q: Point3, u: Vec3, v: Vec3, emit: Color) -> (Quad, Quad) {
    let mat: Arc<dyn Material> = Arc::new(DiffuseLight { emit: emit.into() });
    let world   = Quad::new(q, u, v, Arc::clone(&mat));
    let sampler = Quad::new(q, u, v, mat);
    (world, sampler)
}

pub fn build_random_scene() -> SceneData {
    let mut rng = rand::thread_rng();

    let ground: Arc<dyn Hittable> = Arc::new(Sphere::new(
        Point3::new(0.0, -1000.0, 0.0), 1000.0,
        Arc::new(Lambertian { texture: Texture::Checker {
            scale: 10.0,
            even:  Color::new(0.2, 0.3, 0.1),
            odd:   Color::new(0.9, 0.9, 0.9),
        }}),
    ));
    let diamond_r   = 0.7_f32;
    let diamond_pav = diamond_r * 40.75_f32.to_radians().tan();
    let diamond_obj = Diamond::new(
        Point3::new(2.0, diamond_pav, -1.5),
        diamond_r,
        Arc::new(SpectralDielectric { cauchy_b: 2.395, cauchy_c: 0.00585 }),
    );
    let diamond: Arc<dyn Hittable> = Arc::new(diamond_obj);
    // Noise-driven cloud — sits right among the main spheres, unmissable.
    // turb()*0.5 ∈ [0, 0.34], median 0.063; threshold=0.05 keeps the top ~43%.
    // density=8 → avg σ≈0.28, mean free path ~3.6 units, ~1.9 expected scatters across diameter.
    let cloud_boundary: Arc<dyn Hittable> = Arc::new(Sphere::new(
        Point3::new(0.0, 1.5, 0.0), 3.5,
        Arc::new(Lambertian { texture: Texture::from(Color::new(0.5, 0.5, 0.5)) }),
    ));
    let cloud: Arc<dyn Hittable> =
        Arc::new(NoiseMedium::new(cloud_boundary, Color::new(1.0, 0.97, 0.90), 8.0, 0.6, 0.05, 0.85));

    let mut list = HittableList::new();
    list.objects.push(ground);
    list.objects.push(diamond);
    list.objects.push(cloud);

    // Hero spheres
    list.add(Sphere::new(Point3::new( 0.0, 1.0,  0.0), 1.0, Arc::new(Dielectric { ir: 1.5 })));
    list.add(Sphere::new(Point3::new(-4.0, 1.0,  0.0), 1.0,
        Arc::new(PbrMaterial { albedo: Color::new(0.4, 0.2, 0.1), roughness: 0.85, ..Default::default() })));
    list.add(Sphere::new(Point3::new( 4.0, 1.0,  0.0), 1.0,
        Arc::new(PbrMaterial { albedo: Color::new(0.85, 0.65, 0.25), roughness: 0.25, metallic: 1.0, anisotropy: 0.85, ..Default::default() })));
    list.add(Sphere::new(Point3::new(-2.0, 1.0, -2.0), 1.0,
        Arc::new(PearlMaterial {
            base_color:       Color::new(0.98, 0.93, 0.88),
            ior:              1.56,
            film_thickness:   450.0,
            orient_strength:  0.30,
            film_scale:       5.0,
            luster_roughness: 0.05,
        })));
    // Iridescent clearcoat: near-black base under a thin-film dielectric coat.
    // Orbit the camera to watch the rainbow cycle across the surface.
    list.add(Sphere::new(Point3::new(2.0, 1.0, 2.0), 1.0,
        Arc::new(PbrMaterial {
            albedo: Color::new(0.02, 0.01, 0.03),
            roughness: 0.6,
            clearcoat: 0.9,
            film_thickness: 480.0,
            film_ior: 1.45,
            ..Default::default()
        })));

    // SSS marbles near the diamond — volumetric multiple scattering, soft coloured glow.
    // Per-scatter albedos: bright channel ≈ 1, others < 1 so absorption builds
    // colour over ~2 scatters per diameter traversal.
    // Max channel capped at 0.97 so Russian roulette can terminate scatter paths.
    // A channel of 1.0 gives survive probability = 1.0 forever → unbounded depth.
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
    // Beer-Lambert absorption per channel (per unit length).  Tuned for r=0.15 marbles
    // (max path ≈ 0.30): sigma_a ≈ 3 → T ≈ 0.41, sigma_a ≈ 0.2 → T ≈ 0.94.
    // High absorption in the complementary channels deepens the marble's colour.
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
    // Dedicated SSS marbles clustered around the diamond on the ground.
    let dedicated_marbles: &[(usize, Point3)] = &[
        (0, Point3::new(1.5,  0.15, -1.0)),
        (1, Point3::new(2.5,  0.15, -2.0)),
        (2, Point3::new(1.0,  0.15, -2.5)),
        (3, Point3::new(3.0,  0.15, -1.5)),
        (4, Point3::new(2.0,  0.15, -0.8)),
    ];
    for &(idx, center) in dedicated_marbles {
        list.add(Sphere::new(center, 0.15, Arc::new(SSSMaterial {
            albedo:  sss_colors[idx],
            sigma_a: sss_sigma_a[idx],
            ior:     1.5,
            density: 7.0,
            g:       0.30,
        })));
    }

    for a in -15i32..15 {
        for b in -15i32..15 {
            let cx = a as f32 + 0.9 * rng.gen::<f32>();
            let cz = b as f32 + 0.9 * rng.gen::<f32>();
            let ground_pos = Point3::new(cx, 0.2, cz);
            if (ground_pos - Point3::new( 4.0, 0.2, 0.0)).length() < 1.2 { continue; }
            if (ground_pos - Point3::new( 0.0, 0.2, 0.0)).length() < 1.2 { continue; }
            if (ground_pos - Point3::new(-4.0, 0.2, 0.0)).length() < 1.2 { continue; }
            let choose: f32 = rng.gen();
            let mat: Arc<dyn Material> = if choose < 0.60 {
                let idx     = rng.gen_range(0..sss_colors.len());
                let density = rng.gen_range(3.0_f32..9.0);
                Arc::new(SSSMaterial {
                    albedo:  sss_colors[idx],
                    sigma_a: sss_sigma_a[idx],
                    ior:     1.5,
                    density,
                    g:       0.30,
                })
            } else if choose < 0.75 {
                let albedo = Color::random(&mut rng) * Color::random(&mut rng);
                let roughness: f32 = rng.gen_range(0.5..1.0);
                Arc::new(PbrMaterial { albedo, roughness, ..Default::default() })
            } else if choose < 0.92 {
                let albedo    = Color::random_range(0.5, 1.0, &mut rng);
                let roughness: f32 = rng.gen_range(0.0..0.5);
                Arc::new(PbrMaterial { albedo, roughness, metallic: 1.0, ..Default::default() })
            } else {
                Arc::new(Dielectric { ir: 1.5 })
            };
            let center = Point3::new(cx, 0.2, cz);
            list.add(Sphere::new(center, 0.2, mat));
        }
    }

    let mut scene = SceneData {
        world:          Arc::new(BvhTree::from_list(list)) as Arc<dyn Hittable>,
        lights:         HittableList::new(),
        background:     Background::Physical { sun_dir: Vec3::new(-0.4, 0.9, -0.3).unit(), turbidity: 3.0 },
        name:           "Random Spheres",
        cam_init:       SceneCameraParams {
            pos: Point3::new(13.0, 2.0, 3.0), lookat: Point3::new(0.0, 0.0, 0.0),
            vfov: 20.0, aperture: 0.1, focus_dist: 10.0, move_speed: 0.3,
            aperture_blades: 6,
        },
        max_samples:    2000,
        enable_caustics:       true,
        caustic_quad:          None,
        caustic_gather_radius: 0.15,
        photon_map:            None,
    };
    scene.rebuild_caustics();
    scene
}

pub fn build_cornell_box() -> SceneData {
    let mut list = HittableList::new();
    let red:   Arc<dyn Material> = Arc::new(Lambertian { texture: Color::new(0.65, 0.05, 0.05).into() });
    let white: Arc<dyn Material> = Arc::new(Lambertian { texture: Color::new(0.73, 0.73, 0.73).into() });
    let green: Arc<dyn Material> = Arc::new(Lambertian { texture: Color::new(0.12, 0.45, 0.15).into() });

    let (world_light, sampler_light) = emissive_quad(
        Point3::new(343.0, 554.0, 332.0),
        Vec3::new(-130.0, 0.0, 0.0),
        Vec3::new(0.0, 0.0, -105.0),
        Color::new(15.0, 15.0, 15.0),
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

    list.add(Sphere::new(Point3::new(190.0, 90.0, 190.0), 80.0, Arc::new(Dielectric { ir: 1.5 })));

    SceneData {
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
        enable_caustics:       false,
        caustic_quad:          None,
        caustic_gather_radius: 0.15,
        photon_map:            None,
    }
}


pub fn build_nextweek_scene() -> SceneData {
    let mut rng = SmallRng::seed_from_u64(42);
    let mut list = HittableList::new();

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

    let (light_world, light_sampler) = emissive_quad(
        Point3::new(123.0, 554.0, 147.0),
        Vec3::new(300.0,   0.0,   0.0),
        Vec3::new(  0.0,   0.0, 265.0),
        Color::new(7.0, 7.0, 7.0),
    );
    let mut lights = HittableList::new();
    lights.add(light_sampler);
    list.add(light_world);

    let c0 = Point3::new(400.0, 400.0, 200.0);
    list.add(Sphere::new_moving(
        c0, c0 + Vec3::new(30.0, 0.0, 0.0), 50.0,
        Arc::new(Lambertian { texture: Color::new(0.7, 0.3, 0.1).into() }),
    ));

    list.add(Sphere::new(
        Point3::new(260.0, 150.0, 45.0), 50.0,
        Arc::new(Dielectric { ir: 1.5 }),
    ));
    list.add(Sphere::new(
        Point3::new(0.0, 150.0, 145.0), 50.0,
        Arc::new(PbrMaterial { albedo: Color::new(0.8, 0.8, 0.9), metallic: 1.0, roughness: 0.9, ..Default::default() }),
    ));

    let fog_boundary: Arc<dyn Hittable> = Arc::new(Sphere::new(
        Point3::new(360.0, 150.0, 145.0), 70.0,
        Arc::new(Dielectric { ir: 1.5 }),
    ));
    list.objects.push(Arc::clone(&fog_boundary));
    // Noise-driven volume: patchy blue cloud inside the glass sphere.
    // turb()*0.5 is empirically distributed in [0, 0.34] with median 0.063.
    // threshold=0.07 ≈ median → roughly half the sphere is empty (clear glass), half is blue cloud.
    // density=3.0 puts the mean effective density around 0.12, giving ~12 expected scatters/ray.
    list.add(NoiseMedium::new(fog_boundary, Color::new(0.2, 0.4, 0.9), 3.0, 0.05, 0.07, 0.7));

    let mist_boundary: Arc<dyn Hittable> = Arc::new(Sphere::new(
        Point3::new(0.0, 0.0, 0.0), 5000.0,
        Arc::new(Dielectric { ir: 1.5 }),
    ));
    list.add(ConstantMedium::new(mist_boundary, 0.0001, Color::new(1.0, 1.0, 1.0), 0.0));

    // Diamond â€” large gem floating directly below the area light.
    // Radius 100 fills the space between the ceiling and the scene objects.
    // Positioned at x=200, z=200 to clear the perlin sphere at (220,280,300).
    let diamond_r = 100.0_f32;
    list.objects.push(Arc::new(Diamond::new(
        Point3::new(200.0, 430.0, 200.0),
        diamond_r,
        Arc::new(SpectralDielectric { cauchy_b: 2.395, cauchy_c: 0.00585 }),
    )) as Arc<dyn Hittable>);

    // Pearl â€” Akoya-style cream pearl.  film_thickness=450 nm gives a rose-pink
    // orient at normal incidence cycling through blue and green at oblique angles.
    list.add(Sphere::new(
        Point3::new(400.0, 150.0, 270.0), 50.0,
        Arc::new(PearlMaterial {
            base_color:       Color::new(0.98, 0.93, 0.88),
            ior:              1.56,
            film_thickness:   450.0,
            orient_strength:  0.30,
            film_scale:       0.10,
            luster_roughness: 0.05,
        }),
    ));

    let earth_tex = Texture::load("assets/earthmap.jpg")
        .unwrap_or_else(|_| Texture::Checker {
            scale: 2.0,
            even:  Color::new(0.1, 0.3, 0.7),
            odd:   Color::new(0.2, 0.7, 0.2),
        });
    list.add(Sphere::new(
        Point3::new(400.0, 200.0, 400.0), 100.0,
        Arc::new(Lambertian { texture: earth_tex }),
    ));

    let perlin = Arc::new(Perlin::new(&mut rng));
    list.add(Sphere::new(
        Point3::new(220.0, 280.0, 300.0), 80.0,
        Arc::new(Lambertian { texture: Texture::Noise { perlin, scale: 0.2 } }),
    ));

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
    let cluster: Arc<dyn Hittable> = Arc::new(Translate::new(cluster, Vec3::new(-100.0, 270.0, 395.0)));
    list.objects.push(cluster);

    // Light quad geometry â€” reused as the photon-map emitter for caustics.
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
            pos:             Point3::new(478.0, 278.0, -600.0),
            lookat:          Point3::new(278.0, 278.0,    0.0),
            vfov:            40.0,
            aperture:        0.0,
            focus_dist:      10.0,
            move_speed:      8.0,
            aperture_blades: 0,
        },
        max_samples:    2000,
        enable_caustics:       true,
        caustic_quad:          Some(caustic_quad),
        caustic_gather_radius: 10.0,
        photon_map:            None,
    };
    // Static scene: rebuild() is never called from tick(), so build the
    // photon map once here at construction time.
    scene.rebuild_caustics();
    scene
}

