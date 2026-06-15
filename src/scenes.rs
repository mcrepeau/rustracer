use std::sync::Arc;
use crate::aabb::Aabb;
use crate::bvh::BvhTree;
use crate::camera::SceneCameraParams;
use crate::hittable::{Hittable, HittableList, Material};
use crate::material::{Dielectric, DiffuseLight, Lambertian, MarbleMaterial, Metal, PbrMaterial, PearlMaterial, SpectralDielectric, SSSMaterial};
use crate::perlin::Perlin;
use crate::quad::{Quad, make_box};
use crate::renderer::Background;
use crate::scene::{DynamicSphere, SceneData};
use crate::sphere::Sphere;
use crate::texture::Texture;
use crate::diamond::Diamond;
use crate::transform::{Rotate, Translate};
use crate::vec3::{Color, Point3, Vec3};
use crate::volume::ConstantMedium;
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
        Arc::new(SpectralDielectric { ir_red: 2.407, ir_green: 2.417, ir_blue: 2.426 }),
    );
    let diamond_planes = diamond_obj.planes().to_vec();
    let diamond: Arc<dyn Hittable> = Arc::new(diamond_obj);
    let static_objects: Vec<Arc<dyn Hittable>> = vec![ground, diamond];

    let mut dynamic: Vec<DynamicSphere> = Vec::new();

    dynamic.push(DynamicSphere {
        center: Point3::new( 0.0, 1.0, 0.0),
        velocity: Vec3::default(),
        radius: 1.0, mat: Arc::new(Dielectric { ir: 1.5 }),
        restitution: 0.65, is_static: true,
    });
    dynamic.push(DynamicSphere {
        center: Point3::new(-4.0, 1.0, 0.0),
        velocity: Vec3::default(),
        radius: 1.0,
        mat: Arc::new(PbrMaterial { albedo: Color::new(0.4, 0.2, 0.1), roughness: 0.85, metallic: 0.0 }),
        restitution: 0.35, is_static: true,
    });
    dynamic.push(DynamicSphere {
        center: Point3::new( 4.0, 1.0, 0.0),
        velocity: Vec3::default(),
        radius: 1.0,
        mat: Arc::new(PbrMaterial { albedo: Color::new(0.7, 0.6, 0.5), roughness: 0.0, metallic: 1.0 }),
        restitution: 0.80, is_static: true,
    });
    dynamic.push(DynamicSphere {
        center: Point3::new(-2.0, 1.0, -2.0),
        velocity: Vec3::default(),
        radius: 1.0,
        mat: Arc::new(PearlMaterial {
            base_color:      Color::new(0.98, 0.93, 0.88),
            ior:             1.56,
            film_thickness:  450.0,
            orient_strength: 0.30,
            film_scale:      5.0,
        }),
        restitution: 0.50, is_static: true,
    });

    // Glass marbles â€” small glass spheres falling near the diamond.
    // Two flavours:
    //   MarbleMaterial  â€” Perlin swirl pattern visible through the glass (colour
    //                     applied once at the exit surface).
    //   SSSMaterial     â€” volumetric multiple scattering: light diffuses over ~2
    //                     mean-free-paths inside and exits as a soft coloured glow.
    let marble_perlin = Arc::new(Perlin::new(&mut rng));
    // Palette for MarbleMaterial swirl spheres (color1 = ribbon, color2 = clear base).
    let marble_palette: &[(Color, Color)] = &[
        (Color::new(0.05, 0.70, 0.15), Color::new(1.0, 1.0, 1.0)), // cat's-eye green
        (Color::new(0.05, 0.15, 0.85), Color::new(1.0, 1.0, 1.0)), // cobalt blue
        (Color::new(0.90, 0.35, 0.03), Color::new(1.0, 1.0, 1.0)), // amber
        (Color::new(0.85, 0.03, 0.03), Color::new(1.0, 1.0, 1.0)), // ruby red
        (Color::new(0.50, 0.05, 0.85), Color::new(1.0, 1.0, 1.0)), // violet
        (Color::new(0.80, 0.65, 0.00), Color::new(1.0, 1.0, 1.0)), // gold
        (Color::new(0.00, 0.70, 0.70), Color::new(1.0, 1.0, 1.0)), // teal
        (Color::new(0.80, 0.10, 0.50), Color::new(1.0, 1.0, 1.0)), // magenta
    ];
    // Per-scatter scattering albedos for SSSMaterial (bright channel ~ 1, others < 1
    // so absorption builds colour over ~2 scatters per diameter traversal).
    let sss_colors: &[Color] = &[
        Color::new(0.60, 1.00, 0.60),  // jade green
        Color::new(0.40, 0.55, 1.00),  // cobalt blue
        Color::new(1.00, 0.70, 0.15),  // amber
        Color::new(1.00, 0.30, 0.20),  // ruby red
        Color::new(0.55, 0.20, 1.00),  // violet
        Color::new(1.00, 0.85, 0.30),  // gold
        Color::new(0.20, 1.00, 0.85),  // teal
        Color::new(1.00, 0.35, 0.70),  // rose
    ];
    // Dedicated SSS marbles dropped near the diamond â€” these showcase the organic
    // translucent glow most clearly since they are large and well-lit.
    let dedicated_marbles: &[(usize, Point3)] = &[
        (0, Point3::new(1.5,  8.0, -1.0)),
        (1, Point3::new(2.5,  6.0, -2.0)),
        (2, Point3::new(1.0, 10.0, -2.5)),
        (3, Point3::new(3.0,  7.0, -1.5)),
        (4, Point3::new(2.0,  9.0, -0.8)),
    ];
    for &(idx, center) in dedicated_marbles {
        dynamic.push(DynamicSphere {
            center,
            velocity:    Vec3::default(),
            radius:      0.15,
            mat:         Arc::new(SSSMaterial {
                albedo:  sss_colors[idx],
                ior:     1.5,
                density: 7.0,   // Ïƒ_t â‰ˆ 7 â†’ mean free path 0.14 â‰ˆ marble radius
                g:       0.30,  // slightly forward-scattering glass inclusions
            }),
            restitution: 0.60,
            is_static:   false,
        });
    }

    for a in -11i32..11 {
        for b in -11i32..11 {
            let cx = a as f32 + 0.9 * rng.gen::<f32>();
            let cz = b as f32 + 0.9 * rng.gen::<f32>();
            let ground_pos = Point3::new(cx, 0.2, cz);
            if (ground_pos - Point3::new( 4.0, 0.2, 0.0)).length() < 1.2 { continue; }
            if (ground_pos - Point3::new( 0.0, 0.2, 0.0)).length() < 1.2 { continue; }
            if (ground_pos - Point3::new(-4.0, 0.2, 0.0)).length() < 1.2 { continue; }
            let choose: f32 = rng.gen();
            let (mat, restitution): (Arc<dyn Material>, f32) = if choose < 0.40 {
                // Perlin swirl marble
                let (color1, color2) = marble_palette[rng.gen_range(0..marble_palette.len())];
                (Arc::new(MarbleMaterial { ir: 1.5, color1, color2, scale: 8.0, perlin: Arc::clone(&marble_perlin) }), 0.60)
            } else if choose < 0.60 {
                // SSS translucent marble â€” organic coloured glow
                let albedo  = sss_colors[rng.gen_range(0..sss_colors.len())];
                let density = rng.gen_range(5.0_f32..9.0);
                (Arc::new(SSSMaterial { albedo, ior: 1.5, density, g: 0.30 }), 0.60)
            } else if choose < 0.75 {
                let albedo = Color::random(&mut rng) * Color::random(&mut rng);
                let roughness: f32 = rng.gen_range(0.5..1.0);
                (Arc::new(PbrMaterial { albedo, roughness, metallic: 0.0 }), 0.35)
            } else if choose < 0.92 {
                let albedo   = Color::random_range(0.5, 1.0, &mut rng);
                let roughness: f32 = rng.gen_range(0.0..0.5);
                (Arc::new(PbrMaterial { albedo, roughness, metallic: 1.0 }), 0.5 + (1.0 - roughness) * 0.35)
            } else {
                (Arc::new(Dielectric { ir: 1.5 }), 0.65)
            };
            let center = Point3::new(cx, 0.2 + rng.gen_range(3.0..12.0), cz);
            dynamic.push(DynamicSphere { center, velocity: Vec3::default(), radius: 0.2, mat, restitution, is_static: false });
        }
    }

    let mut list = HittableList::new();
    for obj in &static_objects { list.objects.push(Arc::clone(obj)); }
    for ds in &dynamic { list.add(Sphere::new(ds.center, ds.radius, Arc::clone(&ds.mat))); }

    SceneData {
        world:          Arc::new(BvhTree::from_list(list)) as Arc<dyn Hittable>,
        lights:         HittableList::new(),
        background:     Background::Physical { sun_dir: Vec3::new(-0.4, 0.9, -0.3).unit() },
        name:           "Random Spheres",
        cam_init:       SceneCameraParams {
            pos: Point3::new(13.0, 2.0, 3.0), lookat: Point3::new(0.0, 0.0, 0.0),
            vfov: 20.0, aperture: 0.1, focus_dist: 10.0, move_speed: 0.3,
        },
        static_objects,
        dynamic,
        bounds:         None,
        colliders:         vec![],
        convex_colliders:  vec![diamond_planes],
        gravity:           0.03,
        settled:        false,
        paused:         false,
        max_samples:    2000,
        enable_caustics:       true,
        caustic_quad:          None,
        caustic_gather_radius: 0.15,
        photon_map:            None,
        cached_static:         None,
    }
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
    let tall_bbox = tall.bounding_box().unwrap();
    list.objects.push(tall);

    let short = Arc::new(make_box(Point3::new(0.0,0.0,0.0), Point3::new(165.0,165.0,165.0), white)) as Arc<dyn Hittable>;
    let short = Arc::new(Rotate::around_y(short, -18.0)) as Arc<dyn Hittable>;
    let short = Arc::new(Translate::new(short, Vec3::new(130.0, 0.0, 65.0))) as Arc<dyn Hittable>;
    let short_bbox = short.bounding_box().unwrap();
    list.objects.push(short);

    let static_objects = list.objects.clone();

    let dynamic = vec![DynamicSphere {
        center:      Point3::new(190.0, 100.0, 190.0),
        velocity:    Vec3::new(3.0, 5.0, 2.0),
        radius:      80.0,
        mat:         Arc::new(Dielectric { ir: 1.5 }),
        restitution: 1.0,
        is_static:   false,
    }];
    let bounds = Aabb::new(
        Point3::new(1.0, 1.0, 1.0),
        Point3::new(554.0, 554.0, 554.0),
    );

    for ds in &dynamic {
        list.add(Sphere::new(ds.center, ds.radius, Arc::clone(&ds.mat)));
    }

    SceneData {
        world:      Arc::new(BvhTree::from_list(list)) as Arc<dyn Hittable>,
        lights,
        background: Background::Solid(Color::default()),
        name:       "Cornell Box",
        cam_init:   SceneCameraParams {
            pos: Point3::new(278.0, 278.0, -800.0), lookat: Point3::new(278.0, 278.0, 0.0),
            vfov: 40.0, aperture: 0.0, focus_dist: 10.0, move_speed: 8.0,
        },
        static_objects,
        dynamic,
        bounds:            Some(bounds),
        colliders:         vec![tall_bbox, short_bbox],
        convex_colliders:  vec![],
        gravity:           0.0,
        settled:        false,
        paused:         false,
        max_samples:    2000,
        enable_caustics:       false,
        caustic_quad:          None,
        caustic_gather_radius: 0.15,
        photon_map:            None,
        cached_static:         None,
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
        Arc::new(Metal { albedo: Color::new(0.8, 0.8, 0.9), fuzz: 1.0 }),
    ));

    let fog_boundary: Arc<dyn Hittable> = Arc::new(Sphere::new(
        Point3::new(360.0, 150.0, 145.0), 70.0,
        Arc::new(Dielectric { ir: 1.5 }),
    ));
    list.objects.push(Arc::clone(&fog_boundary));
    list.add(ConstantMedium::new(fog_boundary, 0.2, Color::new(0.2, 0.4, 0.9)));

    let mist_boundary: Arc<dyn Hittable> = Arc::new(Sphere::new(
        Point3::new(0.0, 0.0, 0.0), 5000.0,
        Arc::new(Dielectric { ir: 1.5 }),
    ));
    list.add(ConstantMedium::new(mist_boundary, 0.0001, Color::new(1.0, 1.0, 1.0)));

    // Diamond â€” large gem floating directly below the area light.
    // Radius 100 fills the space between the ceiling and the scene objects.
    // Positioned at x=200, z=200 to clear the perlin sphere at (220,280,300).
    let diamond_r = 100.0_f32;
    list.objects.push(Arc::new(Diamond::new(
        Point3::new(200.0, 430.0, 200.0),
        diamond_r,
        Arc::new(SpectralDielectric { ir_red: 2.407, ir_green: 2.417, ir_blue: 2.426 }),
    )) as Arc<dyn Hittable>);

    // Pearl â€” Akoya-style cream pearl.  film_thickness=450 nm gives a rose-pink
    // orient at normal incidence cycling through blue and green at oblique angles.
    list.add(Sphere::new(
        Point3::new(400.0, 150.0, 270.0), 50.0,
        Arc::new(PearlMaterial {
            base_color:      Color::new(0.98, 0.93, 0.88),
            ior:             1.56,
            film_thickness:  450.0,
            orient_strength: 0.30,
            film_scale:      0.10,
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
            pos:        Point3::new(478.0, 278.0, -600.0),
            lookat:     Point3::new(278.0, 278.0,    0.0),
            vfov:       40.0,
            aperture:   0.0,
            focus_dist: 10.0,
            move_speed: 8.0,
        },
        static_objects:   vec![],
        dynamic:          vec![],
        bounds:           None,
        colliders:        vec![],
        convex_colliders: vec![],
        gravity:          0.0,
        settled:        false,
        paused:         false,
        max_samples:    2000,
        enable_caustics:       true,
        caustic_quad:          Some(caustic_quad),
        caustic_gather_radius: 10.0,
        photon_map:            None,
        cached_static:         None,
    };
    // Static scene: rebuild() is never called from tick(), so build the
    // photon map once here at construction time.
    scene.rebuild_caustics();
    scene
}

