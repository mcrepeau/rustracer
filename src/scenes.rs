use std::sync::Arc;
use std::time::Duration;
use crate::aabb::Aabb;
use crate::bvh::BvhTree;
use crate::camera::SceneCameraParams;
use crate::hittable::{Hittable, HittableList, Material};
use crate::material::{BumpMaterial, Dielectric, DiffuseLight, Lambertian, MarbleMaterial, Metal, SpectralDielectric};
use crate::perlin::Perlin;
use crate::quad::{Quad, make_box};
use crate::renderer::Background;
use crate::ring::Ring;
use crate::scene::{DynamicSphere, Orbit, RingData, SceneData};
use crate::sphere::Sphere;
use crate::texture::Texture;
use crate::diamond::Diamond;
use crate::transform::{Rotate, Translate};
use crate::vec3::{Color, Point3, Vec3};
use crate::volume::ConstantMedium;
use rand::Rng;
use rand::rngs::SmallRng;
use rand::SeedableRng;
use std::f32::consts::PI;

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
        restitution: 0.65, is_static: true, orbit: None,
        axial_angle: 0.0, axial_speed: 0.0, axial_tilt: 0.0, ring: None,
        stretch: Vec3::new(1.0, 1.0, 1.0),
    });
    dynamic.push(DynamicSphere {
        center: Point3::new(-4.0, 1.0, 0.0),
        velocity: Vec3::default(),
        radius: 1.0, mat: Arc::new(Lambertian { texture: Color::new(0.4, 0.2, 0.1).into() }),
        restitution: 0.35, is_static: true, orbit: None,
        axial_angle: 0.0, axial_speed: 0.0, axial_tilt: 0.0, ring: None,
        stretch: Vec3::new(1.0, 1.0, 1.0),
    });
    dynamic.push(DynamicSphere {
        center: Point3::new( 4.0, 1.0, 0.0),
        velocity: Vec3::default(),
        radius: 1.0, mat: Arc::new(Metal { albedo: Color::new(0.7, 0.6, 0.5), fuzz: 0.0 }),
        restitution: 0.80, is_static: true, orbit: None,
        axial_angle: 0.0, axial_speed: 0.0, axial_tilt: 0.0, ring: None,
        stretch: Vec3::new(1.0, 1.0, 1.0),
    });

    // Glass marbles — small glass spheres with internal Perlin swirl, falling near the diamond.
    // color2 is pure white so the clear areas of the marble are fully transparent,
    // making the coloured ribbons (color1) stand out sharply.
    let marble_perlin = Arc::new(Perlin::new(&mut rng));
    // Palette used for both the dedicated marbles and the random dropping spheres.
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
    let dedicated_marbles: &[(usize, Point3)] = &[
        (0, Point3::new(1.5,  8.0, -1.0)),
        (1, Point3::new(2.5,  6.0, -2.0)),
        (2, Point3::new(1.0, 10.0, -2.5)),
        (3, Point3::new(3.0,  7.0, -1.5)),
        (4, Point3::new(2.0,  9.0, -0.8)),
    ];
    for &(palette_idx, center) in dedicated_marbles {
        let (color1, color2) = marble_palette[palette_idx];
        dynamic.push(DynamicSphere {
            center,
            velocity:    Vec3::default(),
            radius:      0.15,
            mat:         Arc::new(MarbleMaterial {
                ir: 1.5, color1, color2, scale: 8.0,
                perlin: Arc::clone(&marble_perlin),
            }),
            restitution: 0.60,
            is_static:   false,
            orbit:       None,
            axial_angle: 0.0, axial_speed: 0.0, axial_tilt: 0.0,
            ring:        None,
            stretch:     Vec3::new(1.0, 1.0, 1.0),
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
            let (mat, restitution): (Arc<dyn Material>, f32) = if choose < 0.60 {
                let (color1, color2) = marble_palette[rng.gen_range(0..marble_palette.len())];
                (Arc::new(MarbleMaterial { ir: 1.5, color1, color2, scale: 8.0, perlin: Arc::clone(&marble_perlin) }), 0.60)
            } else if choose < 0.75 {
                (Arc::new(Lambertian { texture: (Color::random(&mut rng) * Color::random(&mut rng)).into() }), 0.35)
            } else if choose < 0.92 {
                let fuzz: f32 = rng.gen_range(0.0..0.5);
                (Arc::new(Metal { albedo: Color::random_range(0.5, 1.0, &mut rng), fuzz }), 0.5 + (1.0 - fuzz) * 0.35)
            } else {
                (Arc::new(Dielectric { ir: 1.5 }), 0.65)
            };
            let center = Point3::new(cx, 0.2 + rng.gen_range(3.0..12.0), cz);
            dynamic.push(DynamicSphere { center, velocity: Vec3::default(), radius: 0.2, mat, restitution, is_static: false, orbit: None, axial_angle: 0.0, axial_speed: 0.0, axial_tilt: 0.0, ring: None, stretch: Vec3::new(1.0, 1.0, 1.0) });
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
        named_bodies:   vec![],
        physics_dt:     Duration::from_millis(16),
        use_oidn_aux:   true,
        cached_static:  None,
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
        orbit:       None,
        axial_angle: 0.0,
        axial_speed: 0.0,
        axial_tilt:  0.0,
        ring:        None,
        stretch:     Vec3::new(1.0, 1.0, 1.0),
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
        named_bodies:   vec![],
        physics_dt:     Duration::from_millis(16),
        use_oidn_aux:   false,
        cached_static:  None,
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

    SceneData {
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
        settled:        true,
        paused:         false,
        max_samples:    2000,
        named_bodies:   vec![],
        physics_dt:     Duration::from_millis(16),
        use_oidn_aux:   false,
        cached_static:  None,
    }
}

pub fn build_solar_system_scene() -> SceneData {
    // ── Sun ───────────────────────────────────────────────────────────────────
    let mut rng = SmallRng::seed_from_u64(7);
    let sun_perlin = Arc::new(Perlin::new(&mut rng));
    let sun_mat: Arc<dyn Material> = Arc::new(DiffuseLight {
        emit: Texture::SolarNoise {
            perlin: sun_perlin,
            scale:  0.8,
            // Values scaled so both hot and cool sit on the visible slope of
            // the ACES curve (no channel fully saturates), giving perceptible
            // granulation contrast while still illuminating distant planets.
            hot:    Color::new(5.0, 3.5, 1.25),
            cool:   Color::new(1.25, 0.45, 0.05),
        },
    });
    let mut lights = HittableList::new();
    lights.add(Sphere::new(Point3::new(0.0, 0.0, 0.0), 12.0, Arc::clone(&sun_mat)));
    let mut static_objects: Vec<Arc<dyn Hittable>> = vec![
        Arc::new(Sphere::new(Point3::new(0.0, 0.0, 0.0), 12.0, sun_mat)),
    ];

    // ── Orbital trail rings ────────────────────────────────────────────────────
    // Hairline dim-emissive rings showing each planet's orbital plane.
    // Normal of an orbit with inclination i and ascending node Ω:
    //   (sin Ω · sin i,  cos i,  −cos Ω · sin i)
    let trail_mat: Arc<dyn Material> = Arc::new(DiffuseLight {
        emit: Color::new(0.05, 0.05, 0.10).into(),
    });
    // (orbital_radius, inclination_deg, asc_node_deg)  — J2000 values
    // Radii use sqrt-compression: r_scene = 35 × sqrt(r_AU)  (see planet calls below).
    let orbit_data: &[(f32, f32, f32)] = &[
        ( 22.0,  7.00,  48.33),  // Mercury  (0.387 AU)
        ( 30.0,  3.39,  76.68),  // Venus    (0.723 AU)
        ( 35.0,  0.00,   0.00),  // Earth    (1.000 AU) — anchor
        ( 43.0,  1.85,  49.56),  // Mars     (1.524 AU)
        ( 80.0,  1.30, 100.49),  // Jupiter  (5.203 AU)
        (108.0,  2.49, 113.64),  // Saturn   (9.539 AU)
        (153.0,  0.77,  74.00),  // Uranus  (19.18  AU)
        (192.0,  1.77, 131.78),  // Neptune (30.07  AU)
    ];
    for &(r, incl_deg, omega_deg) in orbit_data {
        let i = incl_deg.to_radians();
        let n = omega_deg.to_radians();
        let normal = Vec3::new(n.sin() * i.sin(), i.cos(), -n.cos() * i.sin());
        static_objects.push(Arc::new(Ring::new(
            Point3::new(0.0, 0.0, 0.0),
            r - 0.2,
            r + 0.2,
            normal,
            Arc::clone(&trail_mat),
        )) as Arc<dyn Hittable>);
    }

    // ── Planet + moon helper ──────────────────────────────────────────────────
    // Planets are pushed first (indices 0-7); moons reference planet indices.
    let mut dynamic: Vec<DynamicSphere> = Vec::new();

    // Perlin for Jupiter's banded cloud texture (seeded separately from the sun).
    let mut jup_rng = SmallRng::seed_from_u64(12);
    let jup_perlin  = Arc::new(Perlin::new(&mut jup_rng));

    let mut planet = |orbit_r: f32, speed: f32, angle: f32, incl_deg: f32,
                      body_r: f32, axial_speed: f32, axial_tilt_deg: f32,
                      asc_node_deg: f32,
                      mat: Arc<dyn Material>| {
        let incl  = incl_deg.to_radians();
        let omega = asc_node_deg.to_radians();
        let (sa, ca) = angle.sin_cos();
        let (si, ci) = incl.sin_cos();
        let (sn, cn) = omega.sin_cos();
        dynamic.push(DynamicSphere {
            center: Point3::new(
                orbit_r * (ca * cn - sa * sn * ci),
                orbit_r *  sa * si,
                orbit_r * (ca * sn + sa * cn * ci),
            ),
            velocity:    Vec3::default(),
            radius:      body_r,
            mat,
            restitution: 0.0,
            is_static:   false,
            orbit: Some(Orbit { parent_idx: None, radius: orbit_r, speed, angle, inclination: incl, asc_node: omega }),
            axial_angle: 0.0,
            axial_speed,
            axial_tilt: axial_tilt_deg.to_radians(),
            stretch:     Vec3::new(1.0, 1.0, 1.0),
            ring: None,
        });
    };

    // ── Planets (indices 0-7) ─────────────────────────────────────────────────
    // Orbital distances use sqrt-compression: r_scene = 35 × sqrt(r_AU).
    // Earth (1 AU) anchors the scale at r_scene = 35.  This is NOT physically
    // accurate — true distances would place Neptune at ~1053 scene units (30×
    // Earth's orbit), making outer planets sub-pixel from any useful camera
    // position.  The sqrt curve gives outer planets proportionally more room
    // than linear compression while keeping all 8 planets on screen together.
    // True-scale scene radii for reference:  Mercury 13.5 | Venus 25.3 | Mars
    // 53.3 | Jupiter 182 | Saturn 334 | Uranus 671 | Neptune 1053.
    //
    // Orbital speeds: Kepler-consistent with SCENE radii: ω = 0.0003491·(35/r)^1.5.
    // Earth (r=35) is the anchor: 18000 ticks/orbit, 0.0003491 rad/tick.
    // Axial speeds: real sidereal day lengths. Earth day ≈ 49.3 ticks.
    // Axial tilts:  real obliquities. Venus is nearly inverted (177.4°, retrograde axial spin).
    //         orbit_r  speed(Kepler)   angle  incl°  body_r  axial_spd   tilt°   Ω°
    planet( 22.0,  0.0007005,  0.0,   7.00,  0.61, 0.00218,    0.0,  48.33,
        Arc::new(Lambertian { texture: Color::new(0.60, 0.58, 0.55).into() })); // Mercury
    planet( 30.0,  0.0004399,  1.2,   3.39,  1.5, -0.000524,  177.4, 76.68,
        Arc::new(Lambertian { texture: Color::new(0.90, 0.80, 0.50).into() })); // Venus
    planet( 35.0,  0.0003491,  2.5,   0.00,  1.6,  0.1274,    23.4,  0.00,
        Arc::new(Lambertian { texture:
            Texture::load("assets/earthmap.jpg")
                .unwrap_or_else(|_| Color::new(0.20, 0.45, 0.85).into())
        }));                                                                       // Earth
    planet( 43.0,  0.0002562,  0.8,   1.85,  0.85, 0.1241,    25.2, 49.56,
        Arc::new(Lambertian { texture: Color::new(0.78, 0.32, 0.12).into() })); // Mars
    planet( 80.0,  0.0001011,  3.5,   1.30,  5.5,  0.3083,     3.1, 100.49,
        Arc::new(Lambertian { texture: Texture::PlanetBands {
            perlin:    Arc::clone(&jup_perlin),
            scale:     1.0,
            band_freq: 12.0,
            hot:  Color::new(0.72, 0.44, 0.18),
            cool: Color::new(0.88, 0.80, 0.62),
        }})); // Jupiter
    planet(108.0,  0.0000644,  1.8,   2.49,  4.5,  0.2873,    26.7, 113.64,
        Arc::new(Lambertian { texture: Color::new(0.88, 0.80, 0.55).into() })); // Saturn
    planet(153.0,  0.0000382,  4.2,   0.77,  3.0, -0.1776,    97.8, 74.00,
        Arc::new(Lambertian { texture: Color::new(0.50, 0.88, 0.88).into() })); // Uranus
    planet(192.0,  0.0000272,  2.9,   1.77,  2.9,  0.1900,    28.3, 131.78,
        Arc::new(Lambertian { texture: Color::new(0.18, 0.28, 0.80).into() })); // Neptune

    // Saturn's rings (index 5).
    // Ring normal matches Saturn's axial tilt: Rotate::around_x(26.7°) maps the
    // pole to (0, cos(26.7°), sin(26.7°)), so the equatorial ring has that normal.
    // Saturn's ring geometry scaled to Saturn's scene radius (4.5):
    //   inner_r = 1.24 × 4.5 = 5.58  (C ring inner edge, real 1.24 Saturn radii)
    //   outer_r = 2.27 × 4.5 = 10.22 (A ring outer edge, real 2.27 Saturn radii)
    // Band thresholds are proportional fractions of the real ring widths (km):
    //   C(17,342) B(25,500) Cassini(4,700) A-inner(14,575) Encke(325) A-outer
    let saturn_tilt = 26.7_f32.to_radians();
    let ring_mat: Arc<dyn Material> = Arc::new(Lambertian {
        texture: Texture::RingBands(Arc::new(vec![
            (0.28, Color::new(0.62, 0.55, 0.40)), // C ring:           28% — dusty brown
            (0.69, Color::new(0.92, 0.88, 0.75)), // B ring:           69% — brightest, creamy
            (0.76, Color::new(0.06, 0.05, 0.04)), // Cassini division: 76% — dark gap
            (0.95, Color::new(0.82, 0.76, 0.62)), // A ring:           95% — medium brightness
            (0.96, Color::new(0.10, 0.09, 0.07)), // Encke gap:        96% — narrow dark band
            (1.00, Color::new(0.76, 0.71, 0.58)), // outer A ring:     100% — slightly fainter
        ])),
    });
    dynamic[5].ring = Some(RingData {
        inner_r: 5.58,
        outer_r: 10.22,
        normal:  Vec3::new(0.0, saturn_tilt.cos(), saturn_tilt.sin()),
        mat:     ring_mat,
    });

    // ── Asteroid belt ─────────────────────────────────────────────────────────
    // ~45 small rocky bodies between Mars (r=43) and Jupiter (r=80).
    // Real belt: 2.2–3.2 AU → sqrt-compressed to ~52–63 scene units.
    // Speeds follow the same Kepler formula as planets, with ±15% scatter.
    let mut belt_rng = SmallRng::seed_from_u64(999);
    // Shared Perlin instance: each asteroid sits at a different location in the
    // noise field, so they all look distinct even with a single noise table.
    let mut bump_rng = SmallRng::seed_from_u64(7331);
    let asteroid_perlin = Arc::new(Perlin::new(&mut bump_rng));
    for _ in 0..45 {
        let orbit_r  = 52.0_f32 + belt_rng.gen::<f32>() * 12.0;
        let angle    = belt_rng.gen::<f32>() * 2.0 * PI;
        let incl     = (belt_rng.gen::<f32>() - 0.5) * 0.3;
        let asc_node = belt_rng.gen::<f32>() * 2.0 * PI;
        // Kepler speed: same formula as planets, with ±15% random scatter.
        let speed    = 0.0003491 * (35.0_f32 / orbit_r).powf(1.5)
                       * (0.85 + belt_rng.gen::<f32>() * 0.30);
        let radius   = 0.18 + belt_rng.gen::<f32>() * 0.28;
        let g        = 0.38 + belt_rng.gen::<f32>() * 0.22;
        let red      = belt_rng.gen::<f32>() * 0.09;
        let base_mat: Arc<dyn Material> = Arc::new(Lambertian {
            texture: Color::new(g + red, g - red * 0.3, g * 0.82).into(),
        });
        let mat: Arc<dyn Material> = Arc::new(BumpMaterial {
            inner:    base_mat,
            perlin:   Arc::clone(&asteroid_perlin),
            scale:    4.0,
            strength: 0.5,
        });
        // Random ellipsoidal stretch — gives each asteroid a distinct potato/pebble shape.
        let sx = 0.5 + belt_rng.gen::<f32>() * 1.0; // 0.5–1.5
        let sy = 0.3 + belt_rng.gen::<f32>() * 0.8; // 0.3–1.1  (often flatter in Y)
        let sz = 0.5 + belt_rng.gen::<f32>() * 1.0; // 0.5–1.5
        let (sa, ca) = angle.sin_cos();
        let (si, ci) = incl.sin_cos();
        let (sn, cn) = asc_node.sin_cos();
        dynamic.push(DynamicSphere {
            center:      Point3::new(
                orbit_r * (ca * cn - sa * sn * ci),
                orbit_r *  sa * si,
                orbit_r * (ca * sn + sa * cn * ci),
            ),
            velocity:    Vec3::default(),
            radius,
            mat,
            restitution: 0.0,
            is_static:   false,
            orbit:       Some(Orbit { parent_idx: None, radius: orbit_r, speed, angle, inclination: incl, asc_node }),
            axial_angle: 0.0,
            axial_speed: 0.0,
            axial_tilt:  0.0,
            ring:        None,
            stretch:     Vec3::new(sx, sy, sz),
        });
    }

    // ── Moons ─────────────────────────────────────────────────────────────────
    // Orbital speeds: real sidereal periods at 49.28 ticks/day (18000/365.25).
    // incl_deg / asc_node_deg: orbital plane relative to the ecliptic.
    //   For moons that orbit in their parent's equatorial plane, inclination = planet
    //   axial_tilt and ascending_node = 180°.  This comes from inverting the Keplerian
    //   normal formula: n=(sin Ω sin i, cos i, −cos Ω sin i) = (0, cos t, sin t) when
    //   Ω = 180° and i = t (the planet's axial tilt).
    let mut moon = |parent_idx: usize, orbit_r: f32, speed: f32, angle: f32,
                    body_r: f32, incl_deg: f32, asc_node_deg: f32,
                    mat: Arc<dyn Material>| {
        let p = dynamic[parent_idx].center;
        dynamic.push(DynamicSphere {
            center: Point3::new(p.x + orbit_r * angle.cos(), p.y, p.z + orbit_r * angle.sin()),
            velocity:    Vec3::default(),
            radius:      body_r,
            mat,
            restitution: 0.0,
            is_static:   false,
            orbit: Some(Orbit {
                parent_idx:  Some(parent_idx),
                radius:      orbit_r,
                speed,
                angle,
                inclination: incl_deg.to_radians(),
                asc_node:    asc_node_deg.to_radians(),
            }),
            axial_angle: 0.0,
            axial_speed: 0.0,
            axial_tilt:  0.0,
            ring:        None,
            stretch:     Vec3::new(1.0, 1.0, 1.0),
        });
    };

    // Galilean moon orbit radii scaled so Io visibly clears Jupiter (r=5.5).
    // Proportional scaling from pre-inflation values: factor 8.0/6.5 ≈ 1.23.
    // Titan orbits in Saturn's equatorial plane (tilt 26.7°, Ω=180°).
    // Uranian moons orbit in Uranus's equatorial plane (tilt 97.8°, Ω=180°).
    // Triton: retrograde orbital direction captured by negative speed.
    //              parent  orbit_r   speed    angle  body_r  incl°  Ω°
    moon(2,   3.2, 0.00467,  0.0,  0.42,   5.1, 125.0, Arc::new(Lambertian { texture: Color::new(0.72, 0.72, 0.70).into() })); // Moon
    moon(4,   8.0, 0.07200,  0.0,  0.50,   0.0,   0.0, Arc::new(Lambertian { texture: Color::new(0.90, 0.82, 0.30).into() })); // Io
    moon(4,  11.0, 0.03590,  2.1,  0.44,   0.0,   0.0, Arc::new(Lambertian { texture: Color::new(0.88, 0.90, 0.92).into() })); // Europa
    moon(4,  15.0, 0.01783,  0.5,  0.66,   0.0,   0.0, Arc::new(Lambertian { texture: Color::new(0.58, 0.55, 0.48).into() })); // Ganymede
    moon(4,  19.0, 0.00764,  1.8,  0.60,   0.0,   0.0, Arc::new(Lambertian { texture: Color::new(0.38, 0.35, 0.30).into() })); // Callisto
    moon(5,   6.8, 0.00799,  1.0,  0.52,  26.7, 180.0, Arc::new(Lambertian { texture: Color::new(0.80, 0.62, 0.35).into() })); // Titan
    moon(7,   4.2, -0.02170, 3.0,  0.36,   0.0,   0.0, Arc::new(Lambertian { texture: Color::new(0.78, 0.72, 0.75).into() })); // Triton (retrograde)
    moon(6,   5.0, 0.01464,  0.0,  0.20,  97.8, 180.0, Arc::new(Lambertian { texture: Color::new(0.72, 0.70, 0.68).into() })); // Titania
    moon(6,   6.5, 0.00947,  2.8,  0.19,  97.8, 180.0, Arc::new(Lambertian { texture: Color::new(0.52, 0.49, 0.45).into() })); // Oberon

    // Named bodies for camera follow mode (indices stable after all pushes).
    let n = dynamic.len(); // = 8 planets + 45 asteroids + 9 moons = 62
    let named_bodies = vec![
        (0, "Mercury"), (1, "Venus"),    (2, "Earth"),    (3, "Mars"),
        (4, "Jupiter"), (5, "Saturn"),   (6, "Uranus"),   (7, "Neptune"),
        (n-9, "Moon"),  (n-8, "Io"),     (n-7, "Europa"), (n-6, "Ganymede"),
        (n-5, "Callisto"), (n-4, "Titan"), (n-3, "Triton"),
        (n-2, "Titania"), (n-1, "Oberon"),
    ];

    // ── Build world ───────────────────────────────────────────────────────────
    let mut list = HittableList::new();
    for obj in &static_objects { list.objects.push(Arc::clone(obj)); }
    for ds in &dynamic { list.add(Sphere::new(ds.center, ds.radius, Arc::clone(&ds.mat))); }

    SceneData {
        world:      Arc::new(BvhTree::from_list(list)) as Arc<dyn Hittable>,
        lights,
        background: Background::Stars,
        name:       "Solar System",
        cam_init:   SceneCameraParams {
            // ~22° elevation: shows orbital-plane depth and makes the star field dramatic.
            // Pulled back to fit Neptune's wider orbit (r=192 vs old r=130).
            pos:        Point3::new(0.0, 110.0, 260.0),
            lookat:     Point3::new(0.0,   0.0,   0.0),
            vfov:       55.0,
            aperture:   0.0,
            focus_dist: 10.0,
            move_speed: 4.0,
        },
        static_objects,
        dynamic,
        bounds:           None,
        colliders:        vec![],
        convex_colliders: vec![],
        gravity:          0.0,
        settled:       false,
        paused:        false,
        max_samples:   4000,
        named_bodies,
        // Large physics_dt lets the path tracer accumulate many samples
        // between each planet-position update (planets barely move per tick).
        physics_dt:    Duration::from_millis(500),
        use_oidn_aux:  false,
        cached_static: None,
    }
}
