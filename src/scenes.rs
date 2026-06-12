use std::sync::Arc;
use std::time::Duration;
use crate::aabb::Aabb;
use crate::bvh::BvhTree;
use crate::camera::SceneCameraParams;
use crate::hittable::{Hittable, HittableList, Material};
use crate::material::{Dielectric, DiffuseLight, Lambertian, Metal};
use crate::mesh::load_obj;
use crate::perlin::Perlin;
use crate::quad::{Quad, make_box};
use crate::renderer::Background;
use crate::scene::{DynamicSphere, Orbit, RingData, SceneData};
use crate::sphere::Sphere;
use crate::texture::Texture;
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
    let static_objects = vec![ground];

    let mut dynamic: Vec<DynamicSphere> = Vec::new();

    dynamic.push(DynamicSphere {
        center: Point3::new( 0.0, 1.0, 0.0),
        velocity: Vec3::default(),
        radius: 1.0, mat: Arc::new(Dielectric { ir: 1.5 }),
        restitution: 0.65, is_static: true, orbit: None,
        axial_angle: 0.0, axial_speed: 0.0, axial_tilt: 0.0, ring: None,
    });
    dynamic.push(DynamicSphere {
        center: Point3::new(-4.0, 1.0, 0.0),
        velocity: Vec3::default(),
        radius: 1.0, mat: Arc::new(Lambertian { texture: Color::new(0.4, 0.2, 0.1).into() }),
        restitution: 0.35, is_static: true, orbit: None,
        axial_angle: 0.0, axial_speed: 0.0, axial_tilt: 0.0, ring: None,
    });
    dynamic.push(DynamicSphere {
        center: Point3::new( 4.0, 1.0, 0.0),
        velocity: Vec3::default(),
        radius: 1.0, mat: Arc::new(Metal { albedo: Color::new(0.7, 0.6, 0.5), fuzz: 0.0 }),
        restitution: 0.80, is_static: true, orbit: None,
        axial_angle: 0.0, axial_speed: 0.0, axial_tilt: 0.0, ring: None,
    });

    for a in -11i32..11 {
        for b in -11i32..11 {
            let cx = a as f32 + 0.9 * rng.gen::<f32>();
            let cz = b as f32 + 0.9 * rng.gen::<f32>();
            let ground_pos = Point3::new(cx, 0.2, cz);
            if (ground_pos - Point3::new( 4.0, 0.2, 0.0)).length() < 1.2 { continue; }
            if (ground_pos - Point3::new( 0.0, 0.2, 0.0)).length() < 1.2 { continue; }
            if (ground_pos - Point3::new(-4.0, 0.2, 0.0)).length() < 1.2 { continue; }
            let choose: f32 = rng.gen();
            let (mat, restitution): (Arc<dyn Material>, f32) = if choose < 0.80 {
                (Arc::new(Lambertian { texture: (Color::random(&mut rng) * Color::random(&mut rng)).into() }), 0.35)
            } else if choose < 0.95 {
                let fuzz: f32 = rng.gen_range(0.0..0.5);
                (Arc::new(Metal { albedo: Color::random_range(0.5, 1.0, &mut rng), fuzz }), 0.5 + (1.0 - fuzz) * 0.35)
            } else {
                (Arc::new(Dielectric { ir: 1.5 }), 0.65)
            };
            let center = Point3::new(cx, 0.2 + rng.gen_range(3.0..12.0), cz);
            dynamic.push(DynamicSphere { center, velocity: Vec3::default(), radius: 0.2, mat, restitution, is_static: false, orbit: None, axial_angle: 0.0, axial_speed: 0.0, axial_tilt: 0.0, ring: None });
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
        bounds:    None,
        colliders: vec![],
        gravity:     0.03,
        settled:     false,
        paused:      false,
        max_samples: 2000,
        physics_dt:  Duration::from_millis(16),
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
        bounds:    Some(bounds),
        colliders: vec![tall_bbox, short_bbox],
        gravity:     0.0,
        settled:     false,
        paused:      false,
        max_samples: 2000,
        physics_dt:  Duration::from_millis(16),
    }
}

pub fn build_mesh_scene() -> SceneData {
    const MODEL_PATH: &str = "assets/model.obj";

    let mat: Arc<dyn Material> =
        Arc::new(Lambertian { texture: Color::new(0.8, 0.8, 0.75).into() });

    let (mesh_bvh, cam_init) = match load_obj(MODEL_PATH, 1.0, mat) {
        Err(e) => {
            println!("  Could not load '{MODEL_PATH}': {e}");
            println!("  Drop any OBJ file there and press 3 to view it.");
            let mut list = HittableList::new();
            list.add(Sphere::new(Point3::new(0.0, 1.0, 0.0), 1.0,
                Arc::new(Metal { albedo: Color::new(0.8, 0.85, 0.9), fuzz: 0.02 })));
            (BvhTree::from_list(list), SceneCameraParams {
                pos: Point3::new(0.0, 2.0, 6.0), lookat: Point3::new(0.0, 1.0, 0.0),
                vfov: 40.0, aperture: 0.0, focus_dist: 6.0, move_speed: 0.3,
            })
        }
        Ok(mesh_list) => {
            let bb = mesh_list.bounding_box().unwrap();
            let cx = (bb.min.x + bb.max.x) * 0.5;
            let cy = (bb.min.y + bb.max.y) * 0.5;
            let cz = (bb.min.z + bb.max.z) * 0.5;
            let size = (bb.max.x - bb.min.x).max(bb.max.y - bb.min.y).max(bb.max.z - bb.min.z);
            println!("  center ({cx:.2}, {cy:.2}, {cz:.2}), extent {size:.2}");
            let cam_pos = Point3::new(cx, cy + size * 0.3, cz + size * 1.8);
            (BvhTree::from_list(mesh_list), SceneCameraParams {
                pos: cam_pos, lookat: Point3::new(cx, cy, cz),
                vfov: 40.0, aperture: 0.0, focus_dist: size * 2.0, move_speed: size * 0.02,
            })
        }
    };

    let mut list = HittableList::new();
    list.add(Quad::new(
        Point3::new(-500.0, 0.0, 500.0), Vec3::new(1000.0, 0.0, 0.0), Vec3::new(0.0, 0.0, -1000.0),
        Arc::new(Lambertian { texture: Texture::Checker {
            scale: 1.0, even: Color::new(0.15, 0.15, 0.15), odd: Color::new(0.85, 0.85, 0.85),
        }}),
    ));
    let (overhead_world, overhead_sampler) = emissive_quad(
        Point3::new(-200.0, 500.0, -200.0),
        Vec3::new(400.0, 0.0, 0.0),
        Vec3::new(0.0, 0.0, 400.0),
        Color::new(6.0, 6.0, 6.0),
    );
    let mut lights = HittableList::new();
    lights.add(overhead_sampler);
    list.add(overhead_world);
    list.add(mesh_bvh);

    SceneData {
        world:      Arc::new(BvhTree::from_list(list)) as Arc<dyn Hittable>,
        lights,
        background: Background::Solid(Color::new(0.05, 0.07, 0.12)),
        name:       "Mesh",
        cam_init,
        static_objects: vec![],
        dynamic:        vec![],
        bounds:         None,
        colliders:      vec![],
        gravity:        0.0,
        settled:        false,
        paused:         false,
        max_samples:    2000,
        physics_dt:     Duration::from_millis(16),
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
        static_objects: vec![],
        dynamic:        vec![],
        bounds:         None,
        colliders:      vec![],
        gravity:        0.0,
        settled:        true,
        paused:         false,
        max_samples:    2000,
        physics_dt:     Duration::from_millis(16),
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
    lights.add(Sphere::new(Point3::new(0.0, 0.0, 0.0), 8.0, Arc::clone(&sun_mat)));
    let static_objects: Vec<Arc<dyn Hittable>> = vec![
        Arc::new(Sphere::new(Point3::new(0.0, 0.0, 0.0), 8.0, sun_mat)),
    ];

    // ── Planet + moon helper ──────────────────────────────────────────────────
    // Planets are pushed first (indices 0-7); moons reference planet indices.
    let mut dynamic: Vec<DynamicSphere> = Vec::new();

    let mut planet = |orbit_r: f32, speed: f32, angle: f32, incl: f32,
                      body_r: f32, axial_speed: f32, axial_tilt_deg: f32,
                      mat: Arc<dyn Material>| {
        let a = angle;
        dynamic.push(DynamicSphere {
            center: Point3::new(orbit_r * a.cos(), 0.0, orbit_r * a.sin()),
            velocity:    Vec3::default(),
            radius:      body_r,
            mat,
            restitution: 0.0,
            is_static:   false,
            orbit: Some(Orbit { parent_idx: None, radius: orbit_r, speed, angle, inclination: incl }),
            axial_angle: 0.0,
            axial_speed,
            axial_tilt: axial_tilt_deg.to_radians(),
            ring: None,
        });
    };

    // ── Planets (indices 0-7) ─────────────────────────────────────────────────
    // Orbital speeds: Earth year = 18000 ticks (10 min @ 60 fps).
    // Axial speeds:   Earth day ≈ 49.3 ticks → 2π/49.3 ≈ 0.1274 rad/tick.
    // Axial tilts:    real obliquities in degrees (Rotate::around_x so the pole
    //                 appears at (0, cos(tilt), sin(tilt)) in world space).
    planet(18.0,  0.001448,  0.0,  0.10, 0.8,  0.00218,   0.0,
        Arc::new(Lambertian { texture: Color::new(0.60, 0.58, 0.55).into() })); // Mercury  ~0°
    planet(26.0,  0.000566,  1.2,  0.05, 1.5, -0.000524,  0.0,
        Arc::new(Lambertian { texture: Color::new(0.90, 0.80, 0.50).into() })); // Venus    retrograde
    planet(35.0,  0.0003491, 2.5,  0.03, 1.6,  0.1274,   23.4,
        Arc::new(Lambertian { texture:
            Texture::load("assets/earthmap.jpg")
                .unwrap_or_else(|_| Color::new(0.20, 0.45, 0.85).into())
        }));                                                                      // Earth   23.4°
    planet(47.0,  0.0001855, 0.8,  0.08, 1.0,  0.1241,   25.2,
        Arc::new(Lambertian { texture: Color::new(0.78, 0.32, 0.12).into() })); // Mars    25.2°
    planet(65.0,  0.0000294, 3.5,  0.02, 4.5,  0.3083,    3.1,
        Arc::new(Lambertian { texture: Color::new(0.80, 0.62, 0.40).into() })); // Jupiter  3.1°
    planet(87.0,  0.0000118, 1.8,  0.04, 3.5,  0.2873,   26.7,
        Arc::new(Lambertian { texture: Color::new(0.88, 0.80, 0.55).into() })); // Saturn  26.7°
    planet(108.0, 0.00000415, 4.2, 0.12, 2.2, -0.1776,   97.8,
        Arc::new(Lambertian { texture: Color::new(0.50, 0.88, 0.88).into() })); // Uranus  97.8° (on side)
    planet(130.0, 0.00000212, 2.9, 0.03, 2.1,  0.1900,   28.3,
        Arc::new(Lambertian { texture: Color::new(0.18, 0.28, 0.80).into() })); // Neptune 28.3°

    // Saturn's rings (index 5).
    // Ring normal matches Saturn's axial tilt: Rotate::around_x(26.7°) maps the
    // pole to (0, cos(26.7°), sin(26.7°)), so the equatorial ring has that normal.
    let saturn_tilt = 26.7_f32.to_radians();
    let ring_mat: Arc<dyn Material> = Arc::new(Lambertian {
        texture: Texture::RingBands(Arc::new(vec![
            (0.20, Color::new(0.62, 0.55, 0.40)), // C ring:           faint, dusty brown
            (0.55, Color::new(0.92, 0.88, 0.75)), // B ring:           brightest, creamy white
            (0.63, Color::new(0.06, 0.05, 0.04)), // Cassini division: dark gap
            (0.82, Color::new(0.82, 0.76, 0.62)), // A ring:           medium brightness
            (0.85, Color::new(0.10, 0.09, 0.07)), // Encke gap:        narrow dark band
            (0.97, Color::new(0.75, 0.70, 0.57)), // outer A ring:     slightly fainter
            (1.00, Color::new(0.48, 0.44, 0.36)), // outer fringe:     very faint
        ])),
    });
    dynamic[5].ring = Some(RingData {
        inner_r: 5.0,
        outer_r: 9.0,
        normal:  Vec3::new(0.0, saturn_tilt.cos(), saturn_tilt.sin()),
        mat:     ring_mat,
    });

    // ── Moons ─────────────────────────────────────────────────────────────────
    // Moon speeds scaled to the same 18000-ticks-per-Earth-year base.
    let mut moon = |parent_idx: usize, orbit_r: f32, speed: f32, angle: f32,
                    body_r: f32, mat: Arc<dyn Material>| {
        let p = dynamic[parent_idx].center;
        dynamic.push(DynamicSphere {
            center: Point3::new(p.x + orbit_r * angle.cos(), p.y, p.z + orbit_r * angle.sin()),
            velocity:    Vec3::default(),
            radius:      body_r,
            mat,
            restitution: 0.0,
            is_static:   false,
            orbit: Some(Orbit { parent_idx: Some(parent_idx), radius: orbit_r, speed, angle, inclination: 0.0 }),
            axial_angle: 0.0,
            axial_speed: 0.0,
            axial_tilt:  0.0,
            ring:        None,
        });
    };

    moon(2, 3.2, 0.00467, 0.0,  0.42, Arc::new(Lambertian { texture: Color::new(0.72, 0.72, 0.70).into() })); // Moon     (~22.5 min)
    moon(4, 6.5, 0.0720,  0.0,  0.50, Arc::new(Lambertian { texture: Color::new(0.90, 0.82, 0.30).into() })); // Io       (~1.5 min)
    moon(4, 9.0, 0.0359,  2.1,  0.44, Arc::new(Lambertian { texture: Color::new(0.88, 0.90, 0.92).into() })); // Europa   (~2.9 min)
    moon(5, 6.8, 0.00799, 1.0,  0.52, Arc::new(Lambertian { texture: Color::new(0.80, 0.62, 0.35).into() })); // Titan    (~13 min)
    moon(7, 4.2, 0.0217,  3.0,  0.36, Arc::new(Lambertian { texture: Color::new(0.78, 0.72, 0.75).into() })); // Triton   (~4.8 min)

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
            pos:        Point3::new(0.0, 140.0, 110.0),
            lookat:     Point3::new(0.0,   0.0,   0.0),
            vfov:       55.0,
            aperture:   0.0,
            focus_dist: 10.0,
            move_speed: 3.0,
        },
        static_objects,
        dynamic,
        bounds:    None,
        colliders: vec![],
        gravity:     0.0,
        settled:     false,
        paused:      false,
        max_samples: 4000,
        // Large physics_dt lets the path tracer accumulate many samples
        // between each planet-position update (planets barely move per tick).
        physics_dt:  Duration::from_millis(500),
    }
}
