use rand::Rng;
use crate::vec3::{Point3, Vec3};

pub struct Perlin {
    ranvec: Box<[Vec3; 256]>,
    perm_x: Box<[usize; 256]>,
    perm_y: Box<[usize; 256]>,
    perm_z: Box<[usize; 256]>,
}

fn generate_perm(rng: &mut impl Rng) -> Box<[usize; 256]> {
    let mut p = Box::new([0usize; 256]);
    for i in 0..256 { p[i] = i; }
    for i in (1..256).rev() {
        let j = rng.gen_range(0..=i);
        p.swap(i, j);
    }
    p
}

impl Perlin {
    pub fn new(rng: &mut impl Rng) -> Self {
        let mut ranvec = Box::new([Vec3::default(); 256]);
        for v in ranvec.iter_mut() {
            *v = Vec3::random_range(-1.0, 1.0, rng).unit();
        }
        Self {
            ranvec,
            perm_x: generate_perm(rng),
            perm_y: generate_perm(rng),
            perm_z: generate_perm(rng),
        }
    }

    pub fn noise(&self, p: Point3) -> f32 {
        let u = p.x - p.x.floor();
        let v = p.y - p.y.floor();
        let w = p.z - p.z.floor();
        let u = u * u * (3.0 - 2.0 * u);
        let v = v * v * (3.0 - 2.0 * v);
        let w = w * w * (3.0 - 2.0 * w);

        let i = p.x.floor() as i32;
        let j = p.y.floor() as i32;
        let k = p.z.floor() as i32;

        let mut c = [[[Vec3::default(); 2]; 2]; 2];
        for di in 0..2i32 {
            for dj in 0..2i32 {
                for dk in 0..2i32 {
                    let idx = self.perm_x[((i + di) & 255) as usize]
                            ^ self.perm_y[((j + dj) & 255) as usize]
                            ^ self.perm_z[((k + dk) & 255) as usize];
                    c[di as usize][dj as usize][dk as usize] = self.ranvec[idx];
                }
            }
        }
        trilinear_interp(c, u, v, w)
    }

    pub fn turb(&self, p: Point3, depth: u32) -> f32 {
        let mut accum  = 0.0f32;
        let mut temp_p = p;
        let mut weight = 1.0f32;
        for _ in 0..depth {
            accum  += weight * self.noise(temp_p);
            weight *= 0.5;
            temp_p  = Point3::new(temp_p.x * 2.0, temp_p.y * 2.0, temp_p.z * 2.0);
        }
        accum.abs()
    }
}

#[allow(clippy::needless_range_loop)]
fn trilinear_interp(c: [[[Vec3; 2]; 2]; 2], u: f32, v: f32, w: f32) -> f32 {
    let mut accum = 0.0f32;
    for i in 0..2usize {
        for j in 0..2usize {
            for k in 0..2usize {
                let fi = i as f32;
                let fj = j as f32;
                let fk = k as f32;
                let weight = Vec3::new(u - fi, v - fj, w - fk);
                accum += (fi * u + (1.0 - fi) * (1.0 - u))
                       * (fj * v + (1.0 - fj) * (1.0 - v))
                       * (fk * w + (1.0 - fk) * (1.0 - w))
                       * c[i][j][k].dot(weight);
            }
        }
    }
    accum
}
