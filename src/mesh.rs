use std::sync::Arc;
use crate::aabb::Aabb;
use crate::vec3::{Point3, Vec3};
use crate::ray::Ray;
use crate::hittable::{HitRecord, Hittable, HittableList, Material};

// ── Triangle ──────────────────────────────────────────────────────────────────

pub struct Triangle {
    v:   [Point3; 3],
    n:   [Vec3; 3],         // per-vertex normals (interpolated at hit)
    uv:  [(f32, f32); 3],
    mat: Arc<dyn Material>,
}

impl Triangle {
    pub fn new(v: [Point3; 3], n: [Vec3; 3], uv: [(f32, f32); 3], mat: Arc<dyn Material>) -> Self {
        Self { v, n, uv, mat }
    }
}

impl Hittable for Triangle {
    fn hit(&self, r: &Ray, t_min: f32, t_max: f32) -> Option<HitRecord<'_>> {
        let e1 = self.v[1] - self.v[0];
        let e2 = self.v[2] - self.v[0];
        let h  = r.direction.cross(e2);
        let a  = e1.dot(h);
        if a.abs() < 1e-8 { return None; }   // ray parallel to triangle

        let f = 1.0 / a;
        let s = r.origin - self.v[0];
        let u = f * s.dot(h);
        if !(0.0..=1.0).contains(&u) { return None; }

        let q = s.cross(e1);
        let v = f * r.direction.dot(q);
        if v < 0.0 || u + v > 1.0 { return None; }

        let t = f * e2.dot(q);
        if t < t_min || t > t_max { return None; }

        // Barycentric weights: w for v[0], u for v[1], v for v[2]
        let w = 1.0 - u - v;
        let normal  = (w * self.n[0]    + u * self.n[1]    + v * self.n[2]).unit();
        let tex_u   =  w * self.uv[0].0 + u * self.uv[1].0 + v * self.uv[2].0;
        let tex_v   =  w * self.uv[0].1 + u * self.uv[1].1 + v * self.uv[2].1;

        let mut rec = HitRecord::new(r.at(t), t, &*self.mat, r, normal);
        rec.u = tex_u;
        rec.v = tex_v;
        Some(rec)
    }

    fn bounding_box(&self) -> Option<Aabb> {
        let min = Point3::new(
            self.v[0].x.min(self.v[1].x).min(self.v[2].x),
            self.v[0].y.min(self.v[1].y).min(self.v[2].y),
            self.v[0].z.min(self.v[1].z).min(self.v[2].z),
        );
        let max = Point3::new(
            self.v[0].x.max(self.v[1].x).max(self.v[2].x),
            self.v[0].y.max(self.v[1].y).max(self.v[2].y),
            self.v[0].z.max(self.v[1].z).max(self.v[2].z),
        );
        Some(Aabb::new(min, max).pad())
    }
}

// ── OBJ loader ────────────────────────────────────────────────────────────────

/// Load an OBJ file. All vertex positions are multiplied by `scale`.
/// Returns a flat HittableList of triangles — wrap in BvhTree before use.
pub fn load_obj(
    path: &str,
    scale: f32,
    mat: Arc<dyn Material>,
) -> Result<HittableList, tobj::LoadError> {
    let (models, _) = tobj::load_obj(path, &tobj::LoadOptions {
        triangulate:  true,
        single_index: true,
        ..Default::default()
    })?;

    let mut list = HittableList::new();
    let mut tri_count = 0usize;

    for model in &models {
        let mesh = &model.mesh;
        let pos  = &mesh.positions;
        let nrm  = &mesh.normals;
        let tex  = &mesh.texcoords;

        for chunk in mesh.indices.chunks_exact(3) {
            let (i0, i1, i2) = (chunk[0] as usize, chunk[1] as usize, chunk[2] as usize);

            let v = [
                Point3::new(pos[3*i0] * scale, pos[3*i0+1] * scale, pos[3*i0+2] * scale),
                Point3::new(pos[3*i1] * scale, pos[3*i1+1] * scale, pos[3*i1+2] * scale),
                Point3::new(pos[3*i2] * scale, pos[3*i2+1] * scale, pos[3*i2+2] * scale),
            ];

            let max_idx = i0.max(i1).max(i2);
            let n = if nrm.len() >= 3 * (max_idx + 1) {
                [
                    Vec3::new(nrm[3*i0], nrm[3*i0+1], nrm[3*i0+2]),
                    Vec3::new(nrm[3*i1], nrm[3*i1+1], nrm[3*i1+2]),
                    Vec3::new(nrm[3*i2], nrm[3*i2+1], nrm[3*i2+2]),
                ]
            } else {
                let face_n = (v[1] - v[0]).cross(v[2] - v[0]).unit();
                [face_n; 3]
            };

            let uv = if tex.len() >= 2 * (max_idx + 1) {
                [
                    (tex[2*i0], tex[2*i0+1]),
                    (tex[2*i1], tex[2*i1+1]),
                    (tex[2*i2], tex[2*i2+1]),
                ]
            } else {
                [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]
            };

            list.add(Triangle::new(v, n, uv, Arc::clone(&mat)));
            tri_count += 1;
        }
    }

    println!("  {tri_count} triangles from {path}");
    Ok(list)
}
