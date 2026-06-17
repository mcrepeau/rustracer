use std::sync::Arc;

use crate::bvh::BvhTree;
use crate::hittable::{Hittable, HittableList, Material};
use crate::triangle::Triangle;
use crate::vec3::{Point3, Vec3};

/// Load an OBJ file and return a BVH over its triangles, all sharing `mat`.
///
/// Vertex normals and UV coordinates are used when present in the file;
/// otherwise face normals are computed and UVs default to barycentric coords.
/// All meshes in the file are merged into a single BVH.
pub fn load_obj(path: &str, mat: Arc<dyn Material>) -> Result<Arc<dyn Hittable>, String> {
    let (models, _) = tobj::load_obj(path, &tobj::GPU_LOAD_OPTIONS)
        .map_err(|e| format!("cannot load OBJ '{path}': {e}"))?;

    let mut list = HittableList::new();

    for model in &models {
        let mesh = &model.mesh;
        let pos  = &mesh.positions;   // flat [x,y,z, x,y,z, ...]
        let nor  = &mesh.normals;     // flat [nx,ny,nz, ...], may be empty
        let tex  = &mesh.texcoords;   // flat [u,v, ...], may be empty
        let idx  = &mesh.indices;

        let has_normals   = !nor.is_empty();
        let has_texcoords = !tex.is_empty();

        let get_pos = |i: usize| Point3::new(pos[3*i], pos[3*i+1], pos[3*i+2]);
        let get_nor = |i: usize| Vec3::new(nor[3*i], nor[3*i+1], nor[3*i+2]);
        let get_uv  = |i: usize| (tex[2*i], tex[2*i+1]);

        // Accumulate per-vertex tangents from UV-space derivatives (dPos/dU).
        let n_verts = pos.len() / 3;
        let mut tangent_sum = vec![Vec3::default(); n_verts];
        if has_texcoords {
            for chunk in idx.chunks(3) {
                let (i0, i1, i2) = (chunk[0] as usize, chunk[1] as usize, chunk[2] as usize);
                let e1 = get_pos(i1) - get_pos(i0);
                let e2 = get_pos(i2) - get_pos(i0);
                let (du1, dv1) = { let uv = get_uv(i1); let uv0 = get_uv(i0); (uv.0 - uv0.0, uv.1 - uv0.1) };
                let (du2, dv2) = { let uv = get_uv(i2); let uv0 = get_uv(i0); (uv.0 - uv0.0, uv.1 - uv0.1) };
                let denom = du1 * dv2 - dv1 * du2;
                if denom.abs() < 1e-8 { continue; }
                let t = (e1 * dv2 - e2 * dv1) * (1.0 / denom);
                tangent_sum[i0] = tangent_sum[i0] + t;
                tangent_sum[i1] = tangent_sum[i1] + t;
                tangent_sum[i2] = tangent_sum[i2] + t;
            }
        }
        let get_tan = |i: usize| {
            let t = tangent_sum[i];
            if t.length_squared() > 1e-8 { t.unit() } else { Vec3::new(1.0, 0.0, 0.0) }
        };

        for chunk in idx.chunks(3) {
            let (i0, i1, i2) = (chunk[0] as usize, chunk[1] as usize, chunk[2] as usize);
            let v0 = get_pos(i0);
            let v1 = get_pos(i1);
            let v2 = get_pos(i2);

            // Degenerate triangles (zero-area) produce a zero cross product;
            // skip them so unit() doesn't produce NaN.
            let face_n_raw = (v1 - v0).cross(v2 - v0);
            if face_n_raw.length_squared() < 1e-16 { continue; }
            let face_n = face_n_raw.unit();

            let n0 = if has_normals { get_nor(i0) } else { face_n };
            let n1 = if has_normals { get_nor(i1) } else { face_n };
            let n2 = if has_normals { get_nor(i2) } else { face_n };

            let uv0 = if has_texcoords { get_uv(i0) } else { (0.0, 0.0) };
            let uv1 = if has_texcoords { get_uv(i1) } else { (1.0, 0.0) };
            let uv2 = if has_texcoords { get_uv(i2) } else { (0.5, 1.0) };

            let t0 = get_tan(i0);
            let t1 = get_tan(i1);
            let t2 = get_tan(i2);

            list.add(Triangle::new(v0, v1, v2, n0, n1, n2, t0, t1, t2, uv0, uv1, uv2, Arc::clone(&mat)));
        }
    }

    if list.objects.is_empty() {
        return Err(format!("OBJ file '{path}' contains no triangles"));
    }

    Ok(Arc::new(BvhTree::from_list(list)))
}
