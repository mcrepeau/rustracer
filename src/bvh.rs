use std::sync::Arc;
use crate::aabb::Aabb;
use crate::hittable::{HitRecord, Hittable, HittableList};
use crate::ray::Ray;

const NUM_BUCKETS: usize = 12;

// 8 children per node when AVX2 is available, 4 otherwise.
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
const QBVH_WIDTH: usize = 8;
#[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
const QBVH_WIDTH: usize = 4;

// SAH split depth to reach QBVH_WIDTH leaves: log2(WIDTH).
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
const QBVH_DEPTH: usize = 3;
#[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
const QBVH_DEPTH: usize = 2;

type ObjsBoxes = (Vec<Arc<dyn Hittable>>, Vec<Aabb>);

// N-wide BVH node (N = QBVH_WIDTH). AABBs stored SoA ([axis][child]) for SIMD intersection.
struct QbvhNode {
    min:      [[f32; QBVH_WIDTH]; 3],
    max:      [[f32; QBVH_WIDTH]; 3],
    // ≥ 0 → inner node at nodes[v]
    // < 0, ≠ MIN → leaf primitive at objects[(-v - 1)]
    // MIN → unused slot
    children: [i32; QBVH_WIDTH],
}

pub struct BvhTree {
    nodes:   Vec<QbvhNode>,
    objects: Vec<Arc<dyn Hittable>>,
    bbox:    Aabb,
}

fn surface_area(b: &Aabb) -> f32 {
    let d = b.max - b.min;
    2.0 * (d.x * d.y + d.y * d.z + d.z * d.x)
}

fn sah_partition(
    objs:  &[Arc<dyn Hittable>],
    boxes: &[Aabb],
) -> (ObjsBoxes, ObjsBoxes) {
    let span = objs.len();
    debug_assert!(span >= 2);

    let mut c_min = [f32::INFINITY;     3];
    let mut c_max = [f32::NEG_INFINITY; 3];
    for b in boxes {
        for axis in 0..3 {
            let c = (b.min[axis] + b.max[axis]) * 0.5;
            c_min[axis] = c_min[axis].min(c);
            c_max[axis] = c_max[axis].max(c);
        }
    }

    let node_bbox = boxes.iter().copied().reduce(|a, b| Aabb::surrounding(&a, &b)).unwrap();
    let node_sa   = surface_area(&node_bbox);

    let mut best_cost  = f32::INFINITY;
    let mut best_axis  = 0usize;
    let mut best_split = 0usize;

    for axis in 0..3usize {
        let extent = c_max[axis] - c_min[axis];
        if extent < 1e-6 { continue; }

        let mut bucket_bbox  = [None::<Aabb>; NUM_BUCKETS];
        let mut bucket_count = [0u32; NUM_BUCKETS];
        for b in boxes {
            let c  = (b.min[axis] + b.max[axis]) * 0.5;
            let t  = (c - c_min[axis]) / extent;
            let bi = ((NUM_BUCKETS as f32 * t) as usize).min(NUM_BUCKETS - 1);
            bucket_count[bi] += 1;
            bucket_bbox[bi]   = Some(match bucket_bbox[bi] {
                None    => *b,
                Some(a) => Aabb::surrounding(&a, b),
            });
        }

        // Prefix sweep: left-side surface area and count for each split candidate.
        let mut left_sa    = [0.0f32; NUM_BUCKETS];
        let mut left_count = [0u32;   NUM_BUCKETS];
        let mut acc_box: Option<Aabb> = None;
        let mut acc_n = 0u32;
        for i in 0..NUM_BUCKETS {
            acc_n += bucket_count[i];
            if let Some(b) = bucket_bbox[i] {
                acc_box = Some(match acc_box { None => b, Some(a) => Aabb::surrounding(&a, &b) });
            }
            left_count[i] = acc_n;
            left_sa[i]    = acc_box.map_or(0.0, |b| surface_area(&b));
        }

        // Single backward pass evaluates the right side without suffix arrays.
        acc_box = None;
        acc_n   = 0;
        for split in (0..NUM_BUCKETS - 1).rev() {
            acc_n += bucket_count[split + 1];
            if let Some(b) = bucket_bbox[split + 1] {
                acc_box = Some(match acc_box { None => b, Some(a) => Aabb::surrounding(&a, &b) });
            }
            let nl = left_count[split];
            let nr = acc_n;
            if nl == 0 || nr == 0 { continue; }
            let right_sa = acc_box.map_or(0.0, |b| surface_area(&b));
            let cost = 1.0 + (left_sa[split] * nl as f32 + right_sa * nr as f32) / node_sa;
            if cost < best_cost { best_cost = cost; best_axis = axis; best_split = split; }
        }
    }

    let extent = c_max[best_axis] - c_min[best_axis];
    let bucket_id = |b: &Aabb| -> usize {
        if extent < 1e-6 { return 0; }
        let c = (b.min[best_axis] + b.max[best_axis]) * 0.5;
        let t = (c - c_min[best_axis]) / extent;
        ((NUM_BUCKETS as f32 * t) as usize).min(NUM_BUCKETS - 1)
    };

    // Sort indices by bucket; no Arc clones until the final gather.
    let bid: Vec<usize> = boxes.iter().map(bucket_id).collect();
    let mut order: Vec<usize> = (0..span).collect();
    order.sort_by_key(|&i| bid[i]);
    let mid_off = order.partition_point(|&i| bid[i] <= best_split);
    let mid = if mid_off == 0 || mid_off == span { span / 2 } else { mid_off };

    let (li, ri) = order.split_at(mid);
    let lo: Vec<Arc<dyn Hittable>> = li.iter().map(|&i| Arc::clone(&objs[i])).collect();
    let lb: Vec<Aabb>               = li.iter().map(|&i| boxes[i]).collect();
    let ro: Vec<Arc<dyn Hittable>> = ri.iter().map(|&i| Arc::clone(&objs[i])).collect();
    let rb: Vec<Aabb>               = ri.iter().map(|&i| boxes[i]).collect();
    ((lo, lb), (ro, rb))
}

// Recursively split to produce up to 2^max_depth groups for a single QBVH node.
fn split_recursive(
    objs:      &[Arc<dyn Hittable>],
    boxes:     &[Aabb],
    depth:     usize,
    max_depth: usize,
    out:       &mut Vec<ObjsBoxes>,
) {
    if depth == max_depth || objs.len() <= 1 {
        out.push((objs.iter().map(Arc::clone).collect(), boxes.to_vec()));
        return;
    }
    let ((lo, lb), (ro, rb)) = sah_partition(objs, boxes);
    split_recursive(&lo, &lb, depth + 1, max_depth, out);
    split_recursive(&ro, &rb, depth + 1, max_depth, out);
}

// Produce up to QBVH_WIDTH SAH groups for one QBVH node.
fn split_n(objs: &[Arc<dyn Hittable>], boxes: &[Aabb]) -> Vec<ObjsBoxes> {
    if objs.len() <= QBVH_WIDTH {
        return objs.iter().zip(boxes.iter())
            .map(|(o, b)| (vec![Arc::clone(o)], vec![*b]))
            .collect();
    }
    let mut groups = Vec::with_capacity(QBVH_WIDTH);
    split_recursive(objs, boxes, 0, QBVH_DEPTH, &mut groups);
    groups
}

impl QbvhNode {
    fn from_children(children: [i32; QBVH_WIDTH], bboxes: &[Aabb; QBVH_WIDTH]) -> Self {
        let mut min = [[0.0f32; QBVH_WIDTH]; 3];
        let mut max = [[0.0f32; QBVH_WIDTH]; 3];
        for c in 0..QBVH_WIDTH {
            for axis in 0..3 {
                min[axis][c] = bboxes[c].min[axis];
                max[axis][c] = bboxes[c].max[axis];
            }
        }
        Self { min, max, children }
    }

    fn sentinel() -> Self {
        Self {
            min:      [[0.0; QBVH_WIDTH]; 3],
            max:      [[0.0; QBVH_WIDTH]; 3],
            children: [i32::MIN; QBVH_WIDTH],
        }
    }
}

impl BvhTree {
    pub fn from_list(list: HittableList) -> Self {
        let objs = list.objects;
        assert!(!objs.is_empty(), "BvhTree::from_list requires at least one object");
        let boxes: Vec<Aabb> = objs.iter()
            .map(|o| o.bounding_box().unwrap_or_default())
            .collect();
        let mut nodes   = Vec::with_capacity(objs.len());
        let mut objects = Vec::with_capacity(objs.len());
        let (root, bbox) = Self::build(&objs, &boxes, &mut nodes, &mut objects);
        if root < 0 {
            let mut children = [i32::MIN; QBVH_WIDTH];
            children[0] = root;
            let mut bboxes = [Aabb::default(); QBVH_WIDTH];
            bboxes[0] = boxes[0];
            nodes.push(QbvhNode::from_children(children, &bboxes));
        }
        Self { nodes, objects, bbox }
    }

    fn build(
        objs:  &[Arc<dyn Hittable>],
        boxes: &[Aabb],
        nodes: &mut Vec<QbvhNode>,
        prims: &mut Vec<Arc<dyn Hittable>>,
    ) -> (i32, Aabb) {
        if objs.len() == 1 {
            let idx = prims.len() as i32;
            prims.push(Arc::clone(&objs[0]));
            return (-(idx + 1), boxes[0]);
        }

        let groups = split_n(objs, boxes);

        let my_idx = nodes.len();
        nodes.push(QbvhNode::sentinel());

        let mut children = [i32::MIN; QBVH_WIDTH];
        let mut bboxes   = [Aabb::default(); QBVH_WIDTH];
        let mut node_bbox: Option<Aabb> = None;

        for (i, (g_objs, g_boxes)) in groups.into_iter().enumerate() {
            let (child_data, child_bbox) = Self::build(&g_objs, &g_boxes, nodes, prims);
            children[i] = child_data;
            bboxes[i]   = child_bbox;
            node_bbox   = Some(match node_bbox {
                None    => child_bbox,
                Some(b) => Aabb::surrounding(&b, &child_bbox),
            });
        }

        nodes[my_idx] = QbvhNode::from_children(children, &bboxes);
        (my_idx as i32, node_bbox.unwrap())
    }
}

// Test all 8 child AABBs simultaneously using AVX2.
// Returns (hit_mask, t_near) where bit c = 1 means child c was hit.
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[target_feature(enable = "avx,avx2")]
unsafe fn test_children_avx2(
    node:  &QbvhNode,
    ro:    [f32; 3],
    id:    [f32; 3],
    t_min: f32,
    t_max: f32,
) -> (u32, [f32; 8]) {
    use std::arch::x86_64::*;

    // Build validity mask: bit c = 1 where children[c] != i32::MIN.
    let ch       = _mm256_loadu_si256(node.children.as_ptr() as *const __m256i);
    let sentinel = _mm256_set1_epi32(i32::MIN);
    let invalid  = _mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpeq_epi32(ch, sentinel))) as u32;
    let valid    = (!invalid) & 0xff;

    // Slab test: all 8 children in parallel, 3 axes.
    let mut t0 = _mm256_set1_ps(t_min);
    let mut t1 = _mm256_set1_ps(t_max);
    for axis in 0..3usize {
        let orig = _mm256_set1_ps(ro[axis]);
        let inv  = _mm256_set1_ps(id[axis]);
        let da   = _mm256_mul_ps(_mm256_sub_ps(_mm256_loadu_ps(node.min[axis].as_ptr()), orig), inv);
        let db   = _mm256_mul_ps(_mm256_sub_ps(_mm256_loadu_ps(node.max[axis].as_ptr()), orig), inv);
        t0 = _mm256_max_ps(t0, _mm256_min_ps(da, db));
        t1 = _mm256_min_ps(t1, _mm256_max_ps(da, db));
    }

    // _CMP_LE_OQ = 18: t0 <= t1, ordered quiet (no FP exception on NaN)
    let hit      = _mm256_cmp_ps::<18>(t0, t1);
    let hit_mask = _mm256_movemask_ps(hit) as u32;

    let mut t_near = [0.0f32; 8];
    _mm256_storeu_ps(t_near.as_mut_ptr(), t0);
    (hit_mask & valid, t_near)
}

// Test all 4 child AABBs simultaneously using SSE2 (x86_64 without AVX2).
// Returns (hit_mask, t_near) where bit c of hit_mask = 1 means child c was hit.
#[cfg(all(target_arch = "x86_64", not(target_feature = "avx2")))]
#[target_feature(enable = "sse2")]
unsafe fn test_children_sse(
    node:  &QbvhNode,
    ro:    [f32; 3],
    id:    [f32; 3],
    t_min: f32,
    t_max: f32,
) -> (u32, [f32; 4]) {
    use std::arch::x86_64::*;

    // Build validity mask: bit c = 1 where children[c] != i32::MIN.
    let ch       = _mm_loadu_si128(node.children.as_ptr() as *const __m128i);
    let sentinel = _mm_set1_epi32(i32::MIN);
    let invalid  = _mm_movemask_ps(_mm_castsi128_ps(_mm_cmpeq_epi32(ch, sentinel))) as u32;
    let valid    = (!invalid) & 0xf;

    // Slab test: all 4 children in parallel, 3 axes.
    let mut t0 = _mm_set1_ps(t_min);
    let mut t1 = _mm_set1_ps(t_max);
    for axis in 0..3usize {
        let orig = _mm_set1_ps(ro[axis]);
        let inv  = _mm_set1_ps(id[axis]);
        let da = _mm_mul_ps(_mm_sub_ps(_mm_loadu_ps(node.min[axis].as_ptr()), orig), inv);
        let db = _mm_mul_ps(_mm_sub_ps(_mm_loadu_ps(node.max[axis].as_ptr()), orig), inv);
        t0 = _mm_max_ps(t0, _mm_min_ps(da, db));
        t1 = _mm_min_ps(t1, _mm_max_ps(da, db));
    }

    // hit when t0 <= t1
    let hit_mask = _mm_movemask_ps(_mm_cmple_ps(t0, t1)) as u32;

    let mut t_near = [0.0f32; 4];
    _mm_storeu_ps(t_near.as_mut_ptr(), t0);
    (hit_mask & valid, t_near)
}

// Test all 4 child AABBs simultaneously using ARM NEON.
// Returns (hit_mask, t_near) where bit c of hit_mask = 1 means child c was hit.
// NEON has no movemask: AND each comparison lane with its bit-weight (1/2/4/8)
// then horizontally sum to collapse into a 4-bit integer.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn test_children_neon(
    node:  &QbvhNode,
    ro:    [f32; 3],
    id:    [f32; 3],
    t_min: f32,
    t_max: f32,
) -> (u32, [f32; 4]) {
    use std::arch::aarch64::*;

    // Build validity mask: bit c = 1 where children[c] != i32::MIN.
    let ch        = vld1q_s32(node.children.as_ptr());
    let sentinel  = vdupq_n_s32(i32::MIN);
    let eq_mask   = vceqq_s32(ch, sentinel);     // 0xFFFF…=sentinel=invalid
    let lane_bits = [1u32, 2, 4, 8];
    let weights   = vld1q_u32(lane_bits.as_ptr());
    let invalid   = vaddvq_u32(vandq_u32(eq_mask, weights));
    let valid     = (!invalid) & 0xf;

    // Slab test: all 4 children in parallel, 3 axes.
    let mut t0 = vdupq_n_f32(t_min);
    let mut t1 = vdupq_n_f32(t_max);
    for axis in 0..3usize {
        let orig = vdupq_n_f32(ro[axis]);
        let inv  = vdupq_n_f32(id[axis]);
        let da   = vmulq_f32(vsubq_f32(vld1q_f32(node.min[axis].as_ptr()), orig), inv);
        let db   = vmulq_f32(vsubq_f32(vld1q_f32(node.max[axis].as_ptr()), orig), inv);
        t0 = vmaxq_f32(t0, vminq_f32(da, db));
        t1 = vminq_f32(t1, vmaxq_f32(da, db));
    }

    // hit when t0 <= t1
    let hit_mask = vaddvq_u32(vandq_u32(vcleq_f32(t0, t1), weights));

    let mut t_near = [0.0f32; 4];
    vst1q_f32(t_near.as_mut_ptr(), t0);
    (hit_mask & valid, t_near)
}

// Scalar fallback for targets without SIMD support.
#[allow(dead_code, clippy::needless_range_loop)]
fn test_children_scalar(
    node:  &QbvhNode,
    ro:    [f32; 3],
    id:    [f32; 3],
    t_min: f32,
    t_max: f32,
) -> (u32, [f32; QBVH_WIDTH]) {
    let mut mask   = 0u32;
    let mut t_near = [0.0f32; QBVH_WIDTH];
    for c in 0..QBVH_WIDTH {
        if node.children[c] == i32::MIN { continue; }
        let mut t0 = t_min;
        let mut t1 = t_max;
        for axis in 0..3 {
            let da = (node.min[axis][c] - ro[axis]) * id[axis];
            let db = (node.max[axis][c] - ro[axis]) * id[axis];
            t0 = t0.max(da.min(db));
            t1 = t1.min(da.max(db));
        }
        if t0 <= t1 { mask |= 1 << c; t_near[c] = t0; }
    }
    (mask, t_near)
}

impl Hittable for BvhTree {
    #[allow(clippy::needless_range_loop)]
    fn hit(&self, r: &Ray, t_min: f32, t_max: f32) -> Option<HitRecord<'_>> {
        if self.nodes.is_empty() { return None; }

        let mut stack = [0i32; 128];
        let mut top   = 1usize;
        let mut best: Option<HitRecord<'_>> = None;
        let mut closest = t_max;

        let ro = [r.origin.x,  r.origin.y,  r.origin.z];
        let id = [r.inv_dir.x, r.inv_dir.y, r.inv_dir.z];

        while top > 0 {
            top -= 1;
            let entry = stack[top];

            if entry < 0 {
                let prim_idx = (-entry - 1) as usize;
                if let Some(rec) = self.objects[prim_idx].hit(r, t_min, closest) {
                    closest = rec.t;
                    best    = Some(rec);
                }
                continue;
            }

            let node = &self.nodes[entry as usize];

            // N-wide AABB test: compile-time selected SIMD path.
            #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
            let (mask, t_near) = unsafe { test_children_avx2(node, ro, id, t_min, closest) };
            #[cfg(all(target_arch = "x86_64", not(target_feature = "avx2")))]
            let (mask, t_near) = unsafe { test_children_sse(node, ro, id, t_min, closest) };
            #[cfg(target_arch = "aarch64")]
            let (mask, t_near) = unsafe { test_children_neon(node, ro, id, t_min, closest) };
            #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
            let (mask, t_near) = test_children_scalar(node, ro, id, t_min, closest);

            if mask == 0 { continue; }

            // Collect hits sorted far-to-near so nearest is popped first.
            let mut hits   = [(f32::INFINITY, i32::MIN); QBVH_WIDTH];
            let mut n_hits = 0usize;
            for c in 0..QBVH_WIDTH {
                if mask & (1 << c) != 0 {
                    hits[n_hits] = (t_near[c], node.children[c]);
                    n_hits += 1;
                }
            }
            hits[..n_hits].sort_unstable_by(|a, b| b.0.total_cmp(&a.0));

            for &(_, child) in &hits[..n_hits] {
                debug_assert!(top < stack.len(), "BVH traversal stack overflow at depth {top}");
                stack[top] = child;
                top += 1;
            }
        }

        best
    }

    fn any_hit(&self, r: &Ray, t_min: f32, t_max: f32) -> bool {
        if self.nodes.is_empty() { return false; }

        let mut stack = [0i32; 128];
        let mut top   = 1usize;

        let ro = [r.origin.x,  r.origin.y,  r.origin.z];
        let id = [r.inv_dir.x, r.inv_dir.y, r.inv_dir.z];

        while top > 0 {
            top -= 1;
            let entry = stack[top];

            if entry < 0 {
                if self.objects[(-entry - 1) as usize].any_hit(r, t_min, t_max) {
                    return true;
                }
                continue;
            }

            let node = &self.nodes[entry as usize];

            // N-wide AABB test — same SIMD paths as hit(), t_max is fixed (no shrinkage).
            #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
            let (mask, _) = unsafe { test_children_avx2(node, ro, id, t_min, t_max) };
            #[cfg(all(target_arch = "x86_64", not(target_feature = "avx2")))]
            let (mask, _) = unsafe { test_children_sse(node, ro, id, t_min, t_max) };
            #[cfg(target_arch = "aarch64")]
            let (mask, _) = unsafe { test_children_neon(node, ro, id, t_min, t_max) };
            #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
            let (mask, _) = test_children_scalar(node, ro, id, t_min, t_max);

            if mask == 0 { continue; }

            // No sort needed — any order is fine when one hit is enough.
            for c in 0..QBVH_WIDTH {
                if mask & (1 << c) != 0 {
                    debug_assert!(top < stack.len(), "BVH traversal stack overflow at depth {top}");
                    stack[top] = node.children[c];
                    top += 1;
                }
            }
        }

        false
    }

    fn bounding_box(&self) -> Option<Aabb> {
        if self.nodes.is_empty() { None } else { Some(self.bbox) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vec3::{Point3, Vec3};
    use crate::hittable::{HitRecord, HittableList, Material, ScatterRecord};
    use crate::ray::Ray;

    struct DummyMat;
    impl Material for DummyMat {
        fn scatter(&self, _: &Ray, _: &HitRecord<'_>, _: &mut dyn rand::RngCore) -> Option<ScatterRecord> { None }
    }

    fn unit_sphere_at(center: Point3) -> crate::sphere::Sphere {
        crate::sphere::Sphere::new(center, 1.0, Arc::new(DummyMat))
    }

    fn ray_from(origin: Point3, direction: Vec3) -> Ray {
        Ray::new(origin, direction)
    }

    #[test]
    fn bvh_hits_sphere_on_axis() {
        let mut list = HittableList::new();
        list.add(unit_sphere_at(Point3::new(0.0, 0.0, 0.0)));
        let bvh = BvhTree::from_list(list);
        let r   = ray_from(Point3::new(0.0, 0.0, -5.0), Vec3::new(0.0, 0.0, 1.0));
        assert!(bvh.hit(&r, 0.001, f32::INFINITY).is_some(), "ray through sphere center should hit");
    }

    #[test]
    fn bvh_misses_ray_beside_sphere() {
        let mut list = HittableList::new();
        list.add(unit_sphere_at(Point3::new(0.0, 0.0, 0.0)));
        let bvh = BvhTree::from_list(list);
        // Ray aimed 2 units to the side — clears the radius-1 sphere.
        let r = ray_from(Point3::new(2.0, 0.0, -5.0), Vec3::new(0.0, 0.0, 1.0));
        assert!(bvh.hit(&r, 0.001, f32::INFINITY).is_none(), "ray beside sphere should miss");
    }

    #[test]
    fn bvh_any_hit_matches_hit() {
        let mut list = HittableList::new();
        list.add(unit_sphere_at(Point3::new(0.0, 0.0, 0.0)));
        let bvh   = BvhTree::from_list(list);
        let hit   = ray_from(Point3::new(0.0, 0.0, -5.0), Vec3::new(0.0, 0.0, 1.0));
        let miss  = ray_from(Point3::new(2.0, 0.0, -5.0), Vec3::new(0.0, 0.0, 1.0));
        assert!( bvh.any_hit(&hit,  0.001, f32::INFINITY), "any_hit should return true through sphere");
        assert!(!bvh.any_hit(&miss, 0.001, f32::INFINITY), "any_hit should return false beside sphere");
    }

    #[test]
    fn bvh_finds_nearest_of_multiple_spheres() {
        // Two spheres along the Z axis: one at z=-3, one at z=3.
        // Ray from z=-10 along +Z should hit the nearer one first (t ≈ 6).
        let mut list = HittableList::new();
        list.add(unit_sphere_at(Point3::new(0.0, 0.0, -3.0)));
        list.add(unit_sphere_at(Point3::new(0.0, 0.0,  3.0)));
        let bvh = BvhTree::from_list(list);
        let r   = ray_from(Point3::new(0.0, 0.0, -10.0), Vec3::new(0.0, 0.0, 1.0));
        let rec = bvh.hit(&r, 0.001, f32::INFINITY).expect("should hit at least one sphere");
        assert!(rec.t < 8.0, "nearest sphere should be hit first, t = {}", rec.t);
    }

    #[test]
    fn bvh_all_spheres_in_grid_are_reachable() {
        // Build a 3×3 grid of spheres spaced 10 units apart and verify each is hittable.
        let mut list = HittableList::new();
        let positions: Vec<Point3> = (-1..=1).flat_map(|x| (-1..=1).map(move |y| {
            Point3::new(x as f32 * 10.0, y as f32 * 10.0, 0.0)
        })).collect();
        for &p in &positions {
            list.add(unit_sphere_at(p));
        }
        let bvh = BvhTree::from_list(list);
        for p in positions {
            let r = ray_from(Point3::new(p.x, p.y, -20.0), Vec3::new(0.0, 0.0, 1.0));
            assert!(
                bvh.hit(&r, 0.001, f32::INFINITY).is_some(),
                "sphere at ({}, {}) should be reachable", p.x, p.y,
            );
        }
    }
}
