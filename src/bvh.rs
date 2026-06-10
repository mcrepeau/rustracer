use std::sync::Arc;
use crate::aabb::Aabb;
use crate::hittable::{HitRecord, Hittable, HittableList};
use crate::ray::Ray;
use crate::vec3::Point3;

const NUM_BUCKETS: usize = 12;

// 4-wide BVH node. AABBs stored SoA ([axis][child]) for autovectorization.
struct QbvhNode {
    min: [[f32; 4]; 3],
    max: [[f32; 4]; 3],
    // ≥ 0 → inner node at nodes[v]
    // < 0, ≠ MIN → leaf primitive at objects[(-v - 1)]
    // MIN → unused slot
    children: [i32; 4],
}

pub struct BvhTree {
    nodes:   Vec<QbvhNode>,
    objects: Vec<Arc<dyn Hittable>>,
}

fn surface_area(b: &Aabb) -> f32 {
    let d = b.max - b.min;
    2.0 * (d.x * d.y + d.y * d.z + d.z * d.x)
}

// SAH binary partition. Returns (left_objs, left_boxes, right_objs, right_boxes).
fn sah_partition(
    objs:  &[Arc<dyn Hittable>],
    boxes: &[Aabb],
) -> (Vec<Arc<dyn Hittable>>, Vec<Aabb>, Vec<Arc<dyn Hittable>>, Vec<Aabb>) {
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

        let mut right_sa    = [0.0f32; NUM_BUCKETS];
        let mut right_count = [0u32;   NUM_BUCKETS];
        acc_box = None;
        acc_n   = 0;
        for i in (0..NUM_BUCKETS).rev() {
            acc_n += bucket_count[i];
            if let Some(b) = bucket_bbox[i] {
                acc_box = Some(match acc_box { None => b, Some(a) => Aabb::surrounding(&a, &b) });
            }
            right_count[i] = acc_n;
            right_sa[i]    = acc_box.map_or(0.0, |b| surface_area(&b));
        }

        for split in 0..NUM_BUCKETS - 1 {
            let nl = left_count[split];
            let nr = right_count[split + 1];
            if nl == 0 || nr == 0 { continue; }
            let cost = 1.0 + (left_sa[split] * nl as f32 + right_sa[split + 1] * nr as f32) / node_sa;
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

    let mut pairs: Vec<(Arc<dyn Hittable>, Aabb, usize)> = objs.iter()
        .zip(boxes.iter())
        .map(|(o, b)| (Arc::clone(o), *b, bucket_id(b)))
        .collect();
    pairs.sort_by_key(|(_, _, id)| *id);
    let mid_off = pairs.partition_point(|(_, _, id)| *id <= best_split);
    let mid = if mid_off == 0 || mid_off == span { span / 2 } else { mid_off };

    let (lp, rp) = pairs.split_at(mid);
    let lo: Vec<Arc<dyn Hittable>> = lp.iter().map(|(o, _, _)| Arc::clone(o)).collect();
    let lb: Vec<Aabb>               = lp.iter().map(|(_, b, _)| *b).collect();
    let ro: Vec<Arc<dyn Hittable>> = rp.iter().map(|(o, _, _)| Arc::clone(o)).collect();
    let rb: Vec<Aabb>               = rp.iter().map(|(_, b, _)| *b).collect();
    (lo, lb, ro, rb)
}

// Two rounds of SAH splitting yield up to 4 child groups for a QBVH node.
fn split_four(
    objs:  &[Arc<dyn Hittable>],
    boxes: &[Aabb],
) -> Vec<(Vec<Arc<dyn Hittable>>, Vec<Aabb>)> {
    if objs.len() <= 4 {
        return objs.iter().zip(boxes.iter())
            .map(|(o, b)| (vec![Arc::clone(o)], vec![*b]))
            .collect();
    }
    let (lo, lb, ro, rb) = sah_partition(objs, boxes);
    let mut groups = Vec::with_capacity(4);
    for (sub_o, sub_b) in [(&lo[..], &lb[..]), (&ro[..], &rb[..])] {
        if sub_o.len() < 2 {
            groups.push((sub_o.iter().map(Arc::clone).collect(), sub_b.to_vec()));
        } else {
            let (ll, lb2, rl, rb2) = sah_partition(sub_o, sub_b);
            groups.push((ll, lb2));
            groups.push((rl, rb2));
        }
    }
    groups
}

impl QbvhNode {
    fn from_children(children: [i32; 4], bboxes: &[Aabb; 4]) -> Self {
        let mut min = [[0.0f32; 4]; 3];
        let mut max = [[0.0f32; 4]; 3];
        for c in 0..4 {
            for axis in 0..3 {
                min[axis][c] = bboxes[c].min[axis];
                max[axis][c] = bboxes[c].max[axis];
            }
        }
        Self { min, max, children }
    }

    fn sentinel() -> Self {
        Self { min: [[0.0; 4]; 3], max: [[0.0; 4]; 3], children: [i32::MIN; 4] }
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
        let (root, _) = Self::build(&objs, &boxes, &mut nodes, &mut objects);
        if root < 0 {
            // Single primitive: wrap in a trivial root node so nodes[0] is always valid.
            let mut children = [i32::MIN; 4];
            children[0] = root;
            let bboxes = [boxes[0], Aabb::default(), Aabb::default(), Aabb::default()];
            nodes.push(QbvhNode::from_children(children, &bboxes));
        }
        Self { nodes, objects }
    }

    // Returns (child_data, bbox):
    //   child_data ≥ 0 → QBVH inner node pushed at nodes[child_data]
    //   child_data < 0 → leaf primitive pushed at objects[(-child_data - 1)]
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

        let groups = split_four(objs, boxes);

        let my_idx = nodes.len();
        nodes.push(QbvhNode::sentinel()); // placeholder; filled after children are built

        let mut children  = [i32::MIN; 4];
        let mut bboxes    = [Aabb::default(); 4];
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

impl Hittable for BvhTree {
    fn hit(&self, r: &Ray, t_min: f32, t_max: f32) -> Option<HitRecord<'_>> {
        if self.nodes.is_empty() { return None; }

        // Stack stores child_data values: ≥ 0 = node index, < 0 = leaf primitive
        let mut stack = [0i32; 128];
        let mut top   = 1usize; // stack[0] = 0 (root node)
        let mut best: Option<HitRecord<'_>> = None;
        let mut closest = t_max;

        let ro = [r.origin.x,  r.origin.y,  r.origin.z];
        let id = [r.inv_dir.x, r.inv_dir.y, r.inv_dir.z];

        while top > 0 {
            top -= 1;
            let entry = stack[top];

            if entry < 0 {
                // Leaf: test primitive directly
                let prim_idx = (-entry - 1) as usize;
                if let Some(rec) = self.objects[prim_idx].hit(r, t_min, closest) {
                    closest = rec.t;
                    best    = Some(rec);
                }
                continue;
            }

            let node = &self.nodes[entry as usize];

            // Test all 4 child AABBs; collect hits with entry t
            let mut hits   = [(f32::INFINITY, i32::MIN); 4];
            let mut n_hits = 0usize;

            for c in 0..4 {
                let child = node.children[c];
                if child == i32::MIN { continue; }

                let mut t0 = t_min;
                let mut t1 = closest;
                for axis in 0..3 {
                    let da = (node.min[axis][c] - ro[axis]) * id[axis];
                    let db = (node.max[axis][c] - ro[axis]) * id[axis];
                    t0 = t0.max(da.min(db));
                    t1 = t1.min(da.max(db));
                }
                if t0 <= t1 {
                    hits[n_hits] = (t0, child);
                    n_hits += 1;
                }
            }

            // Insertion-sort descending by tmin: push far children first so near
            // children sit at the top of the stack and are visited first.
            for i in 1..n_hits {
                let key = hits[i];
                let mut j = i;
                while j > 0 && hits[j - 1].0 < key.0 {
                    hits[j] = hits[j - 1];
                    j -= 1;
                }
                hits[j] = key;
            }

            for k in 0..n_hits {
                stack[top] = hits[k].1;
                top += 1;
            }
        }

        best
    }

    fn bounding_box(&self) -> Option<Aabb> {
        let root = self.nodes.first()?;
        let mut mn = [f32::INFINITY;     3];
        let mut mx = [f32::NEG_INFINITY; 3];
        for c in 0..4 {
            if root.children[c] == i32::MIN { continue; }
            for axis in 0..3 {
                mn[axis] = mn[axis].min(root.min[axis][c]);
                mx[axis] = mx[axis].max(root.max[axis][c]);
            }
        }
        if mn[0] == f32::INFINITY { return None; }
        Some(Aabb::new(
            Point3::new(mn[0], mn[1], mn[2]),
            Point3::new(mx[0], mx[1], mx[2]),
        ))
    }
}
