use std::sync::Arc;
use crate::aabb::Aabb;
use crate::hittable::{HitRecord, Hittable, HittableList};
use crate::ray::Ray;

const NUM_BUCKETS: usize = 12;

struct FlatNode {
    bbox: Aabb, // 24 bytes
    /// ≥ 0 → inner node; right child is at nodes[data as usize], left child at current_idx + 1.
    /// < 0 → leaf; primitive is objects[(-data - 1) as usize].
    data: i32,  //  4 bytes  →  FlatNode is 28 bytes, packed in a contiguous Vec
}

pub struct BvhTree {
    nodes:   Vec<FlatNode>,
    objects: Vec<Arc<dyn Hittable>>,
}

fn surface_area(b: &Aabb) -> f32 {
    let d = b.max - b.min;
    2.0 * (d.x * d.y + d.y * d.z + d.z * d.x)
}

impl BvhTree {
    pub fn from_list(list: HittableList) -> Self {
        let objs = list.objects;
        assert!(!objs.is_empty(), "BvhTree::from_list requires at least one object");
        let boxes: Vec<Aabb> = objs.iter()
            .map(|o| o.bounding_box().unwrap_or_default())
            .collect();
        let mut nodes   = Vec::with_capacity(objs.len() * 2);
        let mut objects = Vec::with_capacity(objs.len());
        Self::build(&objs, &boxes, &mut nodes, &mut objects);
        Self { nodes, objects }
    }

    fn build(
        objs:  &[Arc<dyn Hittable>],
        boxes: &[Aabb],
        nodes: &mut Vec<FlatNode>,
        prims: &mut Vec<Arc<dyn Hittable>>,
    ) {
        let span = objs.len();

        let node_bbox = boxes.iter().copied()
            .reduce(|a, b| Aabb::surrounding(&a, &b))
            .unwrap_or_default();

        if span == 1 {
            let idx = prims.len() as i32;
            prims.push(Arc::clone(&objs[0]));
            nodes.push(FlatNode { bbox: boxes[0], data: -(idx + 1) });
            return;
        }

        if span == 2 {
            let my = nodes.len();
            nodes.push(FlatNode { bbox: node_bbox, data: 0 }); // right child filled below
            let li = prims.len() as i32; prims.push(Arc::clone(&objs[0]));
            nodes.push(FlatNode { bbox: boxes[0], data: -(li + 1) });
            nodes[my].data = nodes.len() as i32; // right child starts here
            let ri = prims.len() as i32; prims.push(Arc::clone(&objs[1]));
            nodes.push(FlatNode { bbox: boxes[1], data: -(ri + 1) });
            return;
        }

        // Centroid range per axis.
        let mut c_min = [f32::INFINITY;     3];
        let mut c_max = [f32::NEG_INFINITY; 3];
        for b in boxes {
            for axis in 0..3 {
                let c = (b.min[axis] + b.max[axis]) * 0.5;
                c_min[axis] = c_min[axis].min(c);
                c_max[axis] = c_max[axis].max(c);
            }
        }

        let node_sa        = surface_area(&node_bbox);
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
                bucket_bbox[bi] = Some(match bucket_bbox[bi] {
                    None    => *b,
                    Some(a) => Aabb::surrounding(&a, b),
                });
            }

            // Left prefix sweep.
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

            // Right suffix sweep.
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
                let cost = 1.0
                    + (left_sa[split] * nl as f32 + right_sa[split + 1] * nr as f32) / node_sa;
                if cost < best_cost { best_cost = cost; best_axis = axis; best_split = split; }
            }
        }

        // Partition using precomputed bucket IDs from already-computed boxes.
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
        let l_objs:  Vec<Arc<dyn Hittable>> = lp.iter().map(|(o, _, _)| Arc::clone(o)).collect();
        let l_boxes: Vec<Aabb>               = lp.iter().map(|(_, b, _)| *b).collect();
        let r_objs:  Vec<Arc<dyn Hittable>> = rp.iter().map(|(o, _, _)| Arc::clone(o)).collect();
        let r_boxes: Vec<Aabb>               = rp.iter().map(|(_, b, _)| *b).collect();

        let my = nodes.len();
        nodes.push(FlatNode { bbox: node_bbox, data: 0 }); // right child filled after left subtree

        Self::build(&l_objs, &l_boxes, nodes, prims);

        nodes[my].data = nodes.len() as i32; // right child starts here
        Self::build(&r_objs, &r_boxes, nodes, prims);
    }

    fn hit_node<'a>(&'a self, r: &Ray, idx: usize, t_min: f32, t_max: f32) -> Option<HitRecord<'a>> {
        let node = &self.nodes[idx];
        if !node.bbox.hit(r, t_min, t_max) { return None; }
        if node.data < 0 {
            return self.objects[(-node.data - 1) as usize].hit(r, t_min, t_max);
        }
        let left  = self.hit_node(r, idx + 1, t_min, t_max);
        let t_mid = left.as_ref().map_or(t_max, |h| h.t);
        let right = self.hit_node(r, node.data as usize, t_min, t_mid);
        right.or(left)
    }
}

impl Hittable for BvhTree {
    fn hit(&self, r: &Ray, t_min: f32, t_max: f32) -> Option<HitRecord<'_>> {
        self.hit_node(r, 0, t_min, t_max)
    }

    fn bounding_box(&self) -> Option<Aabb> {
        self.nodes.first().map(|n| n.bbox)
    }
}
