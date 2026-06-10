use std::sync::Arc;
use crate::aabb::Aabb;
use crate::hittable::{HitRecord, Hittable, HittableList};
use crate::ray::Ray;

const NUM_BUCKETS: usize = 12;

pub struct BvhNode {
    left:  Arc<dyn Hittable>,
    right: Arc<dyn Hittable>,
    bbox:  Aabb,
}

fn surface_area(b: &Aabb) -> f32 {
    let d = b.max - b.min;
    2.0 * (d.x * d.y + d.y * d.z + d.z * d.x)
}

impl BvhNode {
    pub fn from_list(list: HittableList) -> Self {
        let mut objects = list.objects;
        let len = objects.len();
        Self::build(&mut objects, 0, len)
    }

    fn build(objects: &mut Vec<Arc<dyn Hittable>>, start: usize, end: usize) -> Self {
        let span = end - start;

        // Collect bounding boxes once to avoid repeated virtual dispatch during SAH.
        let boxes: Vec<Aabb> = objects[start..end]
            .iter()
            .map(|o| o.bounding_box().unwrap_or_default())
            .collect();

        let node_bbox = boxes.iter().copied()
            .reduce(|a, b| Aabb::surrounding(&a, &b))
            .unwrap_or_default();

        if span == 1 {
            return Self {
                left:  Arc::clone(&objects[start]),
                right: Arc::clone(&objects[start]),
                bbox:  node_bbox,
            };
        }
        if span == 2 {
            return Self {
                left:  Arc::clone(&objects[start]),
                right: Arc::clone(&objects[start + 1]),
                bbox:  node_bbox,
            };
        }

        // Centroid range for each axis.
        let mut c_min = [f32::INFINITY;     3];
        let mut c_max = [f32::NEG_INFINITY; 3];
        for b in &boxes {
            for axis in 0..3 {
                let c = (b.min[axis] + b.max[axis]) * 0.5;
                if c < c_min[axis] { c_min[axis] = c; }
                if c > c_max[axis] { c_max[axis] = c; }
            }
        }

        let node_sa   = surface_area(&node_bbox);
        let mut best_cost  = f32::INFINITY;
        let mut best_axis  = 0usize;
        let mut best_split = 0usize; // split after this bucket index

        for axis in 0..3usize {
            let extent = c_max[axis] - c_min[axis];
            if extent < 1e-6 { continue; }

            let mut bucket_bbox  = [None::<Aabb>; NUM_BUCKETS];
            let mut bucket_count = [0u32; NUM_BUCKETS];

            for b in &boxes {
                let c  = (b.min[axis] + b.max[axis]) * 0.5;
                let t  = (c - c_min[axis]) / extent;
                let bi = ((NUM_BUCKETS as f32 * t) as usize).min(NUM_BUCKETS - 1);
                bucket_count[bi] += 1;
                bucket_bbox[bi] = Some(match bucket_bbox[bi] {
                    None    => *b,
                    Some(a) => Aabb::surrounding(&a, b),
                });
            }

            // Left prefix: cumulative SA and count sweeping left → right.
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

            // Right suffix: cumulative SA and count sweeping right → left.
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

            // Evaluate the NUM_BUCKETS-1 candidate split positions.
            for split in 0..NUM_BUCKETS - 1 {
                let nl = left_count[split];
                let nr = right_count[split + 1];
                if nl == 0 || nr == 0 { continue; }
                let cost = 1.0
                    + (left_sa[split] * nl as f32 + right_sa[split + 1] * nr as f32)
                    / node_sa;
                if cost < best_cost {
                    best_cost  = cost;
                    best_axis  = axis;
                    best_split = split;
                }
            }
        }

        // Partition objects by their bucket assignment on the best axis.
        // Use precomputed boxes to avoid calling bounding_box() again via virtual dispatch.
        let extent = c_max[best_axis] - c_min[best_axis];
        let bucket_from_box = |b: &Aabb| -> usize {
            if extent < 1e-6 { return 0; }
            let c = (b.min[best_axis] + b.max[best_axis]) * 0.5;
            let t = (c - c_min[best_axis]) / extent;
            ((NUM_BUCKETS as f32 * t) as usize).min(NUM_BUCKETS - 1)
        };

        let mut pairs: Vec<(Arc<dyn Hittable>, usize)> = objects[start..end]
            .iter()
            .zip(boxes.iter())
            .map(|(obj, b)| (Arc::clone(obj), bucket_from_box(b)))
            .collect();
        pairs.sort_by_key(|(_, id)| *id);
        let mid_off = pairs.partition_point(|(_, id)| *id <= best_split);
        for (i, (obj, _)) in pairs.into_iter().enumerate() {
            objects[start + i] = obj;
        }

        let mid = if mid_off == 0 || mid_off == span {
            start + span / 2  // degenerate: all centroids coincide, fall back to median
        } else {
            start + mid_off
        };

        let left  = Arc::new(Self::build(objects, start, mid)) as Arc<dyn Hittable>;
        let right = Arc::new(Self::build(objects, mid,   end)) as Arc<dyn Hittable>;
        Self { left, right, bbox: node_bbox }
    }
}

impl Hittable for BvhNode {
    fn hit(&self, r: &Ray, t_min: f32, t_max: f32) -> Option<HitRecord<'_>> {
        if !self.bbox.hit(r, t_min, t_max) { return None; }
        let left_hit = self.left.hit(r, t_min, t_max);
        if Arc::ptr_eq(&self.left, &self.right) { return left_hit; }
        let t_max_right = left_hit.as_ref().map_or(t_max, |h| h.t);
        let right_hit   = self.right.hit(r, t_min, t_max_right);
        right_hit.or(left_hit)
    }

    fn bounding_box(&self) -> Option<Aabb> { Some(self.bbox) }
}
