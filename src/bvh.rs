use std::sync::Arc;
use rand::Rng;
use crate::aabb::Aabb;
use crate::hittable::{HitRecord, Hittable, HittableList};
use crate::ray::Ray;

pub struct BvhNode {
    left:  Arc<dyn Hittable>,
    right: Arc<dyn Hittable>,
    bbox:  Aabb,
}

impl BvhNode {
    pub fn from_list(list: HittableList) -> Self {
        let mut objects = list.objects;
        let len = objects.len();
        Self::build(&mut objects, 0, len)
    }

    fn build(objects: &mut Vec<Arc<dyn Hittable>>, start: usize, end: usize) -> Self {
        let axis = rand::thread_rng().gen_range(0usize..3);
        let span = end - start;

        let (left, right): (Arc<dyn Hittable>, Arc<dyn Hittable>) = match span {
            1 => (Arc::clone(&objects[start]), Arc::clone(&objects[start])),
            2 => {
                if centroid(objects[start].as_ref(), axis) <= centroid(objects[start+1].as_ref(), axis) {
                    (Arc::clone(&objects[start]), Arc::clone(&objects[start+1]))
                } else {
                    (Arc::clone(&objects[start+1]), Arc::clone(&objects[start]))
                }
            }
            _ => {
                objects[start..end].sort_by(|a, b| {
                    centroid(a.as_ref(), axis)
                        .partial_cmp(&centroid(b.as_ref(), axis))
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                let mid = start + span / 2;
                (
                    Arc::new(Self::build(objects, start, mid)) as Arc<dyn Hittable>,
                    Arc::new(Self::build(objects, mid,   end)) as Arc<dyn Hittable>,
                )
            }
        };

        let bbox = match (left.bounding_box(), right.bounding_box()) {
            (Some(l), Some(r)) => Aabb::surrounding(&l, &r),
            _ => Aabb::default(),
        };
        Self { left, right, bbox }
    }
}

fn centroid(obj: &dyn Hittable, axis: usize) -> f32 {
    obj.bounding_box()
        .map(|bb| (bb.min[axis] + bb.max[axis]) * 0.5)
        .unwrap_or(0.0)
}

impl Hittable for BvhNode {
    fn hit(&self, r: &Ray, t_min: f32, t_max: f32) -> Option<HitRecord> {
        if !self.bbox.hit(r, t_min, t_max) { return None; }
        let left_hit   = self.left.hit(r, t_min, t_max);
        let t_max_right = left_hit.as_ref().map_or(t_max, |h| h.t);
        let right_hit  = self.right.hit(r, t_min, t_max_right);
        right_hit.or(left_hit)
    }

    fn bounding_box(&self) -> Option<Aabb> { Some(self.bbox) }
}
