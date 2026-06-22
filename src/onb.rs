use crate::vec3::Vec3;

pub struct Onb {
    pub u: Vec3,
    pub v: Vec3,
    pub w: Vec3,
}

impl Onb {
    pub fn from_w(n: Vec3) -> Self {
        let w = n.unit();
        let a = if w.x.abs() > 0.9 { Vec3::new(0.0, 1.0, 0.0) } else { Vec3::new(1.0, 0.0, 0.0) };
        let v = w.cross(a).unit();
        let u = w.cross(v);
        Self { u, v, w }
    }

    pub fn local(&self, a: Vec3) -> Vec3 {
        a.x * self.u + a.y * self.v + a.z * self.w
    }
}
