use crate::vector::Vec3;

#[derive(Clone, Copy, Debug)]
pub struct Ray {
    pub origin: Vec3,    // 起点
    pub direction: Vec3, // 方向
}

impl Ray {
    pub fn at(&self, t: f64) -> Vec3 {
        self.origin + self.direction * t
    }
}
