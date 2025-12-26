use crate::hitrecord::{Hittable, XZRect};
use crate::vector::Vec3;
use rand::Rng;
use std::sync::Arc;

pub trait Light: Sync + Send {
    fn sample(&self) -> (Vec3, Vec3);

    fn emitted(&self, u: f64, v: f64, p: &Vec3) -> Vec3;

    fn area(&self) -> f64;
}

pub struct LightSource {
    geometry: Arc<dyn Hittable>,
    light: Arc<dyn Light>,
}

impl Light for XZRect {
    fn sample(&self) -> (Vec3, Vec3) {
        let mut rng = rand::thread_rng();
        let x = rng.gen_range(self.x0..self.x1);
        let z = rng.gen_range(self.z0..self.z1);
        let y = self.k;
        let point = Vec3::new(x, y, z);
        let normal = Vec3::new(0.0, 1.0, 0.0); // 法线朝上
        (point, normal)
    }

    fn emitted(&self, u: f64, v: f64, p: &Vec3) -> Vec3 {
        self.material.emitted(u, v, p)
    }

    fn area(&self) -> f64 {
        (self.x1 - self.x0) * (self.z1 - self.z0)
    }
}
