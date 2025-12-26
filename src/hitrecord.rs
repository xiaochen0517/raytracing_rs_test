use crate::material::Material;
use crate::ray::Ray;
use crate::vector::Vec3;
use rand::Rng;
use std::sync::Arc;

pub enum ScatterResult {
    Absorbed,    // 光线被吸收
    Scattered,   // 光线被散射
    Transmitted, // 光线被透射
}

#[derive(Clone)]
pub struct HitRecord {
    pub point: Vec3,
    pub normal: Vec3,
    pub t: f64,
    pub front_face: bool,                    // 是否从外部击中
    pub material: Option<Arc<dyn Material>>, // 材质
}

impl HitRecord {
    pub fn new() -> Self {
        HitRecord {
            point: Vec3::new(0.0, 0.0, 0.0),
            normal: Vec3::new(0.0, 0.0, 0.0),
            t: 0.0,
            front_face: true,
            material: None,
        }
    }

    // 设置法线方向（确保总是指向外侧）
    pub fn set_face_normal(&mut self, ray: &Ray, outward_normal: Vec3) {
        self.front_face = ray.direction.dot(&outward_normal) < 0.0;
        self.normal = if self.front_face {
            outward_normal
        } else {
            outward_normal * -1.0
        };
    }
}

// 可以被光线击中的物体
pub trait Hittable: Sync {
    fn hit(&self, ray: &Ray, t_min: f64, t_max: f64, rec: &mut HitRecord) -> bool;
}

pub struct Sphere {
    pub center: Vec3,
    pub radius: f64,
    pub material: Arc<dyn Material>,
}

impl Sphere {
    pub fn new(center: Vec3, radius: f64, material: Arc<dyn Material>) -> Self {
        Sphere {
            center,
            radius,
            material,
        }
    }
}

impl Hittable for Sphere {
    fn hit(&self, ray: &Ray, t_min: f64, t_max: f64, rec: &mut HitRecord) -> bool {
        let oc = ray.origin - self.center;
        let a = ray.direction.dot(&ray.direction);
        let half_b = oc.dot(&ray.direction);
        let c = oc.dot(&oc) - self.radius * self.radius;

        let discriminant = half_b * half_b - a * c;
        if discriminant < 0.0 {
            return false;
        }

        let sqrtd = discriminant.sqrt();
        let mut root = (-half_b - sqrtd) / a;

        if root < t_min || root > t_max {
            root = (-half_b + sqrtd) / a;
            if root < t_min || root > t_max {
                return false;
            }
        }

        rec.t = root;
        rec.point = ray.at(rec.t);
        let outward_normal = (rec.point - self.center) * (1.0 / self.radius);
        rec.set_face_normal(ray, outward_normal);
        rec.material = Some(self.material.clone());

        true
    }
}

pub struct HittableList {
    pub objects: Vec<Box<dyn Hittable>>,
}

impl HittableList {
    pub fn new() -> Self {
        HittableList {
            objects: Vec::new(),
        }
    }

    pub fn add(&mut self, object: Box<dyn Hittable>) {
        self.objects.push(object);
    }
}

impl Hittable for HittableList {
    fn hit(&self, ray: &Ray, t_min: f64, t_max: f64, rec: &mut HitRecord) -> bool {
        let mut temp_rec = HitRecord::new();
        let mut hit_anything = false;
        let mut closest_so_far = t_max;

        // 遍历所有物体，找最近的碰撞
        for object in &self.objects {
            let is_hit = object.hit(ray, t_min, closest_so_far, &mut temp_rec);
            if is_hit {
                hit_anything = true;
                closest_so_far = temp_rec.t; // 更新最近距离
                *rec = temp_rec.clone(); // 保存碰撞记录
            }
        }

        hit_anything
    }
}

pub struct XYRect {
    pub x0: f64,
    pub x1: f64,
    pub y0: f64,
    pub y1: f64,
    pub k: f64, // z 坐标
    pub material: Arc<dyn Material>,
    pub flip_normal: bool,
}

impl XYRect {
    pub fn new(x0: f64, x1: f64, y0: f64, y1: f64, k: f64, material: Arc<dyn Material>) -> Self {
        XYRect {
            x0,
            x1,
            y0,
            y1,
            k,
            material,
            flip_normal: false,
        }
    }

    pub fn new_flipped(
        x0: f64,
        x1: f64,
        y0: f64,
        y1: f64,
        k: f64,
        material: Arc<dyn Material>,
    ) -> Self {
        XYRect {
            x0,
            x1,
            y0,
            y1,
            k,
            material,
            flip_normal: true,
        }
    }
}

impl Hittable for XYRect {
    fn hit(&self, ray: &Ray, t_min: f64, t_max: f64, rec: &mut HitRecord) -> bool {
        if ray.direction.z.abs() < 1e-8 {
            return false; // 光线平行于平面
        }
        // 计算光线与 z=k 平面的交点
        let t = (self.k - ray.origin.z) / ray.direction.z;

        if t < t_min || t > t_max {
            return false;
        }

        // 计算交点的 x, y 坐标
        let x = ray.origin.x + t * ray.direction.x;
        let y = ray.origin.y + t * ray.direction.y;

        // 检查是否在矩形范围内
        if x < self.x0 || x > self.x1 || y < self.y0 || y > self.y1 {
            return false;
        }

        // 记录碰撞信息
        rec.t = t;
        rec.point = ray.at(t);
        // 根据 flip_normal 决定法线方向
        let outward_normal = if self.flip_normal {
            Vec3::new(0.0, 0.0, -1.0) // z 轴负方向
        } else {
            Vec3::new(0.0, 0.0, 1.0) // z 轴正方向
        };
        rec.set_face_normal(ray, outward_normal);
        rec.material = Some(self.material.clone());

        true
    }
}

pub struct XZRect {
    pub x0: f64,
    pub x1: f64,
    pub z0: f64,
    pub z1: f64,
    pub k: f64, // y 坐标
    pub material: Arc<dyn Material>,
}

impl XZRect {
    pub fn new(x0: f64, x1: f64, z0: f64, z1: f64, k: f64, material: Arc<dyn Material>) -> Self {
        XZRect {
            x0,
            x1,
            z0,
            z1,
            k,
            material,
        }
    }

    pub fn sample_point(&self) -> (Vec3, Vec3) {
        let mut rng = rand::thread_rng();
        let random_x: f64 = rng.gen_range(self.x0..self.x1);
        let random_z: f64 = rng.gen_range(self.z0..self.z1);
        let point = Vec3::new(random_x, self.k, random_z);
        let normal = Vec3::new(0.0, 1.0, 0.0); // y 轴正方向
        (point, normal)
    }
}

impl Hittable for XZRect {
    fn hit(&self, ray: &Ray, t_min: f64, t_max: f64, rec: &mut HitRecord) -> bool {
        if ray.direction.y.abs() < 1e-8 {
            return false; // 光线平行于平面
        }
        let t = (self.k - ray.origin.y) / ray.direction.y;

        if t < t_min || t > t_max {
            return false;
        }

        let x = ray.origin.x + t * ray.direction.x;
        let z = ray.origin.z + t * ray.direction.z;

        if x < self.x0 || x > self.x1 || z < self.z0 || z > self.z1 {
            return false;
        }

        rec.t = t;
        rec.point = ray.at(t);
        let outward_normal = Vec3::new(0.0, 1.0, 0.0); // y 轴正方向
        rec.set_face_normal(ray, outward_normal);
        rec.material = Some(self.material.clone());

        true
    }
}

pub struct YZRect {
    pub y0: f64,
    pub y1: f64,
    pub z0: f64,
    pub z1: f64,
    pub k: f64, // x 坐标
    pub material: Arc<dyn Material>,
}

impl YZRect {
    pub fn new(y0: f64, y1: f64, z0: f64, z1: f64, k: f64, material: Arc<dyn Material>) -> Self {
        YZRect {
            y0,
            y1,
            z0,
            z1,
            k,
            material,
        }
    }
}

impl Hittable for YZRect {
    fn hit(&self, ray: &Ray, t_min: f64, t_max: f64, rec: &mut HitRecord) -> bool {
        if ray.direction.x.abs() < 1e-8 {
            return false; // 光线平行于平面
        }
        let t = (self.k - ray.origin.x) / ray.direction.x;

        if t < t_min || t > t_max {
            return false;
        }

        let y = ray.origin.y + t * ray.direction.y;
        let z = ray.origin.z + t * ray.direction.z;

        if y < self.y0 || y > self.y1 || z < self.z0 || z > self.z1 {
            return false;
        }

        rec.t = t;
        rec.point = ray.at(t);
        let outward_normal = Vec3::new(1.0, 0.0, 0.0); // x 轴正方向
        rec.set_face_normal(ray, outward_normal);
        rec.material = Some(self.material.clone());

        true
    }
}
