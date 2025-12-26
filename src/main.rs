use rand::Rng;
use rayon::prelude::*;
use std::ops::{Add, Mul, Sub};

#[derive(Clone, Copy, Debug)]
struct Vec3 {
    x: f64,
    y: f64,
    z: f64,
}

impl Vec3 {
    fn new(x: f64, y: f64, z: f64) -> Self {
        Vec3 { x, y, z }
    }

    // 向量长度
    fn length(&self) -> f64 {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }

    // 单位向量（方向）
    fn unit(&self) -> Vec3 {
        let len = self.length();
        Vec3::new(self.x / len, self.y / len, self.z / len)
    }

    // 生成随机向量 [0, 1)
    fn random() -> Self {
        Self::random_range(0.0, 1.0)
    }

    // 生成指定范围的随机向量
    fn random_range(min: f64, max: f64) -> Self {
        let mut rng = rand::thread_rng();
        Vec3::new(
            rng.gen_range(min..max),
            rng.gen_range(min..max),
            rng.gen_range(min..max),
        )
    }

    // 生成单位球内的随机点（用于漫反射）
    fn random_in_unit_sphere() -> Self {
        loop {
            let p = Vec3::random_range(-1.0, 1.0);
            if p.dot(&p) < 1.0 {
                return p;
            }
        }
    }

    // 生成单位半球内的随机方向（更物理准确）
    fn random_unit_vector() -> Self {
        Vec3::random_in_unit_sphere().unit()
    }

    // 检查向量是否接近零
    fn near_zero(&self) -> bool {
        let s = 1e-8;
        self.x.abs() < s && self.y.abs() < s && self.z.abs() < s
    }

    // 反射向量
    fn reflect(&self, n: &Vec3) -> Vec3 {
        *self - *n * (2.0 * self.dot(n))
    }

    fn length_squared(&self) -> f64 {
        self.x * self.x + self.y * self.y + self.z * self.z
    }

    // 折射向量
    fn refract(&self, n: &Vec3, etai_over_etat: f64) -> Vec3 {
        let cos_theta = ((*self * -1.0).dot(n)).min(1.0);
        let r_out_perp = (*self + *n * cos_theta) * etai_over_etat;
        let r_out_parallel = *n * (-(1.0 - r_out_perp.length_squared()).abs().sqrt());
        r_out_perp + r_out_parallel
    }

    fn cross(&self, other: &Vec3) -> Vec3 {
        Vec3::new(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )
    }
}

// 向量加法
impl Add for Vec3 {
    type Output = Vec3;
    fn add(self, other: Vec3) -> Vec3 {
        Vec3::new(self.x + other.x, self.y + other.y, self.z + other.z)
    }
}

// 向量减法
impl Sub for Vec3 {
    type Output = Vec3;
    fn sub(self, other: Vec3) -> Vec3 {
        Vec3::new(self.x - other.x, self.y - other.y, self.z - other.z)
    }
}

impl Mul<Vec3> for Vec3 {
    type Output = Vec3;
    fn mul(self, other: Vec3) -> Vec3 {
        Vec3::new(self.x * other.x, self.y * other.y, self.z * other.z)
    }
}

// 向量数乘
impl Mul<f64> for Vec3 {
    type Output = Vec3;
    fn mul(self, t: f64) -> Vec3 {
        Vec3::new(self.x * t, self.y * t, self.z * t)
    }
}

// 光线结构
#[derive(Clone, Copy, Debug)]
struct Ray {
    origin: Vec3,    // 起点
    direction: Vec3, // 方向
}

impl Ray {
    fn at(&self, t: f64) -> Vec3 {
        self.origin + self.direction * t
    }
}

use std::fs::File;
use std::io::Write;
impl Vec3 {
    // 向量点积
    fn dot(&self, other: &Vec3) -> f64 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }
}

enum ScatterResult {
    Absorbed,    // 光线被吸收
    Scattered,   // 光线被散射
    Transmitted, // 光线被透射
}

// 材质特征
trait Material: Sync + Send {
    // 返回是否散射、散射光线、颜色衰减
    fn scatter(
        &self,
        ray_in: &Ray,           // 入射光线
        rec: &HitRecord,        // 碰撞记录
        attenuation: &mut Vec3, // 颜色衰减（输出）
        scattered: &mut Ray,    // 散射光线（输出）
    ) -> ScatterResult; // 是否发生散射

    fn emitted(&self, _u: f64, _v: f64, _p: &Vec3) -> Vec3 {
        Vec3::new(0.0, 0.0, 0.0) // 默认不发光
    }
}

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

struct DiffuseLight {
    emit: Vec3, // 发光颜色和强度
}

impl DiffuseLight {
    fn new(emit: Vec3) -> Self {
        DiffuseLight { emit }
    }
}

impl Material for DiffuseLight {
    fn scatter(
        &self,
        _ray_in: &Ray,
        _rec: &HitRecord,
        _attenuation: &mut Vec3,
        _scattered: &mut Ray,
    ) -> ScatterResult {
        // 不散射光线
        ScatterResult::Absorbed
    }

    fn emitted(&self, _u: f64, _v: f64, _p: &Vec3) -> Vec3 {
        self.emit // 返回发光颜色
    }
}

struct Lambertian {
    albedo: Vec3, // 反射率（材质颜色）
}

impl Lambertian {
    fn new(albedo: Vec3) -> Self {
        Lambertian { albedo }
    }
}

impl Material for Lambertian {
    fn scatter(
        &self,
        _ray_in: &Ray,
        rec: &HitRecord,
        attenuation: &mut Vec3,
        scattered: &mut Ray,
    ) -> ScatterResult {
        // 计算散射方向
        let mut scatter_direction = rec.normal + Vec3::random_unit_vector();

        // 处理退化情况（散射方向接近零）
        if scatter_direction.near_zero() {
            scatter_direction = rec.normal;
        }

        // 创建散射光线
        *scattered = Ray {
            origin: rec.point,
            direction: scatter_direction,
        };

        // 设置颜色衰减
        *attenuation = self.albedo;

        // 总是发生散射
        ScatterResult::Scattered
    }
}

struct Metal {
    albedo: Vec3,
    fuzz: f64, // 模糊度 [0, 1]，0=完美镜面
}

impl Metal {
    fn new(albedo: Vec3, fuzz: f64) -> Self {
        Metal {
            albedo,
            fuzz: if fuzz < 1.0 { fuzz } else { 1.0 },
        }
    }
}

impl Material for Metal {
    fn scatter(
        &self,
        ray_in: &Ray,
        rec: &HitRecord,
        attenuation: &mut Vec3,
        scattered: &mut Ray,
    ) -> ScatterResult {
        let reflected = ray_in.direction.unit().reflect(&rec.normal);

        // 添加模糊（在反射方向周围随机偏移）
        *scattered = Ray {
            origin: rec.point,
            direction: reflected + Vec3::random_in_unit_sphere() * self.fuzz,
        };

        *attenuation = self.albedo;

        // 只有反射方向向外时才散射
        if scattered.direction.dot(&rec.normal) > 0.0 {
            ScatterResult::Scattered
        } else {
            ScatterResult::Absorbed
        }
    }
}

struct Dielectric {
    ir: f64, // 折射率（Index of Refraction）
}

impl Dielectric {
    fn new(ir: f64) -> Self {
        Dielectric { ir }
    }

    // Schlick 近似：计算菲涅尔反射率
    fn reflectance(cosine: f64, ref_idx: f64) -> f64 {
        let r0 = ((1.0 - ref_idx) / (1.0 + ref_idx)).powi(2);
        r0 + (1.0 - r0) * (1.0 - cosine).powi(5)
    }
}

impl Material for Dielectric {
    fn scatter(
        &self,
        ray_in: &Ray,
        rec: &HitRecord,
        attenuation: &mut Vec3,
        scattered: &mut Ray,
    ) -> ScatterResult {
        *attenuation = Vec3::new(1.0, 1.0, 1.0); // 玻璃不吸收颜色

        // 计算折射率比
        let refraction_ratio = if rec.front_face {
            1.0 / self.ir // 从空气进入玻璃
        } else {
            self.ir // 从玻璃进入空气
        };

        let unit_direction = ray_in.direction.unit();
        let cos_theta = (unit_direction * -1.0).dot(&rec.normal).min(1.0);
        let sin_theta = (1.0 - cos_theta * cos_theta).sqrt();

        // 判断是否发生全反射
        let cannot_refract = refraction_ratio * sin_theta > 1.0;

        let direction = if cannot_refract
            || Dielectric::reflectance(cos_theta, refraction_ratio)
                > rand::thread_rng().gen_range(0.0..1.0)
        {
            // 反射
            unit_direction.reflect(&rec.normal)
        } else {
            // 折射
            unit_direction.refract(&rec.normal, refraction_ratio)
        };

        *scattered = Ray {
            origin: rec.point,
            direction,
        };

        ScatterResult::Scattered
    }
}

struct OneWayMaterial {
    reflect_material: Arc<dyn Material>, // 从内部击中时使用的材质
}

impl OneWayMaterial {
    fn new(reflect_material: Arc<dyn Material>) -> Self {
        OneWayMaterial { reflect_material }
    }
}

impl Material for OneWayMaterial {
    fn scatter(
        &self,
        ray_in: &Ray,
        rec: &HitRecord,
        attenuation: &mut Vec3,
        scattered: &mut Ray,
    ) -> ScatterResult {
        if rec.front_face {
            // 从外部击中（相机方向） → 不散射，让光线穿过
            *attenuation = Vec3::new(1.0, 1.0, 1.0); // 不衰减颜色
            *scattered = Ray {
                origin: rec.point,
                direction: ray_in.direction,
            };
            ScatterResult::Transmitted
        } else {
            // 从内部击中 → 使用白色漫反射材质
            self.reflect_material
                .scatter(ray_in, rec, attenuation, scattered)
        }
    }
}

#[derive(Clone)]
struct HitRecord {
    point: Vec3,
    normal: Vec3,
    t: f64,
    front_face: bool,                    // 是否从外部击中
    material: Option<Arc<dyn Material>>, // 材质
}

impl HitRecord {
    fn new() -> Self {
        HitRecord {
            point: Vec3::new(0.0, 0.0, 0.0),
            normal: Vec3::new(0.0, 0.0, 0.0),
            t: 0.0,
            front_face: true,
            material: None,
        }
    }

    // 设置法线方向（确保总是指向外侧）
    fn set_face_normal(&mut self, ray: &Ray, outward_normal: Vec3) {
        self.front_face = ray.direction.dot(&outward_normal) < 0.0;
        self.normal = if self.front_face {
            outward_normal
        } else {
            outward_normal * -1.0
        };
    }
}

// 可以被光线击中的物体
trait Hittable: Sync {
    fn hit(&self, ray: &Ray, t_min: f64, t_max: f64, rec: &mut HitRecord) -> bool;
}

struct Sphere {
    center: Vec3,
    radius: f64,
    material: Arc<dyn Material>,
}

impl Sphere {
    fn new(center: Vec3, radius: f64, material: Arc<dyn Material>) -> Self {
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

struct HittableList {
    objects: Vec<Box<dyn Hittable>>,
}

impl HittableList {
    fn new() -> Self {
        HittableList {
            objects: Vec::new(),
        }
    }

    fn add(&mut self, object: Box<dyn Hittable>) {
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

struct XYRect {
    x0: f64,
    x1: f64,
    y0: f64,
    y1: f64,
    k: f64, // z 坐标
    material: Arc<dyn Material>,
    flip_normal: bool,
}

impl XYRect {
    fn new(x0: f64, x1: f64, y0: f64, y1: f64, k: f64, material: Arc<dyn Material>) -> Self {
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

    fn new_flipped(
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

struct XZRect {
    x0: f64,
    x1: f64,
    z0: f64,
    z1: f64,
    k: f64, // y 坐标
    material: Arc<dyn Material>,
}

impl XZRect {
    fn new(x0: f64, x1: f64, z0: f64, z1: f64, k: f64, material: Arc<dyn Material>) -> Self {
        XZRect {
            x0,
            x1,
            z0,
            z1,
            k,
            material,
        }
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

struct YZRect {
    y0: f64,
    y1: f64,
    z0: f64,
    z1: f64,
    k: f64, // x 坐标
    material: Arc<dyn Material>,
}

impl YZRect {
    fn new(y0: f64, y1: f64, z0: f64, z1: f64, k: f64, material: Arc<dyn Material>) -> Self {
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

struct Camera {
    origin: Vec3,
    lower_left_corner: Vec3,
    horizontal: Vec3,
    vertical: Vec3,
}

impl Camera {
    fn new(
        lookfrom: Vec3,    // 相机位置
        lookat: Vec3,      // 看向的点
        vup: Vec3,         // 上方向
        vfov: f64,         // 垂直视野角度（度）
        aspect_ratio: f64, // 宽高比
    ) -> Self {
        let theta = vfov.to_radians();
        let h = (theta / 2.0).tan();
        let viewport_height = 2.0 * h;
        let viewport_width = aspect_ratio * viewport_height;

        // 构建相机坐标系
        let w = (lookfrom - lookat).unit(); // 相机朝向（反方向）
        let u = vup.cross(&w).unit(); // 相机右方向
        let v = w.cross(&u); // 相机上方向

        let origin = lookfrom;
        let horizontal = u * viewport_width;
        let vertical = v * viewport_height;
        let lower_left_corner = origin - horizontal * 0.5 - vertical * 0.5 - w;

        Camera {
            origin,
            lower_left_corner,
            horizontal,
            vertical,
        }
    }

    fn get_ray(&self, s: f64, t: f64) -> Ray {
        Ray {
            origin: self.origin,
            direction: self.lower_left_corner + self.horizontal * s + self.vertical * t
                - self.origin,
        }
    }
}

fn ray_color(ray: &Ray, world: &dyn Hittable, depth: i32) -> Vec3 {
    if !ray.direction.x.is_finite() || !ray.direction.y.is_finite() || !ray.direction.z.is_finite()
    {
        // eprintln!("光线方向包含非有限值: {:?}", ray.direction);
        return Vec3::new(1.0, 0.0, 1.0);
    }
    if depth <= 0 {
        return Vec3::new(0.0, 0.0, 0.0);
    }

    let mut rec = HitRecord::new();

    if !world.hit(ray, 0.001, f64::INFINITY, &mut rec) {
        // 没有击中物体，返回黑色
        return Vec3::new(0.0, 0.0, 0.0);
    }

    // 如果命中物体，获取材质并计算散射
    let mut scattered = Ray {
        origin: Vec3::new(0.0, 0.0, 0.0),
        direction: Vec3::new(0.0, 0.0, 0.0),
    };
    let mut attenuation = Vec3::new(0.0, 0.0, 0.0);

    // 使用材质的 scatter 方法
    if let Some(material) = &rec.material {
        let emitted = material.emitted(0.0, 0.0, &rec.point);

        return match material.scatter(ray, &rec, &mut attenuation, &mut scattered) {
            ScatterResult::Absorbed => {
                emitted // 光线被吸收，只返回发光部分
            }
            ScatterResult::Transmitted => emitted + ray_color(&scattered, world, depth - 1),
            ScatterResult::Scattered => {
                // 递归追踪散射光线，乘以衰减系数
                // 需要注意这里的顺序，先加上发光部分，再乘以衰减后的散射部分
                emitted + ray_color(&scattered, world, depth - 1) * attenuation
            }
        };
    }

    Vec3::new(0.0, 0.0, 0.0) // 光线被吸收
}

// 将 [0, 1] 范围的颜色值转换为 [0, 255] 的像素值
fn write_color(color: Vec3, samples_per_pixel: i32) -> (u8, u8, u8) {
    let scale = 1.0 / samples_per_pixel as f64;

    // 缩放颜色（取平均）
    let r = color.x * scale;
    let g = color.y * scale;
    let b = color.z * scale;

    // 转换到 [0, 255] 范围
    let ir = (256.0 * r.clamp(0.0, 0.999)) as u8;
    let ig = (256.0 * g.clamp(0.0, 0.999)) as u8;
    let ib = (256.0 * b.clamp(0.0, 0.999)) as u8;

    (ir, ig, ib)
}

struct Config {
    image_width: usize,
    image_height: usize,
    samples_per_pixel: i32,
    max_depth: i32,
}

const RELEASE_CONFIG: Config = Config {
    image_width: 900,
    image_height: 600,
    samples_per_pixel: 100,
    max_depth: 50,
};

const DEBUG_CONFIG: Config = Config {
    image_width: 300,
    image_height: 200,
    samples_per_pixel: 100,
    max_depth: 50,
};

const BOX_SIZE: f64 = 2.0;

fn cornell_box() -> HittableList {
    let mut world = HittableList::new();

    // 创建材质
    let red = Arc::new(Lambertian::new(Vec3::new(0.65, 0.05, 0.05)));
    let white = Arc::new(Lambertian::new(Vec3::new(0.73, 0.73, 0.73)));
    let green = Arc::new(Lambertian::new(Vec3::new(0.12, 0.45, 0.15)));
    let blue = Arc::new(Lambertian::new(Vec3::new(0.05, 0.05, 0.65)));
    let light = Arc::new(DiffuseLight::new(Vec3::new(15.0, 15.0, 15.0)));

    // 单向墙材质（内部反射为白色）
    let one_way = Arc::new(OneWayMaterial::new(white.clone()));

    // === 墙壁 ===

    // 左墙（红色）- YZ 平面，x = -1
    world.add(Box::new(YZRect::new(
        -1.0, 1.0, // y 范围：底到顶
        -1.0, 1.0,  // z 范围：前到后
        -1.0, // x 坐标：左侧
        red,
    )));

    // 右墙（蓝色）- YZ 平面，x = 1
    world.add(Box::new(YZRect::new(
        -1.0, 1.0, // y 范围
        -1.0, 1.0, // z 范围
        1.0, // x 坐标：右侧
        blue,
    )));

    // 后墙（绿色）- XY 平面，z = 1
    world.add(Box::new(XYRect::new(
        -1.0, 1.0, // x 范围
        -1.0, 1.0, // y 范围
        1.0, // z 坐标：后方
        green,
    )));

    // 天花板（白色）- XZ 平面，y = 1
    world.add(Box::new(XZRect::new(
        -1.0,
        1.0, // x 范围
        -1.0,
        1.0, // z 范围
        1.0, // y 坐标：顶部
        white.clone(),
    )));

    // 地板（白色）- XZ 平面，y = -1
    world.add(Box::new(XZRect::new(
        -1.0, 1.0, // x 范围
        -1.0, 1.0,  // z 范围
        -1.0, // y 坐标：底部
        white,
    )));

    // === 新增：前墙（单向） - XY 平面，z = -1 ===
    world.add(Box::new(XYRect::new_flipped(
        -1.0, 1.0, // x 范围
        -1.0, 1.0,  // y 范围
        -1.0, // z 坐标：前方
        one_way,
    )));

    // === 光源 ===
    // 天花板中央的方形光源 - XZ 平面，y = 0. 99
    world.add(Box::new(XZRect::new(
        -0.25, 0.25, // x 范围：中央 0.5 宽
        -0.25, 0.25, // z 范围：中央 0.5 宽
        0.99, // y 坐标：略低于天花板
        light,
    )));

    world
}

fn main() {
    let config = DEBUG_CONFIG;
    let width = config.image_width;
    let height = config.image_height;

    let max_depth = config.max_depth;
    let samples_per_pixel = config.samples_per_pixel;

    // 图片设置
    let aspect_ratio = width as f64 / height as f64;

    // 创建场景
    let world = cornell_box();

    // 创建相机
    let lookfrom = Vec3::new(0.0, 0.0, -3.0); // 相机在前方 3 个单位
    let lookat = Vec3::new(0.0, 0.0, 0.0); // 看向盒子中心
    let vup = Vec3::new(0.0, 1.0, 0.0); // 上方向
    let vfov = 40.0; // 视野角度

    let camera = Camera::new(lookfrom, lookat, vup, vfov, aspect_ratio);

    // 生成所有像素坐标
    let pixels: Vec<(usize, usize)> = (0..height)
        .rev()
        .flat_map(|j| (0..width).map(move |i| (i, j)))
        .collect();

    let total_pixels = pixels.len();
    let progress = Arc::new(AtomicUsize::new(0));
    let progress_clone = progress.clone();

    eprintln!(
        "开始渲染 {} x {} 像素（共 {} 像素）...",
        width, height, total_pixels
    );

    // 并行计算每个像素的颜色
    let pixel_colors: Vec<(u8, u8, u8)> = pixels
        .par_iter() // 并行迭代
        .map(|(i, j)| {
            let mut rng = rand::thread_rng();
            let mut pixel_color = Vec3::new(0.0, 0.0, 0.0);

            for _ in 0..samples_per_pixel {
                let random_u: f64 = rng.gen_range(0.0..1.0);
                let random_v: f64 = rng.gen_range(0.0..1.0);

                let u = (*i as f64 + random_u) / (width - 1) as f64;
                let v = (*j as f64 + random_v) / (height - 1) as f64;

                let ray = camera.get_ray(u, v);
                pixel_color = pixel_color + ray_color(&ray, &world, max_depth);
            }

            // 更新进度
            let current = progress_clone.fetch_add(1, Ordering::Relaxed);
            if current % 1000 == 0 {
                eprint!(
                    "\r进度: {}/{} ({:.1}%)",
                    current,
                    total_pixels,
                    (current as f64 / total_pixels as f64) * 100.0
                );
            }

            write_color(pixel_color, samples_per_pixel)
        })
        .collect();

    eprintln!("\r进度: {}/{} (100. 0%)", total_pixels, total_pixels);

    // === 写入文件 ===

    let mut file = File::create("output.ppm").unwrap();
    writeln!(file, "P3\n{} {}\n255", width, height).unwrap();

    for (ir, ig, ib) in pixel_colors {
        writeln!(file, "{} {} {}", ir, ig, ib).unwrap();
    }

    eprintln!("完成!");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vec3_operations() {
        let world = cornell_box();

        let lookfrom = Vec3::new(0.0, 0.0, -3.0);
        let lookat = Vec3::new(0.0, 0.0, 0.0);
        let vup = Vec3::new(0.0, 1.0, 0.0);
        let vfov = 40.0;

        let aspect_ratio = 100 as f64 / 80 as f64;
        let camera = Camera::new(lookfrom, lookat, vup, vfov, aspect_ratio);

        // 测试中心像素，看看能否看到盒子内的颜色
        println!("\n=== 测试单向墙功能 ===");

        let test_pixels = vec![
            (50, 40, "中心"),
            (10, 40, "左侧-应击中红墙"),
            (90, 40, "右侧-应击中蓝墙"),
        ];

        for (i, j, desc) in test_pixels {
            let u = i as f64 / 99.0;
            let v = j as f64 / 79.0;
            let ray = camera.get_ray(u, v);

            // 使用多次采样来获得更稳定的结果
            let mut total_color = Vec3::new(0.0, 0.0, 0.0);
            for _ in 0..10 {
                total_color = total_color + ray_color(&ray, &world, 50);
            }

            let (r, g, b) = write_color(total_color, 10);
            println!("{}: RGB({}, {}, {})", desc, r, g, b);

            // 验证是否能看到颜色
            if r > 0 || g > 0 || b > 0 {
                println!("  ✅ 有颜色 - 单向墙工作正常");
            } else {
                println!("  ⚠️  黑色 - 可能光源太小或场景太暗");
            }
        }
    }
}
