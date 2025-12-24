use rand::Rng;
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

// 材质特征
trait Material {
    // 返回是否散射、散射光线、颜色衰减
    fn scatter(
        &self,
        ray_in: &Ray,           // 入射光线
        rec: &HitRecord,        // 碰撞记录
        attenuation: &mut Vec3, // 颜色衰减（输出）
        scattered: &mut Ray,    // 散射光线（输出）
    ) -> bool; // 是否发生散射
}

use std::rc::Rc;

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
    ) -> bool {
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

        true // 总是发生散射
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
    ) -> bool {
        let reflected = ray_in.direction.unit().reflect(&rec.normal);

        // 添加模糊（在反射方向周围随机偏移）
        *scattered = Ray {
            origin: rec.point,
            direction: reflected + Vec3::random_in_unit_sphere() * self.fuzz,
        };

        *attenuation = self.albedo;

        // 只有反射方向向外时才散射
        scattered.direction.dot(&rec.normal) > 0.0
    }
}

#[derive(Clone)]
struct HitRecord {
    point: Vec3,
    normal: Vec3,
    t: f64,
    front_face: bool,                            // 是否从外部击中
    material: Option<std::rc::Rc<dyn Material>>, // 材质
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
trait Hittable {
    fn hit(&self, ray: &Ray, t_min: f64, t_max: f64, rec: &mut HitRecord) -> bool;
}

struct Sphere {
    center: Vec3,
    radius: f64,
    material: Rc<dyn Material>,
}

impl Sphere {
    fn new(center: Vec3, radius: f64, material: Rc<dyn Material>) -> Self {
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

// 检测光线是否击中球体
fn hit_sphere(center: &Vec3, radius: f64, ray: &Ray) -> f64 {
    let oc = ray.origin - *center;
    let a = ray.direction.dot(&ray.direction);
    let b = 2.0 * oc.dot(&ray.direction);
    let c = oc.dot(&oc) - radius * radius;
    let discriminant = b * b - 4.0 * a * c;

    if discriminant < 0.0 {
        -1.0 // 没碰到，返回 -1
    } else {
        // 返回最近的交点距离
        (-b - discriminant.sqrt()) / (2.0 * a)
    }
}

fn ray_color(ray: &Ray, world: &dyn Hittable, depth: i32) -> Vec3 {
    if depth <= 0 {
        return Vec3::new(0.0, 0.0, 0.0);
    }

    let mut rec = HitRecord::new();

    if world.hit(ray, 0.001, f64::INFINITY, &mut rec) {
        let mut scattered = Ray {
            origin: Vec3::new(0.0, 0.0, 0.0),
            direction: Vec3::new(0.0, 0.0, 0.0),
        };
        let mut attenuation = Vec3::new(0.0, 0.0, 0.0);

        // 使用材质的 scatter 方法
        if let Some(material) = &rec.material {
            if material.scatter(ray, &rec, &mut attenuation, &mut scattered) {
                // 递归追踪散射光线，乘以衰减系数
                return ray_color(&scattered, world, depth - 1) * attenuation;
            }
        }

        return Vec3::new(0.0, 0.0, 0.0); // 光线被吸收
    }

    // 背景
    let unit_direction = ray.direction.unit();
    let t = 0.5 * (unit_direction.y + 1.0);
    Vec3::new(1.0, 1.0, 1.0) * (1.0 - t) + Vec3::new(0.5, 0.7, 1.0) * t
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

fn main() {
    // 图片设置
    let width = 400;
    let height = 225;
    let aspect_ratio = width as f64 / height as f64;

    // 相机设置
    let viewport_height = 2.0;
    let viewport_width = viewport_height * aspect_ratio;
    let focal_length = 1.0;

    let origin = Vec3::new(0.0, 0.0, 0.0);
    let horizontal = Vec3::new(viewport_width, 0.0, 0.0);
    let vertical = Vec3::new(0.0, viewport_height, 0.0);
    let lower_left = origin - horizontal * 0.5 - vertical * 0.5 - Vec3::new(0.0, 0.0, focal_length);

    let material_ground = Rc::new(Lambertian::new(Vec3::new(0.8, 0.8, 0.0)));
    let material_center = Rc::new(Lambertian::new(Vec3::new(0.7, 0.3, 0.3)));
    let material_left = Rc::new(Metal::new(Vec3::new(0.8, 0.8, 0.8), 0.3));
    let material_right = Rc::new(Metal::new(Vec3::new(0.8, 0.6, 0.2), 1.0));

    // 创建场景
    let mut world = HittableList::new();

    // 添加小球（中心的球）
    world.add(Box::new(Sphere::new(
        Vec3::new(0.0, 0.0, -1.0),
        0.5,
        material_center,
    )));
    // 添加左侧球体
    world.add(Box::new(Sphere::new(
        Vec3::new(-1.5, 0.5, -1.5),
        0.5,
        material_left,
    )));
    // 添加右侧球体
    world.add(Box::new(Sphere::new(
        Vec3::new(1.0, 0.0, -1.5),
        0.5,
        material_right,
    )));
    // 添加地面（一个巨大的球）
    world.add(Box::new(Sphere::new(
        Vec3::new(0.0, -100.5, -1.0),
        100.0,
        material_ground,
    )));

    // 渲染
    let mut file = File::create("output.ppm").unwrap();
    writeln!(file, "P3\n{} {}\n255", width, height).unwrap();

    let max_depth = 100;
    let samples_per_pixel = 100;

    let mut rng = rand::thread_rng();

    for j in (0..height).rev() {
        for i in 0..width {
            let mut pixel_color = Vec3::new(0.0, 0.0, 0.0); // 累加颜色

            // 对当前像素进行多次采样
            for _ in 0..samples_per_pixel {
                // 在像素内随机偏移
                let random_u: f64 = rng.gen_range(0.0..1.0);
                let random_v: f64 = rng.gen_range(0.0..1.0);

                let u = (i as f64 + random_u) / (width - 1) as f64;
                let v = (j as f64 + random_v) / (height - 1) as f64;

                let direction = lower_left + horizontal * u + vertical * v - origin;
                let ray = Ray { origin, direction };

                // 累加颜色
                pixel_color = pixel_color + ray_color(&ray, &world, max_depth);
            }

            // 计算平均颜色并写入文件
            let (ir, ig, ib) = write_color(pixel_color, samples_per_pixel);
            writeln!(file, "{} {} {}", ir, ig, ib).unwrap();
        }
        eprint!("\r进度: {}/{}", height - j, height);
    }

    eprintln!("\n完成!");
}
