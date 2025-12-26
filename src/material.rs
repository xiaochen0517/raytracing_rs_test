// 材质特征
pub trait Material: Sync + Send {
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

    // (采样点，法线)
    fn sample_point(&self) -> (Vec3, Vec3) {
        (Vec3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0)) // 默认返回原点
    }

    fn power(&self) -> f64 {
        0.0 // 默认无功率
    }
}

use crate::ray::Ray;
use crate::vector::Vec3;
use crate::{HitRecord, ScatterResult};
use rand::Rng;
use std::sync::Arc;

pub struct DiffuseLight {
    pub emit: Vec3, // 发光颜色和强度
}

impl DiffuseLight {
    pub fn new(emit: Vec3) -> Self {
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

    fn power(&self) -> f64 {
        self.emit.x + self.emit.y + self.emit.z // 光源强度
    }
}

pub struct Lambertian {
    pub albedo: Vec3, // 反射率（材质颜色）
}

impl Lambertian {
    pub fn new(albedo: Vec3) -> Self {
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

pub struct Metal {
    pub albedo: Vec3,
    pub fuzz: f64, // 模糊度 [0, 1]，0=完美镜面
}

impl Metal {
    pub fn new(albedo: Vec3, fuzz: f64) -> Self {
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

pub struct Dielectric {
    pub ir: f64, // 折射率（Index of Refraction）
}

impl Dielectric {
    pub fn new(ir: f64) -> Self {
        Dielectric { ir }
    }

    // Schlick 近似：计算菲涅尔反射率
    pub fn reflectance(cosine: f64, ref_idx: f64) -> f64 {
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

pub struct OneWayMaterial {
    reflect_material: Arc<dyn Material>, // 从内部击中时使用的材质
}

impl OneWayMaterial {
    pub fn new(reflect_material: Arc<dyn Material>) -> Self {
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
