mod camera;
mod hitrecord;
mod light;
mod material;
mod ray;
mod vector;

use crate::ray::Ray;
use crate::vector::Vec3;
use rand::Rng;
use rayon::prelude::*;
use std::ops::{Add, Div, Mul, Sub};

use crate::camera::Camera;
use crate::hitrecord::{
    HitRecord, Hittable, HittableList, ScatterResult, Sphere, XYRect, XZRect, YZRect,
};
use crate::light::Light;
use crate::material::{DiffuseLight, Lambertian, Material, Metal, OneWayMaterial};
use std::fs::File;
use std::io::Write;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

fn sample_direct_light(rec: &HitRecord, world: &dyn Hittable, lights: &[Arc<dyn Light>]) -> Vec3 {
    if lights.is_empty() {
        return Vec3::new(0.0, 0.0, 0.0); // 没有光源，返回黑色
    }

    // 遍历所有光源，累加贡献
    let mut total_light = Vec3::new(0.0, 0.0, 0.0);

    for light in lights {
        // 1. 从光源采样一个点
        let (light_point, light_normal) = light.sample();
        // 2. 计算碰撞点到光源点的方向和距离
        let to_light = light_point - rec.point;
        let distance_to_light = to_light.length();
        let direction_to_light = to_light.unit();
        // 3. 检查是否是背面光照
        let cos_surface = rec.normal.dot(&direction_to_light);
        if cos_surface <= 0.0 {
            continue; // 背面光照，跳过
        }
        // 4. 检查光线是否被遮挡
        let shadow_ray = Ray {
            origin: rec.point,
            direction: direction_to_light,
        };
        let mut shadow_rec = HitRecord::new();
        if world.hit(
            &shadow_ray,
            0.001,
            distance_to_light - 0.001,
            &mut shadow_rec,
        ) {
            continue; // 被遮挡，跳过
        }
        // 5. 计算光源的发光强度
        let light_power = light.emitted(0.0, 0.0, &light_point);
        // 6. 计算光源面积和 PDF
        let light_area = light.area();
        let pdf = 1.0 / light_area; // 均匀采样概率密度函数
        // 7. 计算直接光照贡献 L = Le * cos(theta) / (distance^2 * pdf)
        let direct_light =
            light_power * cos_surface / (distance_to_light * distance_to_light * pdf);
        total_light = total_light + direct_light;
    }

    total_light
}

fn ray_color(ray: &Ray, world: &dyn Hittable, lights: &[Arc<dyn Light>], depth: i32) -> Vec3 {
    if !ray.direction.x.is_finite() || !ray.direction.y.is_finite() || !ray.direction.z.is_finite()
    {
        // eprintln!("光线方向包含非有限值: {:?}", ray.direction);
        return Vec3::new(0.0, 0.0, 0.0);
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
            ScatterResult::Transmitted => emitted + ray_color(&scattered, world, lights, depth - 1),
            ScatterResult::Scattered => {
                let direct_light = sample_direct_light(&rec, world, &lights);
                let indirect_light = ray_color(&scattered, world, lights, depth - 1);
                emitted + (direct_light + indirect_light) * attenuation
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
    samples_per_pixel: 300,
    max_depth: 30,
};

const DEBUG_CONFIG: Config = Config {
    image_width: 300,
    image_height: 200,
    samples_per_pixel: 20,
    max_depth: 10,
};

const BOX_SIZE: f64 = 2.0;

fn cornell_box() -> (HittableList, Vec<Arc<dyn Light>>) {
    let mut world = HittableList::new();

    let mut lights: Vec<Arc<dyn Light>> = Vec::new();

    // 创建材质
    let red = Arc::new(Lambertian::new(Vec3::new(0.65, 0.05, 0.05)));
    let white = Arc::new(Lambertian::new(Vec3::new(0.73, 0.73, 0.73)));
    let green = Arc::new(Lambertian::new(Vec3::new(0.12, 0.45, 0.15)));
    let blue = Arc::new(Lambertian::new(Vec3::new(0.05, 0.05, 0.65)));
    let light = Arc::new(DiffuseLight::new(Vec3::new(10.0, 10.0, 10.0)));

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
        -0.25,
        0.25, // x 范围：中央 0.5 宽
        -0.25,
        0.25, // z 范围：中央 0.5 宽
        0.99, // y 坐标：略低于天花板
        light.clone(),
    )));

    let light_rect = Arc::new(XZRect::new(
        -0.25, 0.25, // x 范围：中央 0.5 宽
        -0.25, 0.25, // z 范围：中央 0.5 宽
        0.99, // y 坐标：略低于天花板
        light,
    ));
    lights.push(light_rect);

    (world, lights)
}

fn main() {
    // let config = DEBUG_CONFIG;
    let config = RELEASE_CONFIG;
    let width = config.image_width;
    let height = config.image_height;

    let max_depth = config.max_depth;
    let samples_per_pixel = config.samples_per_pixel;

    // 图片设置
    let aspect_ratio = width as f64 / height as f64;

    // 创建场景
    let (mut world, lights) = cornell_box();

    // 添加金属球体
    world.add(Box::new(Sphere::new(
        Vec3::new(0.0, -0.5 + 0.2, 0.5),
        0.2,
        Arc::new(Metal::new(Vec3::new(0.8, 0.6, 0.2), 0.1)),
    )));

    // 添加玻璃球体
    world.add(Box::new(Sphere::new(
        Vec3::new(0.0, -0.2, -1.0),
        0.3,
        Arc::new(material::Dielectric::new(1.5)),
    )));

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
                pixel_color = pixel_color + ray_color(&ray, &world, &lights, max_depth);
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
        let (world, lights) = cornell_box();

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
                total_color = total_color + ray_color(&ray, &world, &lights, 50);
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
