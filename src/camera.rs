use crate::ray::Ray;
use crate::vector::Vec3;

pub struct Camera {
    pub origin: Vec3,
    pub lower_left_corner: Vec3,
    pub horizontal: Vec3,
    pub vertical: Vec3,
}

impl Camera {
    pub fn new(
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

    pub fn get_ray(&self, s: f64, t: f64) -> Ray {
        Ray {
            origin: self.origin,
            direction: self.lower_left_corner + self.horizontal * s + self.vertical * t
                - self.origin,
        }
    }
}
