use rand::Rng;
use std::ops::{Add, Div, Mul, Sub};

#[derive(Clone, Copy, Debug)]
pub(crate) struct Vec3 {
    pub(crate) x: f64,
    pub(crate) y: f64,
    pub(crate) z: f64,
}

impl Vec3 {
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Vec3 { x, y, z }
    }

    // 向量长度
    pub fn length(&self) -> f64 {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }

    // 单位向量（方向）
    pub fn unit(&self) -> Vec3 {
        let len = self.length();
        Vec3::new(self.x / len, self.y / len, self.z / len)
    }

    // 生成随机向量 [0, 1)
    pub fn random() -> Self {
        Self::random_range(0.0, 1.0)
    }

    // 生成指定范围的随机向量
    pub fn random_range(min: f64, max: f64) -> Self {
        let mut rng = rand::thread_rng();
        Vec3::new(
            rng.gen_range(min..max),
            rng.gen_range(min..max),
            rng.gen_range(min..max),
        )
    }

    // 生成单位球内的随机点（用于漫反射）
    pub fn random_in_unit_sphere() -> Self {
        loop {
            let p = Vec3::random_range(-1.0, 1.0);
            if p.dot(&p) < 1.0 {
                return p;
            }
        }
    }

    // 生成单位半球内的随机方向（更物理准确）
    pub fn random_unit_vector() -> Self {
        Vec3::random_in_unit_sphere().unit()
    }

    // 检查向量是否接近零
    pub fn near_zero(&self) -> bool {
        let s = 1e-8;
        self.x.abs() < s && self.y.abs() < s && self.z.abs() < s
    }

    // 反射向量
    pub fn reflect(&self, n: &Vec3) -> Vec3 {
        *self - *n * (2.0 * self.dot(n))
    }

    pub fn length_squared(&self) -> f64 {
        self.x * self.x + self.y * self.y + self.z * self.z
    }

    // 折射向量
    pub fn refract(&self, n: &Vec3, etai_over_etat: f64) -> Vec3 {
        let cos_theta = ((*self * -1.0).dot(n)).min(1.0);
        let r_out_perp = (*self + *n * cos_theta) * etai_over_etat;
        let r_out_parallel = *n * (-(1.0 - r_out_perp.length_squared()).abs().sqrt());
        r_out_perp + r_out_parallel
    }

    pub fn cross(&self, other: &Vec3) -> Vec3 {
        Vec3::new(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )
    }

    // 向量点积
    pub fn dot(&self, other: &Vec3) -> f64 {
        self.x * other.x + self.y * other.y + self.z * other.z
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

impl Div<f64> for Vec3 {
    type Output = Vec3;
    fn div(self, t: f64) -> Vec3 {
        Vec3::new(self.x / t, self.y / t, self.z / t)
    }
}

// 向量数乘
impl Mul<f64> for Vec3 {
    type Output = Vec3;
    fn mul(self, t: f64) -> Vec3 {
        Vec3::new(self.x * t, self.y * t, self.z * t)
    }
}

impl Mul<&Vec3> for Vec3 {
    type Output = Vec3;

    fn mul(self, rhs: &Vec3) -> Self::Output {
        Vec3::new(self.x * rhs.x, self.y * rhs.y, self.z * rhs.z)
    }
}
