use crate::vector::Vec3;

/// 双边滤波器降噪
pub fn bilateral_filter(
    pixels: &[(u8, u8, u8)],
    width: usize,
    height: usize,
    spatial_sigma: f64, // 空间距离权重(推荐 2.0-3.0)
    color_sigma: f64,   // 颜色差异权重(推荐 25.0-50.0)
    radius: usize,      // 滤波半径(推荐 2-3)
) -> Vec<(u8, u8, u8)> {
    let mut result = Vec::with_capacity(pixels.len());

    for y in 0..height {
        for x in 0..width {
            let center_idx = y * width + x;
            let center_color = pixels[center_idx];

            let mut sum_r = 0.0;
            let mut sum_g = 0.0;
            let mut sum_b = 0.0;
            let mut total_weight = 0.0;

            // 遍历邻域
            for dy in -(radius as i32)..=(radius as i32) {
                for dx in -(radius as i32)..=(radius as i32) {
                    let ny = (y as i32 + dy).clamp(0, height as i32 - 1) as usize;
                    let nx = (x as i32 + dx).clamp(0, width as i32 - 1) as usize;
                    let neighbor_idx = ny * width + nx;
                    let neighbor_color = pixels[neighbor_idx];

                    // 1. 计算空间距离权重(高斯)
                    let spatial_dist = ((dx * dx + dy * dy) as f64).sqrt();
                    let spatial_weight = (-spatial_dist * spatial_dist
                        / (2.0 * spatial_sigma * spatial_sigma))
                        .exp();

                    // 2. 计算颜色差异权重
                    let color_diff = color_distance(center_color, neighbor_color);
                    let color_weight =
                        (-color_diff * color_diff / (2.0 * color_sigma * color_sigma)).exp();

                    // 3. 组合权重
                    let weight = spatial_weight * color_weight;

                    sum_r += neighbor_color.0 as f64 * weight;
                    sum_g += neighbor_color.1 as f64 * weight;
                    sum_b += neighbor_color.2 as f64 * weight;
                    total_weight += weight;
                }
            }

            // 归一化
            let filtered_r = (sum_r / total_weight).clamp(0.0, 255.0) as u8;
            let filtered_g = (sum_g / total_weight).clamp(0.0, 255.0) as u8;
            let filtered_b = (sum_b / total_weight).clamp(0.0, 255.0) as u8;

            result.push((filtered_r, filtered_g, filtered_b));
        }
    }

    result
}

/// 计算两个颜色的欧式距离
#[inline]
fn color_distance(c1: (u8, u8, u8), c2: (u8, u8, u8)) -> f64 {
    let dr = c1.0 as f64 - c2.0 as f64;
    let dg = c1.1 as f64 - c2.1 as f64;
    let db = c1.2 as f64 - c2.2 as f64;
    (dr * dr + dg * dg + db * db).sqrt()
}
